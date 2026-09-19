import type { Request, Response } from 'express';
import pino from 'pino';
import { REQUEST_HEADER, USER_HEADER } from '@lumina/contract';
import { env } from './env.js';
import { sseHeaders } from './sse.js';

const log = pino({ level: env.logLevel });

/**
 * Forward this request to the agent service and send its answer back unchanged.
 *  - JSON routes: same status, same body.
 *  - The ask stream: every chunk is written to the browser the moment it arrives. No
 *    buffering, no re-parsing of SSE frames, no compression, or TTFT fails for no visible reason.
 *  - Agent unreachable → 502. Never a 2xx when the upstream failed.
 *  - Browser disconnects → the upstream request is aborted, so the agent can stop working.
 */
export async function proxy(req: Request, res: Response): Promise<void> {
  const requestId = String(res.locals.requestId);
  const upstreamAbort = new AbortController();
  res.on('close', () => upstreamAbort.abort()); // tab closed (or response finished): stop upstream

  const headers: Record<string, string> = { [REQUEST_HEADER]: requestId };
  if (res.locals.userId) headers[USER_HEADER] = res.locals.userId;
  const hasBody = req.method !== 'GET' && req.method !== 'DELETE';
  // An upload is multipart: pass the raw bytes through untouched (the agent's multer parses
  // them). Everything else was already parsed and validated here, so it is re-sent as JSON.
  const incomingType = req.header('content-type') ?? '';
  const multipart = hasBody && incomingType.startsWith('multipart/');
  if (multipart) headers['content-type'] = incomingType;
  else if (hasBody) headers['content-type'] = 'application/json';

  let upstream: globalThis.Response;
  try {
    upstream = await fetch(`${env.agentUrl}${req.originalUrl}`, {
      method: req.method,
      headers,
      body: multipart ? (req as unknown as ReadableStream) : hasBody ? JSON.stringify(req.body ?? {}) : undefined,
      // Required by Node's fetch whenever the body is a stream.
      ...(multipart ? { duplex: 'half' } : {}),
      signal: upstreamAbort.signal
    } as RequestInit);
  } catch (err) {
    log.error({ err, requestId }, 'agent unreachable');
    res.status(502).json({ error: `agent service unreachable: ${(err as Error).message}`, status: 502, requestId });
    return;
  }

  const type = upstream.headers.get('content-type') ?? '';

  // ---- the stream: pass each chunk straight through
  if (type.startsWith('text/event-stream') && upstream.body) {
    res.status(upstream.status);
    sseHeaders(res);
    const reader = upstream.body.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(value);
      }
    } catch (err) {
      // The browser left (we aborted) or the agent died mid-stream. Headers are gone, so the
      // only honest thing left is to close; the agent's own run log records what happened.
      if (!upstreamAbort.signal.aborted) log.error({ err, requestId }, 'stream from agent broke');
    }
    res.end();
    return;
  }

  // ---- everything else: same status, same body
  const body = await upstream.text();
  res.status(upstream.status);
  if (type) res.setHeader('content-type', type);
  res.send(body);
}
