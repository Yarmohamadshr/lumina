import type { Request, Response } from 'express';
import pino from 'pino';
import { AskBody, DoneEvent, newId, SourcesEvent, ThreadId, unresolvedCitations, type SseEventName } from '@lumina/contract';
import { env } from './env.js';
import { costUsd } from './llm.js';
import { research } from './loop.js';
import { buildSources, streamAnswer } from './answer.js';
import { sseHeaders, sseSend } from './sse.js';

const log = pino({ level: env.logLevel });

/**
 * POST /threads/:threadId/ask — streams  trace* → sources → token* → done.
 * Any provider failure ends the run with an `error` event (or a real 502 if nothing was sent yet).
 */
export async function ask(req: Request, res: Response): Promise<void> {
  const started = Date.now();

  // Validate here too, not only in the gateway (defense in depth, DESIGN.md Q2).
  const thread = ThreadId.safeParse(req.params.threadId);
  const body = AskBody.safeParse(req.body);
  if (!thread.success) {
    res.status(400).json({ error: 'threadId must look like thr_…', status: 400 });
    return;
  }
  if (!body.success) {
    res.status(400).json({ error: body.error.issues.map((i) => i.message).join('; '), status: 400 });
    return;
  }
  const { query, depth } = body.data;

  // Open the stream lazily: if the very first provider call fails, we can still answer a real 502.
  let streaming = false;
  const send = (event: SseEventName, data: unknown) => {
    if (!streaming) {
      sseHeaders(res);
      streaming = true;
    }
    sseSend(res, event, data);
  };

  try {
    // 1. research: one trace event per tool call
    const found = await research(query, depth, (ev) => send('trace', ev));

    // 2. sources, BEFORE the first token (validated against the contract)
    const sources = SourcesEvent.parse(buildSources(found.pages, query));
    send('sources', sources);

    // 3. the answer, token by token
    let ttftMs = 0;
    const answer = await streamAnswer(query, found.pages, (text) => {
      if (!ttftMs) ttftMs = Date.now() - started;
      send('token', { text });
    });

    // 4. every [n] must resolve to a source retrieved in THIS request
    const dangling = unresolvedCitations(answer.text, sources);
    if (dangling.length) throw new Error(`answer cited [${dangling.join('], [')}] but no such source was retrieved`);

    // 5. done
    const tokens = { in: found.tokens.in + answer.tokens.in, out: found.tokens.out + answer.tokens.out };
    send(
      'done',
      DoneEvent.parse({
        answerId: newId('ans'),
        latencyMs: Date.now() - started,
        ttftMs,
        model: env.llmModel,
        tokens,
        costUsd: costUsd(tokens),
        searchCached: false,
        terminated: found.terminated,
        depth
      })
    );
    res.end();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error({ err, threadId: thread.data }, 'ask failed');
    if (!streaming) {
      res.status(502).json({ error: message, status: 502 });
      return;
    }
    send('error', { status: 502, error: message });
    res.end();
  }
}
