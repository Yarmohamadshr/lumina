import type { Response } from 'express';
import type { SseEventName } from '@lumina/contract';

/**
 * Same helper as the gateway's. Headers that stop proxies buffering the stream, and a flush
 * after every frame, or tokens arrive all at once at the end and TTFT fails for no visible reason.
 * Never put compression() in front of the ask route.
 */
export function sseHeaders(res: Response): void {
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
}

/** One SSE frame. The blank line at the end is what tells the client the frame is complete. */
export function sseSend(res: Response, event: SseEventName, data: unknown): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}
