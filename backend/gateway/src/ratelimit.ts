import type { NextFunction, Request, Response } from 'express';
import { env } from './env.js';

const WINDOW_MS = 60_000;

/** userId → the times of their recent requests (only the last minute is kept). */
const recent = new Map<string, number[]>();

/**
 * Per-user rate limit, a sliding window of one minute → 429 with Retry-After and resetsAt.
 * Applied to the ask route: that is the one that spends money. It protects traffic, not the
 * budget; the deep daily cap (the budget) lives in the agent service, next to the spending.
 * In memory, so it is per gateway process: fine for one Fly machine.
 */
export function rateLimit(_req: Request, res: Response, next: NextFunction): void {
  const userId: string = res.locals.userId;
  const now = Date.now();
  const times = (recent.get(userId) ?? []).filter((t) => now - t < WINDOW_MS);

  if (times.length >= env.rateLimitPerMinute) {
    const resetsAtMs = (times[0] ?? now) + WINDOW_MS; // when the oldest request leaves the window
    res.setHeader('Retry-After', String(Math.ceil((resetsAtMs - now) / 1000)));
    res.status(429).json({
      error: `rate limit: ${env.rateLimitPerMinute} asks per minute`,
      status: 429,
      resetsAt: new Date(resetsAtMs).toISOString(),
      requestId: String(res.locals.requestId)
    });
    recent.set(userId, times);
    return;
  }

  times.push(now);
  recent.set(userId, times);
  next();
}
