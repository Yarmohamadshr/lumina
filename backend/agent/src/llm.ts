import Anthropic from '@anthropic-ai/sdk';
import { secrets } from './env.js';

/** The one Claude client for this process. The key never leaves env.ts and this line. */
export const llm = new Anthropic({ apiKey: secrets.anthropic, maxRetries: 2 });

/**
 * claude-sonnet-5 list prices, USD per million tokens. Cached input is the whole point of the
 * caching in loop.ts: writing the cache costs a little more than fresh input, reading it costs a
 * tenth. Counting cached reads at full price would make the run log overstate what we actually pay.
 */
const USD_PER_MTOK = { in: 2, out: 10, cacheWrite: 2.5, cacheRead: 0.2 };

export type TokenUse = { in: number; out: number; cacheWrite?: number; cacheRead?: number };

export function costUsd(t: TokenUse): number {
  return (
    (t.in * USD_PER_MTOK.in +
      t.out * USD_PER_MTOK.out +
      (t.cacheWrite ?? 0) * USD_PER_MTOK.cacheWrite +
      (t.cacheRead ?? 0) * USD_PER_MTOK.cacheRead) /
    1_000_000
  );
}
