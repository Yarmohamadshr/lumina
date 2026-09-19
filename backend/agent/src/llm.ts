import Anthropic from '@anthropic-ai/sdk';
import { secrets } from './env.js';

/** The one Claude client for this process. The key never leaves env.ts and this line. */
export const llm = new Anthropic({ apiKey: secrets.anthropic, maxRetries: 2 });

/**
 * claude-sonnet-5 list prices, USD per million tokens. Cached input is the whole point of the
 * caching in loop.ts: writing the cache costs a little more than fresh input, reading it costs a
 * tenth. Counting cached reads at full price would make the run log overstate what we actually pay.
 */
const SONNET = { in: 2, out: 10, cacheWrite: 2.5, cacheRead: 0.2 };
/** claude-haiku-4-5: exactly half of Sonnet 5 (quick answers and the triage run on it). */
const HAIKU = { in: 1, out: 5, cacheWrite: 1.25, cacheRead: 0.1 };
const priceOf = (model: string) => (model.includes('haiku') ? HAIKU : SONNET);

export type TokenUse = { in: number; out: number; cacheWrite?: number; cacheRead?: number };

/** USD for these tokens on this model. Unknown models are priced as Sonnet: never under-report. */
export function costUsd(t: TokenUse, model = 'claude-sonnet-5'): number {
  const p = priceOf(model);
  return (
    (t.in * p.in + t.out * p.out + (t.cacheWrite ?? 0) * p.cacheWrite + (t.cacheRead ?? 0) * p.cacheRead) / 1_000_000
  );
}
