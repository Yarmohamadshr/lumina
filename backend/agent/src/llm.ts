import Anthropic from '@anthropic-ai/sdk';
import { secrets } from './env.js';

/** The one Claude client for this process. The key never leaves env.ts and this line. */
export const llm = new Anthropic({ apiKey: secrets.anthropic, maxRetries: 2 });

/** claude-sonnet-5 list prices, USD per million tokens. */
const USD_PER_MTOK = { in: 2, out: 10 };

export function costUsd(tokens: { in: number; out: number }): number {
  return (tokens.in * USD_PER_MTOK.in + tokens.out * USD_PER_MTOK.out) / 1_000_000;
}
