import OpenAI from 'openai';
import { EMBEDDING_DIMS } from '@lumina/contract';
import { env, secrets } from './env.js';

const openai = new OpenAI({ apiKey: secrets.openai, maxRetries: 2 });

/**
 * Turn text into an embedding: a list of 1536 numbers that captures its MEANING, so two ways
 * of saying the same thing land near each other. That is what makes recall semantic —
 * "how do I call Tavily?" can find "I want code examples, not prose" without sharing a word.
 * Errors are not caught: a provider failure must fail loud, never return a zero vector.
 */
export async function embed(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({ model: env.embeddingModel, input: text });
  const vector = res.data[0]?.embedding;
  if (!vector || vector.length !== EMBEDDING_DIMS) {
    throw new Error(`embedding failed: expected ${EMBEDDING_DIMS} dims, got ${vector?.length ?? 0}`);
  }
  return vector;
}

/**
 * Many texts in ONE call: a 60-page PDF is ~150 chunks, and 150 separate calls would be slow
 * and rate-limited. OpenAI returns one vector per input; `index` says which, so re-sort by it.
 * Same rule as embed(): any missing or wrong-sized vector fails the whole batch, loudly.
 */
export async function embedMany(texts: string[]): Promise<number[][]> {
  if (!texts.length) return [];
  const res = await openai.embeddings.create({ model: env.embeddingModel, input: texts });
  const vectors = [...res.data].sort((a, b) => a.index - b.index).map((d) => d.embedding);
  if (vectors.length !== texts.length || vectors.some((v) => v?.length !== EMBEDDING_DIMS)) {
    throw new Error(`embedding failed: expected ${texts.length} vectors of ${EMBEDDING_DIMS} dims`);
  }
  return vectors;
}
