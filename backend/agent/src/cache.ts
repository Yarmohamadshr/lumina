import { createHash } from 'node:crypto';
import { COLLECTIONS, type SearchCacheDoc } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { webSearch, type SearchHit } from './search.js';

/** Tier 1: this process's memory. Small, instant, gone on restart. */
const LRU_MAX = 200;
const lru = new Map<string, { hits: SearchHit[]; expiresAt: number }>();

const normalize = (query: string) => query.trim().toLowerCase().replace(/\s+/g, ' ');

/** sha256 of (normalized query + provider): same question, same provider, same row. */
const keyFor = (query: string) => createHash('sha256').update(`${normalize(query)}|${env.searchProvider}`).digest('hex');

/**
 * "What happened today", "the latest release", "in 2026" — answers that go stale in hours.
 * These skip the cache in both directions: a slow fresh answer beats a fast wrong one (DESIGN Q4).
 */
export function isTimeSensitive(query: string): boolean {
  const year = new Date().getFullYear();
  return new RegExp(`\\b(today|todays|tonight|now|current|currently|latest|newest|breaking|this (week|month|year)|${year}|${year + 1})\\b`, 'i').test(
    query
  );
}

function lruGet(key: string): SearchHit[] | null {
  const entry = lru.get(key);
  if (!entry) return null;
  if (entry.expiresAt <= Date.now()) {
    lru.delete(key);
    return null;
  }
  lru.delete(key); // re-insert so the most recently used entry is last
  lru.set(key, entry);
  return entry.hits;
}

function lruSet(key: string, hits: SearchHit[], expiresAt: number): void {
  lru.set(key, { hits, expiresAt });
  if (lru.size > LRU_MAX) lru.delete(lru.keys().next().value as string); // oldest out
}

/**
 * web_search with its two-tier cache: memory → Mongo → Tavily.
 * `cached` is what the done event's `searchCached` is built from.
 */
export async function cachedWebSearch(query: string): Promise<{ hits: SearchHit[]; cached: boolean }> {
  if (isTimeSensitive(query)) return { hits: await webSearch(query), cached: false };

  const key = keyFor(query);
  const fromLru = lruGet(key);
  if (fromLru) return { hits: fromLru, cached: true };

  const collection = (await db()).collection<SearchCacheDoc>(COLLECTIONS.searchCache);
  const row = await collection.findOne({ _id: key });
  // Mongo's TTL sweeper runs about once a minute, so an expired row can still be here.
  if (row && new Date(row.expiresAt).getTime() > Date.now()) {
    const hits = row.results as unknown as SearchHit[];
    lruSet(key, hits, new Date(row.expiresAt).getTime());
    return { hits, cached: true };
  }

  const hits = await webSearch(query); // a provider error throws: never cached, never hidden
  const expiresAt = new Date(Date.now() + env.searchCacheTtlSeconds * 1000);
  lruSet(key, hits, expiresAt.getTime());
  // Not awaited: the answer does not need the cache row, and the write was ~100 ms on the TTFT path.
  // A failed write only means a future miss, so it is logged, never swallowed silently.
  collection
    .updateOne(
      { _id: key },
      { $set: { provider: env.searchProvider, query: normalize(query), results: hits, expiresAt, createdAt: new Date() } },
      { upsert: true }
    )
    .catch((err: unknown) => console.error('searchCache write failed', (err as Error).message));
  return { hits, cached: false };
}
