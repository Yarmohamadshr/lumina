import type { Request, Response } from 'express';
import { COLLECTIONS, type RunDoc, type StatsResponse } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import type { RequestExtras } from './requests.js';

const startOfToday = () => {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
};

const p95 = (values: number[]): number => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)] ?? 0;
};

/**
 * GET /stats — computed from the `requests` rows and the run logs, never from counters kept in
 * memory: a counter drifts the first time a code path forgets to increment it, and then the
 * dashboard is confidently wrong. Today only, so the numbers mean something.
 */
export async function stats(_req: Request, res: Response): Promise<void> {
  const userId: string = res.locals.userId;
  const since = startOfToday();
  const database = await db();

  const asks = await database
    .collection<{ ttftMs?: number; searchCached?: boolean } & RequestExtras>(COLLECTIONS.requests)
    .find({ route: { $regex: '/ask$' }, createdAt: { $gte: since.toISOString() } })
    .toArray();
  const runs = await database
    .collection<RunDoc>(COLLECTIONS.runs)
    .find({ createdAt: { $gte: since.toISOString() } })
    .toArray();

  const withCacheInfo = asks.filter((r) => typeof r.searchCached === 'boolean');
  const body: StatsResponse = {
    requests: await database.collection(COLLECTIONS.requests).countDocuments({ createdAt: { $gte: since.toISOString() } }),
    answers: runs.length,
    searchCacheHitRatePct: withCacheInfo.length
      ? (withCacheInfo.filter((r) => r.searchCached).length / withCacheInfo.length) * 100
      : 0,
    ttftP95Ms: p95(asks.flatMap((r) => (typeof r.ttftMs === 'number' && r.ttftMs > 0 ? [r.ttftMs] : []))),
    costUsdToday: runs.reduce((sum, r) => sum + (r.costUsd ?? 0), 0),
    // The deep daily cap counts from the runs themselves: one source of truth (DESIGN.md Q4).
    deepToday: runs.filter((r) => r.depth === 'deep' && r.userId === userId).length,
    deepDailyCap: env.deepDailyCap
  };
  res.json(body);
}
