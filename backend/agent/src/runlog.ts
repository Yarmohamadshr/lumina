import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { COLLECTIONS, RunDoc, RunLog } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';

/**
 * How many deep searches this user has run today. Counted from the run logs themselves, not from a
 * separate counter (DESIGN.md Q4): one source of truth, and it survives a restart, which a variable
 * in memory does not.
 */
export async function deepRunsToday(userId: string): Promise<number> {
  const since = new Date();
  since.setHours(0, 0, 0, 0);
  return (await db())
    .collection(COLLECTIONS.runs)
    .countDocuments({ userId, depth: 'deep', createdAt: { $gte: since.toISOString() } });
}

/**
 * Save one run log, written once per answer, including failed ones.
 *   runs/<requestId>.json : exactly the RunLog shape quality/check.mjs reads
 *   Mongo `runs`          : the same plus ids and query; the deep daily cap counts from here
 * Both are validated against the contract first, so a malformed log fails here, not in a gate.
 */
export async function saveRun(run: RunDoc): Promise<void> {
  const doc = RunDoc.parse(run);
  const file = RunLog.parse(doc); // zod drops the extra fields → the exact file shape
  await Promise.all([
    writeFile(join(env.runsDir, `${doc.requestId}.json`), JSON.stringify(file, null, 2)),
    (await db()).collection(COLLECTIONS.runs).insertOne({ ...doc })
  ]);
}
