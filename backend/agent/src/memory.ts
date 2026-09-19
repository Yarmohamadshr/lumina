import { randomUUID } from 'node:crypto';
import type { Request, Response } from 'express';
import { COLLECTIONS, SEARCH_INDEXES, type ListMemoryResponse, type MemoryDoc } from '@lumina/contract';
import { db } from './db.js';
import { embed } from './embed.js';

const memories = async () => (await db()).collection<MemoryDoc>(COLLECTIONS.memories);

/** What recall injects at most: enough to be useful, small enough to stay cheap (doc 08). */
const RECALL_LIMIT = 5;

/** save_memory: a durable fact or preference about this user. */
export async function saveMemory(userId: string, text: string, sourceThread?: string): Promise<string> {
  const _id = `mem_${randomUUID()}`;
  await (await memories()).insertOne({
    _id,
    userId,
    text: text.trim(),
    embedding: await embed(text),
    ...(sourceThread ? { sourceThread } : {}),
    createdAt: new Date()
  });
  return _id;
}

/**
 * recall_memory: semantic search over THIS user's memories.
 *
 * `filter: { userId }` sits INSIDE $vectorSearch, never in a later $match. A $match afterwards
 * searches everyone's memories first and then throws away what is not yours, so the top-100
 * candidates can all belong to other people and you get zero results — while also having read
 * their data. Same mistake, same cost, as the RAG one that makes recall@5 = 0.
 */
export async function recallMemory(userId: string, query: string, limit = RECALL_LIMIT): Promise<MemoryDoc[]> {
  const col = await memories();
  const [found, fresh] = await Promise.all([
    col
      .aggregate<MemoryDoc>([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.memoriesVector,
            path: 'embedding',
            queryVector: await embed(query),
            numCandidates: 100,
            limit,
            filter: { userId }
          }
        },
        { $project: { embedding: 0 } } // 1536 numbers we do not need back
      ])
      .toArray(),
    // Read-your-write, as with RAG's probe: the vector index lags a new memory by a few seconds. A
    // preference saved in thread A and asked about in thread B 3 s later was NOT recalled (the answer
    // came back in Java for a Python user). So the last few minutes are read straight from the
    // collection, which has no lag, and merged in.
    col
      .find({ userId, createdAt: { $gte: new Date(Date.now() - FRESH_MS) } }, { projection: { embedding: 0 } })
      .sort({ createdAt: -1 })
      .limit(limit)
      .toArray()
  ]);
  const seen = new Set(found.map((m) => m._id));
  return [...found, ...fresh.filter((m) => !seen.has(m._id))].slice(0, limit + 2);
}

/** How far back a memory counts as "maybe not indexed yet". */
const FRESH_MS = 10 * 60_000;

/** GET /memory — everything this user has stored, newest first. */
export async function listMemory(_req: Request, res: Response): Promise<void> {
  const rows = await (await memories())
    .find({ userId: res.locals.userId }, { projection: { embedding: 0 } })
    .sort({ createdAt: -1 })
    .limit(200)
    .toArray();
  const body: ListMemoryResponse = {
    memories: rows.map((m) => ({
      id: m._id,
      text: m.text,
      ...(m.sourceThread ? { sourceThread: m.sourceThread } : {}),
      createdAt: new Date(m.createdAt).toISOString()
    }))
  };
  res.json(body);
}

/** DELETE /memory/:memoryId — 204, or 404 if it is not this user's. */
export async function deleteMemory(req: Request, res: Response): Promise<void> {
  const result = await (await memories()).deleteOne({ _id: req.params.memoryId ?? '', userId: res.locals.userId });
  if (!result.deletedCount) {
    res.status(404).json({ error: 'memory not found', status: 404 });
    return;
  }
  res.status(204).end();
}
