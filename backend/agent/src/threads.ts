import type { NextFunction, Request, Response } from 'express';
import {
  COLLECTIONS,
  CreateThreadBody,
  newId,
  ThreadId,
  USER_HEADER,
  type GetThreadResponse,
  type ListThreadsResponse,
  type MessageDoc,
  type ThreadDoc
} from '@lumina/contract';
import { db } from './db.js';

const DEFAULT_TITLE = 'New thread';

const threads = async () => (await db()).collection<ThreadDoc>(COLLECTIONS.threads);
export const messages = async () => (await db()).collection<MessageDoc>(COLLECTIONS.messages);

const iso = (d: string | Date) => new Date(d).toISOString();

/** 401 without X-User-Id. The gateway checks this too; the agent does not trust that it did. */
export function requireUser(req: Request, res: Response, next: NextFunction): void {
  const userId = req.header(USER_HEADER)?.trim();
  if (!userId) {
    res.status(401).json({ error: `missing ${USER_HEADER} header`, status: 401 });
    return;
  }
  res.locals.userId = userId;
  next();
}

/**
 * The thread, only if it belongs to this user. Someone else's thread and a missing thread
 * both come back null, so a 404 never reveals that another user's thread exists.
 */
export async function findThread(threadId: string, userId: string): Promise<ThreadDoc | null> {
  if (!ThreadId.safeParse(threadId).success) return null;
  return (await threads()).findOne({ _id: threadId, userId });
}

/**
 * The last few turns of a thread, as Claude message objects, so a follow-up question makes sense
 * on its own. Bounded on purpose (DESIGN.md Q4): the full thread stays in Mongo, but sending 40
 * turns to Claude on every ask would blow both the token budget and the cost SLA.
 */
const HISTORY_TURNS = 6;

export async function recentHistory(threadId: string): Promise<{ role: 'user' | 'assistant'; content: string }[]> {
  const rows = await (await messages())
    .find({ threadId }, { projection: { role: 1, content: 1, createdAt: 1 } })
    .sort({ createdAt: -1 })
    .limit(HISTORY_TURNS)
    .toArray();
  return rows
    .reverse() // newest-first from Mongo, oldest-first for Claude
    .filter((m) => m.content.trim())
    .map((m) => ({ role: m.role, content: m.content }));
}

/** A new thread's title is set from its first question, so the sidebar is readable. */
export async function titleFromFirstQuestion(threadId: string, query: string): Promise<void> {
  await (await threads()).updateOne({ _id: threadId, title: DEFAULT_TITLE }, { $set: { title: query.slice(0, 80) } });
}

/** POST /threads → 201 { threadId } */
export async function createThread(req: Request, res: Response): Promise<void> {
  const body = CreateThreadBody.safeParse(req.body ?? {});
  if (!body.success) {
    res.status(400).json({ error: body.error.issues.map((i) => i.message).join('; '), status: 400 });
    return;
  }
  const threadId = newId('thr');
  await (await threads()).insertOne({
    _id: threadId,
    userId: res.locals.userId,
    title: body.data.title ?? DEFAULT_TITLE,
    createdAt: new Date()
  });
  res.status(201).json({ threadId });
}

/** GET /threads → this user's threads, newest first. */
export async function listThreads(_req: Request, res: Response): Promise<void> {
  const rows = await (await threads())
    .find({ userId: res.locals.userId })
    .sort({ createdAt: -1 })
    .limit(100)
    .toArray();
  const body: ListThreadsResponse = {
    threads: rows.map((t) => ({ threadId: t._id, title: t.title, createdAt: iso(t.createdAt) }))
  };
  res.json(body);
}

/** GET /threads/:threadId → its messages, oldest first. 404 if it is not this user's. */
export async function getThread(req: Request, res: Response): Promise<void> {
  const thread = await findThread(req.params.threadId ?? '', res.locals.userId);
  if (!thread) {
    res.status(404).json({ error: 'thread not found', status: 404 });
    return;
  }
  const rows = await (await messages()).find({ threadId: thread._id }).sort({ createdAt: 1 }).toArray();
  const body: GetThreadResponse = {
    threadId: thread._id,
    title: thread.title,
    messages: rows.map((m) => ({
      role: m.role,
      content: m.content,
      sources: m.sources,
      ...(m.answerId ? { answerId: m.answerId } : {}),
      ...(m.done ? { done: m.done } : {}),
      createdAt: iso(m.createdAt)
    }))
  };
  res.json(body);
}
