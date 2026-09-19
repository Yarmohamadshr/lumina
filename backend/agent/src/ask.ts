import { randomUUID } from 'node:crypto';
import type { Request, Response } from 'express';
import pino from 'pino';
import {
  AskBody,
  DoneEvent,
  newId,
  SourcesEvent,
  ThreadId,
  unresolvedCitations,
  type MessageDoc,
  type RunLog,
  type SseEventName,
  type TraceEvent,
  type Terminated
} from '@lumina/contract';
import { env } from './env.js';
import { costUsd } from './llm.js';
import { research, searchSpaceOnly, type ResearchContext } from './loop.js';
import { quickSearch } from './quick.js';
import { deepSearch } from './deep.js';
import { deepRunsToday } from './runlog.js';
import { buildSources, streamAnswer } from './answer.js';
import { saveRun } from './runlog.js';
import { findThread, messages, recentHistory, titleFromFirstQuestion } from './threads.js';
import { sseHeaders, sseSend } from './sse.js';
import { findSpace } from './spaces.js';
import { indexedTitles } from './retrieve.js';

const log = pino({ level: env.logLevel });

/**
 * POST /threads/:threadId/ask — streams  trace* → sources → token* → done.
 * Any provider failure ends the run with an `error` event (or a real 502 if nothing was sent yet).
 * Every run, done or failed, is saved as a run log.
 */
export async function ask(req: Request, res: Response): Promise<void> {
  const started = Date.now();
  const requestId: string = res.locals.requestId; // set by requestLog, from the gateway's header

  // Validate here too, not only in the gateway (defense in depth, DESIGN.md Q2).
  const userId: string = res.locals.userId; // set by requireUser → 401 before we get here
  const thread = ThreadId.safeParse(req.params.threadId);
  const body = AskBody.safeParse(req.body);
  if (!thread.success) {
    res.status(400).json({ error: 'threadId must look like thr_…', status: 400, requestId });
    return;
  }
  if (!body.success) {
    res.status(400).json({ error: body.error.issues.map((i) => i.message).join('; '), status: 400, requestId });
    return;
  }
  const { query, depth, mode, spaceId } = body.data;

  // The thread must exist and be this user's. Checked before anything is streamed or spent.
  if (!(await findThread(thread.data, userId))) {
    res.status(404).json({ error: 'thread not found', status: 404, requestId });
    return;
  }
  // The Space, if one is attached: it must be this user's. mode "docs" without one has nothing to search.
  if (mode === 'docs' && !spaceId) {
    res.status(400).json({ error: 'mode "docs" needs a spaceId', status: 400, requestId });
    return;
  }
  const space = spaceId ? await findSpace(spaceId, userId) : null;
  if (spaceId && !space) {
    res.status(404).json({ error: 'space not found', status: 404, requestId });
    return;
  }

  // The spend gate, BEFORE any provider call: deep costs several times more, so a user over their
  // daily cap is refused having spent nothing. It lives here in the AGENT, which is not publicly
  // reachable — a cap on the edge is a cap you can bypass by calling the agent directly.
  if (depth === 'deep') {
    const used = await deepRunsToday(userId);
    if (used >= env.deepDailyCap) {
      const resetsAt = new Date();
      resetsAt.setHours(24, 0, 0, 0); // midnight: when today's count starts again
      res.status(429).json({
        error: `deep search daily cap reached (${env.deepDailyCap} per day)`,
        status: 429,
        resetsAt: resetsAt.toISOString(),
        requestId
      });
      return;
    }
  }

  // Read the history BEFORE storing this question, or the question would appear twice:
  // once as history and once as the live query.
  const history = await recentHistory(thread.data);
  await (await messages()).insertOne({
    _id: `msg_${randomUUID()}`,
    threadId: thread.data,
    userId,
    role: 'user',
    content: query,
    sources: [],
    createdAt: new Date()
  });
  await titleFromFirstQuestion(thread.data, query);

  // Open the stream lazily: if the very first provider call fails, we can still answer a real 502.
  let streaming = false;
  const send = (event: SseEventName, data: unknown) => {
    if (!streaming) {
      sseHeaders(res);
      streaming = true;
    }
    sseSend(res, event, data);
  };

  // What the run log records, filled in as the run goes.
  const toolCalls: RunLog['toolCalls'] = [];
  const tokens = { in: 0, out: 0, cacheWrite: 0, cacheRead: 0 };
  let terminated: Terminated = 'error';
  let answerId: string | undefined;
  let assistantMessage: MessageDoc | undefined; // saved after the stream closes

  try {
    // 1. research. Quick: one loop, with the thread's recent turns for context.
    // Deep: plan → isolated subagents in parallel → merge. Either way one trace event per tool
    // call, and on a deep run every step carries the sub-question it serves.
    const onTrace = (ev: TraceEvent) => {
      toolCalls.push({ name: ev.tool, ok: ev.ok, ms: ev.ms, ...(ev.error ? { error: ev.error } : {}) });
      send('trace', ev);
    };
    const ctx: ResearchContext = {
      userId,
      threadId: thread.data,
      history,
      mode,
      ...(space && mode !== 'web' ? { space: { spaceId: space._id, name: space.name, titles: await indexedTitles(space._id, userId) } } : {})
    };
    // Router: deep → plan + agentic subagents · quick "docs" → straight to the Space · quick web
    // (or auto with no Space) → the fast pipeline, no Claude call before the search · quick auto
    // WITH a Space → the loop, because choosing between documents and web is a real decision.
    const found =
      depth === 'deep'
        ? await deepSearch(query, ctx, (subQuestions) => send('plan', { subQuestions }), onTrace)
        : mode === 'docs'
          ? await searchSpaceOnly(query, ctx, onTrace)
          : !ctx.space
            ? await quickSearch(query, ctx, onTrace)
            : await research(query, depth, ctx, onTrace);
    const memories: string[] = 'memories' in found && Array.isArray(found.memories) ? found.memories : [];
    const plan = 'plan' in found ? found.plan : undefined;
    tokens.in += found.tokens.in;
    tokens.out += found.tokens.out;
    tokens.cacheWrite += found.tokens.cacheWrite;
    tokens.cacheRead += found.tokens.cacheRead;

    // 2. sources, BEFORE the first token (validated against the contract)
    const sources = SourcesEvent.parse(buildSources(found.pages, query));
    send('sources', sources);

    // 3. the answer, token by token
    let ttftMs = 0;
    const answer = await streamAnswer(
      query,
      found.pages,
      plan,
      (text) => {
        if (!ttftMs) ttftMs = Date.now() - started;
        send('token', { text });
      },
      memories
    );
    tokens.in += answer.tokens.in;
    tokens.out += answer.tokens.out;

    // 4. every [n] must resolve to a source retrieved in THIS request
    const dangling = unresolvedCitations(answer.text, sources);
    if (dangling.length) throw new Error(`answer cited [${dangling.join('], [')}] but no such source was retrieved`);

    // 5. done
    terminated = found.terminated;
    answerId = newId('ans');
    const done = DoneEvent.parse({
      answerId,
      latencyMs: Date.now() - started,
      ttftMs,
      model: env.llmModel,
      tokens: { in: tokens.in + tokens.cacheWrite + tokens.cacheRead, out: tokens.out },
      costUsd: costUsd(tokens),
      searchCached: 'searchCached' in found ? found.searchCached : false,
      terminated,
      depth,
      ...(plan ? { subQuestions: plan.length } : {})
    });
    // What /stats reads off this request's row.
    res.locals.requestExtras = { ttftMs, searchCached: 'searchCached' in found ? found.searchCached : false };
    send('done', done);
    res.end();

    // 6. the answer joins the thread (saved below, after the try/catch)
    assistantMessage = {
      _id: `msg_${randomUUID()}`,
      threadId: thread.data,
      userId,
      role: 'assistant',
      content: answer.text,
      answerId,
      sources,
      done,
      createdAt: new Date()
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error({ err, requestId, threadId: thread.data }, 'ask failed');
    if (!streaming) {
      res.status(502).json({ error: message, status: 502, requestId });
    } else {
      send('error', { status: 502, error: message });
      res.end();
    }
  }

  // 7. persist. The response is already complete, so a failure here can only be logged, loudly.
  // The answer message, only for a finished run, so GET /threads/:id can show it again.
  if (assistantMessage) {
    try {
      await (await messages()).insertOne(assistantMessage);
    } catch (err) {
      log.error({ err, requestId }, 'assistant message NOT saved');
    }
  }
  // The run log: always, whether the run finished or failed.
  try {
    await saveRun({
      requestId,
      userId,
      threadId: thread.data,
      answerId,
      query,
      tokens: tokens.in + tokens.cacheWrite + tokens.cacheRead + tokens.out,
      wallClockSec: (Date.now() - started) / 1000,
      costUsd: costUsd(tokens),
      terminated,
      depth,
      toolCalls,
      createdAt: new Date().toISOString()
    });
  } catch (err) {
    log.error({ err, requestId }, 'run log NOT saved');
  }
}
