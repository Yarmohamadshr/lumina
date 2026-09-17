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
  type Terminated
} from '@lumina/contract';
import { env } from './env.js';
import { costUsd } from './llm.js';
import { research } from './loop.js';
import { buildSources, streamAnswer } from './answer.js';
import { saveRun } from './runlog.js';
import { findThread, messages, titleFromFirstQuestion } from './threads.js';
import { sseHeaders, sseSend } from './sse.js';

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
  const { query, depth } = body.data;

  // The thread must exist and be this user's. Checked before anything is streamed or spent.
  if (!(await findThread(thread.data, userId))) {
    res.status(404).json({ error: 'thread not found', status: 404, requestId });
    return;
  }
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
  const tokens = { in: 0, out: 0 };
  let terminated: Terminated = 'error';
  let answerId: string | undefined;
  let assistantMessage: MessageDoc | undefined; // saved after the stream closes

  try {
    // 1. research: one trace event per tool call
    const found = await research(query, depth, (ev) => {
      toolCalls.push({ name: ev.tool, ok: ev.ok, ms: ev.ms, ...(ev.error ? { error: ev.error } : {}) });
      send('trace', ev);
    });
    tokens.in += found.tokens.in;
    tokens.out += found.tokens.out;

    // 2. sources, BEFORE the first token (validated against the contract)
    const sources = SourcesEvent.parse(buildSources(found.pages, query));
    send('sources', sources);

    // 3. the answer, token by token
    let ttftMs = 0;
    const answer = await streamAnswer(query, found.pages, (text) => {
      if (!ttftMs) ttftMs = Date.now() - started;
      send('token', { text });
    });
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
      tokens,
      costUsd: costUsd(tokens),
      searchCached: found.searchCached,
      terminated,
      depth
    });
    // What /stats reads off this request's row.
    res.locals.requestExtras = { ttftMs, searchCached: found.searchCached };
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
      tokens: tokens.in + tokens.out,
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
