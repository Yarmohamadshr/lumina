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
  type Source,
  type SubQuestion,
  type RunLog,
  type SseEventName,
  type TraceEvent,
  type Terminated
} from '@lumina/contract';
import { env } from './env.js';
import { costUsd } from './llm.js';
import { research, searchSpaceOnly, type FetchedSource, type ResearchContext } from './loop.js';
import { quickFinish, quickStart } from './quick.js';
import { deepSearch } from './deep.js';
import { deepRunsToday } from './runlog.js';
import { buildSources, streamAnswer } from './answer.js';
import { saveRun } from './runlog.js';
import { findThread, messages, recentHistory, titleFromFirstQuestion } from './threads.js';
import { sseHeaders, sseSend } from './sse.js';
import { findSpace } from './spaces.js';
import { indexedTitles } from './retrieve.js';

const log = pino({ level: env.logLevel });

/** How long past the model's first words the gate still waits for the live citation check. */
const CHECK_GRACE_MS = 150;

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
  // Priced per model as tokens come in: a quick run is Haiku throughout, a deep run mostly Sonnet.
  let spentUsd = 0;
  const phases: Record<string, number> = {};
  let answerModel = env.llmModel;
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
    let ttftMs = 0;
    const onToken = (text: string) => {
      if (!ttftMs) ttftMs = Date.now() - started;
      send('token', { text });
    };
    const addTokens = (t: { in: number; out: number; cacheWrite?: number; cacheRead?: number }, model: string = env.llmModel) => {
      spentUsd += costUsd(t, model);
      tokens.in += t.in;
      tokens.out += t.out;
      tokens.cacheWrite += t.cacheWrite ?? 0;
      tokens.cacheRead += t.cacheRead ?? 0;
    };

    // Router: deep → plan + agentic subagents · quick "docs" → straight to the Space · quick web
    // (or auto with no Space) → the fast pipeline · quick auto WITH a Space → the loop, because
    // choosing between documents and web is a real decision.
    let found: { pages: FetchedSource[]; terminated: 'done' | 'cap'; searchCached?: boolean; plan?: SubQuestion[] };
    let sources: Source[];
    let answer: { text: string; tokens: { in: number; out: number } };

    if (depth === 'quick' && mode !== 'docs' && !ctx.space) {
      // Fast pipeline, with a SPECULATIVE answer: Claude starts writing from the search of the
      // question as asked while Haiku's triage finishes. Nothing reaches the user until the triage
      // agrees; then sources, then the held tokens. Haiku's ~0.8 s hides inside Claude's own start.
      // Phase timings (ms since the request started) → the request log, to see where TTFT goes.
      const at = () => Date.now() - started;
      const start = await quickStart(query, ctx, onTrace);
      phases.pages = at();
      void start.decision.then(() => (phases.triage = at()), () => undefined);
      void start.check.then(() => (phases.check = at()), () => undefined);
      sources = SourcesEvent.parse(buildSources(start.pages, query));
      const first = sources;
      // The gate opens only when BOTH agree: Haiku (no rewrite needed) and the live citation check
      // (every chosen page really shows its quote). Either one saying no cancels the speculation.
      // Production timings showed the gate opening exactly 1,000 ms after the pages — the live check's
      // timeout (one of two sites is usually slow from Fly) — while the model's first words had been
      // ready for ~450 ms. So the check gets until the model is ready (+150 ms): a check that finishes
      // in time still swaps a bad page out; one still running counts as "unknown", as a timeout did.
      let modelReady!: () => void;
      const ready = new Promise<void>((r) => (modelReady = r));
      const checkOrReady = Promise.race([
        start.check,
        ready.then(() => new Promise<null>((r) => setTimeout(() => r(null), CHECK_GRACE_MS)))
      ]);
      let result = await streamAnswer(
        query,
        start.pages,
        undefined,
        onToken,
        start.memories,
        { gate: Promise.all([start.decision, checkOrReady]).then(([d, replaced]) => !d.rewrite && !replaced), onFirstText: () => modelReady(), onOpen: () => {
            phases.gate = at();
            send('sources', first);
          }
        },
        env.quickAnswerModel
      );
      answerModel = env.quickAnswerModel;
      if (result.firstTextAt) phases.modelFirst = result.firstTextAt - started;
      if (result.aborted) phases.restarted = 1;
      const decision = await start.decision;
      // What the gate actually decided on: a check still running at the gate did not replace anything.
      const replaced = result.aborted ? await start.check : null;
      addTokens(decision.tokens, env.plannerModel);
      addTokens(result.tokens, env.quickAnswerModel); // a cancelled answer still cost its input
      let pages = start.pages;
      let searchCached = start.searchCached;
      if (result.aborted) {
        if (decision.rewrite) {
          // The question needed its standalone form: search again and answer from THAT.
          const snippetQuery = `${query} ${decision.rewrite}`;
          const fin = await quickFinish(decision.rewrite, snippetQuery, onTrace, start.steps());
          pages = fin.pages;
          searchCached = searchCached && fin.searchCached;
          sources = SourcesEvent.parse(buildSources(pages, snippetQuery));
        } else {
          // A chosen page failed the live check: answer from the pages that passed.
          pages = replaced ?? start.pages;
          sources = SourcesEvent.parse(buildSources(pages, query));
        }
        send('sources', sources);
        result = await streamAnswer(query, pages, undefined, onToken, start.memories, undefined, env.quickAnswerModel);
        addTokens(result.tokens, env.quickAnswerModel);
      }
      found = { pages, terminated: 'done', searchCached };
      answer = result;
    } else {
      const res =
        depth === 'deep'
          ? await deepSearch(query, ctx, (subQuestions) => send('plan', { subQuestions }), onTrace)
          : mode === 'docs'
            ? await searchSpaceOnly(query, ctx, onTrace)
            : await research(query, depth, ctx, onTrace);
      addTokens(res.tokens);
      found = {
        pages: res.pages,
        terminated: res.terminated,
        ...('searchCached' in res ? { searchCached: res.searchCached } : {}),
        ...('plan' in res ? { plan: res.plan } : {})
      };
      // 2. sources, BEFORE the first token (validated against the contract)
      sources = SourcesEvent.parse(buildSources(found.pages, query));
      send('sources', sources);
      // 3. the answer, token by token
      answer = await streamAnswer(query, found.pages, found.plan, onToken);
      addTokens(answer.tokens);
    }
    const plan = found.plan;

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
      model: answerModel,
      tokens: { in: tokens.in + tokens.cacheWrite + tokens.cacheRead, out: tokens.out },
      costUsd: spentUsd,
      searchCached: found.searchCached ?? false,
      terminated,
      depth,
      ...(plan ? { subQuestions: plan.length } : {})
    });
    // What /stats reads off this request's row.
    res.locals.requestExtras = { ttftMs, searchCached: found.searchCached ?? false, ...(Object.keys(phases).length ? { phases } : {}) };
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
      costUsd: spentUsd,
      terminated,
      depth,
      toolCalls,
      createdAt: new Date().toISOString()
    });
  } catch (err) {
    log.error({ err, requestId }, 'run log NOT saved');
  }
}
