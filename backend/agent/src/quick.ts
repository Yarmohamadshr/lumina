import type Anthropic from '@anthropic-ai/sdk';
import type { TraceEvent } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { cachedWebSearch } from './cache.js';
import { fetchPage, PageUnreadableError, type SearchHit } from './search.js';
import { recallMemory, saveMemory } from './memory.js';
import type { FetchedSource, ResearchContext, ResearchResult } from './loop.js';

/**
 * QUICK web search as a fixed pipeline (DESIGN change, Day 9): no Claude call stands between the
 * question and the first search. The agent loop made 4 Claude calls in a row before answering
 * (~2.5 s each, even at low effort), so its TTFT was 10–20 s against a 2.5 s SLA. Here:
 *
 *   in parallel:  web_search(question) · recall_memory(question) · Haiku: rewrite? save?
 *   then:         fetch the top pages in parallel
 *   then:         the answer streams (answer.ts)
 *
 * Deep search stays agentic: that is the gear where model-driven research earns its latency.
 */

/** Pages the answer reads. Two readable pages is what the loop used on almost every quick run. */
const PAGES_WANTED = 2;
/** Fetch a few more than we need, in parallel: one dead link must not cost a second round trip. */
const PAGES_TRIED = 4;
/** Turns of history Haiku sees to resolve "what about its port?" into a standalone query. */
const HISTORY_FOR_REWRITE = 4;

type Triage = { query?: string; saveMemory?: string; tokens: { in: number; out: number } };

const TRIAGE_TOOL: Anthropic.Tool = {
  name: 'triage',
  description: 'Prepare the web search for this message and decide whether it states something durable about the user.',
  input_schema: {
    type: 'object',
    properties: {
      needsRewrite: {
        type: 'boolean',
        description:
          'true ONLY if (a) the new message cannot be understood without the conversation (it says "it", "that", ' +
          '"bigger than that", "and what about…"), or (b) something we know about the user should change WHAT to ' +
          'search (they want Python examples → add "Python"). Otherwise false, however the question is worded.'
      },
      searchQuery: {
        type: 'string',
        description:
          'The message as a standalone web search query. Resolve pronouns and references ("it", "that one") ' +
          'using the conversation, and add what we know about the user when it changes what to look for. ' +
          'If neither applies, repeat the message unchanged.'
      },
      durableFact: {
        type: 'string',
        description:
          'ONLY if the NEW message itself states a lasting preference or fact about the user (how they want ' +
          'answers, what they work with, who they are), that fact in one short sentence. Never something we ' +
          'already know, never what they asked, never anything true only today. Omit otherwise.'
      }
    },
    required: ['needsRewrite', 'searchQuery']
  }
};

/**
 * One small Haiku call, run IN PARALLEL with the first search: turns a follow-up into a standalone
 * query, and spots a durable preference for save_memory. Haiku, because this runs on every quick
 * question and must finish before the search does.
 */
async function triage(query: string, history: Anthropic.MessageParam[], memories: string[]): Promise<Triage> {
  const known = memories.length ? `What we know about this user:\n${memories.map((m) => `- ${m}`).join('\n')}\n\n` : '';
  const convo = history
    .slice(-HISTORY_FOR_REWRITE)
    .map((m) => `${m.role}: ${typeof m.content === 'string' ? m.content : ''}`)
    .join('\n');
  const res = await llm.messages.create({
    model: env.plannerModel,
    max_tokens: 300,
    tools: [TRIAGE_TOOL],
    tool_choice: { type: 'tool', name: 'triage' },
    messages: [{ role: 'user', content: `${known}${convo ? `Conversation so far:\n${convo}\n\n` : ''}New message: ${query}` }]
  });
  const call = res.content.find((b) => b.type === 'tool_use');
  const input = (call?.type === 'tool_use' ? call.input : {}) as { needsRewrite?: boolean; searchQuery?: string; durableFact?: string };
  // A rewrite is used only when Haiku says the message depends on the conversation: rewording a
  // question that already stands alone ("What is X?" → "X") cost a second search for nothing.
  // With memories, Haiku's yes/no alone was not enough: it wrote "read a file line by line in
  // TypeScript" but answered needsRewrite=false, the search stayed generic, and the answer came back
  // in Bash. So a query that ADDS words (the remembered preference) is used either way.
  const q = input.searchQuery?.trim();
  const addsWords = !!q && memories.length > 0 && [...words(q)].some((w) => !words(query).has(w));
  const rewrite = q && (history.length || memories.length) && (input.needsRewrite || addsWords) ? q : undefined;
  return {
    ...(rewrite ? { query: rewrite } : {}),
    ...(input.durableFact?.trim() ? { saveMemory: input.durableFact.trim() } : {}),
    tokens: { in: res.usage.input_tokens, out: res.usage.output_tokens }
  };
}

const norm = (q: string) => q.trim().toLowerCase().replace(/\s+/g, ' ');
const STOP = new Set(['the', 'and', 'for', 'how', 'what', 'with', 'does', 'can', 'you', 'your', 'are', 'use', 'using']);
/** Content words (3+ letters, not stop words), to tell a real addition from a rewording. */
function words(text: string): Set<string> {
  return new Set((text.toLowerCase().match(/[a-z0-9+#.]{3,}/g) ?? []).filter((w) => !STOP.has(w)));
}

export async function quickSearch(query: string, ctx: ResearchContext, emit: (ev: TraceEvent) => void): Promise<ResearchResult & { memories: string[] }> {
  let step = 0;
  let toolCalls = 0;
  let searches = 0;
  let cachedSearches = 0;
  const trace = (tool: TraceEvent['tool'], input: Record<string, unknown>, ok: boolean, ms: number, reason: string, error?: string) => {
    toolCalls++;
    emit({ step: ++step, tool, input, ok, ms, reason, ...(error ? { error } : {}) });
  };
  const timed = async <T>(fn: () => Promise<T>): Promise<[T, number]> => {
    const t0 = Date.now();
    const out = await fn();
    return [out, Date.now() - t0];
  };

  // ---- 1. at once: the search, memory recall, and Haiku. Haiku starts WITHOUT the memories so it
  // is not waiting ~0.25 s on the recall; only a user who HAS memories gets a second, informed call
  // (a stated preference can change what to search). Provider errors are not caught: fail loud.
  const recalled = timed(() => recallMemory(ctx.userId, query));
  const blind = triage(query, ctx.history, []);
  // When the informed call replaces it, nobody awaits `blind`: without this, an error in it would be
  // an unhandled rejection, which crashes the process. When it IS used, the await still rethrows.
  blind.catch(() => undefined);
  const [[memories, memMs], [first, searchMs], plan] = await Promise.all([
    recalled,
    timed(() => cachedWebSearch(query)),
    recalled.then(([m]) => (m.length ? triage(query, ctx.history, m.map((x) => x.text)) : blind))
  ]);
  trace('recall_memory', { query }, true, memMs, 'Checking what I know about this user, in case it changes the answer.');
  searches++;
  if (first.cached) cachedSearches++;
  trace('web_search', { query }, true, searchMs, 'Searching the web for the question as asked; the results include each page\'s text.');

  // A follow-up that only makes sense in context ("and its default port?") gets a second search
  // with the standalone version. A question that already stands alone costs nothing extra.
  let hits: SearchHit[] = first.hits;
  const standalone = plan.query;
  if (standalone && norm(standalone) !== norm(query)) {
    const [second, ms] = await timed(() => cachedWebSearch(standalone));
    searches++;
    if (second.cached) cachedSearches++;
    trace('web_search', { query: standalone }, true, ms, 'The question depends on the conversation or on what I know about this user, so searching its standalone form.');
    hits = second.hits.length ? second.hits : first.hits;
  }

  // Belt and braces: Haiku was told not to re-save what it was shown, but a repeat would pile up.
  const alreadyKnown = plan.saveMemory && memories.some((m) => overlap(m.text, plan.saveMemory!) >= 0.6);
  if (plan.saveMemory && !alreadyKnown) {
    const [, ms] = await timed(() => saveMemory(ctx.userId, plan.saveMemory!, ctx.threadId));
    trace('save_memory', { text: plan.saveMemory }, true, ms, 'The user stated a lasting preference about themselves, so remembering it.');
  }

  // ---- 2. the pages: straight from the search when Tavily returned their text (the usual case),
  // otherwise read in parallel, going as soon as PAGES_WANTED are in.
  const withText = hits.filter((h) => h.content).slice(0, PAGES_WANTED);
  const pages: FetchedSource[] = withText.map((h, i) => ({ n: i + 1, kind: 'web', title: h.title || h.url, url: h.url, text: h.content! }));
  if (pages.length < PAGES_WANTED) {
    const more = await readTopPages(hits.filter((h) => !h.content).slice(0, PAGES_TRIED), trace, PAGES_WANTED - pages.length);
    for (const p of more) pages.push({ ...p, n: pages.length + 1 });
  }

  return {
    messages: [],
    pages,
    terminated: 'done',
    tokens: { in: plan.tokens.in, out: plan.tokens.out, cacheWrite: 0, cacheRead: 0 },
    toolCalls,
    searchCached: searches > 0 && cachedSearches === searches,
    memories: memories.map((m) => m.text)
  };
}

/** Stop waiting for more pages this long after the fetches start, if at least one page is in. */
const FETCH_DEADLINE_MS = 2500;
const FETCH_HARD_DEADLINE_MS = 5000;

/** Share of words two short sentences have in common (Jaccard), to spot a re-saved memory. */
function overlap(a: string, b: string): number {
  const wa = new Set(a.toLowerCase().match(/[a-z0-9]+/g) ?? []);
  const wb = new Set(b.toLowerCase().match(/[a-z0-9]+/g) ?? []);
  const both = [...wa].filter((w) => wb.has(w)).length;
  return both / (new Set([...wa, ...wb]).size || 1);
}

type Trace = (tool: TraceEvent['tool'], input: Record<string, unknown>, ok: boolean, ms: number, reason: string, error?: string) => void;

/**
 * Fetch every candidate at once; resolve with the first PAGES_WANTED readable pages, in the order
 * they arrive — or, at the deadline, with whatever has arrived (if anything). Fetches still running
 * are abandoned and not traced: the trace only shows steps whose result was used or failed.
 * A provider failure (not an unreadable page) rejects: fail loud.
 */
function readTopPages(candidates: SearchHit[], trace: Trace, wanted: number): Promise<FetchedSource[]> {
  return new Promise((resolve, reject) => {
    const pages: FetchedSource[] = [];
    let settled = 0;
    let done = false;
    const t0 = Date.now();
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(pages);
    };
    // Soft deadline: go with what we have. Hard deadline: go even with nothing — an honest "the
    // sources could not be read" beats a user watching a spinner for a slow site.
    const timer = setTimeout(() => {
      if (pages.length) finish();
      else setTimeout(finish, FETCH_HARD_DEADLINE_MS - FETCH_DEADLINE_MS);
    }, FETCH_DEADLINE_MS);
    if (!candidates.length) return finish();

    for (const hit of candidates) {
      fetchPage(hit.url)
        .then((page) => {
          if (done) return;
          pages.push({ n: pages.length + 1, kind: 'web', title: hit.title || page.url, url: page.url, text: page.text });
          trace('fetch_page', { url: hit.url }, true, Date.now() - t0, `Reading "${hit.title}" in full: a search snippet is not enough to answer from.`);
          if (pages.length >= wanted) finish();
        })
        .catch((err: unknown) => {
          if (done) return;
          if (!(err instanceof PageUnreadableError)) {
            done = true;
            clearTimeout(timer);
            return reject(err);
          }
          trace('fetch_page', { url: hit.url }, false, Date.now() - t0, `Reading "${hit.title}".`, err.message);
        })
        .finally(() => {
          if (++settled === candidates.length) finish();
        });
    }
  });
}
