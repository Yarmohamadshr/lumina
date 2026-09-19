import type Anthropic from '@anthropic-ai/sdk';
import type { TraceEvent } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { cachedWebSearch } from './cache.js';
import { fetchPage, pageTextOnly, PageUnreadableError, type SearchHit } from './search.js';
import { pickSnippet } from './answer.js';
import { canonicalUrl, knownVerdict, verifyQuote } from './verify.js';
import { recallMemory, saveMemory } from './memory.js';
import type { FetchedSource, ResearchContext } from './loop.js';

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

export type QuickStart = {
  memories: string[];
  /** Pages from searching the question as asked. The speculative answer starts from these. */
  pages: FetchedSource[];
  searchCached: boolean;
  /**
   * Settles when the live citation check of `pages` is done: null = keep them; otherwise the
   * replacement pages (a chosen page loaded WITHOUT the quoted text: stale, moved, JS-rendered).
   */
  check: Promise<FetchedSource[] | null>;
  /**
   * Settles when Haiku's triage is done (and any memory it found is saved and traced). `rewrite` set
   * → the question depends on the conversation or on a memory: the speculative answer is cancelled
   * and quickFinish() searches again. Rejects on a provider error: the run fails loud.
   */
  decision: Promise<{ rewrite?: string; tokens: { in: number; out: number } }>;
};

type Trace = (tool: TraceEvent['tool'], input: Record<string, unknown>, ok: boolean, ms: number, reason: string, error?: string) => void;

const makeTrace = (emit: (ev: TraceEvent) => void, start = 0): { trace: Trace; next: () => number } => {
  let step = start;
  return {
    trace: (tool, input, ok, ms, reason, error) => emit({ step: ++step, tool, input, ok, ms, reason, ...(error ? { error } : {}) }),
    next: () => step
  };
};

const timed = async <T>(fn: () => Promise<T>): Promise<[T, number]> => {
  const t0 = Date.now();
  const out = await fn();
  return [out, Date.now() - t0];
};

/** How long the live citation check may take. It runs while Haiku and Claude are starting anyway. */
const VERIFY_MS = 1000;

type WebPage = Extract<FetchedSource, { kind: 'web' }>;
const numbered = (pages: WebPage[]): FetchedSource[] => pages.map((p, i) => ({ ...p, n: i + 1 }));

/**
 * Pages to answer from, straight from the search when Tavily sent their text (else read in
 * parallel), one per real page (www./m./?query variants are the same page). Every candidate's
 * snippet is checked against the LIVE page at once, in the background: `check` says whether the
 * first PAGES_WANTED can stand, or which candidates replace the ones that failed.
 */
async function choosePages(hits: SearchHit[], snippetQuery: string, trace: Trace): Promise<{ pages: FetchedSource[]; check: Promise<FetchedSource[] | null> }> {
  const seen = new Set<string>();
  const candidates: WebPage[] = [];
  for (const h of hits) {
    const key = canonicalUrl(h.url);
    if (!h.content || seen.has(key)) continue;
    seen.add(key);
    const text = pageTextOnly(h.content);
    // Already known not to show its quote (a repeated question): skip it before speculating on it.
    if (knownVerdict(h.url, pickSnippet(text, snippetQuery)) === 'missing') continue;
    candidates.push({ n: 0, kind: 'web', title: h.title || h.url, url: h.url, text });
  }
  if (candidates.length < PAGES_WANTED) {
    const unread = hits.filter((h) => !h.content && !seen.has(canonicalUrl(h.url))).slice(0, PAGES_TRIED);
    const read = await readTopPages(unread, trace, PAGES_WANTED - candidates.length);
    candidates.push(...read.filter((p): p is WebPage => p.kind === 'web'));
  }

  const verdicts = candidates.map((c) => verifyQuote(c.url, pickSnippet(c.text, snippetQuery), VERIFY_MS));
  const pages = numbered(candidates.slice(0, PAGES_WANTED));
  const check = (async () => {
    const chosen = await Promise.all(verdicts.slice(0, PAGES_WANTED));
    if (!chosen.includes('missing')) return null;
    // Keep what passed (or could not be checked), fill up from the next candidates that pass.
    const all = await Promise.all(verdicts);
    const keep = candidates.filter((_, i) => all[i] !== 'missing').slice(0, PAGES_WANTED);
    for (const [i, c] of candidates.entries()) {
      if (all[i] === 'missing') trace('fetch_page', { url: c.url }, false, VERIFY_MS, `Checking that "${c.title}" still shows the quoted text.`, 'the live page does not contain the quoted text (stale, moved, or rendered by JavaScript); not cited');
    }
    return numbered(keep);
  })();
  return { pages, check };
}

/** Stage 1: search + recall at once, Haiku alongside. Returns as soon as the pages are in. */
export async function quickStart(query: string, ctx: ResearchContext, emit: (ev: TraceEvent) => void): Promise<QuickStart & { steps: () => number }> {
  const { trace, next } = makeTrace(emit);

  // Haiku starts WITHOUT the memories so it is not waiting on the recall; only a user who HAS
  // memories gets a second, informed call (a stated preference can change what to search).
  const recalled = timed(() => recallMemory(ctx.userId, query));
  const blind = triage(query, ctx.history, []);
  // When the informed call replaces it, nobody awaits `blind`: without this, an error in it would be
  // an unhandled rejection, which crashes the process. When it IS used, the await still rethrows.
  blind.catch(() => undefined);
  const plan = recalled.then(([m]) => (m.length ? triage(query, ctx.history, m.map((x) => x.text)) : blind));
  plan.catch(() => undefined); // awaited through `decision`; see above

  const [[memories, memMs], [first, searchMs]] = await Promise.all([recalled, timed(() => cachedWebSearch(query))]);
  trace('recall_memory', { query }, true, memMs, 'Checking what I know about this user, in case it changes the answer.');
  trace('web_search', { query }, true, searchMs, "Searching the web for the question as asked; the results include each page's text.");
  const { pages, check } = await choosePages(first.hits, query, trace);

  const decision = plan.then(async (p) => {
    // Belt and braces: Haiku was told not to re-save what it was shown, but a repeat would pile up.
    if (p.saveMemory && !memories.some((m) => overlap(m.text, p.saveMemory!) >= 0.6)) {
      const [, ms] = await timed(() => saveMemory(ctx.userId, p.saveMemory!, ctx.threadId));
      trace('save_memory', { text: p.saveMemory }, true, ms, 'The user stated a lasting preference about themselves, so remembering it.');
    }
    const rewrite = p.query && norm(p.query) !== norm(query) ? p.query : undefined;
    return { ...(rewrite ? { rewrite } : {}), tokens: p.tokens };
  });

  return { memories: memories.map((m) => m.text), pages, check, searchCached: first.cached, decision, steps: next };
}

/** Stage 2, only when the triage asked for it: search the standalone form of the question. */
export async function quickFinish(
  rewrite: string,
  /** The text snippets are picked for; ask.ts builds the sources with the same string. */
  snippetQuery: string,
  emit: (ev: TraceEvent) => void,
  stepsSoFar: number
): Promise<{ pages: FetchedSource[]; searchCached: boolean }> {
  const { trace } = makeTrace(emit, stepsSoFar);
  const [second, ms] = await timed(() => cachedWebSearch(rewrite));
  trace('web_search', { query: rewrite }, true, ms, 'The question depends on the conversation or on what I know about this user, so searching its standalone form.');
  // No speculation on this path, so the live check is simply awaited (≤ VERIFY_MS).
  const { pages, check } = await choosePages(second.hits, snippetQuery, trace);
  return { pages: (await check) ?? pages, searchCached: second.cached };
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
