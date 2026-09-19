import type Anthropic from '@anthropic-ai/sdk';
import type { AskMode, AskTool, Depth, Locator, Terminated, TraceEvent } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { toolsFor } from './tools.js';
import { fetchPage, PageUnreadableError } from './search.js';
import { cachedWebSearch } from './cache.js';
import { recallMemory, saveMemory } from './memory.js';
import { hybridSearch, type DocHit } from './retrieve.js';

/** The tools we have built so far. toolsFor(depth) decides which of them the model sees. */
const TOOL_DEFS: Partial<Record<AskTool, Anthropic.Tool>> = {
  web_search: {
    name: 'web_search',
    description: 'Search the web. Returns titles, urls and short teaser snippets. Snippets are NOT enough to answer from.',
    input_schema: {
      type: 'object',
      properties: { query: { type: 'string', description: 'The search query' } },
      required: ['query']
    }
  },
  fetch_page: {
    name: 'fetch_page',
    description: 'Read the full text of one url from the search results. Returns the page numbered [n] for citing.',
    input_schema: {
      type: 'object',
      properties: { url: { type: 'string', description: 'A url from web_search results' } },
      required: ['url']
    }
  },
  search_documents: {
    name: 'search_documents',
    description:
      "Search the user's OWN uploaded documents (their Space) by meaning and by keyword. Returns the best " +
      'matching passages, each numbered [n] with its document and page, ready to cite. Use it for anything ' +
      'the documents might cover, before the web.',
    input_schema: {
      type: 'object',
      properties: { query: { type: 'string', description: 'What to look for; the user question itself works well' } },
      required: ['query']
    }
  },
  // The description is how the model decides when to call it, so the boundary lives here.
  save_memory: {
    name: 'save_memory',
    description:
      'Remember a DURABLE fact or preference about this user for future conversations: how they want answers, ' +
      'what they work with, who they are. Examples: "prefers code examples over prose", "builds in TypeScript", ' +
      '"is a nurse". NEVER save what they asked today, search results, or anything true only right now — ' +
      'that would be a log, not a memory. Only save when the user states something about themselves.',
    input_schema: {
      type: 'object',
      properties: { text: { type: 'string', description: 'The fact or preference, in one short sentence' } },
      required: ['text']
    }
  },
  recall_memory: {
    name: 'recall_memory',
    description:
      'Search what you already know about this user, by meaning. Call this FIRST on any question where a ' +
      'preference or personal detail could change the answer — which is most questions.',
    input_schema: {
      type: 'object',
      properties: { query: { type: 'string', description: 'What to look for, e.g. the user question itself' } },
      required: ['query']
    }
  }
};

const SYSTEM = `You research a question on the web before it is answered.
Today is ${new Date().toISOString().slice(0, 10)}.
- Call recall_memory first: what you know about this user may change how the answer should look.
- If the user states a durable preference or fact about themselves, call save_memory once.
- Start with web_search, then fetch_page the 2-3 most relevant results. Answer only from fetched pages.
- Before each tool call, write ONE short sentence saying why you are making it.
- This is a quick search: ONE search and TWO fetches is usually enough. Stop as soon as you can answer.
- Never repeat a search you have already run, and never call a tool twice with the same input.
- When you have enough, reply with only the word DONE. Do not write the answer here.`;

/**
 * One thing we actually retrieved in this request: a web page we fetched, or a page of the
 * user's document. `n` is the number it will be cited as, in ONE numbering for both kinds.
 */
export type FetchedSource = { n: number; title: string; text: string } & (
  | { kind: 'web'; url: string }
  | { kind: 'doc'; docId: string; locator: Locator }
);

/** Same web page, or same document page → same source. Used to dedupe within and across runs. */
export const sourceKey = (s: FetchedSource): string =>
  s.kind === 'web' ? s.url : `${s.docId}|${JSON.stringify(s.locator)}`;

/** "retrieval-basics.pdf, p. 3" — how a document passage is labelled for the model and the UI. */
export const docLabel = (title: string, locator: Locator): string =>
  locator.page ? `${title}, p. ${locator.page}` : locator.heading ? `${title}, "${locator.heading}"` : `${title}, line ${locator.line}`;

/**
 * The ~`chars` of a passage that best match the query: slide a window over its sentences and keep
 * the one sharing the most query words. The first 900 characters of a page often miss the answer
 * ("The common default is 1.2" sits at character ~1,100 of page 1), and then the model searches
 * again, or goes to the web, for a fact it was already handed.
 */
export function focus(text: string, query: string, chars: number): string {
  if (text.length <= chars) return text;
  const want = new Set(query.toLowerCase().match(/[a-z0-9]{2,}/g) ?? []);
  const sentences = text.split(/(?<=[.!?])\s+/);
  let best = 0;
  let bestScore = -1;
  for (let i = 0; i < sentences.length; i++) {
    let window = '';
    let score = 0;
    for (let j = i; j < sentences.length && window.length < chars; j++) {
      window += `${sentences[j]} `;
      score += (sentences[j]!.toLowerCase().match(/[a-z0-9]{2,}/g) ?? []).filter((w) => want.has(w)).length;
    }
    if (score > bestScore) [best, bestScore] = [i, score];
  }
  const out = sentences.slice(best).join(' ').slice(0, chars);
  return `${best > 0 ? '… ' : ''}${out}${out.length < text.length ? ' …' : ''}`;
}

/**
 * Text the MODEL reads, never the snippet: drop the page's own footnote markers. A Wikipedia page
 * says "…in 1928.[3]", the model copied "[3]" into an answer that had 2 sources, and the run failed
 * with a dangling citation. The citation numbers must only ever be ours.
 */
export const forModel = (text: string): string => text.replace(/\[\d{1,3}\]/g, '');

/** The same shape for a Space's passages: added to `pages` and numbered, deduped by page. */
function addDocHits(pages: FetchedSource[], hits: DocHit[]): FetchedSource[] {
  return hits.map((h) => {
    const existing = pages.find((p) => p.kind === 'doc' && p.docId === h.docId && JSON.stringify(p.locator) === JSON.stringify(h.locator));
    if (existing) return existing;
    const page: FetchedSource = { n: pages.length + 1, kind: 'doc', title: h.title, docId: h.docId, locator: h.locator, text: h.text };
    pages.push(page);
    return page;
  });
}

export type ResearchResult = {
  messages: Anthropic.MessageParam[];
  pages: FetchedSource[];
  terminated: Exclude<Terminated, 'error'>;
  tokens: { in: number; out: number; cacheWrite: number; cacheRead: number };
  toolCalls: number;
  /** true only when EVERY search in this run was a cache hit (the done event's searchCached). */
  searchCached: boolean;
};

/**
 * What we send the model per page. The whole page text is kept for snippets; only this much
 * reaches Claude. Every turn resends the history, and the answer call sends the pages AGAIN,
 * so this number multiplies: at 8000 a 2-page answer cost $0.062, over the $0.05 SLA. At 4000
 * the same answer costs about half that, with no loss of quality on the pages we tested.
 */
const PAGE_CHARS_FOR_MODEL = 2500;

/**
 * A document passage in the LOOP is only there so the model can decide what to do next: enough to
 * see whether the Space covers the question. The answer step gets the full passage. Sending five
 * whole pages here, resent on every turn, pushed a docs+web run to $0.060, over the $0.05 SLA.
 */
const DOC_CHARS_FOR_MODEL = 900;

/**
 * Prompt caching: mark the end of what we have sent so far, so the NEXT call in the loop re-reads
 * that prefix from Anthropic's cache at about 10% of the input price instead of paying full price
 * for the same page text again. A run that fetches 2 pages over 5 calls pays for those pages five
 * times without this. Only the newest breakpoint is kept; the API allows a small number of them.
 */
function markCachePoint(messages: Anthropic.MessageParam[]): void {
  for (const m of messages) {
    if (typeof m.content === 'string') continue;
    for (const block of m.content) delete (block as { cache_control?: unknown }).cache_control;
  }
  const last = messages[messages.length - 1];
  if (!last || typeof last.content === 'string' || !last.content.length) return;
  const block = last.content[last.content.length - 1] as { cache_control?: { type: 'ephemeral' } };
  block.cache_control = { type: 'ephemeral' };
}

/**
 * The research loop: ask Claude → run the tools it asks for → give it the results → repeat,
 * until it says DONE or a cap is hit. Provider errors are NOT caught here: they throw out
 * of this function, and the caller ends the run with terminated:"error" (fail loud).
 */
export type ResearchContext = {
  userId: string;
  threadId: string;
  /** The last few turns of this thread, so a follow-up question makes sense on its own. */
  history: Anthropic.MessageParam[];
  /** Where to look (AskBody.mode). */
  mode: AskMode;
  /** The Space attached to this question, if any, with the titles of its indexed documents. */
  space?: { spaceId: string; name: string; titles: string[] };
};

/**
 * Tool calls and wall clock are capped per RUN, not per subagent. A deep search fans out to
 * several isolated subagents, and they all draw down the same budget, so the object is shared
 * and mutated. Sequential or parallel, 24 calls and 240 s is the whole run's envelope.
 */
export type Budget = { callsUsed: number; maxCalls: number; deadline: number };

export const newBudget = (depth: Depth): Budget => ({
  callsUsed: 0,
  maxCalls: depth === 'deep' ? env.maxToolCallsDeep : env.maxToolCalls,
  deadline: Date.now() + (depth === 'deep' ? env.maxWallClockSecDeep : env.maxWallClockSec) * 1000
});

export type ResearchOptions = {
  /** Shared across a deep run's subagents; a quick run gets its own. */
  budget?: Budget;
  /** Which sub-question this subagent is serving. Tags every trace step and every page it finds. */
  subQuestion?: number;
  /** A subagent's own ceiling, so one greedy branch cannot eat the whole run's budget. */
  maxCallsHere?: number;
  /** Extra instruction for a subagent: it answers ONE part of a larger question. */
  extraSystem?: string;
};

export async function research(
  query: string,
  depth: Depth,
  ctx: ResearchContext,
  emit: (ev: TraceEvent) => void,
  opts: ResearchOptions = {}
): Promise<ResearchResult> {
  const budget = opts.budget ?? newBudget(depth);
  const callsAllowedHere = opts.maxCallsHere ?? Number.POSITIVE_INFINITY;
  let callsHere = 0;
  // A subagent never plans (plan_research belongs to the call that fans out) and never touches
  // memory: the parent run already recalled what it needs, and six subagents each recalling the
  // same thing burned 6 of this run's 24 tool calls for nothing.
  const allowed = opts.subQuestion
    ? toolsFor('quick').filter((t) => t !== 'recall_memory' && t !== 'save_memory')
    : toolsFor(depth);
  // The router, as a gate: no Space → no search_documents; mode "web" → no documents; mode "docs"
  // → no web. What the model cannot see it cannot call, whatever the prompt says.
  const routed = allowed.filter((t) => {
    if (t === 'search_documents') return !!ctx.space && ctx.mode !== 'web';
    if (t === 'web_search' || t === 'fetch_page') return ctx.mode !== 'docs';
    return true;
  });
  const tools = routed.flatMap((name) => TOOL_DEFS[name] ?? []);
  const system = [SYSTEM, spaceInstructions(ctx), opts.extraSystem].filter(Boolean).join('\n');

  const messages: Anthropic.MessageParam[] = [...ctx.history, { role: 'user', content: query }];
  const pages: FetchedSource[] = [];
  const titles = new Map<string, string>(); // url → title, remembered from search results
  const searched = new Map<string, string>(); // normalized query → the results we already returned
  const tokens = { in: 0, out: 0, cacheWrite: 0, cacheRead: 0 };
  let toolCalls = 0;
  let nudged = false; // the "fetch a page first" nudge is sent at most once
  let searches = 0;
  let cachedSearches = 0;
  let searchedDocs = false;

  const result = (terminated: 'done' | 'cap'): ResearchResult => ({
    messages,
    pages,
    terminated,
    tokens,
    toolCalls,
    searchCached: searches > 0 && cachedSearches === searches
  });

  while (true) {
    if (Date.now() > budget.deadline) return result('cap');

    markCachePoint(messages);
    const response = await llm.messages.create({
      model: env.llmModel,
      max_tokens: 2000,
      // Loop turns only pick the next tool; the thinking happens in what the tools return. On
      // Sonnet 5 an omitted `thinking` means adaptive thinking ON at the default (high) effort, and
      // each of 3–4 turns paid for it: ~12 s of a 13.6 s TTFT was the loop's own calls.
      // Quick only: deep subagents keep the default effort they were measured with (2.7–4.5x the
      // sources of quick, inside the 90 s SLA); low effort there could cost sources, not seconds.
      ...(depth === 'quick' ? { output_config: { effort: env.loopEffort } } : {}),
      // The system prompt and the tool list never change during a run, so cache them too.
      system: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }],
      tools,
      messages
    });
    tokens.in += response.usage.input_tokens;
    tokens.out += response.usage.output_tokens;
    tokens.cacheWrite += response.usage.cache_creation_input_tokens ?? 0;
    tokens.cacheRead += response.usage.cache_read_input_tokens ?? 0;

    if (response.stop_reason !== 'tool_use') {
      // A gate, not a prompt: the answer may only use pages fetched in THIS request. Claude stops
      // early in two ways — it searched but never read a page, or (with thread history in front of
      // it) it thinks it already knows and never searches at all. Both end with zero sources and
      // "no sources were provided", so both get one nudge back.
      if (!pages.length && !nudged) {
        nudged = true;
        messages.push({ role: 'assistant', content: response.content });
        messages.push({
          role: 'user',
          content: routed.includes('search_documents') && !searchedDocs
            ? "You have not retrieved anything yet, and the answer can only use what is retrieved in THIS request. Call search_documents now."
            : titles.size
              ? 'You have not fetched any page yet, and the answer can only use pages fetched in this request. Call fetch_page on the most relevant search result.'
              : 'You have not searched yet. Earlier turns in this conversation are context, not sources: the answer may only use pages fetched in THIS request. Call web_search now, then fetch_page.'
        });
        continue;
      }
      return result('done');
    }
    messages.push({ role: 'assistant', content: response.content });

    // Claude's one-sentence "why" before its tool calls becomes the trace step's reason.
    const reason = response.content
      .flatMap((b) => (b.type === 'text' ? [b.text.trim()] : []))
      .join(' ')
      .slice(0, 300);

    const results: Anthropic.ToolResultBlockParam[] = [];
    let capped = false;
    for (const call of response.content) {
      if (call.type !== 'tool_use') continue;
      // Every tool_use needs a tool_result, even the ones the cap stops us from running.
      if (capped || budget.callsUsed >= budget.maxCalls || callsHere >= callsAllowedHere || Date.now() > budget.deadline) {
        capped = true;
        results.push({ type: 'tool_result', tool_use_id: call.id, content: 'Not run: tool-call cap reached.', is_error: true });
        continue;
      }
      budget.callsUsed++;
      callsHere++;
      toolCalls++;
      const t0 = Date.now();
      const out = await runTool(call.name, call.input as Record<string, unknown>);
      emit({
        step: budget.callsUsed,
        tool: call.name as AskTool,
        input: call.input as Record<string, unknown>,
        ok: out.ok,
        ms: Date.now() - t0,
        ...(reason ? { reason } : {}),
        ...(out.ok ? {} : { error: out.content }),
        // Mandatory on a deep run: a merged trace nobody can follow back to a sub-question is a
        // pile, not research. Enforced here, once, so no later merge can drop it.
        ...(opts.subQuestion ? { subQuestion: opts.subQuestion } : {})
      });
      results.push({ type: 'tool_result', tool_use_id: call.id, content: out.content, ...(out.ok ? {} : { is_error: true }) });
    }
    messages.push({ role: 'user', content: results });
    if (capped) {
      // "cap" means the RUN ran out of its envelope. A subagent stopping at its own share is our
      // scheduling, not the run hitting a wall: the run still has budget, so it is not a partial.
      const runOutOfRoom = budget.callsUsed >= budget.maxCalls || Date.now() > budget.deadline;
      return result(runOutOfRoom ? 'cap' : 'done');
    }
  }

  /** Runs one tool. Returns ok:false for a bad input or an unreadable page; rethrows provider failures. */
  async function runTool(name: string, input: Record<string, unknown>): Promise<{ ok: boolean; content: string }> {
    if (name === 'web_search') {
      if (typeof input.query !== 'string' || !input.query.trim()) return { ok: false, content: 'web_search needs a query string' };
      // Claude sometimes fires the same search twice in one turn. The network cost is already
      // covered by the cache, but the tool call and its tokens are not, so answer from what we
      // returned before instead of spending the budget twice.
      const key = input.query.trim().toLowerCase().replace(/\s+/g, ' ');
      const before = searched.get(key);
      if (before !== undefined) return { ok: true, content: `Already searched this. Same results:\n${before}` };

      const { hits, cached } = await cachedWebSearch(input.query);
      searches++;
      if (cached) cachedSearches++;
      for (const h of hits) titles.set(h.url, h.title);
      const content = hits.length
        ? hits.map((h) => `- ${h.title}\n  ${h.url}\n  ${h.snippet.slice(0, 300)}`).join('\n')
        : 'No results.';
      searched.set(key, content);
      return { ok: true, content };
    }

    if (name === 'fetch_page') {
      if (typeof input.url !== 'string' || !input.url.startsWith('http')) return { ok: false, content: 'fetch_page needs a url' };
      const already = pages.find((p) => p.kind === 'web' && p.url === input.url);
      if (already) return { ok: true, content: `Already fetched as [${already.n}].` };
      try {
        const page = await fetchPage(input.url);
        const n = pages.length + 1;
        const title = titles.get(input.url) ?? page.url;
        pages.push({ n, kind: 'web', title, url: page.url, text: page.text });
        // The part of the page that best matches the question, not its first N characters: the
        // top of a page is often navigation and an intro, and the fact asked about sits further down.
        return { ok: true, content: `[${n}] ${title}\n${forModel(focus(page.text, query, PAGE_CHARS_FOR_MODEL))}` };
      } catch (err) {
        if (err instanceof PageUnreadableError) return { ok: false, content: err.message };
        throw err; // provider failure: fail loud
      }
    }

    if (name === 'search_documents') {
      if (typeof input.query !== 'string' || !input.query.trim()) return { ok: false, content: 'search_documents needs a query' };
      if (!ctx.space) return { ok: false, content: 'no Space is attached to this question' };
      searchedDocs = true;
      const key = `docs:${input.query.trim().toLowerCase().replace(/\s+/g, ' ')}`;
      const before = searched.get(key);
      if (before !== undefined) return { ok: true, content: `Already searched this. Same passages:\n${before}` };
      const found = addDocHits(pages, await hybridSearch(ctx.space.spaceId, ctx.userId, input.query));
      const content = found.length
        ? found.map((p) => `[${p.n}] ${p.kind === 'doc' ? docLabel(p.title, p.locator) : p.title}\n${forModel(focus(p.text, String(input.query), DOC_CHARS_FOR_MODEL))}`).join('\n\n')
        : 'No matching passages in this Space.';
      searched.set(key, content);
      return { ok: true, content };
    }

    if (name === 'save_memory') {
      if (typeof input.text !== 'string' || input.text.trim().length < 3) return { ok: false, content: 'save_memory needs text' };
      await saveMemory(ctx.userId, input.text, ctx.threadId);
      return { ok: true, content: `Saved: "${input.text.trim()}"` };
    }

    if (name === 'recall_memory') {
      if (typeof input.query !== 'string' || !input.query.trim()) return { ok: false, content: 'recall_memory needs a query' };
      const hits = await recallMemory(ctx.userId, input.query);
      if (!hits.length) return { ok: true, content: 'Nothing remembered about this user yet.' };
      return { ok: true, content: hits.map((m) => `- ${m.text}`).join('\n') };
    }

    // toolsFor + TOOL_DEFS mean the model never sees another tool; if it names one anyway, refuse.
    return { ok: false, content: `tool ${name} is not available` };
  }
}

/** What the model is told about an attached Space. The tool list is the gate; this is the steer. */
function spaceInstructions(ctx: ResearchContext): string {
  if (!ctx.space || ctx.mode === 'web') return '';
  const docs = ctx.space.titles.length ? ctx.space.titles.map((t) => `"${t}"`).join(', ') : '(none indexed yet)';
  if (ctx.mode === 'docs') {
    return `The user asked you to answer from their Space "${ctx.space.name}" only (documents: ${docs}). Call search_documents; do not use the web.`;
  }
  return `The user attached their Space "${ctx.space.name}" with these documents: ${docs}.
- Call search_documents FIRST on every question: their own documents come before the web.
- If the passages answer the question, stop there. Use web_search and fetch_page as well only when the
  documents do not cover it, or the question is about the outside world (news, markets, other companies).`;
}

/**
 * mode "docs" on a quick run: the user already chose where to look, so there is no routing
 * decision left for a model to make. Search the Space with the question itself — one retrieval,
 * no LLM round trip before it — and hand the passages to the answer step. Faster and cheaper
 * than a loop that would make exactly this call anyway.
 */
export async function searchSpaceOnly(
  query: string,
  ctx: ResearchContext,
  emit: (ev: TraceEvent) => void
): Promise<ResearchResult> {
  if (!ctx.space) throw new Error('searchSpaceOnly needs a Space');
  const t0 = Date.now();
  const pages: FetchedSource[] = [];
  addDocHits(pages, await hybridSearch(ctx.space.spaceId, ctx.userId, query));
  emit({
    step: 1,
    tool: 'search_documents',
    input: { query, spaceId: ctx.space.spaceId },
    ok: true,
    ms: Date.now() - t0,
    reason: `Answering from the Space "${ctx.space.name}" only (mode: docs): searching its documents for the question.`
  });
  return { messages: [], pages, terminated: 'done', tokens: { in: 0, out: 0, cacheWrite: 0, cacheRead: 0 }, toolCalls: 1, searchCached: false };
}
