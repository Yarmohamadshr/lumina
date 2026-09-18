import type Anthropic from '@anthropic-ai/sdk';
import type { AskTool, Depth, Terminated, TraceEvent } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { toolsFor } from './tools.js';
import { fetchPage, PageUnreadableError } from './search.js';
import { cachedWebSearch } from './cache.js';

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
  }
};

const SYSTEM = `You research a question on the web before it is answered.
Today is ${new Date().toISOString().slice(0, 10)}.
- Start with web_search, then fetch_page the 2-3 most relevant results. Answer only from fetched pages.
- Before each tool call, write ONE short sentence saying why you are making it.
- This is a quick search: usually 1-2 searches and 2-4 fetches are enough.
- Never repeat a search you have already run, and never call a tool twice with the same input.
- When you have enough, reply with only the word DONE. Do not write the answer here.`;

/** One page we actually fetched in this request. `n` is the number it will be cited as. */
export type FetchedSource = { n: number; title: string; url: string; text: string };

export type ResearchResult = {
  messages: Anthropic.MessageParam[];
  pages: FetchedSource[];
  terminated: Exclude<Terminated, 'error'>;
  tokens: { in: number; out: number };
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
const PAGE_CHARS_FOR_MODEL = 4000;

/**
 * The research loop: ask Claude → run the tools it asks for → give it the results → repeat,
 * until it says DONE or a cap is hit. Provider errors are NOT caught here: they throw out
 * of this function, and the caller ends the run with terminated:"error" (fail loud).
 */
export async function research(query: string, depth: Depth, emit: (ev: TraceEvent) => void): Promise<ResearchResult> {
  const started = Date.now();
  const maxCalls = depth === 'deep' ? env.maxToolCallsDeep : env.maxToolCalls;
  const maxMs = (depth === 'deep' ? env.maxWallClockSecDeep : env.maxWallClockSec) * 1000;
  const tools = toolsFor(depth).flatMap((name) => TOOL_DEFS[name] ?? []);

  const messages: Anthropic.MessageParam[] = [{ role: 'user', content: query }];
  const pages: FetchedSource[] = [];
  const titles = new Map<string, string>(); // url → title, remembered from search results
  const searched = new Map<string, string>(); // normalized query → the results we already returned
  const tokens = { in: 0, out: 0 };
  let toolCalls = 0;
  let nudged = false; // the "fetch a page first" nudge is sent at most once
  let searches = 0;
  let cachedSearches = 0;

  const result = (terminated: 'done' | 'cap'): ResearchResult => ({
    messages,
    pages,
    terminated,
    tokens,
    toolCalls,
    searchCached: searches > 0 && cachedSearches === searches
  });

  while (true) {
    if (Date.now() - started > maxMs) return result('cap');

    const response = await llm.messages.create({
      model: env.llmModel,
      max_tokens: 2000,
      system: SYSTEM,
      tools,
      messages
    });
    tokens.in += response.usage.input_tokens;
    tokens.out += response.usage.output_tokens;

    if (response.stop_reason !== 'tool_use') {
      // A gate, not a prompt: answers may only use fetched pages. If Claude stops with none
      // although its searches found results, send it back once to read one.
      if (!pages.length && titles.size && !nudged) {
        nudged = true;
        messages.push({ role: 'assistant', content: response.content });
        messages.push({
          role: 'user',
          content: 'You have not fetched any page yet, and the answer can only use fetched pages. Call fetch_page on the most relevant search result.'
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
      if (capped || toolCalls >= maxCalls || Date.now() - started > maxMs) {
        capped = true;
        results.push({ type: 'tool_result', tool_use_id: call.id, content: 'Not run: tool-call cap reached.', is_error: true });
        continue;
      }
      toolCalls++;
      const t0 = Date.now();
      const out = await runTool(call.name, call.input as Record<string, unknown>);
      emit({
        step: toolCalls,
        tool: call.name as AskTool,
        input: call.input as Record<string, unknown>,
        ok: out.ok,
        ms: Date.now() - t0,
        ...(reason ? { reason } : {}),
        ...(out.ok ? {} : { error: out.content })
      });
      results.push({ type: 'tool_result', tool_use_id: call.id, content: out.content, ...(out.ok ? {} : { is_error: true }) });
    }
    messages.push({ role: 'user', content: results });
    if (capped) return result('cap');
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
      const already = pages.find((p) => p.url === input.url);
      if (already) return { ok: true, content: `Already fetched as [${already.n}].` };
      try {
        const page = await fetchPage(input.url);
        const n = pages.length + 1;
        const title = titles.get(input.url) ?? page.url;
        pages.push({ n, title, url: page.url, text: page.text });
        return { ok: true, content: `[${n}] ${title}\n${page.text.slice(0, PAGE_CHARS_FOR_MODEL)}` };
      } catch (err) {
        if (err instanceof PageUnreadableError) return { ok: false, content: err.message };
        throw err; // provider failure: fail loud
      }
    }

    // toolsFor + TOOL_DEFS mean the model never sees another tool; if it names one anyway, refuse.
    return { ok: false, content: `tool ${name} is not available` };
  }
}
