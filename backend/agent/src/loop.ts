import type Anthropic from '@anthropic-ai/sdk';
import type { AskTool, Depth, Terminated, TraceEvent } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { toolsFor } from './tools.js';
import { fetchPage, PageUnreadableError, webSearch } from './search.js';

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
- When you have enough, reply with only the word DONE. Do not write the answer here.`;

/** One page we actually fetched in this request. `n` is the number it will be cited as. */
export type FetchedSource = { n: number; title: string; url: string; text: string };

export type ResearchResult = {
  messages: Anthropic.MessageParam[];
  pages: FetchedSource[];
  terminated: Exclude<Terminated, 'error'>;
  tokens: { in: number; out: number };
  toolCalls: number;
};

/** What we send the model per page: enough to answer from, small enough to stay cheap. */
const PAGE_CHARS_FOR_MODEL = 8000;

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
  const tokens = { in: 0, out: 0 };
  let toolCalls = 0;

  const result = (terminated: 'done' | 'cap'): ResearchResult => ({ messages, pages, terminated, tokens, toolCalls });

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

    if (response.stop_reason !== 'tool_use') return result('done');
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
      const hits = await webSearch(input.query);
      for (const h of hits) titles.set(h.url, h.title);
      if (!hits.length) return { ok: true, content: 'No results.' };
      return { ok: true, content: hits.map((h) => `- ${h.title}\n  ${h.url}\n  ${h.snippet.slice(0, 300)}`).join('\n') };
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
