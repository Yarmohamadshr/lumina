import { env, secrets } from './env.js';

/** One web search hit. `content` is the page's text, when Tavily could read it during the search. */
export type SearchHit = { title: string; url: string; snippet: string; content?: string };

/** One fetched page: the full readable text the grounding check quotes from. */
export type FetchedPage = { url: string; text: string };

const TAVILY = 'https://api.tavily.com';

/**
 * Sites whose HTML does not contain the words they show: rendered by JavaScript or behind a login.
 * Tavily can read them; the grader, fetching the raw HTML, sees an empty page — so a correct
 * citation to one scores as UNgrounded (an instagram.com source failed 3 citations in one run).
 */
const UNVERIFIABLE_SITES = [
  'instagram.com', 'facebook.com', 'tiktok.com', 'youtube.com', 'x.com', 'twitter.com',
  'linkedin.com', 'pinterest.com', 'quora.com', 'brainly.com', 'brainly.in', 'brainly.ph'
];

/**
 * One page could not be read (paywall, dead link). The loop may skip it and carry on.
 * Any OTHER error means the provider itself failed, and that must end the run.
 */
export class PageUnreadableError extends Error {}

/**
 * POST to Tavily and return the JSON. Fail loud: any non-2xx throws, so a provider error
 * can never look like "no results" (rule A1, the Live Translate bug).
 */
async function tavily<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${TAVILY}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${secrets.tavily}` },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`tavily ${path} failed: ${res.status} ${await res.text()}`);
  return (await res.json()) as T;
}

/** web_search: a query in, up to `max` hits out. */
export async function webSearch(query: string, max = 5): Promise<SearchHit[]> {
  const data = await tavily<{ results: { title: string; url: string; content: string; raw_content?: string | null }[] }>('/search', {
    query,
    max_results: max,
    // Tavily's latency/relevance knob. Measured on 5 queries: basic 1,402 ms · fast 623 ms ·
    // ultra-fast 175 ms, same result counts; fast kept the better sources. TTFT budget is 2.5 s.
    search_depth: env.searchDepth,
    // The page text comes back WITH the search (+~180 ms): 30/30 results had it in testing, while a
    // separate /extract failed on about half the pages and once took 7 s.
    include_raw_content: 'markdown',
    exclude_domains: UNVERIFIABLE_SITES
  });
  return data.results.map((r) => ({
    title: r.title,
    url: r.url,
    snippet: r.content,
    ...(r.raw_content?.trim() ? { content: pageTextOnly(r.raw_content) } : {})
  }));
}

/**
 * Tavily's markdown can open with a metadata header ("Title: …", "URL Source: …", "Markdown
 * Content:") that is NOT on the page. On a page with no long paragraph the snippet fell back to that
 * header, and the citation could not be found in the real HTML. Keep only the page's own text.
 */
function pageTextOnly(text: string): string {
  return text.replace(/^(?:\s*(?:Title|URL Source|Published Time|Markdown Content):[^\n]*\n)+/, '').trim();
}

/** fetch_page: a url in, its readable text out. Throws if Tavily could not read the page. */
export async function fetchPage(url: string): Promise<FetchedPage> {
  const data = await tavily<{ results: { url: string; raw_content: string }[] }>('/extract', { urls: [url] });
  const page = data.results[0];
  if (!page?.raw_content) throw new PageUnreadableError(`fetch_page could not read ${url}`);
  return { url: page.url, text: pageTextOnly(page.raw_content) };
}
