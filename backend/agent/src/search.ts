import { secrets } from './env.js';

/** One web search hit, as the loop sees it. */
export type SearchHit = { title: string; url: string; snippet: string };

/** One fetched page: the full readable text the grounding check quotes from. */
export type FetchedPage = { url: string; text: string };

const TAVILY = 'https://api.tavily.com';

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
  const data = await tavily<{ results: { title: string; url: string; content: string }[] }>('/search', {
    query,
    max_results: max
  });
  return data.results.map((r) => ({ title: r.title, url: r.url, snippet: r.content }));
}

/** fetch_page: a url in, its readable text out. Throws if Tavily could not read the page. */
export async function fetchPage(url: string): Promise<FetchedPage> {
  const data = await tavily<{ results: { url: string; raw_content: string }[] }>('/extract', { urls: [url] });
  const page = data.results[0];
  if (!page?.raw_content) throw new Error(`fetch_page could not read ${url}`);
  return { url: page.url, text: page.raw_content };
}
