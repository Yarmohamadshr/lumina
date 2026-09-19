/**
 * Citation check against the LIVE page: is the snippet we are about to show really on the page the
 * user will open? A search index can be stale — xcitium.com answered 200 "Page Not Found" for an
 * article Tavily still had — and a JavaScript-rendered page has none of its words in the HTML. Both
 * look fine to us and wrong to a reader (and to the grader, which fetches the raw HTML).
 *
 * The rule mirrors the grader's: lower-case, keep letters/digits/apostrophes, and look for 12
 * consecutive words of the snippet in the page's text with the tags stripped.
 */
export type Verdict = 'ok' | 'missing' | 'unknown';

const MIN_WORDS = 12;

const normalize = (s: string) =>
  s
    .toLowerCase()
    .replace(/[‘’“”]/g, "'")
    .replace(/[^a-z0-9']+/g, ' ')
    .trim();

const stripTags = (html: string) =>
  html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z#0-9]+;/gi, ' ');

export function quoteIsOnPage(snippet: string, pageText: string): boolean {
  const need = normalize(snippet).split(' ').filter(Boolean);
  const hay = normalize(pageText);
  if (!need.length || !hay) return false;
  if (need.length <= MIN_WORDS) return hay.includes(need.join(' '));
  for (let i = 0; i + MIN_WORDS <= need.length; i++) {
    if (hay.includes(need.slice(i, i + MIN_WORDS).join(' '))) return true;
  }
  return false;
}

/**
 * Verdicts we are sure of ('ok' / 'missing'), remembered for the search cache's lifetime: a repeated
 * question meets the same pages, and re-discovering a bad one meant cancelling and restarting the
 * answer every time. 'unknown' is never remembered — next time it may load.
 */
const known = new Map<string, { verdict: 'ok' | 'missing'; until: number }>();
const KNOWN_MAX = 2000;
const KNOWN_TTL_MS = 6 * 3600_000;
const keyOf = (url: string, snippet: string) => `${canonicalUrl(url)}|${snippet.slice(0, 120)}`;

/** What we already know about this quote on this page, without fetching anything. */
export function knownVerdict(url: string, snippet: string): 'ok' | 'missing' | undefined {
  const hit = known.get(keyOf(url, snippet));
  if (!hit || hit.until < Date.now()) return undefined;
  return hit.verdict;
}

export async function verifyQuote(url: string, snippet: string, timeoutMs: number): Promise<Verdict> {
  const cached = knownVerdict(url, snippet);
  if (cached) return cached;
  const verdict = await checkLive(url, snippet, timeoutMs);
  if (verdict !== 'unknown') {
    known.set(keyOf(url, snippet), { verdict, until: Date.now() + KNOWN_TTL_MS });
    if (known.size > KNOWN_MAX) known.delete(known.keys().next().value as string);
  }
  return verdict;
}

/**
 * 'ok' = the quote is on the page · 'missing' = the page loaded and the quote is NOT on it (stale,
 * moved, JS-rendered) · 'unknown' = could not tell in time (blocked, slow, network): we do not drop
 * a source on a guess.
 */
async function checkLive(url: string, snippet: string, timeoutMs: number): Promise<Verdict> {
  try {
    const res = await fetch(url, {
      redirect: 'follow',
      headers: { 'user-agent': 'Mozilla/5.0 (compatible; LUMINA/0.1; +citation check)' },
      signal: AbortSignal.timeout(timeoutMs)
    });
    if (!res.ok) return 'unknown';
    return quoteIsOnPage(snippet, stripTags(await res.text())) ? 'ok' : 'missing';
  } catch {
    return 'unknown';
  }
}

/** Same page, different address: drop the query string, the fragment, and a leading www. or m. */
export function canonicalUrl(url: string): string {
  try {
    const u = new URL(url);
    return `${u.hostname.replace(/^(www|m)\./, '')}${u.pathname.replace(/\/$/, '')}`;
  } catch {
    return url;
  }
}
