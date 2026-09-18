import type { Source } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import type { FetchedSource } from './loop.js';

/** The grounding check looks for ~12 consecutive tokens of the snippet in the real page. */
const MIN_SNIPPET_WORDS = 12;
const MAX_SNIPPET_CHARS = 400;
const PAGE_CHARS_FOR_MODEL = 3000; // see the note in loop.ts: this number multiplies into the bill

/**
 * Tavily returns markdown, but the grader downloads the real HTML and strips the tags, so it
 * sees only the words a browser shows. Drop images, keep a link's text but not its url, and
 * turn markdown symbols into spaces. The words and their order stay exactly as on the page.
 */
function visibleText(line: string): string {
  return line
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ') // ![alt](image-url) → gone
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // [text](url) → text
    .replace(/[`*|#>]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Pick the passage of a fetched page that best matches the query, copied VERBATIM.
 * A paraphrase here would fail grounding, so we only ever cut, never rewrite.
 */
export function pickSnippet(text: string, query: string): string {
  const queryWords = new Set(query.toLowerCase().match(/[a-z0-9]{3,}/g) ?? []);
  const lines = text.split(/\n+/).map(visibleText).filter(Boolean);
  const passages = lines.filter((p) => p.split(' ').length >= MIN_SNIPPET_WORDS);

  let best = passages[0] ?? lines.join(' ');
  let bestScore = -1;
  for (const p of passages) {
    const words = new Set(p.toLowerCase().match(/[a-z0-9]{3,}/g) ?? []);
    const score = [...queryWords].filter((w) => words.has(w)).length;
    if (score > bestScore) [best, bestScore] = [p, score];
  }
  if (best.length <= MAX_SNIPPET_CHARS) return best;
  const cut = best.slice(0, MAX_SNIPPET_CHARS);
  return cut.slice(0, cut.lastIndexOf(' ')); // end on a whole word; still a verbatim substring
}

/** Step 4: one Source per fetched page, numbered as the page was numbered when fetched. */
export function buildSources(pages: FetchedSource[], query: string): Source[] {
  return pages.map((p) => ({ n: p.n, kind: 'web', title: p.title, url: p.url, snippet: pickSnippet(p.text, query) }));
}

const ANSWER_SYSTEM = `Answer the question using ONLY the numbered sources provided.
- Put a citation like [1] after every factual claim, using only the source numbers given.
- If the sources do not answer the question, say so plainly. Never use outside knowledge.
- Be concise: start with the direct answer, then a few short paragraphs at most. No preamble.
- Write PLAIN TEXT. No markdown: no **bold**, no ## headings, no backticks, no bullet characters.
  The UI renders the text as it arrives, so markdown symbols show up literally and look broken.
  For a list, write one short sentence per line instead.
- Keep the whole answer under 200 words. Answer the question asked; do not summarise the sources.`;

/**
 * Step 5: stream the answer. `onText` is called for every piece of text as it arrives.
 * Errors are not caught: a failed stream rejects, and the route fails loud.
 */
export async function streamAnswer(
  query: string,
  pages: FetchedSource[],
  onText: (text: string) => void
): Promise<{ text: string; tokens: { in: number; out: number } }> {
  const context = pages.length
    ? pages.map((p) => `[${p.n}] ${p.title}\n${p.url}\n${p.text.slice(0, PAGE_CHARS_FOR_MODEL)}`).join('\n\n---\n\n')
    : '(no sources were found)';

  const stream = llm.messages.stream({
    model: env.llmModel,
    max_tokens: 1200,
    thinking: { type: 'disabled' }, // no tools in this call; skipping thinking gets the first token out sooner
    system: ANSWER_SYSTEM,
    messages: [{ role: 'user', content: `Sources:\n\n${context}\n\nQuestion: ${query}` }]
  });
  stream.on('text', onText);

  const final = await stream.finalMessage();
  const text = final.content.flatMap((b) => (b.type === 'text' ? [b.text] : [])).join('');
  return { text, tokens: { in: final.usage.input_tokens, out: final.usage.output_tokens } };
}
