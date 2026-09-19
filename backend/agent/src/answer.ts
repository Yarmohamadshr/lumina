import type { Source, SubQuestion } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { docLabel, focus, forModel, type FetchedSource } from './loop.js';

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

/**
 * Step 4: one Source per fetched page, numbered as the page was numbered when fetched.
 * On a deep run each page carries the sub-question that found it, and it is copied through here —
 * dropping it in the merge is the classic way to lose points on an otherwise correct fan-out.
 */
export function buildSources(pages: (FetchedSource & { subQuestion?: number })[], query: string): Source[] {
  return pages.map((p) => {
    const tag = p.subQuestion ? { subQuestion: p.subQuestion } : {};
    // A document passage IS the retrieved text: its snippet is the chunk text verbatim, so the
    // citation can be checked against exactly what was retrieved, with the page it came from.
    if (p.kind === 'doc') {
      return { n: p.n, kind: 'doc', title: p.title, docId: p.docId, locator: p.locator, snippet: p.text, ...tag };
    }
    return { n: p.n, kind: 'web', title: p.title, url: p.url, snippet: pickSnippet(p.text, query), ...tag };
  });
}

/**
 * Code in an answer (`data[0]`, `rows[1]`) looks exactly like a citation to anything that reads
 * `[n]` — our own dangling-citation check and the grader's regex alike. A prompt rule did not hold
 * (the model still wrote data[0]), so it is enforced here, on the stream: a bracketed number that
 * is CODE — inside a ``` block, or right after a name or `)` — becomes `[ 0 ]`, which is still valid
 * Python and JS. A `[7]` in prose is left exactly as written: if it cites nothing, that is a real
 * failure and the run must still fail loud.
 */
export function codeBracketGuard() {
  let pending = '';
  let inFence = false;
  let prev = '';
  const inCode = (s: string) => s.replace(/\[(\d{1,3})\]/g, '[ $1 ]');
  const inProse = (s: string) =>
    s.replace(/\[(\d{1,3})\]/g, (all, d: string, at: number) => {
      const before = at > 0 ? s[at - 1]! : prev;
      return /[A-Za-z0-9_)]/.test(before) ? `[ ${d} ]` : all;
    });
  const process = (s: string) => {
    let out = '';
    s.split('```').forEach((part, i) => {
      if (i > 0) {
        out += '```';
        inFence = !inFence;
      }
      out += inFence ? inCode(part) : inProse(part);
    });
    if (out) prev = out[out.length - 1]!;
    return out;
  };
  return {
    /** Text in, safe text out. Holds back a tail that could be half a `[12]` or half a fence. */
    push(text: string): string {
      const s = pending + text;
      const tail = /(\[\d{0,3}|`{1,2})$/.exec(s);
      pending = tail ? tail[0] : '';
      return process(tail ? s.slice(0, tail.index) : s);
    },
    flush(): string {
      const s = pending;
      pending = '';
      return process(s);
    }
  };
}

const ANSWER_SYSTEM = `Answer the question using ONLY the numbered sources provided.
- Put a citation like [1] after every factual claim, using only the source numbers given.
- If the sources do not answer the question, say so plainly. Never use outside knowledge.
- Be concise: start with the direct answer, then a few short paragraphs at most. No preamble.
- Write PLAIN TEXT. No markdown: no **bold**, no ## headings, no backticks, no bullet characters.
  The UI renders the text as it arrives, so markdown symbols show up literally and look broken.
  For a list, write one short sentence per line instead.
- Keep the whole answer under 200 words. Answer the question asked; do not summarise the sources.
- Square brackets holding a number are ALWAYS read as citations. In code, never put a number
  directly inside square brackets: write items[i] with a named index, or first = items[:1], never items[0].`;

/**
 * Step 5: stream the answer. `onText` is called for every piece of text as it arrives.
 * Errors are not caught: a failed stream rejects, and the route fails loud.
 */
export async function streamAnswer(
  query: string,
  pages: (FetchedSource & { subQuestion?: number })[],
  plan: SubQuestion[] | undefined,
  onText: (text: string) => void,
  /** What recall_memory returned: how this user wants answers. Shapes the answer, is never cited. */
  memories: string[] = []
): Promise<{ text: string; tokens: { in: number; out: number } }> {
  const context = pages.length
    ? pages
        .map(
          (p) =>
            `[${p.n}] ${p.kind === 'doc' ? `${docLabel(p.title, p.locator)} (the user's own document)` : p.title}` +
            `${p.subQuestion ? ` (sub-question ${p.subQuestion})` : ''}\n${p.kind === 'web' ? `${p.url}\n` : ''}` +
            forModel(focus(p.text, query, PAGE_CHARS_FOR_MODEL))
        )
        .join('\n\n---\n\n')
    : '(no sources were found)';

  // A deep answer is structured by the plan it ran, and says plainly where the evidence is thin.
  // Padding is what the human grader is looking for, so length is not the goal: coverage is.
  const deep = plan?.length
    ? `\nThis was a DEEP search. It researched these sub-questions:\n` +
      plan.map((q) => `${q.i}. ${q.question}`).join('\n') +
      `\nStructure the answer around them: a short direct answer first, then one short section per
sub-question with its number and a heading line, then one line on what is still unknown — only if
something genuinely is. If a sub-question found little, say so instead of padding it.
Keep it under 400 words in total.`
    : '';

  const stream = llm.messages.stream({
    model: env.llmModel,
    max_tokens: plan?.length ? 2000 : 1200,
    thinking: { type: 'disabled' }, // no tools in this call; skipping thinking gets the first token out sooner
    system:
      ANSWER_SYSTEM +
      deep +
      (memories.length
        ? `\nWhat you know about this user (apply it to HOW you answer; it is not a source, never cite it):\n${memories.map((m) => `- ${m}`).join('\n')}`
        : ''),
    messages: [{ role: 'user', content: `Sources:\n\n${context}\n\nQuestion: ${query}` }]
  });
  // What the user sees and what the citation check reads are the SAME guarded text.
  const guard = codeBracketGuard();
  let text = '';
  const emit = (t: string) => {
    if (!t) return;
    text += t;
    onText(t);
  };
  stream.on('text', (t) => emit(guard.push(t)));

  const final = await stream.finalMessage();
  emit(guard.flush());
  return { text, tokens: { in: final.usage.input_tokens, out: final.usage.output_tokens } };
}
