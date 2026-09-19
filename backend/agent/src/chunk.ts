import type { Locator } from '@lumina/contract';
import type { Section } from './parse.js';

export interface Chunk {
  text: string;
  locator: Locator;
  ord: number;
}

/**
 * ~200 words (~270 tokens) per chunk. Small enough that a question lands on the right PAGE
 * (recall@5 is scored by page), big enough to carry the sentence around the answer.
 */
const TARGET_WORDS = 200;
/** ~15 % overlap, so a sentence cut at a boundary is whole in at least one chunk. */
const OVERLAP_WORDS = 30;

const wordCount = (s: string) => s.split(/\s+/).filter(Boolean).length;

/**
 * Split on structure first, then on length:
 *   section (page / heading) → sentences → chunks of about TARGET_WORDS.
 * A chunk never crosses a section, so its locator is exact. Chunks are cut between
 * sentences, never mid-sentence, because a chunk that starts mid-sentence retrieves badly
 * and reads worse in a citation.
 */
export function chunk(sections: Section[]): Chunk[] {
  const out: Chunk[] = [];
  for (const section of sections) {
    const sentences = splitSentences(section.text);
    let current: string[] = [];
    let words = 0;

    const emit = () => {
      const text = current.join(' ').trim();
      if (text) out.push({ text, locator: section.locator, ord: out.length });
    };

    for (const sentence of sentences) {
      const n = wordCount(sentence);
      if (words + n > TARGET_WORDS && words > 0) {
        emit();
        // Carry the last few sentences (up to OVERLAP_WORDS) into the next chunk.
        const carry: string[] = [];
        let carried = 0;
        for (let i = current.length - 1; i >= 0; i--) {
          const w = wordCount(current[i]!);
          if (carried + w > OVERLAP_WORDS) break;
          carry.unshift(current[i]!);
          carried += w;
        }
        current = carry;
        words = carried;
      }
      current.push(sentence);
      words += n;
    }
    emit();
  }
  return out;
}

/**
 * Sentences, with line breaks inside a sentence flattened to spaces. A single "sentence"
 * longer than TARGET_WORDS (a table, a run-on list) is hard-cut by words so no chunk is huge.
 */
function splitSentences(text: string): string[] {
  const flat = text.replace(/\s+/g, ' ').trim();
  if (!flat) return [];
  // A sentence ends at . ! ? followed by a space — so "1.2" and "e.g.x" are not cut in half.
  const parts = flat.split(/(?<=[.!?]["')\]]?)\s+/);
  const out: string[] = [];
  for (const raw of parts) {
    const s = raw.trim();
    if (!s) continue;
    const words = s.split(' ');
    for (let i = 0; i < words.length; i += TARGET_WORDS) out.push(words.slice(i, i + TARGET_WORDS).join(' '));
  }
  return out;
}
