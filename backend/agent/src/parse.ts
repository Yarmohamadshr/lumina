import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';
import type { Locator } from '@lumina/contract';

/**
 * One piece of a document that a citation can point at: a PDF page, a Markdown section,
 * a block of plain-text lines. Chunking never crosses a section, so every chunk inherits
 * exactly one locator — that is what makes "retrieval-basics.pdf, p. 3" possible.
 */
export interface Section {
  text: string;
  locator: Locator;
}

export interface Parsed {
  sections: Section[];
  /** PDFs only. */
  pages?: number;
}

/** Bytes → sections, by type. Throws on a file it cannot read: the worker marks it failed. */
export async function parse(data: Buffer, mimeType: string): Promise<Parsed> {
  if (mimeType === 'application/pdf') return parsePdf(data);
  const text = data.toString('utf8');
  if (mimeType === 'text/markdown') return { sections: parseMarkdown(text) };
  return { sections: parsePlain(text) };
}

/** pdfjs gives text per page; each page is one section with { page }. */
async function parsePdf(data: Buffer): Promise<Parsed> {
  // pdfjs takes ownership of the array it is given, so hand it a copy.
  const pdf = await pdfjs.getDocument({ data: new Uint8Array(data), verbosity: 0, isEvalSupported: false }).promise;
  try {
    const sections: Section[] = [];
    for (let page = 1; page <= pdf.numPages; page++) {
      const content = await (await pdf.getPage(page)).getTextContent();
      const text = content.items
        .map((item) => ('str' in item ? item.str + (item.hasEOL ? '\n' : '') : ''))
        .join('')
        .trim();
      if (text) sections.push({ text, locator: { page } });
    }
    return { sections, pages: pdf.numPages };
  } finally {
    await pdf.destroy();
  }
}

/** A new section at every heading; each carries { heading, line } of where it starts. */
function parseMarkdown(md: string): Section[] {
  const sections: Section[] = [];
  let heading: string | undefined;
  let startLine = 1;
  let buf: string[] = [];

  const flush = () => {
    const text = buf.join('\n').trim();
    if (text) sections.push({ text, locator: heading ? { heading, line: startLine } : { line: startLine } });
    buf = [];
  };

  md.split(/\r?\n/).forEach((line, i) => {
    const h = /^#{1,6}\s+(.+?)\s*#*\s*$/.exec(line);
    if (h) {
      flush();
      heading = h[1]!.trim();
      startLine = i + 1;
    }
    buf.push(line); // the heading line stays in its section: it is useful context for search
  });
  flush();
  return sections;
}

/**
 * Plain text has no pages or headings, so paragraphs are grouped into sections of about
 * PLAIN_SECTION_WORDS, each with { line } where it starts. (One section per paragraph would
 * make one tiny chunk per paragraph, which retrieves precisely and says nothing.)
 */
const PLAIN_SECTION_WORDS = 200;

function parsePlain(text: string): Section[] {
  const sections: Section[] = [];
  let startLine = 1;
  let buf: string[] = [];
  let words = 0;
  const flush = () => {
    const t = buf.join('\n').trim();
    if (t) sections.push({ text: t, locator: { line: startLine } });
    buf = [];
    words = 0;
  };
  text.split(/\r?\n/).forEach((line, i) => {
    // A blank line ends a paragraph; that is the only place a big-enough section is cut.
    if (!line.trim()) {
      if (words >= PLAIN_SECTION_WORDS) flush();
      else if (buf.length) buf.push('');
      return;
    }
    if (!buf.length) startLine = i + 1;
    buf.push(line);
    words += line.split(/\s+/).filter(Boolean).length;
  });
  flush();
  return sections;
}
