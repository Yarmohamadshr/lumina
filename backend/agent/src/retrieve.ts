import { COLLECTIONS, SEARCH_INDEXES, type ChunkDoc, type Locator } from '@lumina/contract';
import { db } from './db.js';
import { embed } from './embed.js';
import { documents } from './spaces.js';

/** One citable passage from a Space: every retrieved chunk of one document PAGE (or section), merged. */
export type DocHit = {
  docId: string;
  title: string;
  locator: Locator;
  /** The chunks' own text, verbatim, in document order. The citation snippet is this text. */
  text: string;
  score: number;
};

/** How many candidates each half contributes to the fusion. */
const CANDIDATES = 20;
/** The RRF constant. 60 is the value from the original paper and the usual default. */
const RRF_K = 60;
/** Passages (pages) returned per search. recall@5 reads the first five. */
export const DOC_HITS = 5;

const chunks = async () => (await db()).collection<ChunkDoc>(COLLECTIONS.chunks);

type Ranked = Pick<ChunkDoc, '_id' | 'docId' | 'text' | 'locator' | 'ord'>;

/**
 * Hybrid retrieval over ONE Space, for ONE user.
 *
 *   meaning half   $vectorSearch  — finds "attrition" for a question about "churn"
 *   keyword half   $search (BM25) — finds "k1" and "BM25", exact terms embeddings blur
 *   fusion         RRF: score = Σ 1 / (60 + rank). Ranks, not raw scores, because a cosine
 *                  score and a BM25 score are on different scales and cannot be added.
 *
 * Both filters sit INSIDE the search stage. A later $match would search every Space first
 * and throw away what is not this one: recall@5 = 0 with everything indexed.
 */
export async function hybridSearch(spaceId: string, userId: string, query: string, k = DOC_HITS): Promise<DocHit[]> {
  const col = await chunks();
  const project = { $project: { _id: 1, docId: 1, text: 1, locator: 1, ord: 1 } };

  const [vector, text] = await Promise.all([
    embed(query).then((queryVector) =>
      col
        .aggregate<Ranked>([
          {
            $vectorSearch: {
              index: SEARCH_INDEXES.chunksVector,
              path: 'embedding',
              queryVector,
              numCandidates: CANDIDATES * 10,
              limit: CANDIDATES,
              filter: { spaceId, userId }
            }
          },
          project
        ])
        .toArray()
    ),
    col
      .aggregate<Ranked>([
        {
          $search: {
            index: SEARCH_INDEXES.chunksText,
            compound: {
              must: [{ text: { query, path: 'text' } }],
              filter: [{ equals: { path: 'spaceId', value: spaceId } }, { equals: { path: 'userId', value: userId } }]
            }
          }
        },
        { $limit: CANDIDATES },
        project
      ])
      .toArray()
  ]);

  // ---- RRF: a chunk ranked well by both halves beats one ranked first by only one
  const fused = new Map<string, { chunk: Ranked; score: number }>();
  for (const list of [vector, text]) {
    list.forEach((chunk, rank) => {
      const entry = fused.get(chunk._id) ?? { chunk, score: 0 };
      entry.score += 1 / (RRF_K + rank + 1);
      fused.set(chunk._id, entry);
    });
  }
  const ranked = [...fused.values()].sort((a, b) => b.score - a.score);

  // ---- group by page: one source per (document, page). Two sources on the same page would
  // share one citation key, and the top 5 would spend two slots on one page.
  const groups = new Map<string, { docId: string; locator: Locator; score: number; parts: Ranked[] }>();
  for (const { chunk, score } of ranked) {
    const key = `${chunk.docId}|${JSON.stringify(chunk.locator)}`;
    const g = groups.get(key);
    if (g) g.parts.push(chunk);
    else if (groups.size < k) groups.set(key, { docId: chunk.docId, locator: chunk.locator, score, parts: [chunk] });
  }

  const titles = new Map(
    (await (await documents()).find({ _id: { $in: [...groups.values()].map((g) => g.docId) } }, { projection: { title: 1 } }).toArray()).map(
      (d) => [d._id, d.title]
    )
  );

  return [...groups.values()].map((g) => ({
    docId: g.docId,
    title: titles.get(g.docId) ?? g.docId,
    locator: g.locator,
    text: joinChunks(g.parts.sort((a, b) => a.ord - b.ord)),
    score: g.score
  }));
}

/**
 * Neighbouring chunks overlap by up to 30 words. Drop the repeated start of each chunk so the
 * merged passage reads once — still a verbatim piece of the document, just without the echo.
 */
function joinChunks(parts: Ranked[]): string {
  let out = parts[0]?.text ?? '';
  for (let i = 1; i < parts.length; i++) {
    const next = parts[i]!.text;
    const adjacent = parts[i]!.ord === parts[i - 1]!.ord + 1;
    let cut = 0;
    if (adjacent) {
      // the longest start of `next` that is also the end of `out`
      const words = next.split(' ');
      for (let n = Math.min(40, words.length - 1); n > 0; n--) {
        if (out.endsWith(words.slice(0, n).join(' '))) {
          cut = n;
          break;
        }
      }
    }
    const rest = next.split(' ').slice(cut).join(' ');
    out += adjacent ? ` ${rest}` : ` … ${rest}`;
  }
  return out;
}

/** Documents in a Space that are ready to search, for the router's prompt. */
export async function indexedTitles(spaceId: string, userId: string): Promise<string[]> {
  return (await (await documents()).find({ spaceId, userId, status: 'indexed' }, { projection: { title: 1 } }).limit(30).toArray()).map(
    (d) => d.title
  );
}
