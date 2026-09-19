/**
 * The jobs worker: its own process (`npm run worker`; on Fly, a second process group in the
 * agent app). It must not run on the thread that streams answers — the bench measures search
 * p95 DURING a 60-page ingest, so a blocking parse would show up as a failed SLA.
 *
 * index_document → GridFS read → parse (page-aware) → chunk → embed → upsert into chunks
 *                → READ-YOUR-WRITE PROBE → status: 'indexed'
 *
 * The only link to the upload route is the `jobs` collection (DESIGN Q3): if this process is
 * down, uploads still get 202 and their jobs wait as `pending`. Nothing is lost.
 */
import { hostname } from 'node:os';
import { ObjectId } from 'mongodb';
import pino from 'pino';
import { COLLECTIONS, SEARCH_INDEXES, type ChunkDoc, type DocStatus, type DocumentDoc, type JobDoc } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { embedMany } from './embed.js';
import { parse } from './parse.js';
import { chunk } from './chunk.js';
import { documents, jobs, uploadsBucket } from './spaces.js';

const log = pino({ level: env.logLevel, base: { svc: 'worker' } });
const workerId = `${hostname()}-${process.pid}`;

/** Idle poll interval (DESIGN Q3: polling, not change streams). */
const POLL_MS = 2000;
/** A `running` job not heard from in this long belongs to a dead worker: the sweeper frees it. */
const STALE_MS = 5 * 60_000;
const SWEEP_EVERY_MS = 60_000;
/** Transient failures (provider down, network) are retried this many times in total. */
const MAX_ATTEMPTS = 3;
/** Chunks per embeddings call. */
const EMBED_BATCH = 64;
/** How long the probe waits for Atlas Search to catch up before failing the document. */
const PROBE_TIMEOUT_MS = 90_000;

const chunks = async () => (await db()).collection<ChunkDoc>(COLLECTIONS.chunks);

/** A failure that no retry can fix (unreadable file, no text): fail the document at once. */
class Permanent extends Error {}

// ---------------------------------------------------------------- claim + sweep

/**
 * Atomic claim: find a pending job AND mark it running in one operation, so two workers can
 * never take the same job. (find-then-update in two steps has a window where both do.)
 */
async function claim(): Promise<JobDoc | null> {
  return (await jobs()).findOneAndUpdate(
    { status: 'pending', kind: 'index_document' },
    { $set: { status: 'running', claimedAt: new Date(), workerId }, $inc: { attempts: 1 } },
    { sort: { createdAt: 1 }, returnDocument: 'after' }
  );
}

/**
 * The janitor: a job stuck `running` with an old claimedAt goes back to `pending` — unless it has
 * already been tried MAX_ATTEMPTS times. A file that kills the worker itself (out of memory on a
 * huge PDF) never reaches handle()'s catch, so without this it would crash the worker forever.
 */
async function sweep(): Promise<void> {
  const stale = { status: 'running' as const, claimedAt: { $lt: new Date(Date.now() - STALE_MS) } };
  const dead = await (await jobs()).find({ ...stale, attempts: { $gte: MAX_ATTEMPTS } }).toArray();
  for (const job of dead) {
    const error = `the worker stopped while indexing this file ${job.attempts} times (too large or malformed?)`;
    await (await jobs()).updateOne({ _id: job._id }, { $set: { status: 'failed', error }, $unset: { workerId: '' } });
    await (await documents()).updateOne({ _id: String(job.payload.docId) }, { $set: { status: 'failed', error } });
    log.error({ jobId: job._id, docId: job.payload.docId }, error);
  }
  const { modifiedCount } = await (await jobs()).updateMany(
    { ...stale, attempts: { $lt: MAX_ATTEMPTS } },
    { $set: { status: 'pending' }, $unset: { workerId: '' } }
  );
  if (modifiedCount) log.warn({ modifiedCount }, 'sweeper returned stale jobs to pending');
}

// ---------------------------------------------------------------- one job

async function run(job: JobDoc): Promise<void> {
  const docId = String(job.payload.docId ?? '');
  const doc = await (await documents()).findOne({ _id: docId });
  if (!doc) throw new Permanent(`document ${docId} not found (deleted?)`);
  if (doc.status === 'indexed') return; // already done by an earlier attempt

  /** Progress for the user, and a heartbeat for the sweeper: every step refreshes claimedAt. */
  const progress = async (status: DocStatus, pct: number, extra: Partial<DocumentDoc> = {}) => {
    await (await documents()).updateOne({ _id: docId }, { $set: { status, pct, ...extra }, $unset: { error: '' } });
    await (await jobs()).updateOne({ _id: job._id }, { $set: { claimedAt: new Date() } });
  };

  // ---- 1. read + parse (cheap, so it is simply re-run on a retry)
  await progress('parsing', 5);
  const bytes = await readUpload(doc.fileId);
  let parsed;
  try {
    parsed = await parse(bytes, doc.mimeType);
  } catch (err) {
    throw new Permanent(`could not parse ${doc.title}: ${(err as Error).message}`);
  }
  const pieces = chunk(parsed.sections);
  if (!pieces.length) {
    throw new Permanent(`no extractable text in ${doc.title} (a scanned PDF needs OCR, which LUMINA does not do)`);
  }
  await progress('embedding', 25, parsed.pages ? { pages: parsed.pages } : {});

  // ---- 2. embed + upsert, skipping chunks a previous attempt already paid for
  const col = await chunks();
  const idOf = (ord: number) => `${docId}:${ord}`;
  const already = new Map(
    (await col.find({ docId }, { projection: { _id: 1, text: 1 } }).toArray()).map((c) => [c._id, c.text])
  );
  const todo = pieces.filter((p) => already.get(idOf(p.ord)) !== p.text);
  log.info({ docId, chunks: pieces.length, reused: pieces.length - todo.length }, 'embedding');

  for (let i = 0; i < todo.length; i += EMBED_BATCH) {
    const batch = todo.slice(i, i + EMBED_BATCH);
    const vectors = await embedMany(batch.map((p) => p.text));
    await col.bulkWrite(
      batch.map((p, j) => ({
        replaceOne: {
          filter: { _id: idOf(p.ord) },
          replacement: {
            _id: idOf(p.ord),
            docId,
            spaceId: doc.spaceId, // from the document row, never from the job payload
            userId: doc.userId,
            text: p.text,
            locator: p.locator,
            ord: p.ord,
            embedding: vectors[j]!,
            createdAt: new Date()
          },
          upsert: true
        }
      }))
    );
    await progress('embedding', 25 + Math.round((65 * Math.min(i + EMBED_BATCH, todo.length)) / todo.length));
  }
  await col.deleteMany({ docId, ord: { $gte: pieces.length } }); // leftovers from an older, longer parse

  // ---- 3. the probe: written is not searchable. Only a successful probe earns `indexed`.
  await progress('embedding', 95);
  const ms = await probe(docId, doc.spaceId);
  await (await documents()).updateOne(
    { _id: docId },
    { $set: { status: 'indexed', pct: 100, chunks: pieces.length }, $unset: { error: '' } }
  );
  log.info({ docId, chunks: pieces.length, pages: parsed.pages, probeMs: ms }, 'indexed');
}

async function readUpload(fileId: string): Promise<Buffer> {
  const parts: Buffer[] = [];
  for await (const part of (await uploadsBucket()).openDownloadStream(new ObjectId(fileId))) parts.push(part as Buffer);
  return Buffer.concat(parts);
}

/**
 * Read-your-write probe. Atlas Search indexes are eventually consistent: insertMany returns
 * before the chunks are findable. So search for one of OUR chunks through BOTH indexes that
 * hybrid retrieval will use — with the filter inside the search stage, exactly like a real
 * query — until it comes back. Search with the chunk's own vector (a perfect match) and its
 * own words, so "not found" can only mean "not indexed yet".
 */
async function probe(docId: string, spaceId: string): Promise<number> {
  const col = await chunks();
  const sample = await col.findOne({ docId }, { sort: { ord: 1 } });
  if (!sample) throw new Error('probe: no chunk was written');
  const words = sample.text.split(/\s+/).slice(0, 12).join(' ');

  const t0 = Date.now();
  let vectorOk = false;
  let textOk = false;
  while (Date.now() - t0 < PROBE_TIMEOUT_MS) {
    if (!vectorOk) {
      const hits = await col
        .aggregate<{ docId: string }>([
          {
            $vectorSearch: {
              index: SEARCH_INDEXES.chunksVector,
              path: 'embedding',
              queryVector: sample.embedding,
              numCandidates: 100,
              limit: 20,
              filter: { spaceId }
            }
          },
          { $project: { docId: 1 } }
        ])
        .toArray();
      vectorOk = hits.some((h) => h.docId === docId);
    }
    if (!textOk) {
      const hits = await col
        .aggregate<{ docId: string }>([
          {
            $search: {
              index: SEARCH_INDEXES.chunksText,
              compound: { must: [{ text: { query: words, path: 'text' } }], filter: [{ equals: { path: 'spaceId', value: spaceId } }] }
            }
          },
          { $limit: 20 },
          { $project: { docId: 1 } }
        ])
        .toArray();
      textOk = hits.some((h) => h.docId === docId);
    }
    if (vectorOk && textOk) return Date.now() - t0;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(
    `read-your-write probe timed out after ${PROBE_TIMEOUT_MS / 1000}s (vector ${vectorOk ? 'ok' : 'missing'}, text ${textOk ? 'ok' : 'missing'})`
  );
}

// ---------------------------------------------------------------- the loop

/** Success → job done. Failure → retry if it might help, else fail the document LOUDLY. */
async function handle(job: JobDoc): Promise<void> {
  const docId = String(job.payload.docId ?? '');
  const t0 = Date.now();
  try {
    await run(job);
    await (await jobs()).updateOne({ _id: job._id }, { $set: { status: 'done' }, $unset: { error: '' } });
    log.info({ jobId: job._id, docId, ms: Date.now() - t0 }, 'job done');
  } catch (err) {
    const message = (err as Error).message || 'unknown error';
    const final = err instanceof Permanent || job.attempts >= MAX_ATTEMPTS;
    log.error({ jobId: job._id, docId, attempt: job.attempts, final, err: message }, 'job failed');
    await (await jobs()).updateOne(
      { _id: job._id },
      { $set: { status: final ? 'failed' : 'pending', error: message }, $unset: { workerId: '' } }
    );
    // Never a silent stall: a document that will not be retried says why, in the list the user sees.
    if (final) await (await documents()).updateOne({ _id: docId }, { $set: { status: 'failed', error: message } });
  }
}

let stopping = false;
for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    log.info({ sig }, 'stopping after the current job');
    stopping = true;
  });
}

async function main(): Promise<void> {
  log.info({ workerId, pollMs: POLL_MS }, 'jobs worker up');
  let lastSweep = 0;
  while (!stopping) {
    try {
      if (Date.now() - lastSweep > SWEEP_EVERY_MS) {
        await sweep();
        lastSweep = Date.now();
      }
      const job = await claim();
      if (job) {
        await handle(job);
        continue; // more may be waiting: don't sleep between jobs
      }
    } catch (err) {
      // Mongo unreachable etc.: log, wait, try again. The job rows are the state, not this loop.
      log.error({ err: (err as Error).message }, 'worker loop error');
    }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
  process.exit(0);
}

void main();
