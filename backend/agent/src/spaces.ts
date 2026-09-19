import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { randomUUID } from 'node:crypto';
import type { NextFunction, Request, Response } from 'express';
import multer from 'multer';
import { GridFSBucket } from 'mongodb';
import {
  ACCEPTED_UPLOAD_TYPES,
  COLLECTIONS,
  CreateSpaceBody,
  GRIDFS_BUCKETS,
  MAX_UPLOAD_BYTES,
  newId,
  SpaceId,
  type CreateSpaceResponse,
  type DocumentDoc,
  type JobDoc,
  type ListDocumentsResponse,
  type ListSpacesResponse,
  type SpaceDoc,
  type UploadDocumentResponse
} from '@lumina/contract';
import { db } from './db.js';

const spaces = async () => (await db()).collection<SpaceDoc>(COLLECTIONS.spaces);
export const documents = async () => (await db()).collection<DocumentDoc>(COLLECTIONS.documents);
export const jobs = async () => (await db()).collection<JobDoc>(COLLECTIONS.jobs);
export const uploadsBucket = async () => new GridFSBucket(await db(), { bucketName: GRIDFS_BUCKETS.uploads });

const bad = (res: Response, status: number, error: string) => res.status(status).json({ error, status });

/** The Space, only if it is this user's. Missing and someone else's look the same: 404. */
export async function findSpace(spaceId: string, userId: string): Promise<SpaceDoc | null> {
  if (!SpaceId.safeParse(spaceId).success) return null;
  return (await spaces()).findOne({ _id: spaceId, userId });
}

// ---------------------------------------------------------------- spaces

/** POST /spaces → 201 {spaceId, name}. A Space is just a label that chunks are filtered by. */
export async function createSpace(req: Request, res: Response): Promise<void> {
  const parsed = CreateSpaceBody.safeParse(req.body ?? {});
  if (!parsed.success) {
    bad(res, 400, parsed.error.issues.map((i) => `${i.path.join('.') || 'body'}: ${i.message}`).join('; '));
    return;
  }
  const space: SpaceDoc = { _id: newId('spc'), userId: res.locals.userId, name: parsed.data.name, createdAt: new Date() };
  await (await spaces()).insertOne(space);
  const body: CreateSpaceResponse = { spaceId: space._id, name: space.name };
  res.status(201).json(body);
}

/** GET /spaces — this user's Spaces, newest first. */
export async function listSpaces(_req: Request, res: Response): Promise<void> {
  const rows = await (await spaces()).find({ userId: res.locals.userId }).sort({ createdAt: -1 }).limit(200).toArray();
  const body: ListSpacesResponse = {
    spaces: rows.map((s) => ({ spaceId: s._id, name: s.name, createdAt: new Date(s.createdAt).toISOString() }))
  };
  res.json(body);
}

// ---------------------------------------------------------------- upload

/**
 * multer keeps the file in memory (≤ 25 MB) so it can go straight to GridFS. Its size error
 * becomes the contract's 413 here, instead of reaching the error handler as a 502.
 */
const multerSingle = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_UPLOAD_BYTES, files: 1 } }).single('file');

export function receiveFile(req: Request, res: Response, next: NextFunction): void {
  multerSingle(req, res, (err: unknown) => {
    if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
      bad(res, 413, `file too large: the limit is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB`);
      return;
    }
    if (err) {
      bad(res, 400, `could not read the upload: ${(err as Error).message}`);
      return;
    }
    next();
  });
}

/**
 * Browsers are unreliable about a .md file's type ('' or application/octet-stream), so the
 * extension decides when the declared type is not one we accept.
 */
const BY_EXTENSION: Record<string, (typeof ACCEPTED_UPLOAD_TYPES)[number]> = {
  pdf: 'application/pdf',
  md: 'text/markdown',
  markdown: 'text/markdown',
  txt: 'text/plain'
};

function mimeOf(file: Express.Multer.File): string | null {
  const declared = file.mimetype.split(';')[0]!.trim().toLowerCase();
  if ((ACCEPTED_UPLOAD_TYPES as readonly string[]).includes(declared)) return declared;
  const ext = file.originalname.split('.').pop()?.toLowerCase() ?? '';
  return BY_EXTENSION[ext] ?? null;
}

/**
 * POST /spaces/:spaceId/documents → 202 {docId, status:"pending"}.
 *
 * Three writes and out: raw file → GridFS, a `pending` documents row, a `jobs` row. No
 * parsing here — a 60-page PDF parsed on this thread would stall every answer being
 * streamed at the same moment (the bench measures exactly that).
 */
export async function uploadDocument(req: Request, res: Response): Promise<void> {
  const userId: string = res.locals.userId;
  const spaceId = req.params.spaceId ?? '';
  if (!(await findSpace(spaceId, userId))) {
    bad(res, 404, 'space not found');
    return;
  }

  const file = req.file;
  if (!file) {
    bad(res, 400, 'file is required (multipart field "file")');
    return;
  }
  if (!file.size) {
    bad(res, 400, 'file is empty');
    return;
  }
  const mimeType = mimeOf(file);
  if (!mimeType) {
    bad(res, 400, `unsupported file type "${file.mimetype}": upload PDF, Markdown or plain text`);
    return;
  }

  // multer decodes the filename as latin1; browsers send UTF-8.
  const title = Buffer.from(file.originalname, 'latin1').toString('utf8') || 'untitled';
  const docId = newId('doc');

  const stream = (await uploadsBucket()).openUploadStream(title, { metadata: { docId, spaceId, userId, mimeType } });
  await pipeline(Readable.from(file.buffer), stream);

  const now = new Date();
  await (await documents()).insertOne({
    _id: docId,
    spaceId,
    userId,
    title,
    mimeType,
    bytes: file.size,
    status: 'pending',
    pct: 0,
    fileId: stream.id.toString(),
    createdAt: now
  });
  // The ONLY way work reaches the worker. If the worker is down, this row waits; nothing is lost.
  await (await jobs()).insertOne({
    _id: `job_${randomUUID()}`,
    kind: 'index_document',
    status: 'pending',
    payload: { docId },
    userId,
    attempts: 0,
    createdAt: now
  });

  const body: UploadDocumentResponse = { docId, status: 'pending' };
  res.status(202).json(body);
}

/** GET /spaces/:spaceId/documents — status + pct, which the UI and the bench poll. */
export async function listDocuments(req: Request, res: Response): Promise<void> {
  const spaceId = req.params.spaceId ?? '';
  if (!(await findSpace(spaceId, res.locals.userId))) {
    bad(res, 404, 'space not found');
    return;
  }
  const rows = await (await documents()).find({ spaceId, userId: res.locals.userId }).sort({ createdAt: 1 }).toArray();
  const body: ListDocumentsResponse = {
    documents: rows.map((d) => ({
      docId: d._id,
      title: d.title,
      status: d.status,
      pct: d.pct,
      ...(d.pages ? { pages: d.pages } : {}),
      ...(d.chunks !== undefined ? { chunks: d.chunks } : {}),
      ...(d.error ? { error: d.error } : {})
    }))
  };
  res.json(body);
}
