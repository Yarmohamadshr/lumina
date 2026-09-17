import type { NextFunction, Request, Response } from 'express';
import pino from 'pino';
import { COLLECTIONS, newId, REQUEST_HEADER, USER_HEADER, type RequestDoc } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';

const log = pino({ level: env.logLevel });

/** Extras the ask handler hangs on res.locals for this row; /stats is computed from them. */
export type RequestExtras = { ttftMs?: number; searchCached?: boolean };
type RequestRow = RequestDoc & RequestExtras;

/**
 * One log line and one `requests` row per request, with the SAME requestId the gateway used,
 * so `grep req_…` finds the request in both services' logs. /stats is then computed from these
 * rows and the run logs, which is what makes its numbers reconcile with reality.
 */
export function requestLog(req: Request, res: Response, next: NextFunction): void {
  const started = Date.now();
  const requestId = req.header(REQUEST_HEADER) || newId('req');
  res.locals.requestId = requestId;
  res.setHeader(REQUEST_HEADER, requestId);

  let recorded = false;
  const record = () => {
    if (recorded) return; // 'finish' and 'close' can both fire
    recorded = true;
    const ms = Date.now() - started;
    const row: RequestRow = {
      requestId,
      userId: req.header(USER_HEADER) || 'anonymous',
      route: `${req.method} ${req.route?.path ?? req.path}`,
      status: res.statusCode,
      ms,
      createdAt: new Date().toISOString(),
      ...(res.locals.requestExtras as RequestExtras | undefined)
    };
    log.info(row, 'request');
    if (req.path === '/health') return; // health is polled constantly; it would drown /stats
    void (async () => {
      try {
        await (await db()).collection<RequestRow>(COLLECTIONS.requests).insertOne(row);
      } catch (err) {
        log.error({ err, requestId }, 'request row NOT saved');
      }
    })();
  };

  res.on('finish', record);
  res.on('close', record);
  next();
}
