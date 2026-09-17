import type { NextFunction, Request, Response } from 'express';
import type { ZodTypeAny } from 'zod';
import { USER_HEADER } from '@lumina/contract';

const requestId = (res: Response) => String(res.locals.requestId);

/** 401 without X-User-Id. Every route except /health and /evals/report.json. */
export function requireUser(req: Request, res: Response, next: NextFunction): void {
  const userId = req.header(USER_HEADER)?.trim();
  if (!userId) {
    res.status(401).json({ error: `missing ${USER_HEADER} header`, status: 401, requestId: requestId(res) });
    return;
  }
  res.locals.userId = userId;
  next();
}

/**
 * 400 with the zod message when the body breaks the contract, checked HERE at the edge,
 * before the request costs anything. The parsed body (with defaults like depth:"quick")
 * replaces the raw one, so the agent receives exactly what the contract describes.
 */
export const validateBody =
  (schema: ZodTypeAny) =>
  (req: Request, res: Response, next: NextFunction): void => {
    const parsed = schema.safeParse(req.body ?? {});
    if (!parsed.success) {
      const error = parsed.error.issues.map((i) => `${i.path.join('.') || 'body'}: ${i.message}`).join('; ');
      res.status(400).json({ error, status: 400, requestId: requestId(res) });
      return;
    }
    req.body = parsed.data;
    next();
  };

/** 400 when a url id has the wrong shape (e.g. a threadId that is not thr_…). */
export const validateParam =
  (name: string, schema: ZodTypeAny) =>
  (req: Request, res: Response, next: NextFunction): void => {
    if (!schema.safeParse(req.params[name]).success) {
      res.status(400).json({ error: `invalid ${name}`, status: 400, requestId: requestId(res) });
      return;
    }
    next();
  };
