/**
 * LUMINA gateway — the software backend. PROVIDED SKELETON: YOU BUILD THIS OUT.
 *
 * What is already here: the server, CORS, the request id, the pino request log, /health
 * (which nests the agent service's health), a 501 for every contract route, and the
 * static hosting of web/dist. That is deliberately the boring half.
 *
 * What you build (backend/gateway/, see README Part 2):
 *   1. X-User-Id enforcement           → 401 without it, on every route but /health
 *   2. zod validation from @lumina/contract → 400 on a bad body, with the zod message
 *   3. a per-user rate limit           → 429
 *   4. the proxy to the agent service, and SSE pass-through for /threads/:id/ask
 *   5. 502 for any upstream failure    → never a 2xx when the agent threw
 *
 * The browser talks ONLY to this service. No provider key is ever read here.
 */
import express from 'express';
import cors from 'cors';
import { pinoHttp } from 'pino-http';
import pino from 'pino';
import { randomUUID } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  AskBody,
  CreateSpaceBody,
  CreateThreadBody,
  HealthResponse,
  MAX_UPLOAD_BYTES,
  REQUEST_HEADER,
  ROUTES,
  SpaceId,
  ThreadId,
  USER_HEADER
} from '@lumina/contract';
import { env } from './env.js';
import { requireUser, validateBody, validateParam } from './checks.js';
import { rateLimit } from './ratelimit.js';
import { proxy } from './proxy.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use(cors({ origin: env.corsOrigins, credentials: false, exposedHeaders: [REQUEST_HEADER] }));

// One request id, reused if the caller sent one, generated if not, forwarded to the agent
// service and logged by both. This is what makes one request greppable end to end.
app.use((req, res, next) => {
  const id = (req.header(REQUEST_HEADER) ?? `req_${randomUUID().slice(0, 12)}`).trim();
  res.locals.requestId = id;
  res.setHeader(REQUEST_HEADER, id);
  next();
});

app.use(
  pinoHttp({
    logger: log,
    genReqId: (_req, res) => String(res.locals.requestId),
    customProps: (req, res) => ({
      requestId: res.locals.requestId,
      userId: req.header(USER_HEADER) ?? null
    }),
    // The ask route is a stream; one line when it closes is the useful line.
    autoLogging: true
  })
);

// JSON everywhere except the multipart upload route, which your handler owns.
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

// ---------------------------------------------------------------- /health (implemented)

app.get('/health', async (_req, res) => {
  let ai: { status: 'ok' | 'down' } & Record<string, unknown> = { status: 'down' };
  try {
    const upstream = await fetch(`${env.agentUrl}/health`, { signal: AbortSignal.timeout(3000) });
    const body = (await upstream.json()) as Record<string, unknown>;
    ai = { ...body, status: upstream.ok ? 'ok' : 'down' };
  } catch (err) {
    // Health tells the truth about a dead dependency. It never pretends.
    ai = { status: 'down', error: (err as Error).message };
  }

  const body: HealthResponse = {
    status: ai.status === 'ok' ? 'ok' : 'degraded',
    model: String(ai.model ?? 'unset'),
    searchProvider: (ai.searchProvider as HealthResponse['searchProvider']) ?? 'tavily',
    vectorStore: (ai.vectorStore as HealthResponse['vectorStore']) ?? 'atlas-vector-search',
    db: (ai.db as HealthResponse['db']) ?? 'down',
    ai
  };
  res.status(ai.status === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- built routes
// Registered BEFORE the 501 loop below: Express uses the first route that matches.

// Express 4 does not catch errors thrown in async handlers; wrap() hands them to the 502 handler.
const wrap =
  (fn: (req: express.Request, res: express.Response) => Promise<void>): express.RequestHandler =>
  (req, res, next) =>
    fn(req, res).catch(next);

// The order of each chain matters: 401 → 400 → 429 → proxy.
// Body before thread: an empty body on an unknown thread is a 400, not a 404 (bench probe).
app.post('/threads', requireUser, validateBody(CreateThreadBody), wrap(proxy));
app.get('/threads', requireUser, wrap(proxy));
app.get('/threads/:threadId', requireUser, validateParam('threadId', ThreadId), wrap(proxy));
app.post('/threads/:threadId/ask', requireUser, validateBody(AskBody), validateParam('threadId', ThreadId), rateLimit, wrap(proxy));

// Contract routes the agent does not build yet: pass them through, so its 501 reaches the UI
// and each agent route lights up in the browser the moment it exists.
app.get('/stats', requireUser, wrap(proxy));
app.get('/memory', requireUser, wrap(proxy));
app.delete('/memory/:memoryId', requireUser, wrap(proxy));
app.post('/spaces', requireUser, validateBody(CreateSpaceBody), wrap(proxy));
app.get('/spaces', requireUser, wrap(proxy));
app.get('/spaces/:spaceId/documents', requireUser, validateParam('spaceId', SpaceId), wrap(proxy));

// Uploads: an obviously oversized body is refused here (413) before a byte reaches the agent.
// The exact 25 MB check is the agent's (multer); the 1 MB slack is multipart framing.
const refuseHugeUpload: express.RequestHandler = (req, res, next) => {
  if (Number(req.header('content-length') ?? 0) > MAX_UPLOAD_BYTES + 1024 * 1024) {
    res.status(413).json({ error: 'file too large: the limit is 25 MB', status: 413, requestId: String(res.locals.requestId) });
    return;
  }
  next();
};
app.post('/spaces/:spaceId/documents', requireUser, validateParam('spaceId', SpaceId), refuseHugeUpload, wrap(proxy));

// ---------------------------------------------------------------- the evals report

/**
 * The Product Evaluation the provided /evals page renders. Built offline by
 * eval/build-report.mjs from reports/bench.json, reports/quality.json and runs/*.json,
 * then baked into the image, so serving it is a file read and nothing is recomputed
 * here. No auth: it is the submission, and a grader opens it without a user id.
 */
const REPORT_PATHS = [
  process.env.EVALS_REPORT_PATH,
  resolve(process.cwd(), '../../reports/report.json'),
  resolve(process.cwd(), 'reports/report.json')
].filter(Boolean) as string[];

app.get('/evals/report.json', (_req, res) => {
  const hit = REPORT_PATHS.find((p) => existsSync(p));
  if (!hit) {
    res.status(404).json({
      error: 'no evals report on this deployment — run eval/build-report.mjs and redeploy',
      status: 404
    });
    return;
  }
  res.type('application/json').send(readFileSync(hit, 'utf8'));
});

// ---------------------------------------------------------------- everything else: 501

/**
 * Every contract route answers 501 until you implement it. The UI renders that as
 * "not implemented yet", so the interface is your progress bar: each route you finish
 * lights up a piece of the product.
 */
const notImplemented = (route: string) => (_req: express.Request, res: express.Response) => {
  res.status(501).json({
    error: `not implemented yet: ${route}. Build it in backend/gateway/src/.`,
    status: 501,
    requestId: String(res.locals.requestId)
  });
};

for (const route of ROUTES) {
  if (route.path === '/health') continue;
  const path = route.path.replace(/:(\w+)/g, ':$1');
  const method = route.method.toLowerCase() as 'get' | 'post' | 'delete';
  app[method](path, notImplemented(`${route.method} ${route.path}`));
}


// ---------------------------------------------------------------- static UI

// In production the gateway serves the built UI, so / and /evals come from one origin.
if (existsSync(env.webDist)) {
  app.use(express.static(env.webDist));
  app.get(/^(?!\/(health|stats|threads|memory|spaces|artifacts|evals)).*/, (_req, res) => {
    res.sendFile(`${env.webDist}/index.html`);
  });
}

app.use((req, res) => {
  res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 });
});

// A thrown error is a 502 with a log line, never a 200 with a plausible body (rule A1).
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  log.error({ err, requestId: res.locals.requestId }, 'gateway error');
  res.status(502).json({ error: err.message, status: 502, requestId: String(res.locals.requestId) });
});

app.listen(env.port, () => {
  log.info(
    { port: env.port, agentUrl: env.agentUrl, cors: env.corsOrigins },
    'gateway up — every route but /health returns 501 until you build it'
  );
});
