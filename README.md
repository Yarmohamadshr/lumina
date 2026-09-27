# LUMINA — a Perplexity-style AI search engine

Ask a question and get a **streamed answer with clickable citations**, built from a live web search or from
your own uploaded documents. Ask a harder one and it **plans sub-questions, researches them in parallel**, and
merges everything into one citation list. It remembers your preferences across conversations.

**▶ Demo video:** https://youtu.be/VYFzADuqQ2I (9 min, the full product in production)

> The production deployment (Vercel + Fly.io + MongoDB Atlas) ran from 2026-09-18 to 2026-09-24 and was shut
> down after evaluation to stop costs. The redeploy commands are under [Deploy](#deploy).

---

## What it does

| Mode | Behaviour |
|---|---|
| **Quick** | Web search → answer streams in about 1.6 s (p95 time to first word), with every citation quoting text that really is on the cited page |
| **Deep** | Plans up to 4 sub-questions, researches each with an **isolated sub-agent in parallel**, merges and renumbers sources; every step and source is tagged with its sub-question |
| **Documents** | Upload PDF/Markdown/text into a *Space*; a background worker parses, chunks, embeds and indexes it; answers cite **document + page** |
| **Memory** | "I prefer TypeScript examples" is saved and recalled in *new* conversations; listed and deletable per user |

## Architecture

```
 Browser (React UI, Vercel)
    │  HTTP + Server-Sent Events
    ▼
 Gateway  (Express, Fly.io, public)      auth header check · validation · rate limit · request ids · SSE pass-through
    │  Fly private network only
    ▼
 Agent service (Express, Fly.io, NO public IP)      the only place that holds provider keys
    ├─ quick pipeline ── Tavily search ∥ memory recall ∥ Haiku triage → speculative answer behind a citation gate
    ├─ deep search ───── Haiku planner → parallel sub-agents (Sonnet tool loops) sharing one budget → merge
    ├─ documents ─────── hybrid retrieval: Atlas Vector Search + BM25, fused with RRF, filtered per Space
    └─ worker (separate process group) ── jobs queue in MongoDB → parse → chunk → embed → read-your-write probe
    │
    ▼
 MongoDB Atlas: threads · messages · memories · documents/chunks (+GridFS) · jobs · search cache (TTL) · run logs
 Providers: Anthropic (Claude Sonnet 5, Haiku 4.5) · OpenAI embeddings · Tavily web search
```

Design rationale, state ownership and trade-offs are in [`DESIGN.md`](DESIGN.md).

## Measured results (production, 40-query benchmark)

| Metric | Result | Target |
|---|---|---|
| Time to first word, p95 | **1,631 ms** | ≤ 2,500 ms |
| Citation grounding (quoted text found on the cited page's raw HTML) | **1.000** (118/118, 0 dangling) | — |
| Document retrieval recall@5 (39 gold questions) | **1.0** | ≥ 0.70 |
| Cost per quick answer | **$0.0052** | ≤ $0.05 |
| Cost per deep answer | **$0.1156** | ≤ $0.35 |
| Deep search: parallel vs sequential | **35 s vs 59 s**, same 8 sources | faster |
| Error rate | **0** | — |

All 16 service-level targets passed. Honest caveat: a *first-time* question takes about 4.5 s to its first word;
the p95 above includes cache hits for repeated questions, as the benchmark's traffic mix specifies.

## Engineering stories

**Time to first word: 24.4 s → 1.6 s.** Quick answers began as a Claude tool loop, which meant several model
round-trips before any text appeared. I rebuilt them as a pipeline: web search, memory recall and a small Haiku
"triage" call run concurrently, then Haiku starts answering *speculatively*. Its tokens are held behind a gate
until a live citation check passes (at most first words + 150 ms), so speed never skips verification.

**Grounding is checked against raw HTML, not the search API's markdown.** Citations looked right in our own
checks but failed the independent one, which downloads each page and strips the tags. Some sites render text
with JavaScript (invisible in raw HTML), and entities like `&rsquo;` change the words. Snippets are now taken from
the visible text and verified with the same algorithm as the checker. Result: 118/118 grounded.

**recall@5 38/39 → 39/39 from one regex.** The sentence splitter treated "1.2" as a sentence end, cutting a chunk
mid-fact. Filters also go **inside** `$vectorSearch`; a later `$match` gives recall 0 while everything looks indexed.

**Cost: sending whole pages to the model cost $0.060 per answer.** Cutting to the first 900 characters dropped it
to $0.044 but missed answers further down the page, so `focus()` now sends the 900 characters that best match
the question: 13 s → 4.9 s and the answers came back.

**A health check that took the site down.** When the agent service stopped, the gateway's `/health` honestly
returned 503, and the platform read that as "this machine is broken" and removed the only gateway from the load
balancer. Instead of a clear "agent down" error, the whole site went dark for about 2 hours. Liveness (is the
process up?) and readiness (are its dependencies up?) are different questions, and the platform check conflated
them. Fix: no platform check on `/health`, so callers get the truthful 503. I wrote it up as a lint rule (D1)
that fails on the exact config that caused it.

**The spend cap lives behind the private network.** The deep-search daily cap is enforced in the agent service,
which has no public IP, so no client can bypass it. It's counted from the run logs, the single source of truth,
rather than a separate counter that could drift.

## Tech stack

TypeScript · Node 24 · Express · React (provided UI) · zod (typed API contract) · MongoDB Atlas (Vector Search,
full-text search, GridFS, TTL indexes) · Anthropic SDK (Claude Sonnet 5, Haiku 4.5) · OpenAI embeddings ·
Tavily · pdf.js · Server-Sent Events · Fly.io (private networking, process groups) · Vercel

## Repository map

| Path | What | Author |
|---|---|---|
| `backend/agent/src/` | Agent service: pipelines, tool loop, deep search, memory, RAG, worker, cache, run logs, stats | **me** |
| `backend/gateway/src/` | Gateway: checks, rate limiting, SSE proxy | **me** |
| `DESIGN.md` | Architecture decisions and trade-offs | **me** |
| `backend/*/Dockerfile`, `fly.toml`, `vercel.json` | Deployment | **me** |
| `quality/` precedents + rule D1 | Lessons from real incidents, as executable rules | **me** (rule framework provided) |
| `web/`, `packages/contract/` | React UI and the typed API contract | provided |
| `benchmark/`, `eval/`, `quality/check.mjs` | Benchmark, gold set, evaluation harness | provided |
| `PRD.md`, `SPEC.md`, `TECHNICAL.md`, `AGENTS.md`, [`docs/BRIEF.md`](docs/BRIEF.md) | The product brief I built against | provided |

## Run it locally

```bash
npm install
cp .env.example .env        # MongoDB Atlas URI + Anthropic, OpenAI and Tavily keys
npm run indexes             # Atlas collections + vector/text search indexes
VITE_API_URL=http://localhost:8787 npm run dev    # UI :5173 · gateway :8787 · agent :8000
npm run worker              # document-indexing worker, in a second terminal
```

Verify: `node benchmark/bench.mjs` (latency, grounding, recall, cache, cost) and `node quality/check.mjs .`

## Deploy

```bash
fly deploy --config backend/agent/fly.toml   --dockerfile backend/agent/Dockerfile   --remote-only --ha=false .
fly deploy --config backend/gateway/fly.toml --dockerfile backend/gateway/Dockerfile --remote-only --ha=false .
npx vercel --prod --yes -b VITE_API_URL=https://<gateway-app>.fly.dev
```

The agent app has no public IP; the gateway reaches it at `app.process.<agent-app>.internal:8000` (the process
group name is needed because the worker shares the app).

---

*Built as the first project of the FDE Agent Engineering Bootcamp (2026). The UI, API contract, benchmark and
evaluation harness were provided as the product specification; the backend services, deployment, design and
the performance and grounding work are mine.*
