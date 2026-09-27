import { config } from 'dotenv';
import { resolve } from 'node:path';

// The single .env at the repo root. Provider keys are read HERE and nowhere else.
config({ path: resolve(process.cwd(), '../../.env') });
config({ path: resolve(process.cwd(), '.env') });

const num = (v: string | undefined, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};
export const env = {
  port: num(process.env.PORT_AGENT ?? process.env.PORT, 8000),
  mongoUri: process.env.MONGODB_URI ?? '',
  mongoDb: process.env.MONGODB_DB ?? 'lumina',
  vectorBackend: (process.env.VECTOR_BACKEND ?? 'atlas-vector-search') as
    | 'atlas-vector-search'
    | 'mongo-cosine-scan',

  llmProvider: process.env.LLM_PROVIDER ?? 'anthropic',
  llmModel: process.env.LLM_MODEL ?? 'claude-sonnet-5',
  /** Deep search's plan is its first paint (SLA: 4 s), so it runs on a faster, cheaper model. */
  plannerModel: process.env.LLM_PLANNER_MODEL ?? 'claude-haiku-4-5',
  /**
   * Writes QUICK web answers (learner's decision, Day 9): first token ~0.5 s vs ~1.06 s on Sonnet 5
   * (5 runs each, same prompt), at half the price. Deep answers and the loop stay on LLM_MODEL.
   */
  quickAnswerModel: process.env.LLM_QUICK_ANSWER_MODEL ?? 'claude-haiku-4-5',
  /** Effort for the research loop's tool-picking turns (Sonnet 5: low | medium | high | xhigh | max). */
  loopEffort: (process.env.LLM_LOOP_EFFORT ?? 'low') as 'low' | 'medium' | 'high' | 'xhigh' | 'max',

  searchProvider: (process.env.SEARCH_PROVIDER ?? 'tavily') as 'tavily' | 'serpapi',
  searchCacheTtlSeconds: num(process.env.SEARCH_CACHE_TTL_SECONDS, 21600),
  /** Tavily search_depth: basic | fast | ultra-fast | advanced (advanced costs 2 credits). */
  searchDepth: process.env.SEARCH_DEPTH ?? 'fast',

  embeddingModel: process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small',

  // Deep search is the expensive gear, so its limits are configuration, not code.
  deepSubQuestionsMin: num(process.env.DEEP_SUB_QUESTIONS_MIN, 3),
  deepSubQuestionsMax: num(process.env.DEEP_SUB_QUESTIONS_MAX, 6),
  deepDailyCap: num(process.env.DEEP_DAILY_CAP, 5),
  /** Subagents in flight at once. 1 = sequential, which is how the parallel gain is measured. */
  deepConcurrency: num(process.env.DEEP_CONCURRENCY, 3),

  // The hard caps from AGENTS.md. Raising these to make a gate pass is the failure mode
  // the caps exist to catch. Two gears, two envelopes.
  maxToolCalls: num(process.env.MAX_TOOL_CALLS, 8),
  maxWallClockSec: num(process.env.MAX_WALL_CLOCK_SEC, 90),
  maxToolCallsDeep: num(process.env.MAX_TOOL_CALLS_DEEP, 24),
  maxWallClockSecDeep: num(process.env.MAX_WALL_CLOCK_SEC_DEEP, 240),

  logLevel: process.env.LOG_LEVEL ?? 'info',
  /** Where the per-answer run logs land. quality/check.mjs reads this folder. */
  runsDir: resolve(process.cwd(), '../../runs')
} as const;

/** Never log or return these. /health names the model; it never echoes a key. */
export const secrets = {
  anthropic: process.env.ANTHROPIC_API_KEY ?? '',
  openai: process.env.OPENAI_API_KEY ?? '',
  tavily: process.env.TAVILY_API_KEY ?? '',
  serpapi: process.env.SERPAPI_API_KEY ?? ''
} as const;
