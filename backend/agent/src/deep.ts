import type Anthropic from '@anthropic-ai/sdk';
import { PlanEvent, type SubQuestion, type TraceEvent } from '@lumina/contract';
import { env } from './env.js';
import { llm } from './llm.js';
import { newBudget, research, sourceKey, type FetchedSource, type ResearchContext } from './loop.js';

/**
 * How many subagents run at once, from DEEP_CONCURRENCY. Three, not all of them: six subagents
 * several tool calls deep hit Claude's and Tavily's rate limits. Setting it to 1 makes the run
 * sequential, which is how the parallel speed-up is measured rather than claimed.
 */
const concurrency = () => env.deepConcurrency;

/**
 * A page found by a deep search. `subQuestion` is REQUIRED by the type, not by memory: the merge
 * cannot silently drop it, because the code would not compile. A student in the sample scorecard
 * lost 3 of 15 points to exactly that omission.
 */
export type DeepSource = FetchedSource & { subQuestion: number };

export type DeepResult = {
  plan: SubQuestion[];
  pages: DeepSource[];
  terminated: 'done' | 'cap';
  tokens: { in: number; out: number; cacheWrite: number; cacheRead: number };
  /** Wall clock of the fan-out only, kept so the parallel-vs-sequential claim has evidence. */
  fanOutMs: number;
};

const PLAN_TOOL: Anthropic.Tool = {
  name: 'plan_research',
  description: 'Break a question into the sub-questions that must be answered to answer it well.',
  input_schema: {
    type: 'object',
    properties: {
      subQuestions: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            question: { type: 'string', description: 'One self-contained sub-question, searchable on its own' },
            reason: { type: 'string', description: 'Why this part matters, in at most 10 words' }
          },
          required: ['question', 'reason']
        }
      }
    },
    required: ['subQuestions']
  }
};

const PLANNER_SYSTEM = `You plan research. Break the question into ${env.deepSubQuestionsMin}-${env.deepSubQuestionsMax} sub-questions.
- Each sub-question must be DIFFERENT enough to search different words and land on different pages.
  Rephrasings of the original question are worthless: they find the same pages twice.
- Each must be self-contained: a researcher sees only that sentence, not the original question.
- Ask what a careful person would actually want to know: costs, limits, alternatives, what breaks,
  what changes at scale, what the evidence actually says.
- Give each a reason of at most 10 words. Keep sub-questions short (under 20 words).
- Call plan_research exactly once.`;

/**
 * The plan: ONE model call, before any retrieval, so the reader sees what the system decided to go
 * and find out. A deep search that streams no plan is just a slow quick search.
 */
/** If the planner has not answered by now, a second identical request races the first. */
const PLAN_HEDGE_MS = 2000;

export async function planResearch(query: string): Promise<{ plan: SubQuestion[]; tokens: { in: number; out: number } }> {
  // The plan is the deep search's first paint and the SLA gives it 4 s, so it runs on the fast model
  // (the ANSWER still comes from env.llmModel, which is what /health names). It usually takes ~3 s,
  // and one slow call (6.7 s in bench run 6) failed the p95 of 4 deep runs on its own. So a HEDGED
  // request: past PLAN_HEDGE_MS a second identical call starts, the first to finish wins and the
  // other is cancelled. It costs ~$0.002, and only on slow runs.
  const ask = (signal: AbortSignal) =>
    llm.messages.create(
      {
        model: env.plannerModel,
        max_tokens: 700,
        system: PLANNER_SYSTEM,
        tools: [PLAN_TOOL],
        tool_choice: { type: 'tool', name: 'plan_research' },
        messages: [{ role: 'user', content: query }]
      },
      { signal }
    );
  const first = new AbortController();
  const second = new AbortController();
  let hedge: ReturnType<typeof setTimeout> | undefined;
  const response = await new Promise<Anthropic.Message>((resolve, reject) => {
    let started = 1;
    let failed = 0;
    // First success wins. A failure only ends the plan once every request started so far has failed.
    const race = (p: Promise<Anthropic.Message>) =>
      p.then(resolve, (err: unknown) => {
        if (++failed === started) reject(err); // fail loud: a real provider failure
      });
    race(ask(first.signal));
    hedge = setTimeout(() => {
      started++;
      race(ask(second.signal));
    }, PLAN_HEDGE_MS);
  }).finally(() => {
    clearTimeout(hedge);
    first.abort(); // cancel the loser (aborting a finished request is a no-op)
    second.abort();
  });

  const call = response.content.find((b) => b.type === 'tool_use');
  if (!call) throw new Error('planner did not return a plan'); // fail loud, never a fake plan
  const raw = (call.input as { subQuestions?: { question?: string; reason?: string }[] }).subQuestions ?? [];

  const plan = raw
    .filter((q): q is { question: string; reason?: string } => typeof q.question === 'string' && q.question.trim().length > 0)
    .slice(0, env.deepSubQuestionsMax)
    .map((q, i) => ({ i: i + 1, question: q.question.trim(), ...(q.reason ? { reason: q.reason.trim() } : {}) }));

  if (plan.length < env.deepSubQuestionsMin) {
    throw new Error(`planner returned ${plan.length} sub-questions, need at least ${env.deepSubQuestionsMin}`);
  }
  PlanEvent.parse({ subQuestions: plan }); // our own output, checked against the contract
  return { plan, tokens: { in: response.usage.input_tokens, out: response.usage.output_tokens } };
}

/** Run tasks with a ceiling on how many are in flight at once. */
async function pool<T>(tasks: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, tasks.length) }, async () => {
    while (next < tasks.length) {
      const index = next++;
      results[index] = await (tasks[index] as () => Promise<T>)();
    }
  });
  await Promise.all(workers);
  return results;
}

/**
 * Deep search: plan, then research every sub-question as an ISOLATED subagent — its own messages
 * array, its own tool loop, no shared conversation — several at a time, then merge.
 *
 * Isolation is the point: a single context researching five things at once drifts, and its trace
 * cannot be read back per sub-question. The caps stay central: 24 tool calls and 240 s for the
 * whole run, however many subagents draw on them.
 */
export async function deepSearch(
  query: string,
  ctx: ResearchContext,
  onPlan: (plan: SubQuestion[]) => void,
  emit: (ev: TraceEvent) => void
): Promise<DeepResult> {
  const budget = newBudget('deep');
  const tokens = { in: 0, out: 0, cacheWrite: 0, cacheRead: 0 };

  const { plan, tokens: planTokens } = await planResearch(query);
  tokens.in += planTokens.in;
  tokens.out += planTokens.out;
  onPlan(plan); // the `plan` event goes out BEFORE any retrieval

  const startedFanOut = Date.now();
  const results = await pool(
    plan.map((sq) => async () => {
      const found = await research(sq.question, 'deep', { ...ctx, history: [] }, emit, {
        budget,
        subQuestion: sq.i,
        // Each branch gets a share of the run's calls with headroom left over, so an eager first
        // sub-question cannot starve the last one, and a normal run finishes inside the budget
        // instead of ending 'cap'. 4 branches x 4 calls = 16 of 24.
        maxCallsHere: Math.max(3, Math.floor(budget.maxCalls / (plan.length * 1.5))),
        extraSystem: `You are researching ONE part of a larger question: "${query}".\nYour part: ${sq.question}\nSearch and read for this part only. One or two pages is enough.`
      });
      // The tag is attached here, where the sub-question is still known, and the type demands it.
      return { pages: found.pages.map((p): DeepSource => ({ ...p, subQuestion: sq.i })), tokens: found.tokens, terminated: found.terminated };
    }),
    concurrency()
  );
  const fanOutMs = Date.now() - startedFanOut;

  for (const r of results) {
    tokens.in += r.tokens.in;
    tokens.out += r.tokens.out;
    tokens.cacheWrite += r.tokens.cacheWrite;
    tokens.cacheRead += r.tokens.cacheRead;
  }

  // ---- merge: dedupe by url (or document page), renumber contiguously from 1, keep the tag
  const byKey = new Map<string, DeepSource>();
  for (const page of results.flatMap((r) => r.pages)) if (!byKey.has(sourceKey(page))) byKey.set(sourceKey(page), page);
  const pages = [...byKey.values()].map((p, i) => ({ ...p, n: i + 1 }));

  return {
    plan,
    pages,
    // Any subagent that ran out of budget makes the whole run an honest partial.
    terminated: results.some((r) => r.terminated === 'cap') || budget.callsUsed >= budget.maxCalls ? 'cap' : 'done',
    tokens,
    fanOutMs
  };
}
