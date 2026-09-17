import { AskTool, DEEP_ONLY_TOOLS, type Depth } from '@lumina/contract';

const DEEP_ONLY: readonly AskTool[] = DEEP_ONLY_TOOLS;

/**
 * The tools the model is ALLOWED to see for this gear. A quick search never receives
 * plan_research, so it cannot call it: a filtered list is a gate, a prompt is a suggestion.
 */
export function toolsFor(depth: Depth): AskTool[] {
  if (depth === 'deep') return [...AskTool.options];
  return AskTool.options.filter((tool) => !DEEP_ONLY.includes(tool));
}
