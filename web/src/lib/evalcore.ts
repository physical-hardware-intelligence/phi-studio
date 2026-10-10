// The Evaluate page's data and its plain-language rules. Import-free, so Node's test runner checks it
// (tests/evalcore.test.ts). The record and every number come from the server (src/phi_studio/evals.py);
// nothing here decides who is better, it only says what the server decided.

export const AXES = ["train", "held-out position", "new object", "distractors", "lighting", "camera moved",
  "table or background", "new instruction", "other"] as const;
export type Axis = (typeof AXES)[number];

export interface Milestone { label: string; points: number }
export interface Condition { id: string; label: string; axis: Axis; control: boolean; refs: Record<string, string> }
export interface Card {
  id?: string; name: string; task: string; success: string; limit_s: number;
  rubric: Milestone[]; failures: string[]; conditions: Condition[];
  created_at?: number | null; locked_at?: number | null;
}
export interface PolicyRef { alias: string; id?: string; name?: string }
export interface Suspect { void: string[]; warn: string[] }
export interface Trial {
  n: number; b: number | null; c: string; alias: string; run_id: string | null; stage: number; score: number;
  success: boolean; failures: string[]; note: string; duration_s: number | null;
  metrics: Record<string, number | string>; health: Record<string, unknown>; suspect: Suspect | Record<string, never>;
  void: { reason: string; at: number } | null; at: number;
}
export interface Bundle { b: number; c: string; order: string[] }
export interface EvalRec {
  schema: 2; id: string; card: Card; policies: PolicyRef[]; blind: boolean; alpha: number; seed: number | null;
  grouped: boolean; reps: number | null; schedule: Bundle[]; trials: Trial[];
  skipped: { b: number; alias: string; reason: string; at: number }[];
  started_at: number; ended_at: number | null; stamps: Record<string, unknown>;
  imported?: { from: string; rows: number; kept: number }; paired?: boolean;
}
export interface Rate {
  n: number; k: number; rate: number | null; ci95: [number, number];
  progress: number | null; progress_ci95: [number | null, number | null];
}
export interface PolicyStats extends Rate {
  median_success_s: number | null; failures: Record<string, number>; voided: number;
  by_axis: Record<string, Rate>; by_condition: Record<string, Rate>;
}
export interface Decision { state: "a" | "b" | "none" | "continue"; at_bundle: number | null; pairs: number; wins_a: number; wins_b: number }
export interface Pair {
  a: string; b: string; paired: boolean; bundles?: number; wins_a?: number; wins_b?: number;
  p_a_better: number | null; decision: Decision | null; bundles_more_about?: number | null;
}
export interface Summary {
  policies: Record<string, PolicyStats>; pairs: Pair[]; letters: Record<string, string> | null; order: string[];
  paired: boolean; alpha: number; comparisons: number; planned_bundles: number; planned_trials: number;
  valid_trials: number; voided: number; skipped: number; complete: boolean; detectable_gap: number | null;
}
export interface NextSlot { b: number; c: string; alias: string; repeat: boolean; condition: Condition | null }
export interface PastEval {
  schema: 1 | 2; id: string; name: string; started_at: number; ended_at: number | null; imported?: unknown;
  policies: string[];
  paired?: boolean;
  results?: {
    policy: string; n: number; k: number; rate: number | null; ci95: [number, number]; progress: number | null;
    letters?: string | null; by_axis?: Record<string, [number, number]>;
  }[];
  n?: number; k?: number; rate?: number | null; ci95?: [number, number];
}
export interface EvalState {
  current: EvalRec | null; summary: Summary | null; next: NextSlot | null; awaiting: boolean;
  cards: Card[]; past: PastEval[]; dir: string; club_csv: string | null;
}

// Sherry Chen's SO-101 rubric and the club's failure tags: the same defaults as the server's blank card.
export function blankCard(): Card {
  return {
    name: "", task: "", success: "", limit_s: 30,
    rubric: [{ label: "Reached the object", points: 0.2 }, { label: "Grasped it", points: 0.4 },
      { label: "Carried it to the target", points: 0.7 }, { label: "Released it", points: 0.8 },
      { label: "Object in place", points: 1 }],
    failures: ["Missed the grasp", "Slipped or dropped", "Pushed or knocked it over", "Wrong object or place",
      "Hesitated or oscillated", "Hit something", "Ran out of time", "Stopped for safety"],
    conditions: [],
  };
}

export const pctOf = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);

/** The person-facing name of a policy: a letter while blind, its name after. */
export function policyName(rec: EvalRec, alias: string): string {
  const p = rec.policies.find((x) => x.alias === alias);
  return p?.name ?? p?.id ?? `Policy ${alias}`;
}

export function milestoneLabel(card: Card, stage: number): string {
  return stage === 0 ? "Nothing reached" : card.rubric[stage - 1]?.label ?? `Milestone ${stage}`;
}

/** One sentence for a comparison, from the server's decision. */
export function decisionText(p: Pair, name: (alias: string) => string): string {
  const a = name(p.a), b = name(p.b);
  const prob = p.p_a_better == null ? "" : `P(${a} better) ${p.p_a_better.toFixed(2)}`;
  if (!p.paired) return `${prob}. Not run in bundles, so this may include day-to-day drift.`;
  const d = p.decision;
  if (!d) return prob;
  if (d.state === "a" || d.state === "b") {
    return `${d.state === "a" ? a : b} is better: decided at bundle ${d.at_bundle} (${d.wins_a} to ${d.wins_b} where only one succeeded).`;
  }
  if (d.state === "none") return `No difference shown within the plan (${d.wins_a} to ${d.wins_b}). That is not proof they are equal.`;
  const more = p.bundles_more_about != null ? ` About ${p.bundles_more_about} more bundles at this gap.` : "";
  return `Undecided after ${p.bundles ?? 0} bundles: ${d.wins_a} to ${d.wins_b} where only one succeeded. ${prob}.${more}`;
}

// Two-sided 5%, power 0.8, normal approximation: the same formula as evalstats.n_two_proportions.
const Z_975 = 1.959963984540054, Z_80 = 0.8416212335729143;
export function trialsToSeparate(p1: number, p2: number): number {
  const pbar = (p1 + p2) / 2;
  const num = Z_975 * Math.sqrt(2 * pbar * (1 - pbar)) + Z_80 * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2));
  return Math.ceil((num / (p1 - p2)) ** 2);
}

/** The smallest gap above 50% that n trials per policy can detect, in steps of 1 point; null if none. */
export function detectableGap(n: number): number | null {
  for (let d = 1; d < 50; d++) if (trialsToSeparate(0.5, 0.5 + d / 100) <= n) return d / 100;
  return null;
}

export interface Plan { bundles: number; trials: number; gap: number | null; minutes: number }

/** What an eval would cost and could show, before it starts. `secondsPerTrial` adds a reset to the limit. */
export function planOf(conditions: number, reps: number, policies: number, limitS: number, resetS = 30): Plan {
  const bundles = conditions * reps;
  const trials = bundles * policies;
  return { bundles, trials, gap: policies > 1 ? detectableGap(bundles) : null, minutes: Math.round((trials * (limitS + resetS)) / 60) };
}

/** The trials of a running eval: done (valid) of planned. */
export function progressOf(rec: EvalRec): { done: number; planned: number } {
  return { done: rec.trials.filter((t) => !t.void).length + rec.skipped.length, planned: rec.schedule.length * rec.policies.length };
}

/** Keyboard: 0 is "nothing reached", 1..9 the milestones. */
export function stageFromKey(key: string, milestones: number): number | null {
  if (!/^[0-9]$/.test(key)) return null;
  const n = Number(key);
  return n <= milestones ? n : null;
}

/** Problems with a card draft the window can show before the server's own check. */
export function draftProblem(c: Card): string | null {
  if (!c.name.trim()) return "Name the card.";
  if (!c.task.trim()) return "Write the task as the policy is told it.";
  if (!c.success.trim()) return "Say what counts as success.";
  if (!c.rubric.length) return "Add at least one milestone.";
  for (let i = 1; i < c.rubric.length; i++) if (!(c.rubric[i].points > c.rubric[i - 1].points)) return "Each milestone must be worth more than the one before it.";
  if (c.rubric[c.rubric.length - 1].points !== 1) return "The last milestone is success, worth 1.";
  if (!c.conditions.length) return "Add at least one start condition.";
  const labels = c.conditions.map((x) => x.label.trim());
  if (labels.some((l) => !l)) return "Describe every condition.";
  if (new Set(labels).size !== labels.length) return "Two conditions have the same description.";
  return null;
}
