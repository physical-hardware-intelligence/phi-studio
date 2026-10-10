// The Evaluate page's rules (src/lib/evalcore.ts). Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  blankCard, decisionText, detectableGap, draftProblem, planOf, policyName, progressOf, stageFromKey, trialsToSeparate,
  type EvalRec, type Pair,
} from "../src/lib/evalcore.ts";

test("trial counts match the server's formula (evalstats.n_two_proportions)", () => {
  assert.equal(trialsToSeparate(0.5, 0.9), 20);
  assert.equal(trialsToSeparate(0.5, 0.8), 39);
  assert.equal(trialsToSeparate(0.6, 0.7), 356);
  assert.equal(detectableGap(20), 0.4);
  assert.equal(detectableGap(5), null);
});

test("a plan says what an eval costs and what gap it can show", () => {
  const p = planOf(3, 4, 2, 30);
  assert.deepEqual([p.bundles, p.trials], [12, 24]);
  assert.equal(p.gap, detectableGap(12));
  assert.equal(p.minutes, 24);
  assert.equal(planOf(3, 4, 1, 30).gap, null); // one policy: nothing to compare
});

const name = (a: string) => `Policy ${a}`;
const pair = (state: "a" | "b" | "none" | "continue", extra: Partial<Pair> = {}): Pair => ({
  a: "A", b: "B", paired: true, bundles: 9, wins_a: 6, wins_b: 1, p_a_better: 0.97,
  decision: { state, at_bundle: state === "a" || state === "b" ? 9 : null, pairs: 7, wins_a: 6, wins_b: 1 }, ...extra,
});

test("comparisons are said in plain words, and an unproven gap is not called a tie", () => {
  assert.match(decisionText(pair("a"), name), /^Policy A is better: decided at bundle 9/);
  assert.match(decisionText(pair("b"), name), /^Policy B is better/);
  assert.match(decisionText(pair("none"), name), /not proof they are equal/);
  assert.match(decisionText(pair("continue", { bundles_more_about: 14 }), name), /Undecided after 9 bundles.*About 14 more/);
  assert.match(decisionText({ ...pair("continue"), paired: false, decision: null }, name), /day-to-day drift/);
});

const rec = (over: Partial<EvalRec> = {}): EvalRec => ({
  schema: 2, id: "x", card: { ...blankCard(), name: "c", task: "t", success: "s", conditions: [] },
  policies: [{ alias: "A" }, { alias: "B", name: "ACT recovery" }], blind: true, alpha: 0.05, seed: 1, grouped: false,
  reps: 1, schedule: [{ b: 1, c: "c1", order: ["A", "B"] }, { b: 2, c: "c2", order: ["B", "A"] }],
  trials: [], skipped: [], started_at: 0, ended_at: null, stamps: {}, ...over,
});

test("a blind policy is a letter; a revealed one has its name", () => {
  const r = rec();
  assert.equal(policyName(r, "A"), "Policy A");
  assert.equal(policyName(r, "B"), "ACT recovery");
});

test("progress counts valid trials and skips, not voided ones", () => {
  const t = { n: 1, b: 1, c: "c1", alias: "A", run_id: "r", stage: 0, score: 0, success: false, failures: [], note: "",
    duration_s: null, metrics: {}, health: {}, suspect: {}, at: 1 };
  const r = rec({ trials: [{ ...t, void: { reason: "camera", at: 2 } }, { ...t, n: 2, void: null }],
    skipped: [{ b: 2, alias: "B", reason: "x", at: 3 }] });
  assert.deepEqual(progressOf(r), { done: 2, planned: 4 });
});

test("number keys pick milestones within the rubric only", () => {
  assert.equal(stageFromKey("0", 5), 0);
  assert.equal(stageFromKey("5", 5), 5);
  assert.equal(stageFromKey("6", 5), null);
  assert.equal(stageFromKey("a", 5), null);
});

test("a card draft names its first problem", () => {
  const c = blankCard();
  assert.equal(draftProblem(c), "Name the card.");
  const ok = { ...c, name: "n", task: "t", success: "s", conditions: [{ id: "c1", label: "Spot 1", axis: "train" as const, control: true, refs: {} }] };
  assert.equal(draftProblem(ok), null);
  assert.match(draftProblem({ ...ok, rubric: [{ label: "a", points: 0.5 }, { label: "b", points: 0.5 }] }) ?? "", /more than/);
  assert.match(draftProblem({ ...ok, conditions: [...ok.conditions, { ...ok.conditions[0], id: "c2" }] }) ?? "", /same description/);
});
