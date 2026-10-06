// Calibrate in the terminal: how a register check reads, and when a run has ended.
import assert from "node:assert/strict";
import { test } from "node:test";
import { read, RunWatch, type Verdict } from "../src/lib/termcal.ts";

const v = (p: Partial<Verdict>): Verdict => ({
  arm: "left_follower", path: "/c/phi_bi_left.json", file: true, exact: false, max_deg: 3.2, worst_joint: "wrist_flex",
  problem: null, unfinished: null, at: 0, ...p,
});

test("a file equal to the motors is done, and is never offered for overwrite", () => {
  assert.deepEqual(read(v({ exact: true, max_deg: 0 })), { tone: "ok", text: "Its file matches the motors.", canSave: false });
});

test("unfinished registers are never saved, even with no file", () => {
  const r = read(v({ file: false, unfinished: "shoulder pan still has the factory range 0 to 4095" }));
  assert.equal(r.canSave, false);
  assert.match(r.text, /unfinished calibration: shoulder pan/);
});

test("a finished calibration with no file or a different file may be saved from the motors", () => {
  assert.equal(read(v({ file: false })).canSave, true);
  const r = read(v({}));
  assert.equal(r.canSave, true);
  assert.match(r.text, /3\.2 deg on wrist flex/);
});

test("a run ends only after it was seen running", () => {
  const w = new RunWatch("left_follower");
  assert.equal(w.update(null), false); // the Run was typed, the shell has not started it yet
  assert.equal(w.update("/opt/x/python /opt/x/bin/lerobot-calibrate --robot.type=so101_follower"), false);
  assert.equal(w.update("lerobot-calibrate --robot.id=phi_bi_left"), false);
  assert.equal(w.update(null), true);
  assert.equal(w.update(null), false); // once
  assert.equal(new RunWatch("x").update("lerobot-teleoperate"), false);
});
