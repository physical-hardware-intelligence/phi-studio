// The port finder's state machine, every case the Set up page relies on. Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { test } from "node:test";
import { advance, portKey, type Wizard } from "../src/lib/portfinder.ts";

const F = "/dev/tty.usbmodemF", L = "/dev/tty.usbmodemL", X = "/dev/tty.usbmodemX";
const start = (base: string[], arms = ["follower", "leader"]): Wizard =>
  ({ arms, i: 0, phase: "unplug", base, extra: [], gone: null, found: {}, problem: null, saving: false });
const run = (w: Wizard, ...steps: string[][]) => steps.reduce(advance, w);

test("unplug then replug under the same name finds the arm", () => {
  const w = run(start([F, L]), [L], [F, L]);
  assert.deepEqual(w.found, { follower: F });
  assert.equal(w.i, 1);
  assert.equal(w.phase, "unplug");
});

test("a replug under a new name takes the new name", () => {
  const w = run(start([F, L]), [L], [L, X]);
  assert.deepEqual(w.found, { follower: X });
});

test("all arms found one by one", () => {
  const w = run(start([F, L]), [L], [F, L], [F], [F, L]);
  assert.deepEqual(w.found, { follower: F, leader: L });
  assert.equal(w.i, 2);
});

test("two ports leaving at once is a problem, not a guess", () => {
  const w = run(start([F, L]), []);
  assert.equal(w.phase, "unplug");
  assert.match(w.problem!, /2 ports went away at once/);
});

test("a port that came and went during the turn is not the arm's", () => {
  const w = run(start([F, L]), [F, L, X], [F, L]);
  assert.equal(w.phase, "unplug");
  assert.equal(w.gone, null);
  assert.equal(w.problem, null);
});

test("unplugging an arm already found is flagged", () => {
  const w = run(start([F, L]), [L], [F, L], [L]);
  assert.equal(w.i, 1);
  assert.equal(w.phase, "unplug");
  assert.match(w.problem!, /Follower.s port, found already/);
});

test("another port leaving during the replug stops the finder", () => {
  // The follower is out; the leader's cable is moved too. Naming anything now could swap the arms.
  const w = run(start([F, L]), [L], [X]);
  assert.deepEqual(w.found, {});
  assert.match(w.problem!, /usbmodemL went away too/);
});

test("a cable plugged in during the unplug is not taken as the replug", () => {
  const w = run(start([F, L]), [L, X], [L, X, F]);
  assert.deepEqual(w.found, { follower: F });
});

test("a saving or finished finder does not move", () => {
  const done = run(start([F, L]), [L], [F, L], [F], [F, L]);
  assert.equal(advance(done, []), done);
  const saving = { ...start([F, L]), saving: true };
  assert.equal(advance(saving, []), saving);
});

test("cu and tty name one device", () => {
  assert.equal(portKey("/dev/cu.usbmodemL"), L);
  assert.equal(portKey(L), L);
  assert.equal(portKey("COM3"), "COM3");
});
