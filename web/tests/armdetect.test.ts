// Detect arms on the Rig setup page: which slot each scanned arm goes in and which calibration ids hold it.
import assert from "node:assert/strict";
import { test } from "node:test";
import { place, type FoundArm, type Role } from "../src/lib/armdetect.ts";

const F = "robots/so_follower/", L = "teleoperators/so_leader/";
const arm = (port: string, role: Role | null, matches: string[], problem: string | null = null): FoundArm => ({
  port, serial: port, ids: [1, 2, 3, 4, 5, 6], clashes: [], torque: false,
  match: matches[0] ?? null, match_deg: matches.length ? 0 : 12.5, matches, role, problem,
});

// The club rig on 2026-10-05: phi_bi_left.json is a link to phi_follower.json, which equals yash_follower.json, so
// the left follower matches three files; the right leader's file is a link to yash_leader.json.
const RIG = [
  arm("/dev/tty.lf", "follower", [`${F}phi_bi_left`, `${F}phi_follower`, `${F}yash_follower`]),
  arm("/dev/tty.rf", "follower", [`${F}phi_bi_right`]),
  arm("/dev/tty.ll", "leader", [`${L}phi_bi_left`]),
  arm("/dev/tty.rl", "leader", [`${L}phi_bi_right`, `${L}yash_leader`]),
];

test("the four club arms land in their slots with the phi_bi files", () => {
  const p = place(RIG);
  assert.equal(p.layout, "bimanual");
  assert.deepEqual(p.ports, { left_follower: "/dev/tty.lf", right_follower: "/dev/tty.rf", left_leader: "/dev/tty.ll", right_leader: "/dev/tty.rl" });
  assert.deepEqual(p.ids, { follower: "phi_bi", leader: "phi_bi" });
  assert.deepEqual(p.unplaced, []);
});

test("an arm that matches no file exactly is left for the person to place, and no ids are offered", () => {
  const p = place([...RIG.slice(0, 3), arm("/dev/tty.rl", null, [])]);
  assert.equal(p.layout, "bimanual");
  assert.equal(p.ports.right_leader, undefined);
  assert.equal(p.ports.left_leader, undefined); // one leader alone has no pair to tell sides by
  assert.deepEqual(p.unplaced.map((a) => a.port).sort(), ["/dev/tty.ll", "/dev/tty.rl"]);
  assert.equal(p.ids, null);
});

test("an arm with a motor problem is never placed", () => {
  const bad = arm("/dev/tty.rf", null, [], "Two motors answer to id 6");
  const p = place([RIG[0], bad, RIG[2], RIG[3]]);
  assert.deepEqual(p.broken, [bad]);
  assert.equal(p.ports.right_follower, undefined);
});

test("two files naming opposite sides for one arm leave it unplaced unless an id decides", () => {
  const both = [arm("/dev/tty.a", "follower", [`${F}x_left`, `${F}y_right`]), arm("/dev/tty.b", "follower", [`${F}x_right`])];
  assert.equal(place(both).ports.left_follower, undefined);
  const p = place(both, { follower: "x" });
  assert.deepEqual([p.ports.left_follower, p.ports.right_follower], ["/dev/tty.a", "/dev/tty.b"]);
});

test("one leader and one follower make a single-arm rig with their own unsided ids", () => {
  const p = place([RIG[0], arm("/dev/tty.l", "leader", [`${L}phi_leader`])], { follower: "phi_follower" });
  assert.equal(p.layout, "single");
  assert.deepEqual(p.ports, { follower: "/dev/tty.lf", leader: "/dev/tty.l" });
  assert.deepEqual(p.ids, { follower: "phi_follower", leader: "phi_leader" });
});
