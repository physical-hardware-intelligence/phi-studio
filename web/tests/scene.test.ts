// The 3D view's pure logic: kinematics against MuJoCo, telemetry timing, saved settings, arm pairing, wrist
// cameras and joint units. Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  basePositions, DEFAULTS, DEGREES, followerSlots, frustumKeys, limitState, parseSettings, readArm, type ArmSlot, type ArmUnits,
  type SceneModel,
} from "../src/lib/sceneCore.ts";
import { bodyPoses, forwardTool, mjQuat } from "../src/scene/kinematics.ts";
import { Clock, Track } from "../src/scene/motion.ts";
import { wristMount } from "../src/lib/sceneCore.ts";

const json = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
// The model exactly as GET /api/scene/model serves it (tests/test_scene_api.py keeps this copy equal to it), and
// MuJoCo's own numbers for 8 poses (tests/fixtures/make_so101_fk_reference.py).
const model = json("./fixtures/so101_model.json") as SceneModel;
const ref = json("../../tests/fixtures/so101_fk_reference.json") as {
  joints: string[]; poses: Record<string, { q: number[]; gripperframe: { pos: number[] }; bodies: Record<string, { pos: number[]; quat: number[] }> }>;
};

// -- kinematics -------------------------------------------------------------------------------------------
test("the fixture's joint order is the model's", () => {
  assert.deepEqual(ref.joints, model.joint_order);
  assert.equal(Object.keys(ref.poses).length, 8);
});

test("tool position matches MuJoCo's gripperframe at all 8 poses to 1e-9 m", () => {
  for (const [name, p] of Object.entries(ref.poses)) {
    const tip = forwardTool(model, p.q);
    const err = Math.hypot(tip[0] - p.gripperframe.pos[0], tip[1] - p.gripperframe.pos[1], tip[2] - p.gripperframe.pos[2]);
    assert.ok(err < 1e-9, `${name}: ${err} m`);
  }
});

test("every body's world position and rotation match MuJoCo at all 8 poses", () => {
  for (const [name, p] of Object.entries(ref.poses)) {
    const got = bodyPoses(model, p.q);
    for (const [body, r] of Object.entries(p.bodies)) {
      const b = got.get(body);
      assert.ok(b, `${name}: no ${body}`);
      assert.ok(Math.hypot(b.p[0] - r.pos[0], b.p[1] - r.pos[1], b.p[2] - r.pos[2]) < 1e-9, `${name} ${body} position`);
      // q and -q are the same rotation, so compare |q . r| with 1. r is MuJoCo's (w, x, y, z).
      const m = mjQuat(r.quat as [number, number, number, number]);
      const dot = Math.abs(b.q[0] * m[0] + b.q[1] * m[1] + b.q[2] * m[2] + b.q[3] * m[3]);
      assert.ok(Math.abs(dot - 1) < 1e-9, `${name} ${body} rotation: |dot| ${dot}`);
    }
  }
});

test("a MuJoCo quaternion (w, x, y, z) becomes three's (x, y, z, w)", () => {
  assert.deepEqual(mjQuat([0.1, 0.2, 0.3, 0.4]), [0.2, 0.3, 0.4, 0.1]);
});

// -- telemetry timing -------------------------------------------------------------------------------------
test("clock offset is the smallest arrival minus send gap", () => {
  const c = new Clock();
  c.add(10.0, 10_005);
  const at = c.add(10.033, 10_035);
  assert.equal(c.offset, 2);
  assert.equal(at, 10_033 + 2);
});

test("clock delay is clamped to 16..250 ms", () => {
  const fast = new Clock();
  for (let i = 0; i < 20; i++) fast.add(i * 0.001, i + 3); // 1 ms apart, never late
  assert.equal(fast.delay, 16);
  const slow = new Clock();
  for (let i = 0; i < 20; i++) slow.add(i * 1.0, i * 1000 + 3); // 1 s apart
  assert.equal(slow.delay, 250);
});

test("a worker clock that jumps back starts the mapping over", () => {
  const c = new Clock();
  for (let i = 0; i < 30; i++) c.add(100 + i / 30, 50_000 + i * 33.3);
  const now = 52_000;
  const at = c.add(5, now); // a new worker: its monotonic clock is far behind the old one
  assert.ok(c.jumped);
  assert.ok(Math.abs(at - now) < 1, `mapped to ${at}, now is ${now}`);
  c.add(5.033, now + 33);
  assert.ok(!c.jumped);
});

test("track interpolates between samples and never extrapolates", () => {
  const t = new Track();
  const out = new Float64Array(6);
  assert.equal(t.sample(0, out), false);
  t.push(0, [0, 0, 0, 0, 0, 0], 0);
  t.push(100, [10, 20, 0, 0, 0, 0], 100);
  t.sample(50, out);
  assert.deepEqual([out[0], out[1]], [5, 10]);
  t.sample(400, out);
  assert.deepEqual([out[0], out[1]], [10, 20]);
  t.sample(-50, out);
  assert.deepEqual([out[0], out[1]], [0, 0]);
});

test("a sample older than the newest one restarts the track", () => {
  const t = new Track();
  const out = new Float64Array(6);
  t.push(100, [1, 0, 0, 0, 0, 0], 0);
  t.push(200, [2, 0, 0, 0, 0, 0], 0);
  t.push(50, [9, 0, 0, 0, 0, 0], 0); // after a clock jump back
  t.sample(40, out);
  assert.equal(out[0], 9);
});

// -- saved settings ---------------------------------------------------------------------------------------
const good = { pos: [0.9, -0.3, 0.4], target: [0.1, 0, 0.1], up: [0, 0, 1], fovy_deg: 50 };

test("missing, corrupt or non-object settings load the defaults", () => {
  for (const raw of [null, "", "{not json", "[]", "null", "42", "\"text\""]) {
    const s = parseSettings(raw);
    assert.equal(s.spacing, null, String(raw));
    assert.deepEqual(s.cameras, {}, String(raw));
    assert.equal(s.ghost, true, String(raw));
  }
});

test("a saved camera with any bad field is dropped, a good one kept", () => {
  const bad = {
    noUp: { pos: good.pos, target: good.target, fovy_deg: 50 },
    noTarget: { pos: good.pos, up: good.up, fovy_deg: 50 },
    shortPos: { ...good, pos: [1, 2] },
    stringPos: { ...good, pos: ["1", 0, 0] },
    nullUp: { ...good, up: null },
    zeroUp: { ...good, up: [0, 0, 0] },
    lookAtSelf: { ...good, target: good.pos },
    fovString: { ...good, fovy_deg: "50" },
    fovZero: { ...good, fovy_deg: 0 },
    fovHuge: { ...good, fovy_deg: 500 },
    notObject: 7,
  };
  const s = parseSettings(JSON.stringify({ cameras: { ...bad, front: good } }));
  assert.deepEqual(Object.keys(s.cameras), ["front"]);
  assert.deepEqual(s.cameras.front, good);
});

test("spacing must be a number in range, not a numeric string", () => {
  assert.equal(parseSettings(JSON.stringify({ spacing: "0.5" })).spacing, null);
  assert.equal(parseSettings(JSON.stringify({ spacing: 5 })).spacing, null);
  assert.equal(parseSettings(JSON.stringify({ spacing: 0.5 })).spacing, 0.5);
});

test("other fields of the wrong type fall back one by one", () => {
  const s = parseSettings(JSON.stringify({
    ghost: "yes", trail: false, video: 0, print: "gold", wristArm: 5, cameras: [good], view: { teleop: "3d", policy: "wide", x: 1 },
  }));
  assert.equal(s.ghost, true);
  assert.equal(s.trail, false);
  assert.equal(s.video, DEFAULTS.video); // falls back to the default, whatever it is
  assert.equal(s.print, "model");
  assert.equal(s.wristArm, null);
  assert.deepEqual(s.cameras, {});
  assert.deepEqual(s.view, { teleop: "3d" });
});

test("parsed settings are a fresh object each time", () => {
  const a = parseSettings(null);
  a.cameras.front = good as never;
  assert.deepEqual(parseSettings(null).cameras, {});
});

// -- arm pairing and layout -------------------------------------------------------------------------------
test("bimanual followers pair with the leader on their side", () => {
  const arms = [
    { name: "left_leader", role: "leader", side: "left" }, { name: "left_follower", role: "follower", side: "left" },
    { name: "right_leader", role: "leader", side: "right" }, { name: "right_follower", role: "follower", side: "right" },
  ];
  assert.deepEqual(followerSlots(null, arms), [
    { name: "left_follower", side: "left", ghost: "left_leader" },
    { name: "right_follower", side: "right", ghost: "right_leader" },
  ]);
});

test("a single follower pairs by name, and telemetry alone is enough", () => {
  assert.deepEqual(followerSlots({ arms: { follower: { role: "follower" }, leader: { role: "leader" } } }, undefined), [
    { name: "follower", side: null, ghost: "leader" },
  ]);
  assert.deepEqual(followerSlots(null, [{ name: "follower", role: "follower", side: null }]), [{ name: "follower", side: null, ghost: null }]);
});

test("bases sit in a row centred on the origin, left arm at +y", () => {
  const slots: ArmSlot[] = [{ name: "right_follower", side: "right", ghost: null }, { name: "left_follower", side: "left", ghost: null }];
  assert.deepEqual(basePositions(slots, 0.4), [[0, -0.2, 0], [0, 0.2, 0]]);
  assert.deepEqual(basePositions([{ name: "follower", side: null, ghost: null }], 0.4), [[0, 0, 0]]);
});

// -- cameras ----------------------------------------------------------------------------------------------
const pair: ArmSlot[] = [{ name: "left_follower", side: "left", ghost: null }, { name: "right_follower", side: "right", ghost: null }];

test("LeRobot's per-arm wrist cameras ride on the matching follower", () => {
  assert.deepEqual(wristMount("left_wrist", "wrist", pair, null), { wrist: true, arm: 0 });
  assert.deepEqual(wristMount("right_wrist", "wrist", pair, null), { wrist: true, arm: 1 });
  assert.deepEqual(wristMount("wrist", "wrist", pair, "right_follower"), { wrist: true, arm: 1 });
  assert.deepEqual(wristMount("wrist", "wrist", pair, null), { wrist: true, arm: 0 });
  assert.deepEqual(wristMount("front", "wrist", pair, null), { wrist: false, arm: null });
  assert.deepEqual(wristMount("middle_wrist", "wrist", pair, null), { wrist: true, arm: null });
});

test("frusta are the rig's cameras, or the phi three before the rig reports any", () => {
  assert.deepEqual(frustumKeys([]), ["front", "wrist", "top"]);
  assert.deepEqual(frustumKeys(["left_wrist", "front"]), ["front", "left_wrist"]);
});

// -- joint units and limits -------------------------------------------------------------------------------
const order = model.joint_order;
const reading = (vals: number[]) => Object.fromEntries(order.map((j, i) => [j, vals[i]]));
const cal = { range_min: 1000, range_max: 3000, drive_mode: 0 };
const m100: ArmUnits = { unit: "m100", calibration: Object.fromEntries(order.slice(0, 5).map((j) => [j, cal])), problem: null };

test("degrees readings map to radians, the gripper 0..100 onto its model range", () => {
  const out = new Float64Array(6);
  assert.equal(readArm(model, reading([90, 0, 0, 0, 0, 50]), DEGREES, out), "ok");
  assert.ok(Math.abs(out[0] - Math.PI / 2) < 1e-12);
  const [lo, hi] = model.joints.gripper.range!;
  assert.ok(Math.abs(out[5] - (lo + (hi - lo) / 2)) < 1e-12);
});

test("-100..100 readings go through the calibrated range, as robot_model.m100_to_degrees does", () => {
  const out = new Float64Array(6);
  assert.equal(readArm(model, reading([50, -50, 0, 0, 0, 50]), m100, out), "ok");
  const deg = (50 / 200) * 2000 * 360 / 4095; // 43.956 degrees
  assert.ok(Math.abs(out[0] - deg * Math.PI / 180) < 1e-12, String(out[0]));
  assert.ok(Math.abs(out[1] + deg * Math.PI / 180) < 1e-12);
  const flipped: ArmUnits = { ...m100, calibration: { ...m100.calibration, shoulder_pan: { ...cal, drive_mode: 1 } } };
  readArm(model, reading([50, 0, 0, 0, 0, 50]), flipped, out);
  assert.ok(Math.abs(out[0] + deg * Math.PI / 180) < 1e-12);
  const [lo, hi] = model.joints.gripper.range!;
  assert.ok(Math.abs(out[5] - (lo + (hi - lo) / 2)) < 1e-12, "the gripper is 0..100 in both modes");
});

test("-100..100 readings without a calibration are not drawn", () => {
  const out = new Float64Array(6);
  const none: ArmUnits = { unit: "m100", calibration: null, problem: "no calibration file" };
  assert.equal(readArm(model, reading([50, 0, 0, 0, 0, 50]), none, out), "unusable");
  const partial: ArmUnits = { unit: "m100", calibration: { shoulder_pan: cal }, problem: null };
  assert.equal(readArm(model, reading([50, 0, 0, 0, 0, 50]), partial, out), "unusable");
  assert.equal(readArm(model, { shoulder_pan: 1 }, DEGREES, out), "incomplete");
});

test("the gripper never tints; wrist_roll past its model range is outside the model, not an error", () => {
  const far = 10; // radians: past every range
  assert.equal(limitState(model, "gripper", far).state, "ok");
  assert.equal(limitState(model, "wrist_roll", far).state, "outside");
  assert.equal(limitState(model, "wrist_roll", model.joints.wrist_roll.range![1] - 0.01).state, "ok");
  assert.equal(limitState(model, "shoulder_pan", far).state, "past");
  assert.equal(limitState(model, "shoulder_pan", model.joints.shoulder_pan.range![1] - 0.01).state, "near");
});
