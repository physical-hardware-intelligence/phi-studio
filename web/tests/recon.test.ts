// Environment reconstruction's pure logic: the binary cloud format and which cloud's arm pose the 3D view draws.
// Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { type ArmPose, LINEAR, parseCloud, type PoseSource, posedFrom, posePlan } from "../src/lib/reconCore.ts";
import { type ArmUnits, DEGREES, readArm, type SceneModel } from "../src/lib/sceneCore.ts";

const model = JSON.parse(readFileSync(new URL("./fixtures/so101_model.json", import.meta.url), "utf8")) as SceneModel;
const read = (pos: Record<string, number>, units: ArmUnits, out: Float64Array) => readArm(model, pos, units, out);

// -- binary format ----------------------------------------------------------------------------------------
// cloud3.pcl is written by src/phi_studio/recon.py pack; tests/test_recon.py keeps it equal to pack's output.
const fixture = () => {
  const b = readFileSync(new URL("./fixtures/cloud3.pcl", import.meta.url));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

test("parses the server's own bytes: counts, positions, colours", () => {
  const c = parseCloud(fixture());
  assert.equal(c.n, 3);
  assert.equal(c.nOff, 2);
  assert.deepEqual(Array.from(c.positions), [0.25, -0.5, 0, 1, 2, 3, -0.125, 0.0625, 0.5]); // exact in float32
  assert.equal(c.colors[0], 0);
  assert.equal(c.colors[2], 1);
  assert.equal(c.colors[1], LINEAR[128]);
  assert.equal(c.colors[5], LINEAR[1]);
});

test("the colour table is three's sRGB to linear", () => {
  assert.equal(LINEAR[0], 0);
  assert.ok(Math.abs(LINEAR[255] - 1) < 1e-6);
  assert.ok(Math.abs(LINEAR[128] - 0.2158605) < 1e-6); // ((128/255 + 0.055) / 1.055) ** 2.4
  assert.ok(Math.abs(LINEAR[10] - 10 / 255 / 12.92) < 1e-7); // the linear toe below 0.04045
  for (let i = 1; i < 256; i++) assert.ok(LINEAR[i] > LINEAR[i - 1]);
});

test("refuses a wrong header, a short body and an impossible arm count", () => {
  const good = fixture();
  const bad = good.slice(0);
  new Uint8Array(bad)[0] = 0x51; // "QCL1"
  assert.throws(() => parseCloud(bad), /not a point cloud/);
  assert.throws(() => parseCloud(good.slice(0, good.byteLength - 1)), /does not match/);
  const over = good.slice(0);
  new DataView(over).setUint32(8, 4, true); // 4 off the arm out of 3
  assert.throws(() => parseCloud(over), /does not match/);
  assert.throws(() => parseCloud(new ArrayBuffer(4)), /not a point cloud/);
});

// -- which pose the view draws -----------------------------------------------------------------------------
const pos = Object.fromEntries(model.joint_order.map((j) => [j, 10])) as Record<string, number>;
const pose = (frame: number): ArmPose => ({ arm: "follower", pos, label: `pose from episode 0, frame ${frame}` });
const cloud = (camera: string, at: number, p: ArmPose | null): PoseSource => ({ camera, at, pose: p });

test("the newest shown dataset cloud sets the pose", () => {
  const clouds = { front: cloud("front", 10, pose(60)), top: cloud("top", 20, pose(80)) };
  assert.equal(posedFrom(clouds, {})?.label, "pose from episode 0, frame 80");
  assert.equal(posedFrom(clouds, { top: false })?.label, "pose from episode 0, frame 60"); // hidden: skipped
});

test("live clouds carry no pose, and no shown pose means live again", () => {
  assert.equal(posedFrom({ front: cloud("front", 30, null) }, {}), null);
  assert.equal(posedFrom({ front: cloud("front", 10, pose(60)), top: cloud("top", 30, null) }, {})?.label, "pose from episode 0, frame 60");
  assert.equal(posedFrom({ front: cloud("front", 10, pose(60)) }, { front: false }), null);
  assert.equal(posedFrom({}, {}), null);
});

test("a degrees pose reads exactly as a live reading would", () => {
  const p = posePlan(pose(60), read);
  const want = new Float64Array(6);
  assert.equal(readArm(model, pos, DEGREES, want), "ok");
  assert.deepEqual(p.q, want);
  assert.equal(p.text, "pose from episode 0, frame 60");
});

test("a -100..100 pose uses the calibration the live arm uses", () => {
  const calibration = Object.fromEntries(model.joint_order.map((j) => [j, { range_min: 1000, range_max: 3000, drive_mode: 0 }]));
  const units: ArmUnits = { unit: "m100", calibration, problem: null };
  const p = posePlan({ ...pose(60), units }, read);
  const want = new Float64Array(6);
  assert.equal(readArm(model, pos, units, want), "ok");
  assert.deepEqual(p.q, want);
  assert.notDeepEqual(p.q, posePlan(pose(60), read).q); // not read as degrees
});

test("an arm the view would not draw live is not drawn posed, and the pill says why", () => {
  const units: ArmUnits = { unit: "m100", calibration: null, problem: "reads -100 to 100 and has no calibration file, so its pose is not drawn" };
  const p = posePlan({ ...pose(60), units }, read);
  assert.equal(p.q, null);
  assert.equal(p.text, "pose from episode 0, frame 60 reads -100 to 100 and has no calibration file, so it is not drawn");
  const partial = posePlan({ ...pose(60), pos: { shoulder_pan: 1 } }, read);
  assert.equal(partial.q, null);
  assert.match(partial.text, /missing a joint reading, so it is not drawn$/);
});
