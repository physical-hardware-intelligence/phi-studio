import assert from "node:assert/strict";
import { test } from "node:test";
import { ALIGN_FRESH_S, cameraCheck } from "../src/lib/preflight.ts";

const ago = () => "ago";
const reads = { id: "act", name: "ACT", available: true, note: "", cameras: ["observation.images.top"], dataset: "Parv-09/cubes_v1" };
const now = 1_800_000_000_000;
const at = now / 1000 - 600;
const inLine = { root: "/data/Parv-09/cubes_v1", episode: 0, at, aligned: true, cameras: { "observation.images.top": true } };

test("a policy that reads no camera, or a simulated rig, never waits on align", () => {
  assert.equal(cameraCheck({ ...reads, cameras: [] }, null, false, now, ago).block, null);
  assert.equal(cameraCheck(reads, null, true, now, ago).block, null);
});

test("a real rig must be aligned, to this policy's own data, recently, every camera", () => {
  assert.equal(cameraCheck(reads, null, false, now, ago).block, "Align the cameras first");
  assert.equal(cameraCheck(reads, inLine, false, now, ago).block, null);
  assert.equal(cameraCheck(reads, inLine, false, now, ago).tone, "ok");
  const other = { ...inLine, root: "/data/Parv-09/8bin_v1" };
  assert.match(cameraCheck(reads, other, false, now, ago).block ?? "", /training data/);
  const off = { ...inLine, aligned: false, cameras: { "observation.images.top": false, "observation.images.wrist": true } };
  assert.equal(cameraCheck(reads, off, false, now, ago).state, "1 of 2 off");
  const old = { ...inLine, at: now / 1000 - ALIGN_FRESH_S - 1 };
  assert.match(cameraCheck(reads, old, false, now, ago).block ?? "", /old/);
});

test("without a known training dataset, any in-line align counts", () => {
  assert.equal(cameraCheck({ ...reads, dataset: null }, { ...inLine, root: "/x/other" }, false, now, ago).block, null);
});
