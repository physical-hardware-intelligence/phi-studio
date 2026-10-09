// What onboarding pre-selects (src/lib/onboard.ts). Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { test } from "node:test";
import { configIds, defaultIds, idChoices, seedPorts } from "../src/lib/onboard.ts";
import type { RigStatus } from "../src/lib/rig.ts";

const k = (...ids: string[]) => ids.map((id, i) => ({ id, mtime: 100 - i }));
const L = "/dev/tty.usbmodem5B790807501", F = "/dev/tty.usbmodem5B7B0096441";

// This Mac on 2026-10-09: a stale bimanual config whose ids have no files, the club's single-arm
// ids in ports.local.sh, and the files that do exist.
function mac(over: Partial<RigStatus> = {}): RigStatus {
  return {
    path: "/x/robot-config.yaml", exists: true, name: "Phi Bimanual", layout: "bimanual",
    arms: [
      { key: "left_leader", role: "leader", side: "left", port: L, id: "phi_bimanual_leader_left", calibration: null, calibrated: false, calibrated_at: null },
      { key: "left_follower", role: "follower", side: "left", port: F, id: "phi_bimanual_follower_left", calibration: null, calibrated: false, calibrated_at: null },
    ],
    cameras: [], problems: [], calibration_root: "/x/calibration",
    known: {
      follower: k("phi_bi_follower_right", "phi_bi_follower_left", "phi_follower"),
      leader: k("phi_bi_leader_right", "phi_bi_leader_left", "phi_leader"),
      bi_follower: k("phi_bi_follower"), bi_leader: k("phi_bi_leader"),
    },
    hint: {
      ports: { leader: L, follower: F, left_leader: L, left_follower: F },
      ids: { single: { follower: "phi_follower", leader: "phi_leader" }, bimanual: { follower: "phi_bi_follower", leader: "phi_bi_leader" } },
    },
    ...over,
  };
}

test("a stale config's ids without files are never offered; ports.local.sh's ids with files are", () => {
  assert.deepEqual(defaultIds(mac(), "single"), { follower: "phi_follower", leader: "phi_leader" });
  assert.deepEqual(defaultIds(mac(), "bimanual"), { follower: "phi_bi_follower", leader: "phi_bi_leader" });
});

test("the config's ids win when their files exist and the layout matches", () => {
  const rig = mac({ layout: "single", arms: [
    { key: "leader", role: "leader", side: null, port: L, id: "phi_bi_leader_left", calibration: null, calibrated: true, calibrated_at: 1 },
    { key: "follower", role: "follower", side: null, port: F, id: "phi_bi_follower_left", calibration: null, calibrated: true, calibrated_at: 1 },
  ] });
  assert.deepEqual(defaultIds(rig, "single"), { follower: "phi_bi_follower_left", leader: "phi_bi_leader_left" });
});

test("a single-arm config keeps its ids whole; a bimanual one drops the side LeRobot adds", () => {
  const single = mac({ layout: "single", arms: [
    { key: "follower", role: "follower", side: null, port: F, id: "phi_bi_follower_right", calibration: null, calibrated: true, calibrated_at: 1 },
    { key: "leader", role: "leader", side: null, port: L, id: "phi_bi_leader_right", calibration: null, calibrated: true, calibrated_at: 1 },
  ] });
  assert.deepEqual(configIds(single), { follower: "phi_bi_follower_right", leader: "phi_bi_leader_right" });
  assert.deepEqual(configIds(mac()), { follower: "phi_bimanual_follower", leader: "phi_bimanual_leader" });
});

test("without config or hint, a follower and leader with matching names pair up", () => {
  const rig = mac({ layout: null, arms: [], hint: { ports: {}, ids: {} } });
  assert.deepEqual(defaultIds(rig, "single"), { follower: "phi_bi_follower_right", leader: "phi_bi_leader_right" });
});

test("no matching pair: nothing is pre-selected, never the newest file of each role", () => {
  const rig = mac({ layout: null, arms: [], hint: { ports: {}, ids: {} },
    known: { follower: k("arm_a_follower"), leader: k("arm_b_leader"), bi_follower: [], bi_leader: [] } });
  assert.equal(defaultIds(rig, "single"), null);
  assert.deepEqual(idChoices(rig, "single"), { follower: ["arm_a_follower"], leader: ["arm_b_leader"] });
});

test("switching a bimanual config to one arm seeds the single slots from ports.local.sh", () => {
  assert.deepEqual(seedPorts(mac(), "single", ["leader", "follower"]), { leader: L, follower: F });
  // the config's own ports come first when it already has that layout
  const moved = mac({ hint: { ports: { left_leader: "/dev/tty.usbmodemOLD" }, ids: {} } });
  assert.equal(seedPorts(moved, "bimanual", ["left_leader", "left_follower"]).left_leader, L);
});
