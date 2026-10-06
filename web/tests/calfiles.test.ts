import { test } from "node:test";
import assert from "node:assert/strict";
import { calfileProblems, type CalFile, type CalFilesView } from "../src/lib/calfiles.ts";

const F = "robots/so_follower";
function file(rel: string, o: Partial<CalFile> = {}): CalFile {
  return { rel, id: rel.split("/").pop()!.replace(".json", ""), folder: F, used_by: [], link: null, real: rel, error: null,
    unfinished: null, same_as: [], on_ports: [], mtime: 0, junk: null, unused: !(o.used_by?.length), ...o };
}
function view(files: CalFile[]): CalFilesView {
  return { root: "/cal", files, archives: [], config: true, shared: { source: null, exists: false, rows: [], candidates: [] }, scanned: true };
}

test("two arms using files with one set of numbers is the swapped-calibration fault", () => {
  const l = file(`${F}/l.json`, { used_by: ["left_follower"], same_as: [`${F}/r.json`] });
  const r = file(`${F}/r.json`, { used_by: ["right_follower"], same_as: [`${F}/l.json`] });
  const p = calfileProblems(view([l, r]));
  assert.equal(p.filter((x) => x.level === "danger" && x.text.includes("same numbers")).length, 1);
});

test("an unused alias with the same numbers is not a problem, only an extra file", () => {
  const used = file(`${F}/phi_bi_follower_left.json`, { used_by: ["left_follower"], same_as: [`${F}/phi_follower.json`] });
  const alias = file(`${F}/phi_follower.json`, { same_as: [`${F}/phi_bi_follower_left.json`] });
  const p = calfileProblems(view([used, alias]));
  assert.deepEqual(p.map((x) => x.level), ["warn"]);
  assert.match(p[0].text, /1 file is not used/);
});

test("two arms on one file through a link, a broken file and one file on two ports' motors are dangers", () => {
  const p = calfileProblems(view([
    file(`${F}/a.json`, { used_by: ["left_follower"], link: "b.json", real: `${F}/b.json`, same_as: [`${F}/b.json`] }),
    file(`${F}/b.json`, { used_by: ["right_follower"], same_as: [`${F}/a.json`] }),
    file(`${F}/c.json`, { used_by: ["leader"], error: "bad json" }),
    file(`${F}/d.json`, { used_by: ["follower"], on_ports: ["/dev/tty.x", "/dev/tty.y"] }),
  ]), (k) => k.toUpperCase());
  assert.equal(p.filter((x) => x.level === "danger").length, 3);
  assert.ok(p[0].text.startsWith("LEFT_FOLLOWER and RIGHT_FOLLOWER read one file, b.json"));
});

test("one arm under two ids through a link (phi_bi_left -> phi_follower) is not a fault", () => {
  const link = file(`${F}/phi_bi_left.json`, { used_by: ["left_follower"], link: "phi_follower.json",
    real: `${F}/phi_follower.json`, same_as: [`${F}/phi_follower.json`] });
  const target = file(`${F}/phi_follower.json`, { used_by: ["left_follower"], same_as: [`${F}/phi_bi_left.json`] });
  assert.deepEqual(calfileProblems(view([link, target])), []);
});

test("a rig that uses none of the shared files is told how to switch", () => {
  const v = view([file(`${F}/phi_follower.json`, { used_by: ["follower"] })]);
  v.shared = { source: "/phi/configs/calibration", exists: true, candidates: [], rows: [
    { rel: `${F}/phi_bi_follower_left.json`, id: "phi_bi_follower_left", state: "new", max_deg: null, worst_joint: null, error: null }] };
  const p = calfileProblems(v);
  assert.ok(p.some((x) => x.text.includes("uses none of the shared files (it uses phi_follower)")));
  v.shared.rows[0].rel = `${F}/phi_follower.json`;
  assert.ok(!calfileProblems(v).some((x) => x.text.includes("none of the shared")));
});

test("a shared file whose joint stops are another arm's is a danger and named", () => {
  const L = "teleoperators/so_leader";
  const v = view([file(`${L}/phi_bi_leader_left.json`, { folder: L, used_by: ["left_leader"] })]);
  v.shared = { source: "/phi/configs/calibration", exists: true, candidates: [], rows: [
    { rel: `${L}/phi_bi_leader_left.json`, id: "phi_bi_leader_left", state: "differs", max_deg: 97.9, worst_joint: "gripper",
      error: null, named_for: "left_leader", arm_of: "right_leader", arm_gap: 19.6 }] };
  const p = calfileProblems(v, (k) => k.replace("_", " "));
  assert.ok(p.some((x) => x.level === "danger" && x.text.includes("named for the left leader, but its joint stops are the right leader's")));
  v.shared.rows[0].arm_of = "left_leader";
  assert.ok(!calfileProblems(v).some((x) => x.text.includes("joint stops")));
});

test("two arms' files with one arm's joint stops are a danger", () => {
  const p = calfileProblems(view([
    file(`${F}/l.json`, { used_by: ["left_follower"], same_arm_as: [`${F}/r.json`] }),
    file(`${F}/r.json`, { used_by: ["right_follower"], same_arm_as: [`${F}/l.json`] }),
  ]));
  assert.equal(p.filter((x) => x.level === "danger" && x.text.includes("one arm's joint stops")).length, 1);
});
