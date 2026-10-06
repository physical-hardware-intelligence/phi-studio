import { test } from "node:test";
import assert from "node:assert/strict";
import { calfileProblems, type CalFile, type CalFilesView } from "../src/lib/calfiles.ts";

const F = "robots/so_follower";
function file(rel: string, o: Partial<CalFile> = {}): CalFile {
  return { rel, id: rel.split("/").pop()!.replace(".json", ""), folder: F, used_by: [], link: null, error: null,
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

test("a used link, a broken file and one file on two ports' motors are dangers", () => {
  const p = calfileProblems(view([
    file(`${F}/a.json`, { used_by: ["left_follower"], link: "b.json" }),
    file(`${F}/c.json`, { used_by: ["right_follower"], error: "bad json" }),
    file(`${F}/d.json`, { used_by: ["follower"], on_ports: ["/dev/tty.x", "/dev/tty.y"] }),
  ]), (k) => k.toUpperCase());
  assert.equal(p.filter((x) => x.level === "danger").length, 3);
  assert.ok(p[0].text.startsWith("LEFT_FOLLOWER"));
});
