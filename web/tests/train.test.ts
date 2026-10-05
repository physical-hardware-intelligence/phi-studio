// The Train page's store rules: what a new link resets, which run view wins, and where errors go.
// Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { test } from "node:test";
import { NO_BUSY, TrainStore, newer, type Run } from "../src/lib/traincore.ts";

function rig(online = true) {
  const sent: Record<string, unknown>[] = [];
  const typed: string[] = [];
  const io = {
    online,
    send(msg: Record<string, unknown>) { if (!io.online) return false; sent.push(msg); return true; },
    run(cmd: string) { typed.push(cmd); },
    now: () => new Date(2026, 9, 4, 19, 0, 0),
  };
  return { store: new TrainStore(io), sent, typed, io };
}

const run = (rev: number, state = "RUNNING"): Run => ({
  id: "act-20261004-190000", name: "act", where: "cluster", created: 0, dataset: "a/b", policy: "act", steps: 10,
  batch_size: 8, save_freq: 5, log_freq: 1, parts_planned: 2, push: false, repo_id: "", wandb: false, jobs: [],
  error: null, fetched: null, state, progress: null, fetch: null, rev,
});

test("a new link clears submitting and starting and asks for the state again", () => {
  const { store, sent } = rig();
  store.receive("train_state", { ready: true, settings: null, defaults: null, runs: [], check: { running: false, rows: [], at: null }, data_dir: "/d", busy: { ...NO_BUSY, submit: ["x"] } });
  store.submit();
  store.setForm({ where: "mac" });
  store.startMac();
  assert.equal(store.snap.submitting, true);
  assert.equal(store.snap.starting, true);
  sent.length = 0;
  store.receive("hello", { control: true });
  assert.equal(store.snap.submitting, false);
  assert.equal(store.snap.starting, false);
  assert.deepEqual(store.snap.busy, NO_BUSY);
  assert.deepEqual(sent, [{ cmd: "train_init" }]);
});

test("busy comes from the server, so every window sees a submit in flight", () => {
  const { store } = rig();
  store.receive("train_busy", { busy: { submit: ["r1"], resubmit: [], find: [], cancel: [], fetch: ["r2"] } });
  assert.equal(store.isBusy("r1"), true);
  assert.equal(store.isBusy("r1", "find"), false);
  assert.equal(store.isBusy("r2", "fetch"), true);
  assert.equal(store.isBusy("r2"), false); // a fetch does not block Look for it or Cancel
});

test("an older run view never replaces a newer one", () => {
  const { store } = rig();
  store.receive("train_runs", { runs: [run(5, "COMPLETED")] });
  store.receive("train_log", { run: run(4).id, view: run(4, "RUNNING"), points: [], tail: [], step: 1, total: 10, rate: null, eta_s: null, ended: null });
  assert.equal(store.snap.runs[0].state, "COMPLETED");
  store.receive("train_runs", { runs: [run(3, "PENDING")] });
  assert.equal(store.snap.runs[0].state, "COMPLETED");
  store.receive("train_runs", { runs: [run(9, "FAILED")] });
  assert.equal(store.snap.runs[0].state, "FAILED");
  assert.equal(newer(undefined, run(1)).rev, 1);
});

test("a failed poll does not hide the notice of a failed sbatch", () => {
  const { store } = rig();
  store.receive("train_submitted", { ok: false, message: "sbatch failed for part 2 of 3.", error: { cmd: "sbatch", output: "x" } });
  store.receive("train_error", { what: "poll", message: "Could not reach the cluster." });
  store.receive("train_error", { what: "refresh", message: "Could not reach the cluster for job states." });
  assert.equal(store.snap.notice?.message, "sbatch failed for part 2 of 3.");
  assert.equal(store.snap.background?.message, "Could not reach the cluster for job states.");
  store.receive("train_log", { run: "r", view: run(1), points: [], tail: [], step: 1, total: 10, rate: null, eta_s: null, ended: null });
  assert.equal(store.snap.background, null);
});

test("a submit error clears submitting; a success takes a new stamp, a failure keeps it", () => {
  const { store } = rig();
  const stamp = store.snap.form.stamp;
  store.submit();
  store.receive("train_submitted", { ok: false, message: "no", error: null });
  assert.equal(store.snap.submitting, false);
  assert.equal(store.snap.form.stamp, stamp);
  store.submit();
  store.receive("train_error", { what: "submit", message: "This run is being submitted right now." });
  assert.equal(store.snap.submitting, false);
});

test("settings the server reset are named", () => {
  const { store } = rig();
  store.receive("train_state", { ready: true, settings: null, defaults: null, runs: [], check: { running: false, rows: [], at: null }, data_dir: "/d", settings_dropped: ["partition", "time"] });
  assert.deepEqual(store.snap.settingsDropped, ["partition", "time"]);
  store.receive("train_settings", { settings: null, dropped: [] });
  assert.deepEqual(store.snap.settingsDropped, []);
});
