// The Models page's store rules: which Hub replies it keeps, and when the rollout command is cleared.
// Run: npm test (Node's own runner).
import assert from "node:assert/strict";
import { test } from "node:test";
import { HubStore, downloadView, type Progress, type RolloutAsk } from "../src/lib/hubcore.ts";

function rig(online = true) {
  const sent: Record<string, unknown>[] = [];
  const later: (() => void)[] = [];
  const io = {
    online,
    send(msg: Record<string, unknown>) { if (!io.online) return false; sent.push(msg); return true; },
    localError() {},
    later(fn: () => void) { later.push(fn); },
  };
  const store = new HubStore(io);
  const flush = () => { while (later.length) later.shift()!(); };
  return { store, sent, io, flush };
}

const ASK: RolloutAsk = {
  repo_id: "me/act", revision: "c".repeat(40), task: "t", duration_s: 60, strategy: "base", episodes: 10,
  dataset_repo_id: null, rename_map: {}, upload: false,
};
const job = (state: Progress["state"], message: string | null = null): Progress =>
  ({ repo_id: "me/act", revision: "r1", state, done: 5, total: 10, message, path: null });

test("a late answer to an older search does not replace the newest", () => {
  const { store } = rig();
  store.search("act");
  store.search("smolvla");
  store.receive("hub_search", { query: "smolvla", results: [{ repo_id: "new" }], error: null });
  store.receive("hub_search", { query: "act", results: [{ repo_id: "old" }], error: null });
  assert.equal(store.snap.search?.query, "smolvla");
  assert.equal(store.snap.search?.results[0].repo_id, "new");
});

test("the newest search's answer is kept, and clears the spinner", () => {
  const { store } = rig();
  store.search("act");
  store.receive("hub_search", { query: "act", results: [], error: null });
  assert.equal(store.snap.search?.query, "act");
  assert.equal(store.snap.searching, null);
});

test("a search whose answer was lost in a reconnect is asked again", () => {
  const { store, sent, flush } = rig();
  store.open();
  store.receive("hub_search", { query: "", results: [], error: null });
  store.search("smolvla");
  sent.length = 0;
  store.receive("hello", {});
  flush();
  assert.ok(sent.some((m) => m.cmd === "hub_search" && m.query === "smolvla"));
  sent.length = 0;
  store.receive("hub_search", { query: "smolvla", results: [], error: null });
  store.receive("hello", {});
  flush();
  assert.ok(!sent.some((m) => m.cmd === "hub_search"));
});

test("a failed or cancelled download leaves the button, with the message", () => {
  assert.equal(downloadView(job("running"), "me/act", "r1"), "progress");
  assert.equal(downloadView(job("cancelling"), "me/act", "r1"), "progress");
  assert.equal(downloadView(job("done"), "me/act", "r1"), "progress");
  assert.equal(downloadView(job("error", "disk full"), "me/act", "r1"), "button");
  assert.equal(downloadView(job("cancelled"), "me/act", "r1"), "button");
  assert.equal(downloadView(job("running"), "me/other", "r1"), "button");
  assert.equal(downloadView(null, "me/act", "r1"), "button");
});

test("asking for a new command drops the old one until the answer comes", () => {
  const { store, sent } = rig();
  store.buildRollout(ASK);
  store.receive("hub_rollout", { seq: sent.at(-1)!.seq, cmd: "lerobot-rollout --a", error: null, problems: [], warnings: [] });
  assert.equal(store.snap.rollout?.cmd, "lerobot-rollout --a");
  store.buildRollout({ ...ASK, task: "other" });
  assert.equal(store.snap.rollout, null);
});

test("an edit drops the shown command at once, before the debounced ask, and its late answer", () => {
  const { store, sent } = rig();
  store.buildRollout(ASK);
  const seq = sent.at(-1)!.seq;
  store.receive("hub_rollout", { seq, cmd: "lerobot-rollout --a", error: null, problems: [], warnings: [] });
  store.editingRollout();
  assert.equal(store.snap.rollout, null);
  store.receive("hub_rollout", { seq, cmd: "lerobot-rollout --a", error: null, problems: [], warnings: [] });
  assert.equal(store.snap.rollout, null);
});

test("a crashed rollout request shows its error and no command", () => {
  const { store, sent } = rig();
  store.buildRollout(ASK);
  store.receive("hub_rollout", { seq: sent.at(-1)!.seq, cmd: "lerobot-rollout --a", error: null, problems: [], warnings: [] });
  store.buildRollout({ ...ASK, task: "other" });
  store.receive("error", { cmd: "hub_rollout", message: "Studio could not read the Run panel's request" });
  assert.equal(store.snap.rollout?.cmd, null);
  assert.match(store.snap.rollout?.error ?? "", /could not read/);
});

test("a reconnect drops the command and asks for it again", () => {
  const { store, sent, flush } = rig();
  store.open();
  store.buildRollout(ASK);
  store.receive("hub_rollout", { seq: sent.at(-1)!.seq, cmd: "lerobot-rollout --a", error: null, problems: [], warnings: [] });
  sent.length = 0;
  store.receive("hello", {});
  assert.equal(store.snap.rollout, null);
  flush();
  assert.ok(sent.some((m) => m.cmd === "hub_rollout" && m.task === "t"));
});

test("a model being read when the link drops is read again after the reconnect", () => {
  const { store, sent, flush } = rig();
  store.open();
  store.inspect("me/act");
  sent.length = 0;
  store.receive("hello", {});
  flush();
  assert.ok(sent.some((m) => m.cmd === "hub_inspect" && m.repo_id === "me/act"));
  assert.equal(store.snap.inspecting, "me/act");
  store.receive("hub_model", { asked: "me/act", error: null });
  assert.equal(store.snap.detail?.asked, "me/act");
});

test("a closed model is not read again after a reconnect", () => {
  const { store, sent, flush } = rig();
  store.open();
  store.inspect("me/act");
  store.closeDetail();
  sent.length = 0;
  store.receive("hello", {});
  flush();
  assert.ok(!sent.some((m) => m.cmd === "hub_inspect"));
});

test("a crashed model read ends the wait with its error", () => {
  const { store } = rig();
  store.inspect("me/act");
  store.receive("error", { cmd: "hub_inspect", message: "Studio could not answer hub_inspect" });
  assert.equal(store.snap.inspecting, null);
  assert.match(store.snap.detail?.error ?? "", /could not answer/);
});
