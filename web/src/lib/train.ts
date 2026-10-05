import { useRef, useSyncExternalStore } from "react";
import { studio } from "./studio";
import { terminal } from "./terminal";

// The Train page's state (server: train_api.py). The server builds every command from checked fields;
// this store only sends form values and shows what comes back.

export type Where = "cluster" | "mac";
export interface TrainSettings {
  host: string; remote_base: string; env: string; partition: string; test_partition: string; gres: string;
  exclude: string; time: string; cpus: number; mem: string; remote_user: string;
}
export interface PolicyType { type: string; extra: string | null; missing_here: string[] }
export interface Policies { source: string; version: string | null; policies: PolicyType[]; needs: Record<string, string[]> }
export interface Dataset { id: string; episodes: number | null; frames: number | null; fps: number | null }
export type CheckStatus = "pass" | "warn" | "fail" | "skip";
export interface CheckRun { cmd: string; rc: number; output: string }
export interface CheckRow { id: string; title: string; status: CheckStatus; detail: string; fix: string | null; runs: CheckRun[] }
export interface CheckState {
  running: boolean; rows: CheckRow[]; at: number | null;
  limits?: { free: number | null; submitted: number | null; max_submit: number | null };
}
export interface JobPart {
  part: number; id: string; name: string; state: string; log: string; elapsed?: string; exit?: string; reason?: string;
}
export interface Progress { step: number; total: number; rate: number | null; eta_s: number | null; ended: string | null }
export interface FetchState { state: "running" | "done" | "error"; bytes: number; total: number; path: string; message: string | null }
export interface Run {
  id: string; name: string; where: Where; created: number; dataset: string; policy: string; steps: number;
  batch_size: number; save_freq: number; log_freq: number; parts_planned: number; push: boolean; repo_id: string;
  wandb: boolean; jobs: JobPart[]; error: { part: number; cmd: string; output: string } | null;
  fetched: { path: string; at: number; bytes: number } | null; state: string; progress: Progress | null;
  fetch: FetchState | null; remote_dir?: string; script_path?: string; host?: string; user?: string;
  partition?: string; part_time?: string; local_dir?: string;
}
export interface Point { step: number; sure?: boolean; loss?: number; lr?: number } // sure: false when the step is LeRobot's rounded one
export interface LogView extends Progress { run: string; points: Point[]; tail: string[]; view: Run }
export interface Plan { parts: number; part_time: string; total_hours: number; names: string[] }
export interface Preview {
  ok: boolean; errors: Record<string, string>; run_id?: string; command?: string; script?: string | null;
  plan?: Plan; run_dir?: string; output_dir?: string;
}
export interface TrainNotice {
  id: number; tone: "ok" | "warn" | "danger"; message: string; fix?: string | null; cmd?: string | null; output?: string | null;
}

export interface JobForm {
  where: Where; dataset: string; policy: string; pretrained: string; steps: number; batch_size: number;
  save_freq: number; name: string; push: boolean; repo_id: string; wandb: boolean; parts: number; stamp: string;
}

export interface TrainSnap {
  loaded: boolean;
  ready: boolean; // Studio has a data folder
  settings: TrainSettings | null;
  defaults: TrainSettings | null;
  settingsErrors: Record<string, string>;
  settingsSaved: number; // time of the last save, for a short "Saved"
  runs: Run[];
  policies: Policies | null;
  datasets: Dataset[];
  check: CheckState;
  dataDir: string | null;
  preview: Preview | null;
  logs: Record<string, LogView>;
  notice: TrainNotice | null;
  submitting: boolean;
  starting: boolean; // a run on this Mac, waiting for its command
  form: JobForm; // kept here, not in the page, so moving away and back keeps it
}

/** A run id's time stamp: local time, YYYYMMDD-HHMMSS, set once per form (train.py STAMP). */
export function newStamp(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

type Listener = () => void;

class TrainStore {
  snap: TrainSnap = {
    loaded: false, ready: false, settings: null, defaults: null, settingsErrors: {}, settingsSaved: 0, runs: [],
    policies: null, datasets: [], check: { running: false, rows: [], at: null }, dataDir: null, preview: null,
    logs: {}, notice: null, submitting: false, starting: false,
    form: {
      where: "cluster", dataset: "", policy: "act", pretrained: "", steps: 100000, batch_size: 8, save_freq: 10000,
      name: "", push: false, repo_id: "", wandb: false, parts: 1, stamp: newStamp(),
    },
  };
  private listeners = new Set<Listener>();
  private noticeId = 0;
  private savingSettings = false;

  constructor() {
    const on = (t: string, fn: (m: any) => void) => studio.onMessage(t, fn);
    on("train_state", (m) => this.set({
      loaded: true, ready: m.ready, settings: m.settings, defaults: m.defaults, runs: m.runs, datasets: m.datasets ?? [],
      check: m.check, dataDir: m.data_dir, ...(m.policies ? { policies: m.policies } : {}),
    }));
    on("train_settings", (m) => {
      const saved = this.savingSettings;
      this.savingSettings = false;
      this.set({ settings: m.settings, ...(saved ? { settingsErrors: {}, settingsSaved: Date.now() } : {}) });
      this.preview(); // the script names the run folder, which the settings (and the learnt user) decide
    });
    on("train_policies", (m) => this.set({ policies: m.policies }));
    on("train_preview", (m) => this.set({ preview: m }));
    on("train_check", (m) => this.set({ check: { running: m.running, rows: m.rows, at: m.at, limits: m.limits } }));
    on("train_check_row", (m) => {
      const rows = this.snap.check.rows.filter((r) => r.id !== m.row.id);
      this.set({ check: { ...this.snap.check, rows: [...rows, m.row] } });
    });
    on("train_runs", (m) => this.set({ runs: m.runs }));
    on("train_submitted", (m) => {
      // WHY a new stamp only on success: after a failed submit, Look for it needs the same run id.
      this.set({ submitting: false, ...(m.ok ? { form: { ...this.snap.form, stamp: newStamp() } } : {}) });
      this.notify(m.ok ? "ok" : "danger", m.message, null, m.error?.cmd, m.error?.output);
    });
    on("train_mac_ready", (m) => {
      this.set({ starting: false, form: { ...this.snap.form, stamp: newStamp() } });
      terminal.run(m.command);
      this.notify("ok", "Training started in the terminal below. Its log is kept, so this page charts it too.");
    });
    on("train_log", (m: LogView) => {
      this.set({
        logs: { ...this.snap.logs, [m.run]: m },
        runs: this.snap.runs.map((r) => (r.id === m.run ? m.view : r)),
      });
    });
    on("train_cancelled", (m) => this.notify(m.ok ? "ok" : "danger",
      m.ok ? "Cancel sent for this run's parts that had not ended." : "scancel reported an error.", null, m.cmd, m.output));
    on("train_found", (m) => this.notify(m.found.length ? "ok" : "warn",
      m.found.length
        ? `Found ${m.found.length} job${m.found.length > 1 ? "s" : ""} with this run's names; ${m.new} new to Studio.`
        : "No job with this run's names exists on the cluster from the last two days.", null, m.cmd, m.output));
    on("train_fetch", (m) => this.set({
      runs: this.snap.runs.map((r) => (r.id === m.run ? { ...r, fetch: { state: m.state, bytes: m.bytes, total: m.total, path: m.path, message: m.message } } : r)),
    }));
    on("train_error", (m) => {
      if (m.what === "settings") {
        this.savingSettings = false;
        this.set({ settingsErrors: m.errors ?? { form: m.message } });
        return;
      }
      if (m.what === "submit") this.set({ submitting: false });
      if (m.what === "mac") this.set({ starting: false });
      const fields = m.errors ? Object.values(m.errors as Record<string, string>).join(" ") : m.message;
      this.notify("danger", fields, m.fix, m.cmd, m.output);
    });
    // A command the server could not answer at all (an exception): stop waiting for it.
    on("error", (m) => {
      if (typeof m.cmd === "string" && m.cmd.startsWith("train_")) this.set({ submitting: false, starting: false });
    });
  }

  subscribe = (fn: Listener): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  private set(patch: Partial<TrainSnap>): void {
    this.snap = { ...this.snap, ...patch };
    for (const fn of this.listeners) fn();
  }

  private notify(tone: TrainNotice["tone"], message: string, fix?: string | null, cmd?: string | null, output?: string | null): void {
    this.set({ notice: { id: ++this.noticeId, tone, message, fix, cmd, output } });
  }

  dismiss(): void { this.set({ notice: null }); }

  private sendOr(msg: Record<string, unknown>): boolean {
    if (studio.send(msg)) return true;
    this.notify("danger", "Studio is not connected, so nothing was sent.", "Wait for the link to come back.");
    return false;
  }

  init(): void { studio.send({ cmd: "train_init" }); }
  preview(): void { studio.send({ cmd: "train_preview", job: this.snap.form }); }
  setForm(patch: Partial<JobForm>): void { this.set({ form: { ...this.snap.form, ...patch } }); }
  saveSettings(s: TrainSettings): void {
    this.savingSettings = true;
    if (!this.sendOr({ cmd: "train_settings", settings: s })) this.savingSettings = false;
  }
  runCheck(): void {
    const f = this.snap.form;
    this.sendOr({ cmd: "train_check", job: f.where === "cluster" ? f : null });
  }
  submit(): void { if (this.sendOr({ cmd: "train_submit", job: this.snap.form })) this.set({ submitting: true }); }
  startMac(): void { if (this.sendOr({ cmd: "train_mac_start", job: this.snap.form })) this.set({ starting: true }); }
  refresh(): void { studio.send({ cmd: "train_refresh" }); }
  poll(run: string): void { studio.send({ cmd: "train_poll", run }); }
  cancel(run: string): void { this.sendOr({ cmd: "train_cancel", run }); }
  find(run: string): void { this.sendOr({ cmd: "train_find", run }); }
  fetch(run: string): void { this.sendOr({ cmd: "train_fetch", run }); }
}

export const train = new TrainStore();

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Same caching as useStudio: a selector returning a fresh object must not re-render forever (React #185). */
export function useTrain<T>(select: (s: TrainSnap) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(train.subscribe, () => {
    const value = select(train.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}

// -- words for the table ----------------------------------------------------------------------------
const STATES: Record<string, [string, "ok" | "info" | "warn" | "danger" | "neutral"]> = {
  RUNNING: ["Running", "info"], PENDING: ["Waiting", "neutral"], COMPLETED: ["Done", "ok"],
  TIMEOUT: ["Hit its time limit", "warn"], CANCELLED: ["Cancelled", "neutral"], FAILED: ["Failed", "danger"],
  OUT_OF_MEMORY: ["Out of memory", "danger"], NODE_FAIL: ["Node failed", "danger"], PREEMPTED: ["Preempted", "warn"],
  BOOT_FAIL: ["Node failed to boot", "danger"], DEADLINE: ["Missed its deadline", "warn"],
  "NOT SUBMITTED": ["Not submitted", "danger"], SUBMITTING: ["Submitting", "neutral"], STOPPED: ["Stopped", "neutral"],
  "NOT STARTED": ["Not started", "neutral"], UNKNOWN: ["Unknown", "neutral"],
};
export function stateWords(state: string): [string, "ok" | "info" | "warn" | "danger" | "neutral"] {
  const key = state.split(" ")[0] === "CANCELLED" ? "CANCELLED" : state;
  return STATES[key] ?? [state.charAt(0) + state.slice(1).toLowerCase().replace(/_/g, " "), "neutral"];
}

export function duration(s: number | null): string {
  if (s === null || !Number.isFinite(s)) return "unknown";
  if (s < 90) return `${Math.round(s)} s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${u[i]}`;
}
