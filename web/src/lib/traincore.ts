// The Train page's store (server: src/phi_studio/train_api.py). Import-free, so Node's own test runner
// runs it (web/tests/train.test.ts); train.ts wires it to the Studio link, the terminal and React.
// The server builds every command from checked fields; this store only sends form values and shows
// what comes back.

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
export interface CheckLimits {
  free: number | null; submitted: number | null; max_submit: number | null; partition?: string | null; max_time?: string | null;
}
export interface CheckState { running: boolean; rows: CheckRow[]; at: number | null; limits?: CheckLimits }
export interface JobPart {
  part: number; id: string; name: string; state: string; log: string; elapsed?: string; exit?: string; reason?: string;
}
export interface Progress {
  step: number; total: number; rate: number | null; eta_s: number | null; ended: string | null; stalled?: string | null;
}
export interface FetchState { state: "running" | "done" | "error"; bytes: number; total: number; path: string; message: string | null }
export interface Run {
  id: string; name: string; where: Where; created: number; dataset: string; policy: string; steps: number;
  batch_size: number; save_freq: number; log_freq: number; parts_planned: number; push: boolean; repo_id: string;
  wandb: boolean; jobs: JobPart[]; error: { part: number; cmd: string; output: string } | null;
  fetched: { path: string; at: number; bytes: number; replaced?: string | null } | null; state: string;
  progress: Progress | null; fetch: FetchState | null; remote_dir?: string; script_path?: string; host?: string;
  user?: string; partition?: string; part_time?: string; local_dir?: string; rate?: number;
  remaining_from?: number | null; // after a failed sbatch: the first part the run never got
  rev?: number; // the server stamps each view; a window keeps the newest
}
export interface Point { step: number; sure?: boolean; loss?: number; lr?: number } // sure: false when the step is LeRobot's rounded one
export interface LogView extends Progress { run: string; points: Point[]; tail: string[]; view: Run }
export interface Fit { rate: number | null; from_run: string | null; save_s: number | null; part_s: number; fits: boolean | null }
export interface Plan { parts: number; part_time: string; total_hours: number; names: string[]; fit?: Fit }
export interface Preview {
  ok: boolean; errors: Record<string, string>; run_id?: string; command?: string; script?: string | null;
  plan?: Plan; run_dir?: string; output_dir?: string;
}
export interface TrainNotice {
  id: number; tone: "ok" | "warn" | "danger"; message: string; fix?: string | null; cmd?: string | null; output?: string | null;
}
/** Changes in flight on the server, by kind, each a list of run ids (train_api.py CHANGES). */
export interface Busy { submit: string[]; resubmit: string[]; find: string[]; cancel: string[]; fetch: string[] }
export const NO_BUSY: Busy = { submit: [], resubmit: [], find: [], cancel: [], fetch: [] };

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
  settingsDropped: string[]; // stored fields that failed their check and went back to the default
  runs: Run[];
  policies: Policies | null;
  datasets: Dataset[];
  check: CheckState;
  dataDir: string | null;
  preview: Preview | null;
  logs: Record<string, LogView>;
  notice: TrainNotice | null; // the result of something the person asked for
  background: TrainNotice | null; // a failed poll or refresh: shown apart, so it never hides the notice
  submitting: boolean; // sent, no answer yet; cleared on any answer and on a new link
  starting: boolean; // a run on this Mac, waiting for its command
  busy: Busy; // from the server, the same in every window
  form: JobForm; // kept here, not in the page, so moving away and back keeps it
}

export interface TrainIO {
  send(msg: Record<string, unknown>): boolean;
  run(command: string): void; // type a command into Studio's terminal
  now(): Date;
}

export const TRAIN_MESSAGES = [
  "train_state", "train_settings", "train_policies", "train_preview", "train_check", "train_check_row", "train_runs",
  "train_submitted", "train_resubmitted", "train_mac_ready", "train_log", "train_cancelled", "train_found",
  "train_fetch", "train_error", "train_busy", "error", "hello",
] as const;

/** A run id's time stamp: local time, YYYYMMDD-HHMMSS, set once per form (train.py STAMP). */
export function newStamp(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/** The newer of two views of one run, by the server's stamp. WHY: a poll's answer can be built before
 * a broadcast that arrives first, and must not put back the older parts and states. */
export function newer(have: Run | undefined, got: Run): Run {
  if (!have || have.rev === undefined || got.rev === undefined) return got;
  return got.rev >= have.rev ? got : have;
}

const BACKGROUND = new Set(["poll", "refresh"]);
const NAMES: Record<string, string> = {
  host: "ssh host", remote_base: "Run folder", env: "Environment setup", partition: "Partition",
  test_partition: "Test partition", gres: "GPUs", exclude: "Nodes to avoid", time: "Time per part", cpus: "CPUs",
  mem: "Memory", remote_user: "Cluster user",
};
export const settingName = (k: string) => NAMES[k] ?? k;

type Listener = () => void;

export class TrainStore {
  snap: TrainSnap;
  private listeners = new Set<Listener>();
  private noticeId = 0;
  private savingSettings = false;
  private io: TrainIO;

  constructor(io: TrainIO) {
    this.io = io;
    this.snap = {
      loaded: false, ready: false, settings: null, defaults: null, settingsErrors: {}, settingsSaved: 0,
      settingsDropped: [], runs: [], policies: null, datasets: [], check: { running: false, rows: [], at: null },
      dataDir: null, preview: null, logs: {}, notice: null, background: null, submitting: false, starting: false,
      busy: NO_BUSY,
      form: {
        where: "cluster", dataset: "", policy: "act", pretrained: "", steps: 100000, batch_size: 8, save_freq: 10000,
        name: "", push: false, repo_id: "", wandb: false, parts: 1, stamp: newStamp(io.now()),
      },
    };
  }

  /** One message from the server. */
  receive(type: string, m: any): void {
    switch (type) {
      case "train_state":
        this.set({
          loaded: true, ready: m.ready, settings: m.settings, defaults: m.defaults, datasets: m.datasets ?? [],
          runs: this.merge(m.runs), check: m.check, dataDir: m.data_dir, busy: m.busy ?? NO_BUSY,
          settingsDropped: m.settings_dropped ?? [], ...(m.policies ? { policies: m.policies } : {}),
        });
        break;
      case "train_settings": {
        const saved = this.savingSettings;
        this.savingSettings = false;
        this.set({
          settings: m.settings, ...(m.dropped ? { settingsDropped: m.dropped } : {}),
          ...(saved ? { settingsErrors: {}, settingsSaved: Date.now() } : {}),
        });
        this.preview(); // the script names the run folder, which the settings (and the learnt user) decide
        break;
      }
      case "train_policies": this.set({ policies: m.policies }); break;
      case "train_preview": this.set({ preview: m }); break;
      case "train_check": this.set({ check: { running: m.running, rows: m.rows, at: m.at, limits: m.limits } }); break;
      case "train_check_row": {
        const rows = this.snap.check.rows.filter((r) => r.id !== m.row.id);
        this.set({ check: { ...this.snap.check, rows: [...rows, m.row] } });
        break;
      }
      case "train_runs": this.set({ runs: this.merge(m.runs) }); break;
      case "train_busy": this.set({ busy: { ...NO_BUSY, ...m.busy } }); break;
      case "train_submitted":
        // WHY a new stamp only on success: after a failed submit, Look for it needs the same run id.
        this.set({ submitting: false, ...(m.ok ? { form: { ...this.snap.form, stamp: newStamp(this.io.now()) } } : {}) });
        this.notify(m.ok ? "ok" : "danger", m.message, null, m.error?.cmd, m.error?.output);
        break;
      case "train_resubmitted":
        this.notify(m.ok ? "ok" : "danger", m.message, null, m.error?.cmd, m.error?.output);
        break;
      case "train_mac_ready":
        this.set({ starting: false, form: { ...this.snap.form, stamp: newStamp(this.io.now()) } });
        this.io.run(m.command);
        this.notify("ok", "Training started in the terminal below. Its log is kept, so this page charts it too.");
        break;
      case "train_log": {
        const v = m as LogView;
        this.set({
          logs: { ...this.snap.logs, [v.run]: v },
          runs: this.snap.runs.map((r) => (r.id === v.run ? newer(r, v.view) : r)),
          background: null, // a poll that worked: an older poll error no longer holds
        });
        break;
      }
      case "train_cancelled":
        this.notify(m.ok ? "ok" : "danger",
          m.ok ? "Cancel sent for this run's parts that had not ended." : "scancel reported an error.", null, m.cmd, m.output);
        break;
      case "train_found":
        this.notify(m.found.length ? "ok" : "warn", m.found.length
          ? `Found ${m.found.length} job${m.found.length > 1 ? "s" : ""} with this run's names; ${m.new} new to Studio.`
          : "No job with this run's names exists on the cluster since the day before the run was made.", null, m.cmd, m.output);
        break;
      case "train_fetch":
        this.set({
          runs: this.snap.runs.map((r) => (r.id === m.run
            ? { ...r, fetch: { state: m.state, bytes: m.bytes, total: m.total, path: m.path, message: m.message } } : r)),
        });
        break;
      case "train_error": {
        if (m.what === "settings") {
          this.savingSettings = false;
          this.set({ settingsErrors: m.errors ?? { form: m.message } });
          return;
        }
        if (m.what === "submit") this.set({ submitting: false });
        if (m.what === "mac") this.set({ starting: false });
        const text = m.errors ? Object.values(m.errors as Record<string, string>).join(" ") : m.message;
        if (BACKGROUND.has(m.what)) this.set({ background: this.make("warn", text, m.fix, m.cmd, m.output) });
        else this.notify("danger", text, m.fix, m.cmd, m.output);
        break;
      }
      case "error":
        // A command the server could not answer at all (an exception): stop waiting for it.
        if (typeof m.cmd === "string" && m.cmd.startsWith("train_")) this.set({ submitting: false, starting: false });
        break;
      case "hello":
        // A new link: any answer in flight is gone, and the server's busy list is the truth.
        this.savingSettings = false;
        this.set({ submitting: false, starting: false, busy: NO_BUSY });
        if (this.snap.loaded) this.init();
        break;
    }
  }

  private merge(runs: Run[]): Run[] {
    const have = new Map(this.snap.runs.map((r) => [r.id, r]));
    return runs.map((r) => newer(have.get(r.id), r));
  }

  subscribe = (fn: Listener): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  private set(patch: Partial<TrainSnap>): void {
    this.snap = { ...this.snap, ...patch };
    for (const fn of this.listeners) fn();
  }

  private make(tone: TrainNotice["tone"], message: string, fix?: string | null, cmd?: string | null, output?: string | null): TrainNotice {
    return { id: ++this.noticeId, tone, message, fix, cmd, output };
  }

  private notify(tone: TrainNotice["tone"], message: string, fix?: string | null, cmd?: string | null, output?: string | null): void {
    this.set({ notice: this.make(tone, message, fix, cmd, output) });
  }

  dismiss(): void { this.set({ notice: null }); }
  dismissBackground(): void { this.set({ background: null }); }

  private sendOr(msg: Record<string, unknown>): boolean {
    if (this.io.send(msg)) return true;
    this.notify("danger", "Studio is not connected, so nothing was sent.", "Wait for the link to come back.");
    return false;
  }

  /** Whether a change of this kind (or any change, with no kind) is in flight for a run. */
  isBusy(run: string, kind?: keyof Busy): boolean {
    const b = this.snap.busy;
    if (kind) return b[kind].includes(run);
    return b.submit.includes(run) || b.resubmit.includes(run) || b.find.includes(run) || b.cancel.includes(run);
  }

  init(): void { this.io.send({ cmd: "train_init" }); }
  preview(): void { this.io.send({ cmd: "train_preview", job: this.snap.form }); }
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
  resubmit(run: string): void { this.sendOr({ cmd: "train_resubmit", run }); }
  startMac(): void { if (this.sendOr({ cmd: "train_mac_start", job: this.snap.form })) this.set({ starting: true }); }
  refresh(): void { this.io.send({ cmd: "train_refresh" }); }
  poll(run: string): void { this.io.send({ cmd: "train_poll", run }); }
  cancel(run: string): void { this.sendOr({ cmd: "train_cancel", run }); }
  find(run: string): void { this.sendOr({ cmd: "train_find", run }); }
  fetch(run: string): void { this.sendOr({ cmd: "train_fetch", run }); }
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
