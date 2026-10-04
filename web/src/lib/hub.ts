import { useRef, useSyncExternalStore } from "react";
import { studio } from "./studio";

// The Models page's link to the server's hub commands (src/phi_studio/hub_api.py). Its own store, fed by
// studio.onMessage, so Hub replies never re-render the rest of Studio.

export interface Who { user: string | null; orgs: string[]; error: string | null; checked_at: number; login_cmd: string }
export interface SearchRow {
  repo_id: string; author: string; downloads: number | null; likes: number | null; last_modified: string | null;
  tags: string[]; policy_type: string | null; gated: false | "auto" | "manual" | null; private: boolean | null;
}
export interface Feature { type: string; shape: number[] }
export interface ModelInfo {
  repo_id: string; revision: string; policy_type: string;
  input_features: Record<string, Feature>; output_features: Record<string, Feature>;
  cameras: Record<string, number[]>; state_shape: number[] | null; action_shape: number[] | null;
  files: { path: string; size: number; lfs: boolean }[]; total_size: number; has_weights: boolean;
  last_modified: string | null; private: boolean | null; gated: false | "auto" | "manual" | null;
  card: { datasets: string[]; license: string | null; base_model: string[] };
}
export interface RigCamera { key: string; name: string; size: string | null; usable: boolean }
export interface RigView { error: string | null; cameras: RigCamera[]; joints: number; bimanual?: boolean }
export interface Fit { problems: string[]; warnings: string[]; checked?: boolean }
export interface Detail extends Partial<Fit> {
  asked: string; error: string | null; info?: ModelInfo; rig?: RigView; rename_map?: Record<string, string>; local?: string | null;
}
export interface LocalModel extends Fit {
  repo_id: string; revision: string; path: string; policy_type: string; has_weights: boolean; modified: number; size: number;
  cameras: Record<string, number[]>; state_shape?: number[] | null; action_shape?: number[] | null; error: string | null;
  rename_map: Record<string, string>;
}
export interface Progress {
  repo_id: string; revision: string | null; state: "running" | "cancelling" | "done" | "error" | "cancelled";
  done: number; total: number | null; message: string | null; path: string | null;
}
export interface Rollout extends Fit { seq: number; cmd: string | null; error: string | null }
export interface RolloutAsk {
  repo_id: string; revision: string; task: string; duration_s: number; strategy: string; episodes: number;
  dataset_repo_id: string | null; rename_map: Record<string, string>; upload: boolean;
}

export interface HubSnap {
  who: Who | null;
  whoLoading: boolean;
  search: { query: string; results: SearchRow[]; error: string | null } | null;
  searching: string | null; // the query in flight
  detail: Detail | null;
  inspecting: string | null; // the id in flight
  local: LocalModel[] | null;
  rig: RigView | null;
  download: Progress | null;
  rollout: Rollout | null;
}

type Listener = () => void;

class HubStore {
  snap: HubSnap = {
    who: null, whoLoading: false, search: null, searching: null, detail: null, inspecting: null,
    local: null, rig: null, download: null, rollout: null,
  };
  private listeners = new Set<Listener>();
  private active = 0; // mounted Models pages
  private rolloutSeq = 0;
  query = ""; // the newest search words, asked again after a reconnect if no answer came

  constructor() {
    studio.onMessage("hub_whoami", (m) => this.set({ who: m, whoLoading: false }));
    studio.onMessage("hub_search", (m) => {
      // WHY compare: a slow answer to an older query must not replace the newest one.
      if (m.query !== this.snap.searching && this.snap.searching !== null) return;
      this.set({ search: { query: m.query, results: m.results ?? [], error: m.error }, searching: null });
    });
    studio.onMessage("hub_model", (m) => {
      if (m.asked !== this.snap.inspecting) return;
      this.set({ detail: m, inspecting: null });
    });
    studio.onMessage("hub_local", (m) => this.set({ local: m.models, rig: m.rig, download: m.download ?? this.snap.download }));
    studio.onMessage("hub_progress", (m) => this.set({ download: m }));
    studio.onMessage("hub_rollout", (m) => { if (m.seq === this.rolloutSeq) this.set({ rollout: m }); });
    // A command that crashed on the server answers with an error naming it: clear what waits for it.
    studio.onMessage("error", (m) => {
      if (m.cmd === "hub_search") this.set({ searching: null });
      else if (m.cmd === "hub_inspect") this.set({ inspecting: null });
      else if (m.cmd === "hub_whoami") this.set({ whoLoading: false });
    });
    // A new link (a restarted Studio, a reconnect): replies in flight are gone, so ask again.
    studio.onMessage("hello", () => {
      this.set({ searching: null, inspecting: null, whoLoading: false });
      if (this.active) window.setTimeout(() => this.refresh(), 0);
    });
  }

  subscribe = (fn: Listener): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  private set(p: Partial<HubSnap>): void {
    this.snap = { ...this.snap, ...p };
    for (const fn of this.listeners) fn();
  }

  private send(msg: Record<string, unknown>, what: string): boolean {
    if (studio.send(msg)) return true;
    studio.localError(`${what} needs a link to Studio, and this window is offline`, "Wait for it to reconnect.");
    return false;
  }

  /** The page mounted: who is logged in (cached on the server) and what is on this Mac. */
  open(): () => void {
    this.active++;
    this.refresh();
    return () => { this.active--; };
  }

  // WHY quiet: the page opens before the link does on a fresh load; "hello" asks again, so no error.
  private refresh(): void {
    if (studio.send({ cmd: "hub_whoami" })) this.set({ whoLoading: true });
    studio.send({ cmd: "hub_local" });
    if (!this.snap.search && studio.send({ cmd: "hub_search", query: this.query })) this.set({ searching: this.query });
  }

  checkLogin(): void { if (this.send({ cmd: "hub_whoami", refresh: true }, "Checking the login")) this.set({ whoLoading: true }); }
  loadLocal(): void { this.send({ cmd: "hub_local" }, "Listing models"); }

  search(query: string): void {
    this.query = query;
    if (this.send({ cmd: "hub_search", query }, "Search")) this.set({ searching: query });
  }

  inspect(repoId: string, quiet = false): void {
    const msg = { cmd: "hub_inspect", repo_id: repoId };
    if (quiet ? studio.send(msg) : this.send(msg, "Opening a model")) this.set({ inspecting: repoId, detail: null });
  }

  closeDetail(): void { this.set({ detail: null, inspecting: null }); }

  download(repoId: string, revision: string): void { this.send({ cmd: "hub_download", repo_id: repoId, revision }, "Download"); }
  cancel(): void { this.send({ cmd: "hub_cancel" }, "Cancel"); }

  /** Ask the server for the lerobot-rollout command. Only the newest answer is kept. */
  buildRollout(ask: RolloutAsk): void {
    const seq = ++this.rolloutSeq;
    if (!studio.send({ cmd: "hub_rollout", seq, ...ask })) this.set({ rollout: null });
  }
  clearRollout(): void { this.rolloutSeq++; this.set({ rollout: null }); }
}

export const hub = new HubStore();

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Same caching as useStudio and useTerminal: a selector returning a fresh object must not re-render
 * forever (React error #185). */
export function useHub<T>(select: (s: HubSnap) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(hub.subscribe, () => {
    const value = select(hub.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}

const NOT_MODELS = new Set(["datasets", "spaces", "docs", "models", "settings", "join", "login", "blog", "papers", "collections", "organizations"]);

/** owner/name from a pasted Hub URL or a typed id; null for search words. Matches the server's own parse
 * (hub.py _repo_id), which has the final say. */
export function asRepoId(text: string): string | null {
  const t = text.trim();
  const url = t.match(/^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/([^/\s?#]+)\/([^/\s?#]+)/i);
  if (url) return NOT_MODELS.has(url[1].toLowerCase()) ? null : `${url[1]}/${url[2]}`;
  return /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(t) ? t : null;
}

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "unknown";
  if (n < 1e3) return `${n} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let v = n / 1e3, i = 0;
  while (v >= 1e3 && i < units.length - 1) { v /= 1e3; i++; }
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function fmtCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return "-";
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n);
}

export function fmtDate(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "unknown";
  const d = typeof v === "number" ? new Date(v * 1000) : new Date(v);
  return Number.isNaN(d.getTime()) ? "unknown" : d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}

/** "640x480" from a channels x height x width image shape. */
export function imageSize(shape: number[]): string {
  return shape.length === 3 ? `${shape[2]}x${shape[1]}` : shape.join("x");
}
