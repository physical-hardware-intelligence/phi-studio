// Datasets, episodes and notes: HTTP for bulk data (src/phi_studio/data_api.py), the WebSocket for notes.
import { useSyncExternalStore } from "react";
import { studio } from "./studio";

export type Health = "ok" | "info" | "warn" | "error";
export type Severity = "info" | "warn" | "error";

export interface CameraInfo { key: string; name: string; h: number; w: number; codec: string | null }
export interface DatasetSummary {
  id: string; repo_id: string; name: string; path: string; version: string; robot_type: string | null;
  fps: number; episodes: number; frames: number; duration_s: number; tasks: number;
  cameras: CameraInfo[]; arms: { name: string; so101: boolean }[]; bimanual: boolean;
  dims: Record<string, number>; modified: number; size?: number; notes?: number;
  analysis?: { health: HealthItem[]; flag_counts: Record<string, number>; episodes_by_health: Partial<Record<Health, number>> } | null;
}
export interface HealthItem { severity: Severity; kind: string; text: string }
export interface Arm { name: string; joints: string[]; index: (number | null)[]; so101: boolean }
export interface EpisodeMeta { index: number; length: number; duration: number; tasks: string[] }
export interface DatasetDetail {
  summary: DatasetSummary; tasks: string[]; arms: Arm[]; names: string[]; episodes: EpisodeMeta[]; excluded: number[];
}
export interface Flag {
  kind: string; severity: Severity; text: string; t0: number | null; t1: number | null;
  arm: string | null; joint: string | null; value: number | null; key?: string; dismissed?: boolean;
}
export interface EpisodeMetrics {
  duration_s: number; frames: number; idle_frac: number; idle_start_s: number; idle_end_s: number;
  lag_ms: number | null; track_rms: number | null; speed_peak: number; path_m: number; grasps: number; sparc: number | null;
}
export interface JointMetrics {
  range: number; speed_peak: number; lag_ms: number | null; lag_corr: number; err_rms: number;
  track_rms: number; track_max: number; jerk_rms: number;
}
export interface ArmMetrics {
  path_m?: number; tcp_speed_peak?: number; tcp_speed_mean?: number; sparc?: number; grasps: number; closed_s: number; reach_m?: number;
}
export interface GripEvent { t: number; frame: number; kind: "grasp" | "release"; arm: string | null }
export interface EpisodeAnalysis {
  version: number; metrics: EpisodeMetrics; joints: Record<string, JointMetrics>; arms: Record<string, ArmMetrics>;
  events: GripEvent[]; closed: { t0: number; t1: number; arm: string | null; squeeze: number }[];
  idle: [number, number][]; flags: Flag[]; health: Health;
  series?: { vel: number[][]; acc: number[][]; track: number[][]; err: number[][]; tcp: Record<string, number[][]> };
}
export interface VideoSeg { chunk: number; file: number; from: number; to: number }
export interface EpisodePayload {
  dataset: { id: string; name: string; repo_id: string; fps: number; episodes: number };
  names: string[]; arms: Arm[];
  episode: EpisodeMeta & { videos: Record<string, VideoSeg> };
  t: number[]; state: number[][]; action: number[][];
  analysis: EpisodeAnalysis;
  limits: Record<string, [number, number]>; // model joint ranges, LeRobot units
}
export interface DatasetAnalysis {
  version: number; computed_s?: number;
  episodes: (EpisodeMeta & { metrics: EpisodeMetrics; arms: Record<string, ArmMetrics>; flags: Flag[]; health: Health; grasps_t: number[] })[];
  rest: Record<string, number[]>;
  lengths: number[];
  distributions: Record<string, { edges: number[]; state: number[]; action: number[]; q: number[]; min: number; max: number; model: [number, number] }>;
  workspace: Record<string, { points: number[][]; episode: number[]; min: number[]; max: number[] }>;
  tasks: Record<string, number>; flag_counts: Record<string, number>; health: HealthItem[];
}

export type NoteKind = "note" | "issue" | "bad" | "good";
export interface Note {
  id: string; created: number; updated: number; author: string; kind: NoteKind; severity: Severity;
  status: "open" | "resolved"; dataset: string; dataset_name: string; episode: number | null;
  t0: number | null; t1: number | null; arm: string | null; joint: string | null; tags: string[]; text: string;
  source: "manual" | "flag" | "live";
}

// -- HTTP ------------------------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly fix?: string) { super(message); }
}

export async function api<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { headers: { "X-Studio-Token": studio.token() }, signal, cache: "no-store" });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    let fix: string | undefined;
    try { const j = await r.json(); msg = j.error ?? msg; fix = j.fix; } catch { /* not JSON */ }
    throw new ApiError(msg, r.status, fix);
  }
  return r.json() as Promise<T>;
}

/** A URL a <video> can load: it cannot send headers, so the token rides in the query. */
export function videoUrl(ds: string, key: string, seg: VideoSeg): string {
  return `/api/data/${ds}/video/${encodeURIComponent(key)}/${seg.chunk}/${seg.file}.mp4?token=${encodeURIComponent(studio.token())}`;
}

// -- a tiny cache with subscribers, so pages share one fetch per resource --------------------------
type Entry<T> = { value?: T; error?: ApiError | Error; loading: boolean; promise?: Promise<T> };
const cache = new Map<string, Entry<unknown>>();
const cacheListeners = new Set<() => void>();
const emit = () => cacheListeners.forEach((l) => l());

export function load<T>(path: string, force = false): Promise<T> {
  const e = cache.get(path) as Entry<T> | undefined;
  if (e?.promise && !force) return e.promise;
  const entry: Entry<T> = { ...(e ?? {}), loading: true };
  entry.promise = api<T>(path).then(
    (v) => { cache.set(path, { value: v, loading: false, promise: entry.promise }); emit(); return v; },
    (err) => { cache.set(path, { error: err, loading: false }); emit(); throw err; },
  );
  cache.set(path, entry as Entry<unknown>);
  emit();
  entry.promise.catch(() => undefined);
  return entry.promise;
}

export function useResource<T>(path: string | null): { value?: T; error?: Error; loading: boolean; reload: () => void } {
  const snap = useSyncExternalStore(
    (l) => { cacheListeners.add(l); return () => cacheListeners.delete(l); },
    () => (path ? cache.get(path) : undefined),
  ) as Entry<T> | undefined;
  if (path && !snap) queueMicrotask(() => { if (!cache.has(path)) void load<T>(path); });
  return {
    value: snap?.value, error: snap?.error, loading: !!path && (snap?.loading ?? true),
    reload: () => { if (path) void load<T>(path, true); },
  };
}

export function invalidate(prefix: string): void {
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) cache.delete(k);
  emit();
}

// -- notes ------------------------------------------------------------------------------------------
interface NotesState { byKey: Record<string, Note[]> }
let notes: NotesState = { byKey: {} };
const noteListeners = new Set<() => void>();
const setNotes = (n: NotesState) => { notes = n; noteListeners.forEach((l) => l()); };
const keyOf = (dataset?: string | null, episode?: number | null) => `${dataset ?? "*"}/${episode ?? "*"}`;
// WHY a set of wanted lists, apart from the loaded ones: a page can mount before the socket opens, and a
// request sent then goes nowhere. Every wanted list is asked for again when the server says hello.
const wanted = new Set<string>();
let wired = false;

function ask(key: string): void {
  const [ds, ep] = key.split("/");
  studio.send({ cmd: "notes_list", dataset: ds === "*" ? null : ds, episode: ep === "*" ? null : Number(ep) });
}

function wire(): void {
  if (wired) return;
  wired = true;
  studio.onMessage("notes", (m) => setNotes({ byKey: { ...notes.byKey, [keyOf(m.dataset, m.episode)]: m.items } }));
  studio.onMessage("notes_changed", (m) => {
    for (const k of wanted) if (k.startsWith("*/") || k.startsWith(`${m.dataset}/`)) ask(k);
    if (m.dataset) invalidate(`/api/data/${m.dataset}`);
    invalidate("/api/data/datasets");
  });
  studio.onMessage("hello", () => wanted.forEach(ask));
}

export function useNotes(dataset: string | null, episode: number | null = null): Note[] | null {
  wire();
  const key = keyOf(dataset, episode);
  const list = useSyncExternalStore(
    (l) => { noteListeners.add(l); return () => noteListeners.delete(l); },
    () => notes.byKey[key] ?? null,
  );
  if (!wanted.has(key)) {
    wanted.add(key);
    queueMicrotask(() => ask(key));
  }
  return list;
}

function sendOr(msg: Record<string, unknown>, what: string): boolean {
  wire();
  if (studio.send(msg)) return true;
  studio.localError(`${what} did not reach Studio: this window is offline`, "It will work again once the window reconnects.");
  return false;
}
export function saveNote(note: Partial<Note> & { dataset: string }): boolean { return sendOr({ cmd: "note_save", note }, "The note"); }
export function deleteNote(id: string, dataset: string): boolean { return sendOr({ cmd: "note_delete", id, dataset }, "Deleting the note"); }
export function dismissFlag(dataset: string, episode: number, f: Flag, undo = false): boolean {
  return sendOr({ cmd: "flag_dismiss", dataset, episode, kind: f.kind, key: f.key ?? "", undo }, "Hiding the flag");
}

// -- formatting ---------------------------------------------------------------------------------------
export function fmtDuration(s: number): string {
  if (!Number.isFinite(s)) return "–";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  if (m < 60) return `${m}:${String(r).padStart(2, "0")}`;
  return `${Math.floor(m / 60)} h ${m % 60} m`;
}
export function fmtClock(s: number): string {
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${m}:${r.toFixed(2).padStart(5, "0")}`;
}
export function fmtBytes(n: number | undefined): string {
  if (n === undefined) return "–";
  if (n < 1e6) return `${(n / 1e3).toFixed(0)} kB`;
  if (n < 1e9) return `${(n / 1e6).toFixed(0)} MB`;
  return `${(n / 1e9).toFixed(2)} GB`;
}
export function fmtAgo(unix: number): string {
  const d = Date.now() / 1000 - unix;
  if (d < 90) return "just now";
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400 * 2) return `${Math.round(d / 3600)} h ago`;
  if (d < 86400 * 60) return `${Math.round(d / 86400)} d ago`;
  return new Date(unix * 1000).toLocaleDateString();
}
export const jointLabel = (j: string) => j.replace(/^(left|right)_/, "").replace(/_/g, " ");
export const SHORT: Record<string, string> = {
  shoulder_pan: "pan", shoulder_lift: "lift", elbow_flex: "elbow", wrist_flex: "flex", wrist_roll: "roll", gripper: "grip",
};
export const FLAG_LABEL: Record<string, string> = {
  idle_start: "Idle start", idle_end: "Idle end", no_motion: "No motion", short: "Short", frozen: "Frozen joint",
  jump: "Jump", tracking: "Tracking", squeeze: "Squeeze", no_grasp: "No grasp", video: "Video length", outlier: "Outlier",
};
