// State for environment reconstruction: the depth model's status, local datasets, and the newest point cloud per
// camera. Messages come from src/phi_studio/recon_api.py. No three.js here; scene/pointcloud.ts draws the clouds.
import { useRef, useSyncExternalStore } from "react";
import { basePositions, cameraPose, followerSlots, scene, type CamPose } from "./scene";
import { studio } from "./studio";

export interface FitNumbers { s: number | null; t: number | null; inlier_fraction: number; median_mm: number | null; plane_pixels: number; depth_ratio: number | null }
export interface Estimate { fovy_deg: number; fov: string; placement: string; lens: string }
export interface Source { kind: "dataset" | "live"; name?: string; episode?: number; frame?: number; key: string }
export interface CloudInfo {
  kind: "cloud"; camera: string; id: string; url: string; n: number; n_off_arm: number; bytes: number;
  arm_hidden_by: string | null; fit: FitNumbers; source: Source; estimate: Estimate; ms: Record<string, number>; device: string | null; at: number;
}
export interface Refusal { kind: "refused"; camera: string; message: string; fit?: FitNumbers | null; estimate?: Estimate; model_missing?: boolean; at: number }
export interface Download { state: "idle" | "running" | "done" | "cancelled" | "failed"; done: number; total: number; error: string | null }
export interface Status {
  model: { cached: boolean; repo: string; revision: string; license: string; bytes: number; device: string | null };
  download: Download; running: string | null; pending: string[]; keep: string | null; live: string[];
  limits: { max_points: number; keep_hz: number; workspace_radius_m: number; min_inlier_fraction: number };
}
export interface DatasetInfo { name: string; root: string; episodes: number; frames: number; task: string; cameras: string[]; physical: Record<string, string>; note: string | null }

export interface ReconSettings {
  source: "dataset" | "live";
  root: string | null; episode: number; frame: number; key: string | null; // dataset frame
  liveKey: string | null;
  visible: Record<string, boolean>; // per camera; missing means shown
  size: number; // point size, millimetres
  opacity: number;
  hideArm: boolean;
}

export interface ReconState {
  status: Status | null;
  datasets: DatasetInfo[] | null;
  datasetsError: string | null;
  clouds: Record<string, CloudInfo>;
  refused: Record<string, Refusal>; // the newest refusal per camera, cleared by its next cloud
  asked: Record<string, number>; // camera -> when a capture was asked, until it answers
  settings: ReconSettings;
}

const KEY = "phi-studio-recon-v1";
const DEFAULTS: ReconSettings = { source: "dataset", root: null, episode: 0, frame: 0, key: null, liveKey: null, visible: {}, size: 4, opacity: 1, hideArm: true };
export const SIZE = { min: 1, max: 12 }; // millimetres

function readSettings(): ReconSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const s = { ...DEFAULTS, ...(JSON.parse(raw) as Partial<ReconSettings>) };
    s.size = Math.min(SIZE.max, Math.max(SIZE.min, Number(s.size) || DEFAULTS.size));
    s.opacity = Math.min(1, Math.max(0.1, Number(s.opacity) || 1));
    return s;
  } catch {
    return DEFAULTS; // private window or a corrupt value
  }
}

type Listener = () => void;

class ReconStore {
  snap: ReconState = { status: null, datasets: null, datasetsError: null, clouds: {}, refused: {}, asked: {}, settings: readSettings() };
  private listeners = new Set<Listener>();
  private started = false;

  subscribe = (fn: Listener): (() => void) => { this.listeners.add(fn); return () => this.listeners.delete(fn); };

  private set(p: Partial<ReconState>): void {
    this.snap = { ...this.snap, ...p };
    for (const fn of this.listeners) fn();
  }

  /** Listen for the server's recon messages. Idempotent; the page calls it on mount. */
  start(): void {
    if (this.started) return;
    this.started = true;
    studio.onMessage("recon", (m) => this.onMessage(m));
    studio.onMessage("hello", () => { this.set({ asked: {} }); this.refresh(); }); // a reconnect: what was asked will not answer
  }

  refresh(): void { studio.send({ cmd: "recon_status" }); }
  loadDatasets(): void { studio.send({ cmd: "recon_datasets" }); }

  private onMessage(m: any): void {
    const asked = { ...this.snap.asked };
    switch (m.kind) {
      case "status": this.set({ status: m as Status }); break;
      case "datasets": this.set({ datasets: m.datasets, datasetsError: m.error ?? null }); break;
      case "cloud": {
        delete asked[m.camera];
        const refused = { ...this.snap.refused };
        delete refused[m.camera];
        this.set({ clouds: { ...this.snap.clouds, [m.camera]: m as CloudInfo }, refused, asked });
        break;
      }
      case "refused":
        delete asked[m.camera];
        this.set({ refused: { ...this.snap.refused, [m.camera]: { ...m, at: Date.now() } }, asked });
        if (m.model_missing) this.refresh();
        break;
    }
  }

  update(p: Partial<ReconSettings>): void {
    const settings = { ...this.snap.settings, ...p };
    try { localStorage.setItem(KEY, JSON.stringify(settings)); } catch { /* private window: this page only */ }
    this.set({ settings });
  }

  setVisible(camera: string, on: boolean): void { this.update({ visible: { ...this.snap.settings.visible, [camera]: on } }); }

  /** Drop one camera's cloud from this window. */
  clear(camera: string): void {
    const clouds = { ...this.snap.clouds };
    const refused = { ...this.snap.refused };
    delete clouds[camera];
    delete refused[camera];
    this.set({ clouds, refused });
  }

  /** What the server needs besides the picture: where each camera stands in the scene and where the arms are. */
  private context(): Record<string, unknown> {
    const model = scene.snap.model;
    const settings = scene.snap.settings;
    const poses: Record<string, CamPose & { placed: boolean }> = {};
    if (model) {
      const keys = new Set([...Object.keys(model.camera_defaults), ...Object.keys(settings.cameras)]);
      for (const k of keys) poses[k] = { ...cameraPose(model, settings, k), placed: k in settings.cameras };
    }
    const slots = followerSlots(studio.snap.telemetry, studio.snap.rig?.arms);
    const bases = basePositions(slots, settings.spacing ?? model?.pair_spacing_m ?? 0.4);
    return {
      poses,
      arms: slots.map((s, i) => ({ name: s.name, base: bases[i] })),
      wrist_arm: settings.wristArm ?? slots[0]?.name ?? null,
    };
  }

  /** The scene camera a dataset key shows: its physical name when the dataset says, else the key's last part. */
  sceneCamera(key: string, ds?: DatasetInfo): string { return ds?.physical[key] ?? key.replace(/^observation\.images\./, ""); }

  /** Ask for one cloud. False when this window is offline. */
  capture(camera: string, request: Record<string, unknown>): boolean {
    if (!studio.send({ cmd: "recon_capture", camera, ...request, ...this.context() })) {
      studio.localError("Capture cannot run while this window is offline", "Wait for it to reconnect.");
      return false;
    }
    this.set({ asked: { ...this.snap.asked, [camera]: Date.now() } });
    return true;
  }

  captureDataset(key: string, ds: DatasetInfo): boolean {
    const s = this.snap.settings;
    return this.capture(this.sceneCamera(key, ds), { source: "dataset", root: ds.root, episode: s.episode, frame: s.frame, key });
  }

  captureLive(key: string): boolean { return this.capture(key, { source: "live", key }); }

  keep(key: string | null): void {
    if (key) studio.send({ cmd: "recon_keep", on: true, source: "live", key, camera: key, ...this.context() });
    else studio.send({ cmd: "recon_keep", on: false });
  }

  download(): void { studio.send({ cmd: "recon_download" }); }
  cancelDownload(): void { studio.send({ cmd: "recon_download_cancel" }); }
}

export const reconStore = new ReconStore();

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Same caching as useScene: a selector that builds a fresh object must not re-render forever. */
export function useRecon<T>(select: (s: ReconState) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(reconStore.subscribe, () => {
    const value = select(reconStore.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}

/** "Scale from the table: 3.1 mm median error, 64% of table pixels agree". */
export function fitText(f: FitNumbers): string {
  const agree = `${Math.round(f.inlier_fraction * 100)}% of table pixels agree`;
  return f.median_mm === null ? `Scale from the table: ${agree}` : `Scale from the table: ${f.median_mm.toFixed(1)} mm median error, ${agree}`;
}
