// State for the 3D view: the SO-101 model from the server, each arm's joint units, and this browser's view settings.
// No three.js here: Teleoperate and Run policy import this file eagerly, and three loads only with the viewer.
// The pure logic lives in sceneCore.ts (tested under Node); this file adds the store and the React hook.
import { useRef, useSyncExternalStore } from "react";
import { studio } from "./studio";
import { type ArmUnits, type CamPose, parseSettings, type SceneModel, type Settings, type ViewMode } from "./sceneCore";

export * from "./sceneCore";

const KEY = "phi-studio-scene-v1";

function readSettings(): Settings {
  try {
    return parseSettings(localStorage.getItem(KEY));
  } catch {
    return parseSettings(null); // private window: start from the defaults
  }
}

export interface SceneState {
  model: SceneModel | null;
  error: string | null;
  loading: boolean;
  settings: Settings;
  /** Which user camera is selected for moving, in the 3D view and the camera form. */
  editing: string | null;
  /** Each arm's joint units from robot-config.yaml (scene_api.py scene_arms), by Studio's arm name. Arms not
   * listed read degrees, the unit Studio's rig contract names (src/phi_studio/rig.py JointHealth). */
  units: Record<string, ArmUnits>;
}

type Listener = () => void;

class SceneStore {
  snap: SceneState = { model: null, error: null, loading: false, settings: readSettings(), editing: null, units: {} };
  private listeners = new Set<Listener>();
  private unitsAsked = false;

  subscribe = (fn: Listener): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  private set(p: Partial<SceneState>): void {
    this.snap = { ...this.snap, ...p };
    for (const fn of this.listeners) fn();
  }

  /** Fetches the model once per page load. The JSON is small; the meshes are fetched by the viewer. */
  loadModel(): void {
    if (this.snap.model || this.snap.loading) return;
    this.set({ loading: true, error: null });
    fetch("/api/scene/model", { cache: "no-cache" })
      .then(async (r) => {
        if (!r.ok) throw new Error(`the server answered ${r.status}`);
        this.set({ model: (await r.json()) as SceneModel, loading: false });
      })
      .catch((e: unknown) => this.set({ loading: false, error: `Could not load the arm model: ${e instanceof Error ? e.message : String(e)}` }));
  }

  /** Asks the server for each arm's units, now and whenever robot-config.yaml may have changed. */
  watchUnits(): () => void {
    if (!this.unitsAsked) {
      this.unitsAsked = true;
      studio.onMessage("scene_arms", (m: { arms?: Record<string, ArmUnits> }) => this.set({ units: m.arms ?? {} }));
    }
    let seen: unknown = undefined;
    const ask = () => {
      const s = studio.snap;
      const key = s.link === "open" ? s.files.index?.lerobot ?? null : undefined;
      if (key === seen) return;
      seen = key;
      if (key !== undefined) studio.send({ cmd: "scene_arms" });
    };
    ask();
    return studio.subscribe(ask);
  }

  update(p: Partial<Settings>): void {
    const settings = { ...this.snap.settings, ...p };
    try { localStorage.setItem(KEY, JSON.stringify(settings)); } catch { /* private window: lasts this page only */ }
    this.set({ settings });
  }

  setCamera(key: string, pose: CamPose | null): void {
    const cameras = { ...this.snap.settings.cameras };
    if (pose) cameras[key] = pose; else delete cameras[key];
    this.update({ cameras });
  }

  setView(page: string, mode: ViewMode): void { this.update({ view: { ...this.snap.settings.view, [page]: mode } }); }
  edit(key: string | null): void { this.set({ editing: key }); }
}

export const scene = new SceneStore();

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Same caching as useStudio: a selector that builds a fresh object must not re-render forever (React #185). */
export function useScene<T>(select: (s: SceneState) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(scene.subscribe, () => {
    const value = select(scene.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}
