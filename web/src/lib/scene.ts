// State for the 3D view: the SO-101 model from the server, and this browser's view settings.
// No three.js here: Teleoperate and Run policy import this file eagerly, and three loads only with the viewer.
import { useRef, useSyncExternalStore } from "react";
import type { Telemetry } from "./studio";

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // MuJoCo order: w, x, y, z

/** GET /api/scene/model (src/phi_studio/robot_model.py model_json). Metres, radians, z up, the arm reaches along +x. */
export interface SceneModel {
  model: string;
  source: { repo: string; path: string; commit: string; generator: string; license: string };
  bodies: { name: string; parent: string | null; pos: Vec3; quat: Quat; joints: string[] }[];
  joints: Record<string, { body: string; type: "hinge" | "slide"; axis: Vec3; pos: Vec3; range: [number, number] | null }>;
  geoms: { index: number; body: string; mesh: string; pos: Vec3; quat: Quat; rgba: [number, number, number, number]; material: string | null }[];
  sites: Record<string, { body: string; pos: Vec3; quat: Quat }>;
  meshes: Record<string, { url: string; bytes: number; scale: Vec3 }>;
  joint_order: string[];
  mapping: Record<string, { unit: string; scale: number; offset: number; status: string; source: string }>;
  drives: Record<string, number>; // joint -> index of the servo geom that turns it
  tool_site: string;
  wrist_camera: { key: string; body: string; pos: Vec3; quat: Quat; fovy_deg: number; mount: string; fov: string };
  wrist_camera_bodies: string[];
  camera_defaults: Record<string, CamPose>;
  pair_spacing_m: number;
}

/** A camera the user places: where it is, what it looks at, and its vertical field of view. World frame, metres. */
export interface CamPose { pos: Vec3; target: Vec3; up: Vec3; fovy_deg: number }
export type ViewMode = "both" | "3d" | "cameras";
export type PrintColour = "model" | "white" | "grey" | "black" | "orange" | "red" | "blue";

export interface Settings {
  spacing: number | null; // metres between the two followers; null: the default
  cameras: Record<string, CamPose>; // user-placed cameras that differ from the default
  ghost: boolean; // the leader (or the policy target) as a translucent arm
  trail: boolean;
  frustums: boolean;
  video: boolean; // live video on each frustum's image plane
  autoRotate: boolean;
  print: PrintColour;
  wristArm: string | null; // which follower carries the wrist camera; null: the first
  view: Record<string, ViewMode>; // per page
}

const DEFAULTS: Settings = {
  spacing: null, cameras: {}, ghost: true, trail: true, frustums: true, video: true, autoRotate: false,
  print: "model", wristArm: null, view: {},
};

// Printed-part colours as sRGB, physical material data rather than UI colour. "model" is the MJCF's own material
// colour for every printed part (rgba 1 0.82 0.12, the yellow of TheRobotStudio's renders).
export const PRINT: Record<PrintColour, { name: string; srgb: Vec3 }> = {
  model: { name: "Model yellow (from the MJCF)", srgb: [1, 0.82, 0.12] },
  white: { name: "White PLA", srgb: [0.92, 0.92, 0.9] },
  grey: { name: "Grey PLA", srgb: [0.55, 0.56, 0.58] },
  black: { name: "Black PLA", srgb: [0.09, 0.09, 0.1] },
  orange: { name: "Orange PLA", srgb: [0.96, 0.45, 0.1] },
  red: { name: "Red PLA", srgb: [0.78, 0.12, 0.12] },
  blue: { name: "Blue PLA", srgb: [0.12, 0.33, 0.75] },
};

const KEY = "phi-studio-scene-v1";

function readSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const s = JSON.parse(raw) as Partial<Settings>;
    const out: Settings = { ...DEFAULTS, ...s, cameras: { ...(s.cameras ?? {}) }, view: { ...(s.view ?? {}) } };
    if (!(out.print in PRINT)) out.print = "model";
    if (out.spacing !== null && !(out.spacing >= SPACING.min && out.spacing <= SPACING.max)) out.spacing = null;
    return out;
  } catch {
    return DEFAULTS; // private window or a corrupt value: start from the defaults
  }
}

export const SPACING = { min: 0.15, max: 1.2 }; // metres

export interface SceneState {
  model: SceneModel | null;
  error: string | null;
  loading: boolean;
  settings: Settings;
  /** Which user camera is selected for moving, in the 3D view and the camera form. */
  editing: string | null;
}

type Listener = () => void;

class SceneStore {
  snap: SceneState = { model: null, error: null, loading: false, settings: readSettings(), editing: null };
  private listeners = new Set<Listener>();

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

// -- pure helpers, shared by the viewer and the readouts -------------------------------------------------

/** The camera pose in use: the user's, else the model's starting estimate, else one in front of the arms. */
export function cameraPose(model: SceneModel, settings: Settings, key: string): CamPose {
  return settings.cameras[key] ?? model.camera_defaults[key] ?? { pos: [0.9, -0.4, 0.5], target: [0.15, 0, 0.08], up: [0, 0, 1], fovy_deg: 45 };
}

/** LeRobot reading to MJCF radians, with the one map the server serves (robot_model.joint_maps). Studio's telemetry
 * is degrees for body joints and 0..100 for the gripper (src/phi_studio/rig.py JointHealth), so that is all this
 * takes; the -100..100 mode is converted on the server side when a backend sends it (robot_model.m100_to_degrees). */
export function toRadians(model: SceneModel, joint: string, reading: number): number {
  const m = model.mapping[joint];
  return m ? m.scale * reading + m.offset : 0;
}

export type Limit = "ok" | "near" | "past";
export const NEAR_LIMIT = 0.05; // within 5% of the range from either end

/** Where an angle sits in its joint's model range: near an end, past it, or neither. */
export function limitState(model: SceneModel, joint: string, rad: number): { frac: number; state: Limit } {
  const r = model.joints[joint]?.range;
  if (!r) return { frac: 0.5, state: "ok" };
  const frac = (rad - r[0]) / (r[1] - r[0]);
  return { frac, state: frac < 0 || frac > 1 ? "past" : frac < NEAR_LIMIT || frac > 1 - NEAR_LIMIT ? "near" : "ok" };
}

export interface ArmSlot { name: string; side: "left" | "right" | null; ghost: string | null }

/** The followers to draw, each with the leader that drives it (paired by side, else by name, as the worker does). */
export function followerSlots(t: Telemetry | null, rigArms: { name: string; role: string; side: string | null }[] | undefined): ArmSlot[] {
  const arms = rigArms?.length ? rigArms : Object.entries(t?.arms ?? {}).map(([name, a]) => ({ name, role: a.role, side: null }));
  const leaders = arms.filter((a) => a.role === "leader");
  return arms.filter((a) => a.role === "follower").map((f) => {
    const side = f.side === "left" || f.side === "right" ? f.side : null;
    const lead = side ? leaders.find((l) => l.side === side) : leaders.find((l) => l.name === f.name.replace("follower", "leader"));
    return { name: f.name, side, ghost: lead?.name ?? null };
  });
}

/** Where each follower's base sits, all facing +x: one at the origin; several in a row `spacing` apart, centred on
 * the origin. Left is +y (the operator stands behind the arms, looking along +x), so a left arm goes first. */
export function basePositions(slots: ArmSlot[], spacing: number): Vec3[] {
  const rank = (s: ArmSlot) => (s.side === "left" ? 0 : s.side === "right" ? 2 : 1);
  const order = slots.map((s, i) => ({ s, i })).sort((a, b) => rank(a.s) - rank(b.s) || a.i - b.i);
  const out: Vec3[] = slots.map(() => [0, 0, 0]);
  order.forEach(({ i }, k) => { out[i] = [0, ((slots.length - 1) / 2 - k) * spacing, 0]; });
  return out;
}
