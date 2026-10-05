// The 3D view's pure logic: types, saved-settings parsing, joint units, arm pairing and camera mounts.
// WHY import-free: web/tests run this file under Node's own test runner (npm test), which strips types
// but resolves no bundler imports, and Teleoperate loads it eagerly, so it must not pull in three.js.

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

export const DEFAULTS: Settings = {
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

export const SPACING = { min: 0.15, max: 1.2 }; // metres

export const FOVY = { min: 10, max: 150 }; // degrees: a sane vertical field of view for a placed camera
const VIEWS: ViewMode[] = ["both", "3d", "cameras"];

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);
const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const isVec3 = (x: unknown): x is Vec3 => Array.isArray(x) && x.length === 3 && x.every(isNum);

/** A stored camera pose, or null when any field is unusable: the engine reads every one of them. */
export function validPose(x: unknown): CamPose | null {
  if (!isObj(x) || !isVec3(x.pos) || !isVec3(x.target) || !isVec3(x.up) || !isNum(x.fovy_deg)) return null;
  const { pos, target, up, fovy_deg } = x;
  if (fovy_deg < FOVY.min || fovy_deg > FOVY.max) return null;
  if (Math.hypot(up[0], up[1], up[2]) < 1e-9) return null; // no up direction
  if (Math.hypot(target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]) < 1e-6) return null; // looks at itself
  return { pos: [...pos], target: [...target], up: [...up], fovy_deg };
}

/** Saved settings (localStorage text) to Settings. Every field is checked on its own and falls back to its default
 * when it is the wrong type or out of range; text that is not a JSON object loads all the defaults.
 * WHY so strict: the engine reads these on every load of Teleoperate, Run policy and the 3D view, so one bad value
 * saved by an older build or edited by hand would otherwise break those pages each time. */
export function parseSettings(raw: string | null): Settings {
  const out: Settings = { ...DEFAULTS, cameras: {}, view: {} };
  let s: unknown;
  try { s = raw ? JSON.parse(raw) : null; } catch { return out; }
  if (!isObj(s)) return out;
  if (isNum(s.spacing) && s.spacing >= SPACING.min && s.spacing <= SPACING.max) out.spacing = s.spacing;
  for (const k of ["ghost", "trail", "frustums", "video", "autoRotate"] as const) if (typeof s[k] === "boolean") out[k] = s[k];
  if (typeof s.print === "string" && Object.hasOwn(PRINT, s.print)) out.print = s.print as PrintColour;
  if (typeof s.wristArm === "string") out.wristArm = s.wristArm;
  if (isObj(s.cameras)) {
    for (const [k, v] of Object.entries(s.cameras)) {
      const p = validPose(v);
      if (p) out.cameras[k] = p;
    }
  }
  if (isObj(s.view)) {
    for (const [k, v] of Object.entries(s.view)) if (VIEWS.includes(v as ViewMode)) out.view[k] = v as ViewMode;
  }
  return out;
}

/** The camera pose in use: the user's, else the model's starting estimate, else one in front of the arms. */
export function cameraPose(model: SceneModel, settings: Settings, key: string): CamPose {
  return settings.cameras[key] ?? model.camera_defaults[key] ?? { pos: [0.9, -0.4, 0.5], target: [0.15, 0, 0.08], up: [0, 0, 1], fovy_deg: 45 };
}

/** LeRobot reading to MJCF radians, with the one map the server serves (robot_model.joint_maps). */
export function toRadians(model: SceneModel, joint: string, reading: number): number {
  const m = model.mapping[joint];
  return m ? m.scale * reading + m.offset : 0;
}

/** "outside": past the model's range on a joint where that says nothing about the real arm (shown, not tinted). */
export type Limit = "ok" | "near" | "past" | "outside";
export const NEAR_LIMIT = 0.05; // within 5% of the range from either end

/** Where an angle sits in its joint's model range: near an end, past it, or neither.
 * WHY the two exceptions: the gripper's map is a placeholder (phi group/sim/sim_gate.py:20-23 skips its limit
 * too), so a limit there would be invented; wrist_roll's zero is wherever the wrist sat when calibration was
 * confirmed (phi group/sim/SIM_SETUP.md:37), so its model range is not the arm's range. Neither is tinted. */
export function limitState(model: SceneModel, joint: string, rad: number): { frac: number; state: Limit } {
  const r = model.joints[joint]?.range;
  if (!r) return { frac: 0.5, state: "ok" };
  const frac = (rad - r[0]) / (r[1] - r[0]);
  const past = frac < 0 || frac > 1;
  if (joint === "gripper") return { frac, state: "ok" };
  if (joint === "wrist_roll") return { frac, state: past ? "outside" : "ok" };
  return { frac, state: past ? "past" : frac < NEAR_LIMIT || frac > 1 - NEAR_LIMIT ? "near" : "ok" };
}

export interface ArmSlot { name: string; side: "left" | "right" | null; ghost: string | null }
/** The part of Studio's telemetry this file reads (src/lib/studio.ts Telemetry). */
export interface ArmsReading { arms: Record<string, { role: string }> }

/** The followers to draw, each with the leader that drives it (paired by side, else by name, as the worker does). */
export function followerSlots(t: ArmsReading | null, rigArms: { name: string; role: string; side: string | null }[] | undefined): ArmSlot[] {
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

/** The cameras drawn as frusta: the rig's cameras when it has reported any, else the three a phi rig has. */
export function frustumKeys(have: string[]): string[] {
  return [...new Set(["front", "wrist", "top", ...have])].filter((k) => !have.length || have.includes(k));
}

/** Whether a camera rides on a follower's wrist mount, and which follower (index into `slots`, or null: none fits).
 * `wristKey` ("wrist") goes on the follower the person picked, else the first. LeRobot 0.6.0's bimanual robot names
 * each arm's cameras `left_<key>` and `right_<key>` (bi_so_follower.py:91-98), so `<side>_wrist` goes on the
 * follower of that side (or whose name starts with `<side>_`). */
export function wristMount(key: string, wristKey: string, slots: ArmSlot[], wristArm: string | null): { wrist: boolean; arm: number | null } {
  if (key === wristKey) {
    if (!slots.length) return { wrist: true, arm: null };
    const i = slots.findIndex((s) => s.name === wristArm);
    return { wrist: true, arm: i >= 0 ? i : 0 };
  }
  const m = key.match(/^(.+)_wrist$/);
  if (!m) return { wrist: false, arm: null };
  const i = slots.findIndex((s) => s.side === m[1] || s.name.startsWith(`${m[1]}_`));
  return { wrist: true, arm: i >= 0 ? i : null };
}

/** One body joint's calibration as LeRobot saves it (calibration JSON: range_min, range_max in ticks). */
export interface JointCal { range_min: number; range_max: number; drive_mode: number }
/** What an arm's readings mean: degrees, or LeRobot's -100..100 mode, which needs the calibrated ranges. */
export interface ArmUnits { unit: "degrees" | "m100"; calibration: Record<string, JointCal> | null; problem: string | null }
export const DEGREES: ArmUnits = { unit: "degrees", calibration: null, problem: null };

const TICKS = 4095; // LeRobot divides by max_res = 4095 (motors_bus.py:872), as robot_model.m100_to_degrees does

/** A body-joint reading in LeRobot's -100..100 mode as the degrees DEGREES mode would read. The same formula as
 * src/phi_studio/robot_model.py m100_to_degrees: norm = (raw - min) / (max - min) * 200 - 100, negated when
 * drive_mode is set (motors_bus.py:864-866), and DEGREES mode reads (raw - mid) * 360 / 4095. */
export function m100ToDegrees(value: number, c: JointCal): number {
  const norm = c.drive_mode ? -value : value;
  return (norm / 200) * (c.range_max - c.range_min) * 360 / TICKS;
}

const usableCal = (c: JointCal | undefined): c is JointCal =>
  !!c && isNum(c.range_min) && isNum(c.range_max) && c.range_max > c.range_min && isNum(c.drive_mode);

/** A reading in joint_order, MJCF radians, into `out`. "incomplete": a joint is missing; "unusable": the units
 * cannot be turned into an angle (units.problem says why, or a body joint has no calibrated range). The gripper
 * reads 0..100 in both of LeRobot's modes. */
export function readArm(model: SceneModel, pos: Record<string, number> | undefined, units: ArmUnits, out: Float64Array): "ok" | "incomplete" | "unusable" {
  if (units.problem) return "unusable";
  if (!pos) return "incomplete";
  const order = model.joint_order;
  for (let i = 0; i < order.length; i++) {
    let v = pos[order[i]];
    if (typeof v !== "number" || !Number.isFinite(v)) return "incomplete";
    if (units.unit === "m100" && model.mapping[order[i]]?.unit === "degrees") {
      const c = units.calibration?.[order[i]];
      if (!usableCal(c)) return "unusable";
      v = m100ToDegrees(v, c);
    }
    out[i] = toRadians(model, order[i], v);
  }
  return "ok";
}
