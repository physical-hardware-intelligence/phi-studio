// Environment reconstruction's pure logic, with no three.js or React, so Node's test runner can load it
// (tests/recon.test.ts): the binary cloud format, and which cloud's arm pose the 3D view draws.
import type { ArmUnits } from "./sceneCore";

/** A dataset cloud's arm pose: the follower that shows it, the frame's readings (as telemetry sends them), the
 * units the server read them in (the same ones its arm boxes used), and a label for the arm's name pill. */
export interface ArmPose { arm: string; pos: Record<string, number>; units?: ArmUnits; label: string }
export interface PoseSource { camera: string; at: number; pose: ArmPose | null }

/** sRGB byte to linear float, the formula three uses (ColorManagement SRGBToLinear): three treats vertex colours
 * as linear and converts back to sRGB on output. */
export const LINEAR = Float32Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c < 0.04045 ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4);
});

const MAGIC = 0x50434c31; // "PCL1", read big-endian

/** Parse a cloud from GET /api/recon/cloud/{id} (src/phi_studio/recon.py pack): "PCL1", uint32 n, uint32 n_off_arm,
 * float32 xyz * n, uint8 rgb * n, little-endian. Points off the arm come first. Throws on a wrong header or size. */
export function parseCloud(buf: ArrayBuffer): { n: number; nOff: number; positions: Float32Array; colors: Float32Array } {
  const v = new DataView(buf);
  if (buf.byteLength < 12 || v.getUint32(0, false) !== MAGIC) throw new Error("not a point cloud");
  const n = v.getUint32(4, true), nOff = v.getUint32(8, true);
  if (buf.byteLength !== 12 + 15 * n || nOff > n) throw new Error("point cloud size does not match its header");
  const positions = new Float32Array(buf, 12, 3 * n); // a view, no copy: byte 12 is a multiple of 4
  const rgb = new Uint8Array(buf, 12 + 12 * n, 3 * n);
  const colors = new Float32Array(3 * n);
  for (let i = 0; i < rgb.length; i++) colors[i] = LINEAR[rgb[i]];
  return { n, nOff, positions, colors };
}

/** The pose the 3D view draws: the newest shown cloud that carries one. Null when none does, so the arm goes back
 * to its live readings (or zero). `visible`: per camera, missing means shown. */
export function posedFrom(clouds: Record<string, PoseSource>, visible: Record<string, boolean>): ArmPose | null {
  let best: PoseSource | null = null;
  for (const c of Object.values(clouds)) {
    if (!c.pose || visible[c.camera] === false) continue;
    if (!best || c.at > best.at) best = c;
  }
  return best?.pose ?? null;
}

export type Reader = (pos: Record<string, number>, units: ArmUnits, out: Float64Array) => "ok" | "incomplete" | "unusable";
/** What the engine draws for a pose: joint angles in MJCF radians, or null when the arm must not be drawn. */
export interface Posed { arm: string; q: Float64Array | null; text: string }

const DEGREES: ArmUnits = { unit: "degrees", calibration: null, problem: null };

/** Read a pose the way live readings are read (`read` is sceneCore.readArm with the model). An arm the view would
 * not draw live, such as one that reads -100..100 without a calibration, is not drawn posed either; the pill says
 * why. */
export function posePlan(p: ArmPose, read: Reader): Posed {
  const units = p.units ?? DEGREES;
  const q = new Float64Array(6);
  const got = read(p.pos, units, q);
  if (got === "ok") return { arm: p.arm, q, text: p.label };
  const why = got === "unusable"
    ? (units.problem ?? "reads -100 to 100 without a usable calibration").replace(/, so its pose is not drawn$/, "")
    : "is missing a joint reading";
  return { arm: p.arm, q: null, text: `${p.label} ${why}, so it is not drawn` };
}
