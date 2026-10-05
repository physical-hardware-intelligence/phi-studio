// The SO-101's forward kinematics as the 3D view draws it, in plain numbers. Import-free so web/tests can check it
// against MuJoCo under Node (npm test). engine.ts builds its three.js nodes from the same two pieces, mjQuat and
// hinge, and lets three compose them; forwardTool composes them here the same way: parent * body * hinge.
import type { Quat, SceneModel, Vec3 } from "../lib/sceneCore";

/** A quaternion in three.js order (x, y, z, w). */
export type XYZW = [number, number, number, number];

/** MuJoCo stores (w, x, y, z); three.js wants (x, y, z, w). */
export const mjQuat = (q: Quat): XYZW => [q[1], q[2], q[3], q[0]];

function mul(a: XYZW, b: XYZW): XYZW {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/** v rotated by the unit quaternion q. */
export function rotate(q: XYZW, v: Vec3): Vec3 {
  const [x, y, z, w] = q;
  // t = 2 (q.xyz x v); v' = v + w t + q.xyz x t
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

/** A hinge's own transform: a turn of `angle` about `axis` through `pivot` (both in the body frame). As a rotation
 * about the origin plus a shift: q = axis-angle, p = pivot - q * pivot, so the pivot stays where it is. */
export function hinge(axis: Vec3, pivot: Vec3, angle: number): { q: XYZW; p: Vec3 } {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1, s = Math.sin(angle / 2) / n;
  const q: XYZW = [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
  const r = rotate(q, pivot);
  return { q, p: [pivot[0] - r[0], pivot[1] - r[1], pivot[2] - r[2]] };
}

interface Pose { q: XYZW; p: Vec3 }
const compose = (a: Pose, b: Pose): Pose => {
  const r = rotate(a.q, b.p);
  return { q: mul(a.q, b.q), p: [a.p[0] + r[0], a.p[1] + r[1], a.p[2] + r[2]] };
};

/** World pose of every body's joint frame (where its geoms and children hang) for joint angles `q` (MJCF radians,
 * joint_order), arm base at the origin. */
export function bodyPoses(model: SceneModel, q: ArrayLike<number>): Map<string, Pose> {
  const angle = new Map(model.joint_order.map((j, i) => [j, q[i] ?? 0]));
  const out = new Map<string, Pose>();
  for (const b of model.bodies) {
    let pose: Pose = { q: mjQuat(b.quat), p: [...b.pos] as Vec3 };
    if (b.parent) pose = compose(out.get(b.parent)!, pose);
    for (const j of b.joints) {
      const J = model.joints[j];
      if (J?.type === "hinge") pose = compose(pose, hinge(J.axis, J.pos, angle.get(j) ?? 0));
    }
    out.set(b.name, pose);
  }
  return out;
}

/** World position of the tool site (the gripper frame) for joint angles `q`. */
export function forwardTool(model: SceneModel, q: ArrayLike<number>): Vec3 {
  const site = model.sites[model.tool_site];
  const b = bodyPoses(model, q).get(site.body)!;
  const r = rotate(b.q, site.pos);
  return [b.p[0] + r[0], b.p[1] + r[1], b.p[2] + r[2]];
}
