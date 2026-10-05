// The SO-101 in three.js, built from Studio's model asset (src/phi_studio/assets/so101, made by
// scripts/build_so101_model.py from TheRobotStudio's so101_new_calib MJCF). The same tree as the server's
// forward kinematics (kinematics.py), so the arm on screen and a plotted tool path agree.
// World frame is MuJoCo's: z up, x forward (the way the arm reaches), metres.
import * as THREE from "three";
import { studio } from "./studio";

interface Geom { mesh: string; pos: number[]; quat: number[]; material: string }
interface Site { name: string; pos: number[]; quat: number[] }
interface Body {
  name: string; pos: number[]; quat: number[];
  joint: { name: string; axis: number[]; pos: number[]; range: [number, number] } | null;
  geoms: Geom[]; sites: Site[]; children: Body[];
}
interface ModelJson {
  joints: string[]; gripper_deg: [number, number]; tcp_site: string;
  materials: Record<string, number[]>;
  // v: uint16 positions in the mesh's box; n: int8 normals, split along hard edges at build time; i: triangles
  meshes: Record<string, { v: [number, number]; n: [number, number]; i: [number, number]; i32: boolean; min: number[]; max: number[] }>;
  tree: Body;
}

export interface Model { json: ModelJson; geometries: Map<string, THREE.BufferGeometry>; limits: Record<string, [number, number]> }

let loading: Promise<Model> | null = null;

/** Fetch and decode the model once per page. */
export function loadModel(): Promise<Model> {
  if (loading) return loading;
  loading = (async () => {
    const h = { headers: { "X-Studio-Token": studio.token() } };
    const [jr, br] = await Promise.all([fetch("/api/model/so101/model.json", h), fetch("/api/model/so101/meshes.bin", h)]);
    if (!jr.ok || !br.ok) throw new Error(`The 3D model did not load (${jr.status}/${br.status})`);
    const json = (await jr.json()) as ModelJson;
    const bin = await br.arrayBuffer();
    const geometries = new Map<string, THREE.BufferGeometry>();
    for (const [name, m] of Object.entries(json.meshes)) {
      const q = new Uint16Array(bin, m.v[0], m.v[1] * 3);
      const pos = new Float32Array(q.length);
      for (let k = 0; k < q.length; k++) {
        const a = k % 3;
        pos[k] = m.min[a] + (q[k] / 65535) * (m.max[a] - m.min[a]);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      // WHY normals from the file: computing creased normals here cost seconds of main-thread time per load
      // (scripts/build_so101_model.py does it once). Int8, normalised: -127..127 reads as -1..1.
      g.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(bin, m.n[0], m.n[1] * 3), 3, true));
      g.setIndex(new THREE.BufferAttribute(m.i32 ? new Uint32Array(bin, m.i[0], m.i[1]) : new Uint16Array(bin, m.i[0], m.i[1]), 1));
      g.computeBoundingSphere();
      geometries.set(name, g);
    }
    const limits: Record<string, [number, number]> = {};
    const walk = (b: Body) => {
      if (b.joint) limits[b.joint.name] = b.joint.range;
      b.children.forEach(walk);
    };
    walk(json.tree);
    return { json, geometries, limits };
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

export type ArmStyle = "solid" | "ghost" | "warn";

const PRINTED = new THREE.Color("#d9d9de");
const SERVO = new THREE.Color("#2b2b30");

function material(style: ArmStyle, kind: "printed" | "servo"): THREE.Material {
  if (style === "solid") {
    return new THREE.MeshStandardMaterial({
      color: kind === "printed" ? PRINTED : SERVO, roughness: kind === "printed" ? 0.72 : 0.45, metalness: 0.0,
    });
  }
  const color = style === "ghost" ? "#6b9ef0" : "#d39b3f";
  return new THREE.MeshStandardMaterial({
    color, roughness: 0.6, metalness: 0, transparent: true, opacity: kind === "printed" ? 0.26 : 0.18,
    depthWrite: false, emissive: color, emissiveIntensity: 0.25,
  });
}

const quat = (q: number[]) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // MuJoCo (w,x,y,z)

/** One arm: set its joints in LeRobot units (degrees; gripper 0..100) and it poses itself. */
export class ArmRig {
  readonly root = new THREE.Group();
  private joints = new Map<string, { group: THREE.Object3D; axis: THREE.Vector3 }>();
  private sites = new Map<string, THREE.Object3D>();
  private materials: THREE.Material[] = [];
  private gripperDeg: [number, number];
  readonly order: string[];

  constructor(model: Model, style: ArmStyle = "solid") {
    this.order = model.json.joints;
    this.gripperDeg = model.json.gripper_deg;
    const printed = material(style, "printed");
    const servo = material(style, "servo");
    this.materials.push(printed, servo);
    const build = (b: Body, parent: THREE.Object3D) => {
      const g = new THREE.Group();
      g.name = b.name;
      g.position.set(b.pos[0], b.pos[1], b.pos[2]);
      g.quaternion.copy(quat(b.quat));
      parent.add(g);
      let host: THREE.Object3D = g;
      if (b.joint) {
        // Rotation about an axis through the joint anchor (zero for every SO-101 joint, kept general).
        const anchor = new THREE.Group();
        anchor.position.set(b.joint.pos[0], b.joint.pos[1], b.joint.pos[2]);
        const rot = new THREE.Group();
        rot.name = `joint:${b.joint.name}`;
        const back = new THREE.Group();
        back.position.set(-b.joint.pos[0], -b.joint.pos[1], -b.joint.pos[2]);
        g.add(anchor); anchor.add(rot); rot.add(back);
        host = back;
        this.joints.set(b.joint.name, { group: rot, axis: new THREE.Vector3(...b.joint.axis).normalize() });
      }
      for (const geom of b.geoms) {
        const geometry = model.geometries.get(geom.mesh);
        if (!geometry) continue;
        const rgba = model.json.materials[geom.material] ?? [1, 1, 1, 1];
        const isServo = rgba[0] < 0.3; // the model colours servos near-black and printed parts yellow
        const mesh = new THREE.Mesh(geometry, isServo ? servo : printed);
        mesh.position.set(geom.pos[0], geom.pos[1], geom.pos[2]);
        mesh.quaternion.copy(quat(geom.quat));
        mesh.castShadow = style === "solid";
        mesh.receiveShadow = style === "solid";
        mesh.renderOrder = style === "solid" ? 0 : 1;
        host.add(mesh);
      }
      for (const s of b.sites) {
        const o = new THREE.Object3D();
        o.name = s.name;
        o.position.set(s.pos[0], s.pos[1], s.pos[2]);
        o.quaternion.copy(quat(s.quat));
        host.add(o);
        this.sites.set(s.name, o);
      }
      b.children.forEach((c) => build(c, host));
    };
    build(model.json.tree, this.root);
  }

  /** values: six numbers in LeRobot units, in the model's joint order. Non-finite values are skipped. */
  set(values: ArrayLike<number>): void {
    for (let k = 0; k < this.order.length && k < values.length; k++) {
      const v = values[k];
      if (!Number.isFinite(v)) continue;
      const name = this.order[k];
      const j = this.joints.get(name);
      if (!j) continue;
      const deg = name === "gripper" ? this.gripperDeg[0] + (v / 100) * (this.gripperDeg[1] - this.gripperDeg[0]) : v;
      j.group.quaternion.setFromAxisAngle(j.axis, THREE.MathUtils.degToRad(deg));
    }
  }

  /** World position of a site (e.g. the tool centre point, "gripperframe"). */
  site(name: string, out = new THREE.Vector3()): THREE.Vector3 {
    const s = this.sites.get(name);
    if (!s) return out.set(NaN, NaN, NaN);
    this.root.updateWorldMatrix(true, true);
    return s.getWorldPosition(out);
  }

  dispose(): void {
    this.materials.forEach((m) => m.dispose());
    this.root.removeFromParent();
  }
}
