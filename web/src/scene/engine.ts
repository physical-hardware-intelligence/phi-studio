// The 3D view's renderer: the SO-101 followers drawn from their CAD meshes, moving with telemetry.
// Loaded only with the viewer (dynamic import), so pages without a 3D view never download three.js.
//
// Frames: everything is in the MJCF's own world frame, metres, z up, the arm reaching along +x. The camera's
// up vector is +z (as in MakerWorld's viewer), so no axis swap sits between the model file and the screen.
// Each MJCF body is two nodes: `body` (its fixed pos/quat in the parent) and under it `joint` (the hinge's turn);
// geoms and child bodies hang under `joint`. That is MuJoCo's composition, parent * body * joint, and
// tests/test_robot_model.py checks the same composition against MuJoCo to 1e-6 m.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { HorizontalBlurShader } from "three/addons/shaders/HorizontalBlurShader.js";
import { VerticalBlurShader } from "three/addons/shaders/VerticalBlurShader.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";
import { type Frame, studio, type Telemetry } from "../lib/studio";
import {
  type ArmSlot, type ArmUnits, basePositions, type CamPose, cameraPose, DEGREES, followerSlots, frustumKeys, limitState,
  type Limit, PRINT, readArm, type SceneModel, type Settings, type Vec3, wristMount,
} from "../lib/scene";
import { label } from "../lib/labels";
import { hinge, mjQuat } from "./kinematics";
import { Clock, Track } from "./motion";

export type Preset = "home" | "front" | "side" | "top" | "wrist";
export interface Theme { ground: string; grid: string; ghost: string; trail: string; warn: string; danger: string; frustum: string; selected: string }
export interface EngineOptions {
  compact: boolean;
  labels: HTMLElement; // an overlay the engine fills with name pills, positioned each frame
  onProgress: (loaded: number, total: number) => void;
  onCameraEdit: (key: string, pose: CamPose) => void;
  onSelectCamera: (key: string | null) => void;
  onWristView: (on: boolean) => void;
  onLoadError: (message: string | null) => void; // a mesh did not load; null: loading again
  onContextLost: (lost: boolean) => void; // the browser took the GPU context away, or gave it back
}

const CREASE = THREE.MathUtils.degToRad(35); // smooth across shallow edges, keep CAD edges sharp
const TRAIL_S = 3; // seconds of gripper path drawn
const TRAIL_MAX = 360; // points kept; at most one per 5 ms of motion
const TRAIL_WIDTH = 0.0035; // metres
const STALE_MS = 1000; // a follower with no reading this long is shown as stale
const FOV = 35; // the orbit camera's vertical field of view, degrees: a product-shot lens, little distortion
const VIDEO_W = 320; // the image planes are small, so frames are decoded straight to this width

// Geometry is parsed and creased once per page load and shared by every viewer; each viewer's dispose() frees the
// GPU copies, and three uploads them again for the next viewer. WHY keep the CPU copy: switching between Teleoperate
// and the 3D view otherwise re-parsed 16 MB of STL every time.
const geometryCache = new Map<string, Promise<THREE.BufferGeometry>>();

/** `toMm` turns the file's units into millimetres: the model's mesh scale (file units to metres) times 1000. */
function loadMesh(url: string, toMm: number, onBytes: (n: number) => void): Promise<THREE.BufferGeometry> {
  let p = geometryCache.get(url);
  if (!p) {
    p = fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(`${url.split("/").pop()?.split("?")[0]}, the server answered ${r.status}`);
      const buf = await r.arrayBuffer();
      onBytes(buf.byteLength);
      // WHY in millimetres: toCreasedNormals joins corners on a grid of 1/100 of a unit (BufferGeometryUtils.js,
      // hashMultiplier). Most of these STLs are in metres, where that grid is 1 cm: it fused unrelated corners
      // and took 6.4 s for all meshes, against 54 ms in millimetres (both measured in Node).
      const g = toCreasedNormals(new STLLoader().parse(buf).scale(toMm, toMm, toMm), CREASE).scale(1 / toMm, 1 / toMm, 1 / toMm);
      g.computeBoundingBox();
      g.computeBoundingSphere();
      return g;
    });
    p.catch(() => geometryCache.delete(url)); // a failed fetch may be retried by the next viewer
    geometryCache.set(url, p);
  } else {
    void p.then((g) => onBytes(g.getAttribute("position").count * 50 / 3 + 84)); // the STL's size, for the progress bar
  }
  return p;
}

const v3 = (a: Vec3) => new THREE.Vector3(a[0], a[1], a[2]);
const quat = (q: [number, number, number, number]) => new THREE.Quaternion(...mjQuat(q)); // MuJoCo w,x,y,z
const srgb = (c: Vec3) => new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
/** Drawn: three skips an object when it or any ancestor is hidden. */
const shown = (o: THREE.Object3D): boolean => { for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) return false; return true; };

interface JointNode { node: THREE.Object3D; axis: Vec3; pivot: Vec3 }

/** One arm's node tree. Followers get meshes and per-joint servo materials; the ghost shares one translucent look. */
class ArmView {
  readonly root = new THREE.Group();
  readonly bodies = new Map<string, THREE.Object3D>(); // the joint node of each body: where its children hang
  readonly joints: JointNode[] = []; // in joint_order
  readonly servo = new Map<string, THREE.MeshStandardMaterial>(); // joint -> the material of its driving servo
  readonly meshes: THREE.Mesh[] = [];
  readonly tool = new THREE.Object3D();
  readonly wristCam = new THREE.Object3D();
  readonly wristParts: THREE.Object3D[] = [];
  readonly q = new Float64Array(6);
  limits: Limit[] = ["ok", "ok", "ok", "ok", "ok", "ok"];

  /** `look` gives a geom its material, or several: one mesh per material, drawn in that order (the ghost's depth pass). */
  constructor(model: SceneModel, geoms: Map<string, THREE.BufferGeometry> | null, look: (mesh: string, joint: string | null) => THREE.Material | THREE.Material[]) {
    const drives = new Map(Object.entries(model.drives).map(([j, i]) => [i, j]));
    for (const b of model.bodies) {
      const body = new THREE.Object3D();
      body.name = b.name;
      body.position.copy(v3(b.pos));
      body.quaternion.copy(quat(b.quat));
      const joint = new THREE.Object3D();
      body.add(joint);
      (b.parent ? this.bodies.get(b.parent)! : this.root).add(body);
      this.bodies.set(b.name, joint);
      if (model.wrist_camera_bodies.includes(b.name)) this.wristParts.push(body);
    }
    for (const name of model.joint_order) {
      const j = model.joints[name];
      this.joints.push({ node: this.bodies.get(j.body)!, axis: j.axis, pivot: j.pos });
    }
    if (geoms) {
      for (const g of model.geoms) {
        const geo = geoms.get(g.mesh);
        if (!geo) continue;
        const mats = look(g.mesh, drives.get(g.index) ?? null);
        (Array.isArray(mats) ? mats : [mats]).forEach((mat, pass) => {
          const m = new THREE.Mesh(geo, mat);
          m.position.copy(v3(g.pos));
          m.quaternion.copy(quat(g.quat));
          m.scale.copy(v3(model.meshes[g.mesh].scale));
          m.matrixAutoUpdate = false; // geoms never move inside their body
          m.updateMatrix();
          m.userData.pass = pass;
          this.bodies.get(g.body)!.add(m);
          this.meshes.push(m);
        });
      }
    }
    const site = model.sites[model.tool_site];
    this.tool.position.copy(v3(site.pos));
    this.bodies.get(site.body)!.add(this.tool);
    const wc = model.wrist_camera;
    this.wristCam.position.copy(v3(wc.pos));
    this.wristCam.quaternion.copy(quat(wc.quat));
    this.bodies.get(wc.body)?.add(this.wristCam);
  }

  /** Joint angles in MJCF radians, joint_order. A hinge turns about its axis through its pivot (kinematics.ts,
   * which web/tests/scene.test.ts checks against MuJoCo). */
  setPose(q: ArrayLike<number>): void {
    for (let i = 0; i < this.joints.length; i++) {
      this.q[i] = q[i];
      const j = this.joints[i];
      const h = hinge(j.axis, j.pivot, q[i]);
      j.node.quaternion.set(...h.q);
      j.node.position.set(...h.p);
    }
  }
}

interface Slot { slot: ArmSlot; arm: ArmView; ghost: ArmView | null; track: Track; ghostTrack: Track; seen: boolean; online: boolean; trail: Trail; pill: HTMLElement; problem: string | null }

/** The gripper tip's recent path: a camera-facing ribbon whose alpha fades with age. Fixed buffers, no per-frame allocation. */
class Trail {
  readonly mesh: THREE.Mesh;
  private readonly pts = new Float64Array(TRAIL_MAX * 4); // x, y, z, time (ms)
  private n = 0;
  private head = 0;
  private readonly pos: Float32Array;
  private readonly col: Float32Array;
  private readonly color = new THREE.Color();

  constructor() {
    const g = new THREE.BufferGeometry();
    this.pos = new Float32Array(TRAIL_MAX * 2 * 3);
    this.col = new Float32Array(TRAIL_MAX * 2 * 4);
    const idx = new Uint16Array((TRAIL_MAX - 1) * 6);
    for (let i = 0; i < TRAIL_MAX - 1; i++) idx.set([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2], i * 6);
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("color", new THREE.BufferAttribute(this.col, 4).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    const m = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    this.mesh = new THREE.Mesh(g, m);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 5;
  }

  setColor(c: THREE.Color): void { this.color.copy(c); }

  push(p: THREE.Vector3, t: number): void {
    if (this.n) {
      const last = ((this.head + this.n - 1) % TRAIL_MAX) * 4;
      const dx = p.x - this.pts[last], dy = p.y - this.pts[last + 1], dz = p.z - this.pts[last + 2];
      if (dx * dx + dy * dy + dz * dz < 1e-6) return; // under 1 mm: not worth a point, and a still arm adds none
    }
    const i = ((this.head + this.n) % TRAIL_MAX) * 4;
    if (this.n === TRAIL_MAX) this.head = (this.head + 1) % TRAIL_MAX; else this.n++;
    this.pts[i] = p.x; this.pts[i + 1] = p.y; this.pts[i + 2] = p.z; this.pts[i + 3] = t;
  }

  private drawn = false;

  clear(): void { this.n = 0; this.head = 0; this.mesh.geometry.setDrawRange(0, 0); }

  /** Whether the next frame differs: points are still fading out, or the last drawn ribbon must be cleared. */
  live(): boolean { return this.n > 0 || this.drawn; }

  /** Rebuild the ribbon for this frame. Returns whether anything is drawn. */
  update(now: number, eye: THREE.Vector3): boolean {
    while (this.n && now - this.pts[this.head * 4 + 3] > TRAIL_S * 1000) { this.head = (this.head + 1) % TRAIL_MAX; this.n--; }
    const g = this.mesh.geometry;
    if (this.n < 2) { g.setDrawRange(0, 0); this.drawn = false; return false; }
    const P = (k: number) => ((this.head + k) % TRAIL_MAX) * 4;
    const tx = new THREE.Vector3(), view = new THREE.Vector3(), side = new THREE.Vector3(), c = new THREE.Vector3();
    for (let k = 0; k < this.n; k++) {
      const a = P(Math.max(0, k - 1)), b = P(Math.min(this.n - 1, k + 1)), i = P(k);
      tx.set(this.pts[b] - this.pts[a], this.pts[b + 1] - this.pts[a + 1], this.pts[b + 2] - this.pts[a + 2]);
      c.set(this.pts[i], this.pts[i + 1], this.pts[i + 2]);
      view.subVectors(eye, c);
      side.crossVectors(tx, view);
      const len = side.length();
      if (len > 1e-9) side.multiplyScalar(TRAIL_WIDTH / 2 / len); else side.set(0, 0, 0);
      this.pos.set([c.x + side.x, c.y + side.y, c.z + side.z, c.x - side.x, c.y - side.y, c.z - side.z], k * 6);
      const age = (now - this.pts[i + 3]) / (TRAIL_S * 1000);
      const alpha = 0.85 * Math.pow(Math.max(0, 1 - age), 1.6) * Math.min(1, k / 3);
      this.col.set([this.color.r, this.color.g, this.color.b, alpha, this.color.r, this.color.g, this.color.b, alpha], k * 8);
    }
    (g.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    (g.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
    g.setDrawRange(0, (this.n - 1) * 6);
    this.drawn = true;
    return true;
  }
}

/** A camera drawn as a frustum with its live picture on the image plane. */
class CamView {
  readonly group = new THREE.Group();
  readonly lines: THREE.LineSegments;
  readonly plane: THREE.Mesh;
  readonly pill: HTMLElement;
  readonly canvas = document.createElement("canvas");
  readonly texture: THREE.CanvasTexture;
  private readonly ctx: CanvasRenderingContext2D;
  private busy = false;
  private aspect = 4 / 3;
  fovy = 45;
  depth = 0.12;
  hasFrame = false;
  lastFrame = 0;

  constructor(readonly key: string, readonly wrist: boolean, labels: HTMLElement, onFrame: () => void, private readonly visible: () => boolean) {
    this.canvas.width = VIDEO_W;
    this.canvas.height = Math.round(VIDEO_W / this.aspect);
    this.ctx = this.canvas.getContext("2d")!;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.generateMipmaps = false;
    this.texture.minFilter = THREE.LinearFilter;
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(22 * 3), 3));
    this.lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ transparent: true, opacity: 0.9, toneMapped: false }));
    this.plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: this.texture, side: THREE.DoubleSide, toneMapped: false, transparent: true, opacity: 0.95 }));
    this.plane.userData.camera = key;
    this.group.add(this.lines, this.plane);
    this.pill = document.createElement("div");
    this.pill.className = "scene-pill scene-pill-cam";
    labels.appendChild(this.pill);
    this.shape();
    this.unsub = studio.onFrames(key, (f) => this.frame(f, onFrame));
  }

  private readonly unsub: () => void;

  /** Frustum lines and image plane for the current field of view, depth and picture aspect. */
  shape(): void {
    const hh = this.depth * Math.tan(THREE.MathUtils.degToRad(this.fovy) / 2), hw = hh * this.aspect, d = -this.depth;
    const c = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    const a: number[] = [];
    for (const [x, y] of c) a.push(0, 0, 0, x, y, d); // apex to each corner
    for (let i = 0; i < 4; i++) a.push(c[i][0], c[i][1], d, c[(i + 1) % 4][0], c[(i + 1) % 4][1], d);
    // A small triangle above the top edge marks image up.
    a.push(-hw * 0.3, hh * 1.06, d, 0, hh * 1.3, d, 0, hh * 1.3, d, hw * 0.3, hh * 1.06, d, hw * 0.3, hh * 1.06, d, -hw * 0.3, hh * 1.06, d);
    const attr = this.lines.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.set(a);
    attr.needsUpdate = true;
    this.lines.geometry.computeBoundingSphere();
    this.plane.position.set(0, 0, d);
    this.plane.scale.set(2 * hw, 2 * hh, 1);
  }

  private frame(f: Frame, onFrame: () => void): void {
    this.lastFrame = performance.now();
    if (this.busy || !this.visible()) return; // drop frames while a decode runs, or while nobody can see them
    this.busy = true;
    const aspect = f.w / f.h;
    if (Math.abs(aspect - this.aspect) > 1e-3) {
      this.aspect = aspect;
      this.canvas.height = Math.round(VIDEO_W / aspect);
      this.texture.dispose(); // the GPU texture changes size: let three allocate the new one
      this.shape();
    }
    createImageBitmap(f.jpeg, { resizeWidth: this.canvas.width, resizeHeight: this.canvas.height, resizeQuality: "medium" })
      .then((bmp) => {
        this.ctx.drawImage(bmp, 0, 0);
        bmp.close(); // WHY at once: an ImageBitmap holds decoded pixels until closed or collected
        this.texture.needsUpdate = true;
        this.hasFrame = true;
        onFrame();
      })
      .catch(() => { /* a corrupt frame: skip it */ })
      .finally(() => { this.busy = false; });
  }

  dispose(): void {
    this.unsub();
    this.pill.remove();
    this.texture.dispose();
    this.lines.geometry.dispose();
    (this.lines.material as THREE.Material).dispose();
    this.plane.geometry.dispose();
    (this.plane.material as THREE.Material).dispose();
    this.canvas.width = this.canvas.height = 0;
  }
}

/** A soft dark patch under the arms where they come close to the desk, rendered from below and blurred: the
 * technique of three.js's contact-shadow example. Redrawn only when an arm moved. */
class ContactShadow {
  readonly group = new THREE.Group();
  private rt: THREE.WebGLRenderTarget;
  private rtBlur: THREE.WebGLRenderTarget;
  private readonly cam: THREE.OrthographicCamera;
  private readonly depth: THREE.MeshDepthMaterial;
  private readonly hBlur: THREE.ShaderMaterial;
  private readonly vBlur: THREE.ShaderMaterial;
  private readonly blurPlane: THREE.Mesh;
  private readonly plane: THREE.Mesh;
  static readonly LAYER = 1;

  constructor(w: number, private readonly h0: number, height = 0.35) {
    const h = h0;
    const opts = { type: THREE.HalfFloatType };
    this.rt = new THREE.WebGLRenderTarget(512, 512, opts);
    this.rt.texture.generateMipmaps = false;
    this.rtBlur = new THREE.WebGLRenderTarget(512, 512, opts);
    this.rtBlur.texture.generateMipmaps = false;
    const geo = new THREE.PlaneGeometry(w, h);
    this.plane = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: this.rt.texture, transparent: true, depthWrite: false, opacity: 0.55, toneMapped: false }));
    this.plane.renderOrder = 1;
    this.plane.position.z = 0.0006;
    this.plane.scale.y = -1; // the render from below sees the desk mirrored in y
    this.blurPlane = new THREE.Mesh(geo);
    this.blurPlane.visible = false;
    this.blurPlane.scale.y = -1; // the same mirror, so each blur pass reads and writes the same way up
    this.group.add(this.plane, this.blurPlane);
    // Looks up (+z) from the desk. A camera looks along its own -z, so turn it half a turn about x.
    this.cam = new THREE.OrthographicCamera(-w / 2, w / 2, h / 2, -h / 2, 0, height);
    this.cam.rotation.x = Math.PI;
    this.cam.layers.set(ContactShadow.LAYER);
    this.group.add(this.cam);
    this.depth = new THREE.MeshDepthMaterial();
    this.depth.onBeforeCompile = (shader) => {
      shader.uniforms.darkness = { value: 1.4 };
      shader.fragmentShader = `uniform float darkness;\n${shader.fragmentShader.replace(
        "gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );",
        "gl_FragColor = vec4( vec3( 0.0 ), ( 1.0 - fragCoordZ ) * darkness );",
      )}`;
    };
    this.depth.depthTest = false;
    this.depth.depthWrite = false;
    // Own uniforms: ShaderMaterial keeps the object it is given, and the shader modules are shared.
    this.hBlur = new THREE.ShaderMaterial({ ...HorizontalBlurShader, uniforms: THREE.UniformsUtils.clone(HorizontalBlurShader.uniforms) });
    this.vBlur = new THREE.ShaderMaterial({ ...VerticalBlurShader, uniforms: THREE.UniformsUtils.clone(VerticalBlurShader.uniforms) });
    for (const m of [this.hBlur, this.vBlur]) { m.depthTest = false; m.side = THREE.DoubleSide; } // the camera sees the plane's underside
  }

  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene): void {
    const clearAlpha = renderer.getClearAlpha();
    renderer.setClearAlpha(0);
    scene.overrideMaterial = this.depth;
    renderer.setRenderTarget(this.rt);
    renderer.clear();
    renderer.render(scene, this.cam);
    scene.overrideMaterial = null;
    this.blur(renderer, 2.2);
    this.blur(renderer, 0.9);
    renderer.setRenderTarget(null);
    renderer.setClearAlpha(clearAlpha);
  }

  private blur(renderer: THREE.WebGLRenderer, amount: number): void {
    this.blurPlane.visible = true;
    this.blurPlane.material = this.hBlur;
    this.hBlur.uniforms.tDiffuse.value = this.rt.texture;
    this.hBlur.uniforms.h.value = amount / 256;
    renderer.setRenderTarget(this.rtBlur);
    renderer.render(this.blurPlane, this.cam);
    this.blurPlane.material = this.vBlur;
    this.vBlur.uniforms.tDiffuse.value = this.rtBlur.texture;
    this.vBlur.uniforms.v.value = amount / 256;
    renderer.setRenderTarget(this.rt);
    renderer.render(this.blurPlane, this.cam);
    this.blurPlane.visible = false;
  }

  /** Frees the render targets. Called while the context is lost: once three has made its new context, freeing the
   * old objects makes the browser warn that they belong to another context. */
  release(): void {
    this.rt.dispose();
    this.rtBlur.dispose();
  }

  /** Fresh render targets: after a lost GPU context the old ones hold nothing. */
  renew(): void {
    const opts = { type: THREE.HalfFloatType };
    this.rt = new THREE.WebGLRenderTarget(512, 512, opts);
    this.rt.texture.generateMipmaps = false;
    this.rtBlur = new THREE.WebGLRenderTarget(512, 512, opts);
    this.rtBlur.texture.generateMipmaps = false;
    (this.plane.material as THREE.MeshBasicMaterial).map = this.rt.texture;
  }

  /** The blur plane renders with the camera's layer mask, so it must be on that layer too. */
  init(): void { this.blurPlane.layers.set(ContactShadow.LAYER); }

  /** Centre the patch at (x, y) and make it `h` metres deep in y. WHY not by scaling the group: three r186 builds a
   * camera's view matrix without its world scale (src/cameras/Camera.js:116-126), so the blur passes, which draw a
   * scaled plane through that camera, stretched the shadow 1.29 times per pass (seen in the render target). */
  place(x: number, y: number, h: number): void {
    this.group.position.set(x, y, 0);
    this.cam.top = h / 2;
    this.cam.bottom = -h / 2;
    this.cam.updateProjectionMatrix();
    const k = h / this.h0;
    this.plane.scale.y = -k;
    this.blurPlane.scale.y = -k;
  }

  dispose(): void {
    this.rt.dispose();
    this.rtBlur.dispose();
    this.plane.geometry.dispose();
    (this.plane.material as THREE.Material).dispose();
    this.depth.dispose();
    this.hBlur.dispose();
    this.vBlur.dispose();
  }
}

function gridMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, toneMapped: false,
    uniforms: { uColor: { value: new THREE.Color() }, uMinor: { value: 0.05 }, uMajor: { value: 0.25 }, uRadius: { value: 1.4 }, uOpacity: { value: 1 } },
    vertexShader: /* glsl */ `
      varying vec2 vXY;
      void main() { vec4 w = modelMatrix * vec4(position, 1.0); vXY = w.xy; gl_Position = projectionMatrix * viewMatrix * w; }`,
    // Anti-aliased lines from the distance to the nearest grid line in screen pixels (fwidth), faded with
    // distance from the arms so the desk dissolves into the backdrop instead of ending at an edge.
    fragmentShader: /* glsl */ `
      uniform vec3 uColor; uniform float uMinor; uniform float uMajor; uniform float uRadius; uniform float uOpacity;
      varying vec2 vXY;
      float line(vec2 p, float s) { vec2 c = p / s; vec2 g = abs(fract(c - 0.5) - 0.5) / fwidth(c); return 1.0 - min(min(g.x, g.y), 1.0); }
      void main() {
        float a = max(line(vXY, uMinor) * 0.32, line(vXY, uMajor) * 0.7);
        float fade = 1.0 - smoothstep(uRadius * 0.3, uRadius, length(vXY));
        a *= fade * uOpacity;
        if (a < 0.004) discard;
        gl_FragColor = vec4(uColor, a);
        #include <colorspace_fragment>
      }`,
  });
}

/** The leader drawn as tinted glass: faint where it faces the camera, stronger at its silhouette (a Fresnel rim),
 * so it reads as an outline around the follower instead of a film over it. */
function ghostMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthFunc: THREE.LessEqualDepth, toneMapped: false,
    polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 48,
    uniforms: { uColor: { value: new THREE.Color() }, uBase: { value: 0.07 }, uRim: { value: 0.6 } },
    vertexShader: /* glsl */ `
      varying vec3 vN; varying vec3 vV;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor; uniform float uBase; uniform float uRim;
      varying vec3 vN; varying vec3 vV;
      void main() {
        float f = 1.0 - abs(dot(normalize(vN), normalize(vV)));
        gl_FragColor = vec4(uColor, uBase + uRim * f * f);
        #include <colorspace_fragment>
      }`,
  });
}

function radialAlpha(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "#fff");
  grad.addColorStop(0.32, "#fff");
  grad.addColorStop(1, "#000");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.userData.canvas = c;
  return t;
}

export class Engine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 30);
  private readonly controls: OrbitControls;
  private readonly gizmo: TransformControls;
  private readonly handle = new THREE.Object3D();
  private pmrem: THREE.PMREMGenerator;
  private envTarget: THREE.WebGLRenderTarget;
  private readonly key = new THREE.DirectionalLight(0xffffff, 2.4);
  private readonly fill = new THREE.DirectionalLight(0xffffff, 0.45);
  private readonly ground: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshStandardMaterial>;
  private readonly grid: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  private readonly contact = new ContactShadow(1.8, 1.4);
  private readonly materials = {
    servo: new THREE.MeshStandardMaterial({ color: srgb([0.1, 0.1, 0.1]), roughness: 0.42, metalness: 0 }),
    pla: new THREE.MeshStandardMaterial({ color: srgb(PRINT.model.srgb), roughness: 0.55, metalness: 0 }),
    // WHY the polygon offset: a follower that tracks its leader sits exactly inside the ghost. Pushing the ghost's
    // depth back a little lets the follower win where the two coincide, so the ghost shows only where they differ.
    ghost: ghostMaterial(),
    ghostDepth: new THREE.MeshBasicMaterial({ colorWrite: false, transparent: true, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 48 }),
  };
  private readonly theme: Record<keyof Theme, THREE.Color> = {
    ground: new THREE.Color(), grid: new THREE.Color(), ghost: new THREE.Color(), trail: new THREE.Color(), warn: new THREE.Color(),
    danger: new THREE.Color(), frustum: new THREE.Color(), selected: new THREE.Color(),
  };
  private readonly geoms = new Map<string, THREE.BufferGeometry>();
  private slots: Slot[] = [];
  private slotKey = "";
  private readonly cams = new Map<string, CamView>();
  private settings: Settings;
  private readonly clock = new Clock();
  private lastTelemetry: Telemetry | null = null;
  private readonly unsub: (() => void)[] = [];
  private readonly resize: ResizeObserver;
  private raf = 0;
  private dirty = true;
  private shadowsDirty = true;
  private tween: { from: [THREE.Vector3, THREE.Vector3]; to: [THREE.Vector3, THREE.Vector3]; t0: number; ms: number } | null = null;
  private wristView = false;
  private lastInput = performance.now();
  private lastFrameAt = performance.now();
  private editing: string | null = null;
  private ready = false;
  private labelsAt = 0;
  private framedLive = false;
  private userMoved = false;
  private camKeysSeen = "";
  private disposed = false;
  private lost = false; // the GPU context is gone: draw nothing until the browser restores it
  private units: Record<string, ArmUnits> = {};
  private mounts = new Map<string, Slot>(); // wrist camera key -> the follower it rides on
  private editFrom: Vec3 | null = null; // the selected camera's position when a gizmo press began
  private readonly armMoved = new Set<Slot>(); // followers whose pose changed this frame
  private readonly q = new Float64Array(6);
  private readonly tmp = new THREE.Vector3();
  readonly stats = { frames: 0, renders: 0, frameMs: [] as number[], renderMs: [] as number[], loadMs: 0, meshBytes: 0 };

  constructor(private readonly host: HTMLElement, private readonly model: SceneModel, settings: Settings, private readonly opts: EngineOptions) {
    this.settings = settings;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // MakerWorld caps at 2 too: 3x costs 2.25x the pixels for little
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    // WHY PCFShadowMap: three r186 removed PCFSoftShadowMap (WebGLShadowMap.js:99-104 swaps it for this) and made
    // PCFShadowMap itself soft, sampling a Vogel disk scaled by light.shadow.radius.
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.shadowMap.autoUpdate = false; // redrawn only when an arm moved, as MakerWorld's viewer does
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.className = "scene-canvas";
    host.appendChild(this.renderer.domElement);

    // Image-based light from three's procedural room, made on the GPU: no HDR download.
    [this.pmrem, this.envTarget] = this.environment();
    this.scene.environmentIntensity = 0.65;
    // RoomEnvironment is y-up and this scene is z-up. The renderer samples the environment along
    // transpose(R(environmentRotation)) * direction (WebGLMaterials.js:245), so +90 degrees about x puts the room's
    // ceiling lights overhead.
    this.scene.environmentRotation.set(Math.PI / 2, 0, 0);

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.75, -0.65, 0.5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.minDistance = 0.12;
    this.controls.maxDistance = 4;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(92); // a little below the desk, never under it
    this.controls.target.set(0.12, 0, 0.12);
    this.controls.autoRotateSpeed = 0.75; // about 80 s per turn
    this.controls.addEventListener("start", () => { this.lastInput = performance.now(); this.userMoved = true; this.tween = null; this.controls.autoRotate = false; this.exitWrist(); });
    this.controls.addEventListener("change", () => { this.dirty = true; });

    this.key.position.set(0.55, 0.45, 1.2);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.radius = 4;
    this.key.shadow.bias = -0.0002;
    this.key.shadow.normalBias = 0.0012;
    const sc = this.key.shadow.camera;
    sc.left = -0.75; sc.right = 0.75; sc.top = 0.75; sc.bottom = -0.75; sc.near = 0.2; sc.far = 3;
    this.fill.position.set(-0.7, -0.6, 0.45);
    this.scene.add(this.key, this.key.target, this.fill, this.fill.target);

    const groundMat = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, transparent: true, alphaMap: radialAlpha(), depthWrite: true });
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 3.2), groundMat);
    this.ground.receiveShadow = true;
    this.grid = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 3.2), gridMaterial());
    this.grid.position.z = 0.0003;
    this.grid.renderOrder = 2;
    this.scene.add(this.ground, this.grid, this.contact.group);
    this.contact.init();

    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSize(0.75);
    this.gizmo.addEventListener("dragging-changed", (e) => { this.controls.enabled = !(e as unknown as { value: boolean }).value; });
    this.gizmo.addEventListener("objectChange", () => this.dragCamera());
    this.gizmo.addEventListener("mouseDown", () => { this.editFrom = this.handle.position.toArray() as Vec3; });
    this.gizmo.addEventListener("mouseUp", () => this.commitCamera());
    this.gizmo.addEventListener("change", () => { this.dirty = true; });
    this.scene.add(this.handle, this.gizmo.getHelper());

    this.resize = new ResizeObserver(() => this.fitCanvas());
    this.resize.observe(host);
    this.fitCanvas();
    this.bindPointer();
    this.bindContext();
    this.unsub.push(studio.subscribe(() => this.onStudio()));
    this.raf = requestAnimationFrame(this.loop);
    void this.load();
  }

  // -- loading ------------------------------------------------------------------------------------------
  /** Loads the meshes; a failure goes to onLoadError instead of leaving "Loading" up for good. */
  private async load(): Promise<void> {
    try {
      await this.loadMeshes();
    } catch (e) {
      if (!this.disposed) this.opts.onLoadError(`A part of the arm model did not load (${e instanceof Error ? e.message : String(e)}).`);
    }
  }

  /** Try the meshes again after a failure: the cache dropped the failed ones. */
  retryLoad(): void {
    if (this.ready || this.disposed) return;
    this.opts.onLoadError(null);
    void this.load();
  }

  private async loadMeshes(): Promise<void> {
    const t0 = performance.now();
    const names = [...new Set(this.model.geoms.map((g) => g.mesh))];
    const total = names.reduce((s, n) => s + this.model.meshes[n].bytes, 0);
    let loaded = 0;
    this.opts.onProgress(0, total);
    const got = await Promise.all(names.map((n) => loadMesh(this.model.meshes[n].url, 1000 * this.model.meshes[n].scale[0], (b) => {
      loaded += b;
      if (!this.disposed) this.opts.onProgress(Math.min(loaded, total), total);
    }).then((g) => [n, g] as const)));
    if (this.disposed) return;
    for (const [n, g] of got) this.geoms.set(n, g);
    this.stats.loadMs = performance.now() - t0;
    this.stats.meshBytes = total;
    this.ready = true;
    this.opts.onProgress(total, total);
    this.onStudio();
    this.frame("home", false);
  }

  // -- telemetry ----------------------------------------------------------------------------------------
  private onStudio(): void {
    if (!this.ready) return;
    const s = studio.snap;
    const slots = followerSlots(s.telemetry, s.rig?.arms);
    const key = slots.map((x) => `${x.name}:${x.side}:${x.ghost}`).join("|");
    if (key !== this.slotKey) this.rebuild(slots, key);
    const ck = Object.keys(s.cameras).sort().join("|");
    if (ck !== this.camKeysSeen) { this.camKeysSeen = ck; this.syncCameras(); }
    const t = s.telemetry;
    if (t === this.lastTelemetry) return;
    if (!t) { this.clock.reset(); this.lastTelemetry = null; return; }
    this.lastTelemetry = t;
    const now = performance.now();
    const at = this.clock.add(t.t, now);
    const policy = s.state?.activity === "policy" && t.policy ? t.policy.action : null;
    // WHY no `dirty` here: a reading only changes the picture through the pose the loop draws from it, and the
    // loop compares that pose with the last one drawn. Unchanged readings, which stream at the worker's rate
    // while the arms are still or the rig is disconnected, then cost no frame.
    for (const sl of this.slots) {
      if (this.clock.jumped) { sl.track.clear(); sl.ghostTrack.clear(); }
      const a = t.arms[sl.slot.name];
      sl.online = a?.online ?? false;
      const units = this.units[sl.slot.name] ?? DEGREES;
      const got = readArm(this.model, a?.pos, units, this.q);
      const problem = got === "unusable" ? units.problem ?? "reads -100 to 100 without a usable calibration, so its pose is not drawn" : null;
      if (problem !== sl.problem) { sl.problem = problem; this.applySettings(); this.dirty = true; }
      if (got === "ok") {
        sl.track.push(at, this.q, now);
        if (!sl.seen) { sl.arm.setPose(this.q); this.dirty = this.shadowsDirty = true; } // drawn at once, so the framing below sees it
        sl.seen = true;
      }
      const g = policy?.[sl.slot.name] ?? (sl.slot.ghost ? t.arms[sl.slot.ghost]?.pos : undefined);
      // A policy's target is in the follower's units; a leader's reading in the leader's.
      const gUnits = policy?.[sl.slot.name] ? units : (sl.slot.ghost ? this.units[sl.slot.ghost] : undefined) ?? DEGREES;
      if (readArm(this.model, g, gUnits, this.q) === "ok") sl.ghostTrack.push(at, this.q, now); else sl.ghostTrack.clear();
    }
    // The first reading moves the arms from the zero pose to where they are: frame them there once, unless the
    // person has already moved the view.
    if (!this.framedLive && this.slots.some((x) => x.seen)) {
      this.framedLive = true;
      if (!this.userMoved) this.frame("home", false);
    }
  }

  /** Each arm's joint units (lib/scene.ts SceneState.units). */
  setUnits(units: Record<string, ArmUnits>): void {
    this.units = units;
    for (const s of this.slots) { s.track.clear(); s.ghostTrack.clear(); s.seen = false; }
    this.lastTelemetry = null;
    this.onStudio();
  }

  private rebuild(slots: ArmSlot[], key: string): void {
    for (const s of this.slots) this.dropSlot(s);
    this.slotKey = key;
    this.slots = slots.map((slot) => {
      const servo = new Map<string, THREE.MeshStandardMaterial>();
      const arm = new ArmView(this.model, this.geoms, (mesh, joint) => {
        if (!mesh.startsWith("sts3215") && !mesh.startsWith("wrist_camera_so101")) return this.materials.pla;
        if (!joint) return this.materials.servo;
        const m = this.materials.servo.clone(); // its own, so one joint's warning tints one servo
        servo.set(joint, m);
        return m;
      });
      servo.forEach((m, j) => arm.servo.set(j, m));
      for (const m of arm.meshes) { m.castShadow = true; m.receiveShadow = true; m.layers.enable(ContactShadow.LAYER); }
      let ghost: ArmView | null = null;
      if (slot.ghost) {
        ghost = new ArmView(this.model, this.geoms, () => [this.materials.ghostDepth, this.materials.ghost]);
        // WHY two meshes per geom: the first writes depth only, the second colour with LessEqual, so only the
        // ghost's nearest surface shows, not the parts behind it seen through it.
        for (const m of ghost.meshes) m.renderOrder = 10 + (m.userData.pass as number);
      }
      this.scene.add(arm.root);
      if (ghost) this.scene.add(ghost.root);
      const trail = new Trail();
      trail.setColor(this.theme.trail);
      this.scene.add(trail.mesh);
      const pill = document.createElement("div");
      pill.className = "scene-pill";
      this.opts.labels.appendChild(pill);
      arm.setPose(new Float64Array(6));
      return { slot, arm, ghost, track: new Track(), ghostTrack: new Track(), seen: false, online: true, trail, pill, problem: null };
    });
    this.applySettings();
    this.syncCameras();
    this.shadowsDirty = this.dirty = true;
  }

  private dropSlot(s: Slot): void {
    this.scene.remove(s.arm.root, s.trail.mesh);
    if (s.ghost) this.scene.remove(s.ghost.root);
    s.arm.servo.forEach((m) => m.dispose());
    s.trail.mesh.geometry.dispose();
    (s.trail.mesh.material as THREE.Material).dispose();
    s.pill.remove();
  }

  // -- settings and theme -------------------------------------------------------------------------------
  setSettings(s: Settings): void {
    this.settings = s;
    this.applySettings();
    this.syncCameras();
    this.dirty = this.shadowsDirty = true;
  }

  private applySettings(): void {
    const s = this.settings;
    this.materials.pla.color.copy(srgb(PRINT[s.print].srgb));
    const spacing = s.spacing ?? this.model.pair_spacing_m;
    const bases = basePositions(this.slots.map((x) => x.slot), spacing);
    this.slots.forEach((sl, i) => {
      sl.arm.root.position.copy(v3(bases[i]));
      sl.ghost?.root.position.copy(v3(bases[i]));
      sl.arm.root.visible = !sl.problem; // a pose that cannot be read is not drawn; its label says why
      if (sl.ghost) sl.ghost.root.visible = s.ghost && !sl.problem;
      sl.trail.mesh.visible = s.trail && !sl.problem;
      if (!s.trail || sl.problem) sl.trail.clear();
      if (sl.ghost) for (const p of sl.ghost.wristParts) p.visible = false;
    });
    const span = Math.max(0, (this.slots.length - 1) * spacing);
    this.contact.place(0.12, 0, 1.4 + span);
    this.controls.autoRotate = false;
  }

  setTheme(t: Theme): void {
    for (const k of Object.keys(t) as (keyof Theme)[]) this.theme[k].setStyle(t[k]);
    this.ground.material.color.copy(this.theme.ground);
    this.grid.material.uniforms.uColor.value.copy(this.theme.grid);
    this.materials.ghost.uniforms.uColor.value.copy(this.theme.ghost);
    for (const s of this.slots) s.trail.setColor(this.theme.trail);
    this.slots.forEach((s) => { s.arm.limits = s.arm.limits.map(() => "ok"); }); // re-tint below with the new colours
    for (const c of this.cams.values()) (c.lines.material as THREE.LineBasicMaterial).color.copy(c.key === this.editing ? this.theme.selected : this.theme.frustum);
    this.dirty = true;
  }

  // -- cameras ------------------------------------------------------------------------------------------
  private wristArm(): Slot | undefined {
    return this.slots.find((s) => s.slot.name === this.settings.wristArm) ?? this.slots[0];
  }

  /** Which follower each wrist camera rides on (sceneCore.ts wristMount): "wrist" on the chosen one, and
   * LeRobot's bimanual left_wrist / right_wrist on the follower of that side. */
  private wristMounts(): Map<string, Slot> {
    const slots = this.slots.map((s) => s.slot);
    const out = new Map<string, Slot>();
    for (const k of frustumKeys(Object.keys(studio.snap.cameras))) {
      const m = wristMount(k, this.model.wrist_camera.key, slots, this.settings.wristArm);
      if (m.arm !== null) out.set(k, this.slots[m.arm]);
    }
    return out;
  }

  private syncCameras(): void {
    this.mounts = this.wristMounts();
    // The camera bodies are drawn on each follower that carries a wrist camera.
    for (const sl of this.slots) for (const p of sl.arm.wristParts) p.visible = [...this.mounts.values()].includes(sl);
    const want = this.settings.frustums ? frustumKeys(Object.keys(studio.snap.cameras)) : [];
    for (const [k, c] of this.cams) if (!want.includes(k)) { c.group.removeFromParent(); c.dispose(); this.cams.delete(k); }
    for (const k of want) {
      let c = this.cams.get(k);
      const wrist = wristMount(k, this.model.wrist_camera.key, [], null).wrist;
      if (!c) {
        c = new CamView(k, wrist, this.opts.labels, () => { this.dirty = true; }, () => this.settings.video && !document.hidden);
        (c.lines.material as THREE.LineBasicMaterial).color.copy(this.theme.frustum);
        this.cams.set(k, c);
      }
      if (wrist) {
        c.fovy = this.model.wrist_camera.fovy_deg;
        c.depth = 0.035; // small: at 85 degrees a deeper image plane hides the gripper
        const arm = this.mounts.get(k);
        if (arm && c.group.parent !== arm.arm.wristCam) arm.arm.wristCam.add(c.group);
        if (!arm) c.group.removeFromParent();
      } else {
        const p = cameraPose(this.model, this.settings, k);
        c.fovy = p.fovy_deg;
        c.depth = 0.12;
        this.placeCamera(c.group, p);
        if (c.group.parent !== this.scene) this.scene.add(c.group);
      }
      c.plane.visible = this.settings.video;
      c.shape();
    }
    if (this.editing && !this.cams.has(this.editing)) this.setEditing(null);
    // A pose typed into the form moves the gizmo too; during a drag the gizmo is the source, so leave it.
    else if (this.editing && !this.gizmo.dragging) this.handle.position.copy(v3(cameraPose(this.model, this.settings, this.editing).pos));
    this.dirty = true;
  }

  private placeCamera(o: THREE.Object3D, p: CamPose): void {
    o.position.copy(v3(p.pos));
    o.up.copy(v3(p.up));
    // Object3D.lookAt points a camera's -z at the target only for cameras; build the same basis by hand.
    const m = new THREE.Matrix4().lookAt(v3(p.pos), v3(p.target), v3(p.up));
    o.quaternion.setFromRotationMatrix(m);
  }

  /** Select a user-placed camera for moving with the gizmo; null puts the gizmo away. */
  setEditing(key: string | null): void {
    if (key && this.cams.get(key)?.wrist) key = null; // a wrist camera rides on its arm
    this.editing = key;
    for (const c of this.cams.values()) (c.lines.material as THREE.LineBasicMaterial).color.copy(c.key === key ? this.theme.selected : this.theme.frustum);
    if (key && this.cams.has(key)) {
      const p = cameraPose(this.model, this.settings, key);
      this.handle.position.copy(v3(p.pos));
      this.gizmo.attach(this.handle);
    } else this.gizmo.detach();
    this.dirty = true;
  }

  private dragCamera(): void {
    const k = this.editing, c = k ? this.cams.get(k) : undefined;
    if (!k || !c) return;
    const p = cameraPose(this.model, this.settings, k);
    this.placeCamera(c.group, { ...p, pos: this.handle.position.toArray() as Vec3 });
    this.dirty = true;
  }

  private commitCamera(): void {
    const k = this.editing, from = this.editFrom;
    this.editFrom = null;
    if (!k) return;
    const p = cameraPose(this.model, this.settings, k);
    const r = (x: number) => Math.round(x * 1000) / 1000; // millimetres are plenty for a pose set by hand
    const pos = this.handle.position.toArray().map(r) as Vec3;
    // WHY compare: a click on an axis with no drag still fires mouseUp, and must not mark the pose "placed by you".
    if (from && pos.every((x, i) => Math.abs(x - r(from[i])) < 5e-4)) return;
    this.opts.onCameraEdit(k, { ...p, pos });
  }

  // -- view ---------------------------------------------------------------------------------------------
  private bounds(): THREE.Box3 {
    const box = new THREE.Box3();
    for (const s of this.slots) box.expandByObject(s.arm.root);
    if (box.isEmpty()) box.set(new THREE.Vector3(-0.05, -0.12, 0), new THREE.Vector3(0.4, 0.12, 0.35));
    return box;
  }

  /** The camera distance from the box centre, looking back along `dir`, at which all eight box corners are inside
   * the frustum: for a corner at depth z toward the camera and offsets x, y across the view, the camera must be at
   * least z + |x| / tan(horizontal half-angle) and z + |y| / tan(vertical half-angle) away. Tighter than a sphere. */
  private fitDistance(dir: THREE.Vector3, box: THREE.Box3, c: THREE.Vector3): number {
    const f = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(f, this.camera.up);
    if (right.lengthSq() < 1e-8) right.set(0, -1, 0);
    right.normalize();
    const up = new THREE.Vector3().crossVectors(right, f).normalize();
    const tv = Math.tan(THREE.MathUtils.degToRad(FOV) / 2), th = tv * this.camera.aspect;
    let d = 0;
    const p = new THREE.Vector3();
    for (let i = 0; i < 8; i++) {
      p.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(c);
      const z = p.dot(dir);
      d = Math.max(d, z + Math.abs(p.dot(right)) / th, z + Math.abs(p.dot(up)) / tv);
    }
    return Math.max(0.2, d * 1.12); // a margin for the name pills
  }

  /** Move the orbit camera to a preset, or ride on the wrist camera. Animated unless `animate` is false. */
  frame(p: Preset, animate = true): void {
    if (p === "wrist") { this.enterWrist(); return; }
    this.exitWrist();
    const box = this.bounds(), c = box.getCenter(new THREE.Vector3());
    const dirs: Record<Exclude<Preset, "wrist">, Vec3> = {
      home: [0.78, -0.78, 0.55], front: [1, 0, 0.3], side: [0.08, -1, 0.28], top: [-0.03, 0, 1],
    };
    const dir = v3(dirs[p]).normalize();
    this.goTo(c.clone().addScaledVector(dir, this.fitDistance(dir, box, c)), c, animate);
  }

  /** Keep the view direction, and back off until every arm fits. */
  fit(): void {
    this.exitWrist();
    const box = this.bounds(), c = box.getCenter(new THREE.Vector3());
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.goTo(c.clone().addScaledVector(dir, this.fitDistance(dir, box, c)), c, true);
  }

  private goTo(pos: THREE.Vector3, target: THREE.Vector3, animate: boolean): void {
    if (!animate) {
      this.camera.position.copy(pos);
      this.controls.target.copy(target);
      this.controls.update();
      this.dirty = true;
      return;
    }
    this.tween = { from: [this.camera.position.clone(), this.controls.target.clone()], to: [pos, target], t0: performance.now(), ms: 650 };
    this.lastInput = performance.now();
  }

  /** The wrist camera the Wrist view looks through: the one on the chosen follower, else the first mounted. */
  private wristRide(): { key: string; slot: Slot } | null {
    const pick = this.wristArm();
    let first: { key: string; slot: Slot } | null = null;
    for (const [key, slot] of this.mounts) {
      if (slot === pick) return { key, slot };
      first ??= { key, slot };
    }
    return first ?? (pick ? { key: this.model.wrist_camera.key, slot: pick } : null);
  }

  private enterWrist(): void {
    if (!this.wristRide() || this.wristView) return;
    this.wristView = true;
    this.controls.enabled = false;
    const wc = this.cams.get(this.wristRide()!.key);
    if (wc) wc.group.visible = false; // its own frustum would sit in front of the lens
    this.tween = null;
    this.opts.onWristView(true);
    this.dirty = true;
  }

  private exitWrist(): void {
    if (!this.wristView) return;
    this.wristView = false;
    this.controls.enabled = true;
    for (const c of this.cams.values()) if (c.wrist) c.group.visible = true;
    this.camera.fov = FOV;
    this.camera.up.set(0, 0, 1);
    this.camera.updateProjectionMatrix();
    // Orbit about a point ahead of where the wrist camera was looking.
    const ahead = new THREE.Vector3(0, 0, -0.25).applyQuaternion(this.camera.quaternion).add(this.camera.position);
    this.controls.target.copy(ahead);
    this.opts.onWristView(false);
    this.dirty = true;
  }

  setAutoRotate(on: boolean): void { this.settings = { ...this.settings, autoRotate: on }; this.lastInput = performance.now() - 10_000; }

  // -- pointer: click a frustum to select it ----------------------------------------------------------
  private bindPointer(): void {
    const el = this.renderer.domElement;
    let down: [number, number] | null = null;
    const onDown = (e: PointerEvent) => { down = [e.clientX, e.clientY]; this.lastInput = performance.now(); if (this.wristView) this.frame("home"); };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || this.gizmo.dragging) { down = null; return; }
      down = null;
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(ndc, this.camera);
      const planes = [...this.cams.values()].filter((c) => !c.wrist && c.group.visible).map((c) => c.plane);
      const hit = ray.intersectObjects(planes, false)[0];
      if (hit) this.opts.onSelectCamera(hit.object.userData.camera as string);
    };
    const onWheel = () => { this.lastInput = performance.now(); };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("wheel", onWheel, { passive: true });
    this.unsub.push(() => { el.removeEventListener("pointerdown", onDown); el.removeEventListener("pointerup", onUp); el.removeEventListener("wheel", onWheel); });
  }

  // -- a lost GPU context ------------------------------------------------------------------------------
  /** The light probe from three's procedural room, rendered on the GPU. */
  private environment(): [THREE.PMREMGenerator, THREE.WebGLRenderTarget] {
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const room = new RoomEnvironment();
    const target = pmrem.fromScene(room, 0.04);
    room.dispose();
    this.scene.environment = target.texture;
    return [pmrem, target];
  }

  /** The browser may take the GPU context away (driver reset, too many contexts, GPU switch). three restores its own
   * state when the context comes back (WebGLRenderer.js onContextRestore) and uploads geometry and textures again,
   * but what was rendered on the GPU is gone: the light probe, the contact shadow and the shadow map are made again. */
  private bindContext(): void {
    const el = this.renderer.domElement;
    const onLost = (e: Event) => {
      e.preventDefault(); // WHY: without it the browser never offers the context back
      this.lost = true;
      // WHY: free GPU objects now, against the dead context; after restore they would be freed against the new one.
      this.envTarget.dispose();
      this.pmrem.dispose();
      this.contact.release();
      this.opts.onContextLost(true);
    };
    const onRestored = () => {
      [this.pmrem, this.envTarget] = this.environment();
      this.contact.renew();
      this.lost = false;
      this.dirty = this.shadowsDirty = true;
      this.renderer.shadowMap.needsUpdate = true;
      this.opts.onContextLost(false);
    };
    el.addEventListener("webglcontextlost", onLost);
    el.addEventListener("webglcontextrestored", onRestored);
    this.unsub.push(() => { el.removeEventListener("webglcontextlost", onLost); el.removeEventListener("webglcontextrestored", onRestored); });
  }

  private fitCanvas(): void {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  // -- the frame loop -----------------------------------------------------------------------------------
  private readonly loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(0.1, (now - this.lastFrameAt) / 1000);
    this.lastFrameAt = now;
    this.stats.frames++;
    this.stats.frameMs.push(now);
    if (this.stats.frameMs.length > 240) this.stats.frameMs.shift();
    if (!this.ready || this.lost) return;

    if (this.tween) {
      const u = Math.min(1, (now - this.tween.t0) / this.tween.ms);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; // ease in-out cubic
      this.camera.position.lerpVectors(this.tween.from[0], this.tween.to[0], e);
      this.controls.target.lerpVectors(this.tween.from[1], this.tween.to[1], e);
      if (u >= 1) this.tween = null;
      this.dirty = true;
    }
    const idle = now - this.lastInput > 6000;
    this.controls.autoRotate = this.settings.autoRotate && idle && !this.wristView && !this.tween && !this.editing;
    if (this.controls.enabled && this.controls.update(dt)) this.dirty = true;

    // Draw each arm `delay` ms behind its newest reading, interpolating between readings.
    const t = now - this.clock.delay;
    let moved = false;
    const armMoved = this.armMoved;
    armMoved.clear();
    for (const s of this.slots) {
      if (s.track.sample(t, this.q)) {
        if (this.q.some((v, i) => v !== s.arm.q[i])) { s.arm.setPose(this.q); moved = true; armMoved.add(s); }
      }
      if (s.ghost) {
        const has = this.settings.ghost && !s.problem && s.ghostTrack.sample(t, this.q);
        if (s.ghost.root.visible !== has) { s.ghost.root.visible = has; moved = true; }
        if (has && this.q.some((v, i) => v !== s.ghost!.q[i])) { s.ghost.setPose(this.q); moved = true; }
      }
      this.tint(s);
    }
    if (moved) { this.dirty = true; this.shadowsDirty = true; }

    if (this.wristView) {
      const w = this.wristRide()?.slot;
      if (w) {
        w.arm.wristCam.updateWorldMatrix(true, false);
        w.arm.wristCam.matrixWorld.decompose(this.camera.position, this.camera.quaternion, this.tmp);
        this.camera.fov = this.model.wrist_camera.fovy_deg;
        this.camera.updateProjectionMatrix();
      }
      this.dirty = true;
    }

    // Draw only when the picture changes: a pose moved, a trail is still fading, or something else marked it
    // dirty (the view, a setting, a video frame). A still rig, connected or not, then costs no frames.
    if (!this.dirty && !(this.settings.trail && this.slots.some((s) => s.trail.live()))) {
      if (now - this.labelsAt > 250) this.labels(); // a pill's text changes with time alone ("no reading")
      return;
    }
    this.scene.updateMatrixWorld();
    for (const s of this.slots) {
      if (!this.settings.trail) continue;
      // WHY only when it moved: a still arm adds no point, so its trail fades out and the view stops drawing.
      if (s.seen && armMoved.has(s)) s.trail.push(s.arm.tool.getWorldPosition(this.tmp), now);
      s.trail.update(now, this.camera.position);
    }
    if (this.shadowsDirty) {
      // Ghosts, trails and frustums stay out of the contact shadow: it is only for what touches the desk.
      this.contact.render(this.renderer, this.scene);
      // WHY after the contact pass: the shadow map is drawn with the layers of the camera that triggers it, and
      // the contact camera sees layer 1 only.
      this.renderer.shadowMap.needsUpdate = true;
      this.shadowsDirty = false;
    }
    const r0 = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.stats.renderMs.push(performance.now() - r0);
    if (this.stats.renderMs.length > 240) this.stats.renderMs.shift();
    this.stats.renders++;
    this.dirty = false;
    this.labels();
  };

  /** Warm tint on the servo that drives a joint near its model limit; stronger past it. Changes only on a state change. */
  private tint(s: Slot): void {
    const order = this.model.joint_order;
    for (let i = 0; i < order.length; i++) {
      const st = s.seen ? limitState(this.model, order[i], s.arm.q[i]).state : "ok";
      if (st === s.arm.limits[i]) continue;
      s.arm.limits[i] = st;
      const m = s.arm.servo.get(order[i]);
      if (!m) continue;
      if (st === "ok" || st === "outside") { m.emissive.setRGB(0, 0, 0); m.color.copy(this.materials.servo.color); }
      else {
        const c = st === "past" ? this.theme.danger : this.theme.warn;
        m.color.copy(this.materials.servo.color).lerp(c, 0.55);
        m.emissive.copy(c).multiplyScalar(0.45);
      }
      this.dirty = true;
    }
  }

  /** Name pills: under each arm, above each camera. Positioned by projecting a 3D anchor every drawn frame. */
  private labels(): void {
    const w = this.host.clientWidth, h = this.host.clientHeight, now = performance.now();
    this.labelsAt = now;
    const place = (el: HTMLElement, p: THREE.Vector3, text: string, tone: string) => {
      p.project(this.camera);
      const off = p.z > 1 || Math.abs(p.x) > 1.05 || Math.abs(p.y) > 1.05;
      el.hidden = off || this.wristView;
      if (el.hidden) return;
      el.style.transform = `translate(${((p.x + 1) / 2) * w}px, ${((1 - p.y) / 2) * h}px) translate(-50%, -50%)`;
      if (el.textContent !== text) el.textContent = text;
      if (el.dataset.tone !== tone) el.dataset.tone = tone;
    };
    for (const s of this.slots) {
      const stale = s.seen && now - s.track.lastArrival > STALE_MS;
      const what = s.problem ?? (!s.seen ? "no reading yet, drawn at zero" : !s.online ? "not answering" : stale ? "no reading" : "");
      const p = s.arm.root.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(-0.12, 0, 0)); // behind the base
      place(s.pill, p, what ? `${label(s.slot.name)}: ${what}` : label(s.slot.name), what ? "warn" : "neutral");
    }
    for (const c of this.cams.values()) {
      if (!c.group.parent || !shown(c.group)) { c.pill.hidden = true; continue; } // e.g. on an arm that is not drawn
      const top = new THREE.Vector3(0, c.depth * Math.tan(THREE.MathUtils.degToRad(c.fovy) / 2) * 1.5, -c.depth);
      c.group.localToWorld(top);
      place(c.pill, top, `${label(c.key)}${c.wrist ? " (CAD mount)" : " (placed by you)"}${this.camStatus(c, now)}`, c.key === this.editing ? "accent" : "neutral");
    }
  }

  /** What a camera's label adds about its feed: the same words as the camera tiles (components/CameraTile.tsx),
   * so the 3D view alone tells a dead camera from a slow one. Empty while frames arrive. */
  private camStatus(c: CamView, now: number): string {
    const s = studio.snap, st = s.cameras[c.key];
    if (st && !st.online) return `: ${st.message ?? "No signal"}`;
    if (s.link !== "open") return ": not connected";
    if (!c.lastFrame) return ": waiting for frames";
    return now - c.lastFrame > 1500 ? ": stale" : "";
  }

  // -- debugging and checks ----------------------------------------------------------------------------
  /** Tool-site position (MJCF world, metres) for joint angles in MJCF radians, on an arm at the origin. The same
   * node tree as drawn, without meshes: lets a check compare the browser's kinematics with Python's. */
  toolAt(q: number[]): Vec3 {
    const a = new ArmView(this.model, null, () => this.materials.servo);
    a.setPose(q);
    a.root.updateMatrixWorld(true);
    return a.tool.getWorldPosition(new THREE.Vector3()).toArray() as Vec3;
  }

  fps(): number {
    const f = this.stats.frameMs;
    return f.length > 1 ? ((f.length - 1) * 1000) / (f[f.length - 1] - f[0]) : 0;
  }

  /** three's shared DFG lookup texture, read from a lit material's uniforms; undefined before the first draw. */
  private sharedLut(): THREE.Texture | undefined {
    for (const m of [this.materials.pla, this.materials.servo]) {
      const u = (this.renderer.properties.get(m) as { uniforms?: { dfgLUT?: { value: THREE.Texture | null } } }).uniforms;
      if (u?.dfgLUT?.value) return u.dfgLUT.value;
    }
    return undefined;
  }

  dispose(): { geometries: number; textures: number } {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.unsub.forEach((u) => u());
    this.resize.disconnect();
    this.gizmo.detach();
    this.gizmo.dispose();
    this.controls.dispose();
    for (const s of this.slots) this.dropSlot(s);
    for (const c of this.cams.values()) c.dispose();
    this.cams.clear();
    this.contact.dispose();
    this.key.shadow.dispose();
    this.envTarget.dispose();
    this.pmrem.dispose();
    this.ground.geometry.dispose();
    this.ground.material.alphaMap?.dispose();
    this.ground.material.dispose();
    this.grid.geometry.dispose();
    this.grid.material.dispose();
    // WHY: three r186 shares one DFG lookup texture across every renderer (shaders/DFGLUTData.js getDFGLUT), and each
    // renderer that used it adds a dispose listener to it. renderer.dispose() never removes that listener, so the
    // texture kept every closed renderer, its canvas and its GL context alive. Disposing it runs those listeners; a
    // renderer that still needs it uploads it again on its next draw.
    this.sharedLut()?.dispose();
    Object.values(this.materials).forEach((m) => m.dispose());
    for (const g of this.geoms.values()) g.dispose(); // frees the GPU buffers; the CPU copy stays cached for the next viewer
    // What this renderer still holds after every dispose above: both should be 0 (checked in the browser).
    const left = { geometries: this.renderer.info.memory.geometries, textures: this.renderer.info.memory.textures };
    this.renderer.dispose();
    this.renderer.forceContextLoss(); // WHY: Chrome keeps a few live WebGL contexts per page; return this one now
    this.renderer.domElement.remove();
    this.opts.labels.replaceChildren();
    return left;
  }
}
