// A 3D scene of one or two SO-101s: the measured arm solid, the commanded arm as a ghost, the tool path as a
// trail. Renders only when something changed, so an idle view costs no GPU.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ArmRig, loadModel, type Model } from "./so101";

export type View = "iso" | "front" | "side" | "top";
// WHY these offsets: two arms side by side facing +x (the way they reach), 40 cm apart, as on our bench.
const SIDE_Y: Record<string, number> = { left: 0.2, right: -0.2, "": 0 };

export interface TwinArm { name: string; solid: ArmRig; ghost: ArmRig; trail: THREE.Line | null; dot: THREE.Mesh }

export class TwinScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
  readonly controls: OrbitControls;
  readonly arms = new Map<string, TwinArm>();
  model: Model | null = null;
  private frame = 0;
  private resize: ResizeObserver;
  private cloud: THREE.Points | null = null;
  private disposed = false;
  private ghostVisible = true;
  private current: View = "iso";
  private focus = new THREE.Vector3(0.13, 0, 0.1);
  private radius = 0.32;
  private userMoved = false; // once someone orbits, new data no longer moves their camera

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap; // r186 removed PCFSoft; PCF with a radius is the replacement
    this.renderer.domElement.className = "twin-canvas";
    host.appendChild(this.renderer.domElement);

    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.minDistance = 0.12;
    this.controls.maxDistance = 3;
    this.controls.addEventListener("change", () => this.request());
    this.controls.addEventListener("start", () => { this.userMoved = true; });

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x1a1a1f, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(0.6, -0.8, 1.4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -0.6; key.shadow.camera.right = 0.6;
    key.shadow.camera.top = 0.6; key.shadow.camera.bottom = -0.6;
    key.shadow.camera.near = 0.1; key.shadow.camera.far = 4;
    key.shadow.bias = -0.0004;
    key.shadow.radius = 4;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xb8c8ff, 0.5);
    fill.position.set(-0.8, 0.6, 0.5);
    this.scene.add(fill);

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShadowMaterial({ opacity: 0.28 }));
    ground.receiveShadow = true;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(1.2, 24, 0x5a5a64, 0x34343a);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in xz; our ground is xy (z up)
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.32;
    grid.position.z = -0.0005;
    this.scene.add(grid);

    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(host);
    this.fit();
  }

  async init(names: string[]): Promise<void> {
    this.model = await loadModel();
    if (this.disposed) return;
    for (const name of names) {
      const solid = new ArmRig(this.model, "solid");
      const ghost = new ArmRig(this.model, "ghost");
      const y = SIDE_Y[name] ?? 0;
      solid.root.position.y = y;
      ghost.root.position.y = y;
      ghost.root.visible = this.ghostVisible;
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.006, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0x6b9ef0, depthTest: false, transparent: true }));
      dot.renderOrder = 3;
      dot.visible = false;
      this.scene.add(solid.root, ghost.root, dot);
      this.arms.set(name, { name, solid, ghost, trail: null, dot });
    }
    this.refit();
  }

  view(v: View): void {
    this.current = v;
    this.userMoved = false;
    const dir = { iso: [0.72, -0.62, 0.42], front: [1, 0, 0.16], side: [0.02, -1, 0.18], top: [0.0001, 0, 1] }[v];
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fit = this.radius / Math.sin(Math.min(fov, fov * this.camera.aspect) / 2);
    const p = new THREE.Vector3(...dir).normalize().multiplyScalar(fit * 1.08).add(this.focus);
    this.camera.position.copy(p);
    this.controls.target.copy(this.focus);
    this.controls.update();
    this.request();
  }

  /** Frame everything shown: each arm's reach around its base, its tool path, the workspace cloud. */
  private refit(): void {
    const box = new THREE.Box3();
    this.arms.forEach((a) => {
      const y = a.solid.root.position.y;
      box.expandByPoint(new THREE.Vector3(-0.06, y - 0.08, 0));
      box.expandByPoint(new THREE.Vector3(0.24, y + 0.08, 0.26)); // the folded arm and a little room
      const g = a.trail?.geometry;
      if (g) { g.computeBoundingBox(); if (g.boundingBox) box.union(g.boundingBox); }
    });
    const cg = this.cloud?.geometry;
    if (cg) { cg.computeBoundingBox(); if (cg.boundingBox) box.union(cg.boundingBox); }
    if (box.isEmpty()) return;
    box.getCenter(this.focus);
    this.radius = Math.max(0.12, box.getSize(new THREE.Vector3()).length() / 2);
    if (!this.userMoved) this.view(this.current);
  }

  setState(arm: string, values: ArrayLike<number> | null): void {
    const a = this.arms.get(arm);
    if (!a || !values) return;
    a.solid.set(values);
    this.request();
  }

  setGhost(arm: string, values: ArrayLike<number> | null): void {
    const a = this.arms.get(arm);
    if (!a) return;
    if (values) a.ghost.set(values);
    a.ghost.root.visible = this.ghostVisible && !!values;
    this.request();
  }

  showGhosts(on: boolean): void {
    this.ghostVisible = on;
    this.arms.forEach((a) => { a.ghost.root.visible = on; });
    this.request();
  }

  /** The tool path, in world metres (kinematics.py's frame, offset to the arm's base). */
  setTrail(arm: string, points: number[][] | null, color = 0x6b9ef0): void {
    const a = this.arms.get(arm);
    if (!a) return;
    if (a.trail) { a.trail.geometry.dispose(); (a.trail.material as THREE.Material).dispose(); a.trail.removeFromParent(); a.trail = null; }
    if (points && points.length > 1) {
      const y = SIDE_Y[arm] ?? 0;
      const buf = new Float32Array(points.length * 3);
      points.forEach((p, k) => { buf[k * 3] = p[0]; buf[k * 3 + 1] = p[1] + y; buf[k * 3 + 2] = p[2]; });
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(buf, 3));
      a.trail = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
      a.trail.renderOrder = 2;
      this.scene.add(a.trail);
    }
    this.refit();
    this.request();
  }

  /** Mark the tool point at frame k on the trail. */
  setTrailCursor(arm: string, k: number | null): void {
    const a = this.arms.get(arm);
    if (!a) return;
    const attr = a.trail?.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
    if (k === null || !attr || k < 0 || k >= attr.count) { a.dot.visible = false; this.request(); return; }
    a.dot.position.set(attr.getX(k), attr.getY(k), attr.getZ(k));
    a.dot.visible = true;
    this.request();
  }

  /** A point cloud of tool positions (workspace view), coloured per point. */
  setCloud(points: number[][] | null, colors?: Float32Array, arm = ""): void {
    if (this.cloud) { this.cloud.geometry.dispose(); (this.cloud.material as THREE.Material).dispose(); this.cloud.removeFromParent(); this.cloud = null; }
    if (points && points.length) {
      const y = SIDE_Y[arm] ?? 0;
      const buf = new Float32Array(points.length * 3);
      points.forEach((p, k) => { buf[k * 3] = p[0]; buf[k * 3 + 1] = p[1] + y; buf[k * 3 + 2] = p[2]; });
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(buf, 3));
      if (colors) g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      this.cloud = new THREE.Points(g, new THREE.PointsMaterial({
        size: 0.0035, vertexColors: !!colors, color: colors ? 0xffffff : 0x6b9ef0, transparent: true, opacity: 0.8, depthWrite: false,
      }));
      this.scene.add(this.cloud);
    }
    this.refit();
    this.request();
  }

  request(): void {
    if (this.frame || this.disposed) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.renderer.render(this.scene, this.camera);
    });
  }

  private fit(): void {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (!this.userMoved && this.arms.size) this.view(this.current); // keep the framing as the panel resizes
    this.request();
  }

  dispose(): void {
    this.disposed = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.controls.dispose();
    this.arms.forEach((a) => { a.solid.dispose(); a.ghost.dispose(); a.trail?.geometry.dispose(); });
    this.cloud?.geometry.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
