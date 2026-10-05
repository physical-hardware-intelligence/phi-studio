// Point clouds from environment reconstruction, one layer per camera, drawn in the 3D view's world frame (metres,
// z up, the arm base at the origin). Kept out of engine.ts: the engine only gives a scene, requestRender and onDispose.
//
// Each cloud arrives as binary from GET /api/recon/cloud/{id} (src/phi_studio/recon.py pack): "PCL1", uint32 n,
// uint32 n_off_arm, float32 xyz * n, uint8 rgb * n. Points off the arm come first, so "hide points on the arm" is a
// draw range, not a rebuild.
import * as THREE from "three";
import { type CloudInfo, reconStore, type ReconState } from "../lib/recon";
import { studio } from "../lib/studio";
import type { Engine } from "./engine";

// sRGB byte to linear float: three treats vertex colours as linear and converts to sRGB on output.
const LINEAR = new Float32Array(256).map((_, i) => new THREE.Color().setRGB(i / 255, 0, 0, THREE.SRGBColorSpace).r);

interface Layer { id: string; points: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>; n: number; nOff: number }

/** Parse the binary cloud. Throws on a wrong header or a short body. */
export function parseCloud(buf: ArrayBuffer): { n: number; nOff: number; positions: Float32Array; colors: Float32Array } {
  const v = new DataView(buf);
  if (buf.byteLength < 12 || v.getUint32(0, false) !== 0x50434c31) throw new Error("not a point cloud"); // "PCL1"
  const n = v.getUint32(4, true), nOff = v.getUint32(8, true);
  if (buf.byteLength !== 12 + 15 * n || nOff > n) throw new Error("point cloud size does not match its header");
  const positions = new Float32Array(buf, 12, 3 * n); // a view, no copy: byte 12 is a multiple of 4
  const rgb = new Uint8Array(buf, 12 + 12 * n, 3 * n);
  const colors = new Float32Array(3 * n);
  for (let i = 0; i < rgb.length; i++) colors[i] = LINEAR[rgb[i]];
  return { n, nOff, positions, colors };
}

export class PointLayers {
  readonly group = new THREE.Group();
  private readonly layers = new Map<string, Layer>();
  private readonly loading = new Map<string, { id: string; ctl: AbortController }>(); // camera -> the fetch in flight
  private readonly unsub: () => void;
  private last: ReconState | null = null;
  private disposed = false;

  constructor(private readonly engine: Engine) {
    this.group.name = "point clouds";
    engine.scene.add(this.group);
    this.unsub = reconStore.subscribe(() => this.sync());
    engine.onDispose(() => this.dispose());
    this.sync();
  }

  /** Match the layers to the store: fetch new clouds, drop removed ones, apply the style. */
  private sync(): void {
    if (this.disposed) return;
    const s = reconStore.snap;
    if (s === this.last) return;
    const styleChanged = !this.last || this.last.settings !== s.settings;
    this.last = s;
    for (const cam of [...this.layers.keys()]) if (!s.clouds[cam]) this.drop(cam);
    for (const [cam, l] of [...this.loading]) {
      if (!s.clouds[cam]) { l.ctl.abort(); this.loading.delete(cam); } // removed while it was loading
    }
    for (const [cam, info] of Object.entries(s.clouds)) {
      // WHY check the fetch in flight too: every store change (a status reply, a slider) would otherwise restart it.
      if (this.layers.get(cam)?.id === info.id || this.loading.get(cam)?.id === info.id) continue;
      void this.fetch(cam, info);
    }
    if (styleChanged) this.style();
  }

  private async fetch(cam: string, info: CloudInfo): Promise<void> {
    this.loading.get(cam)?.ctl.abort(); // a newer cloud for this camera: latest wins here too
    const ctl = new AbortController();
    this.loading.set(cam, { id: info.id, ctl });
    try {
      const r = await fetch(info.url, { headers: { "X-Phi-Token": studio.token() }, cache: "no-store", signal: ctl.signal });
      if (!r.ok) {
        // A 404 for a cloud the store already replaced is expected: the newer one is on its way. Anything else is said.
        if (!this.disposed && !ctl.signal.aborted && reconStore.snap.clouds[cam]?.id === info.id) {
          reconStore.fetchFailed(cam, `The points could not be loaded (the server answered ${r.status}). Capture again.`);
        }
        return;
      }
      const cloud = parseCloud(await r.arrayBuffer());
      if (this.disposed || ctl.signal.aborted) return;
      this.drop(cam);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
      g.setAttribute("color", new THREE.BufferAttribute(cloud.colors, 3));
      g.computeBoundingSphere();
      const m = new THREE.PointsMaterial({ vertexColors: true, sizeAttenuation: true, transparent: true, depthWrite: true });
      const points = new THREE.Points(g, m);
      points.name = `points ${cam}`;
      this.layers.set(cam, { id: info.id, points, n: cloud.n, nOff: cloud.nOff });
      this.group.add(points);
      this.style();
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) console.warn(`point cloud ${cam}:`, e);
    } finally {
      if (this.loading.get(cam)?.ctl === ctl) this.loading.delete(cam);
    }
  }

  private style(): void {
    const s = reconStore.snap.settings;
    for (const [cam, l] of this.layers) {
      l.points.visible = s.visible[cam] !== false;
      l.points.material.size = s.size / 1000; // metres: with size attenuation a point is this big in the world
      l.points.material.opacity = s.opacity;
      l.points.material.depthWrite = s.opacity >= 1; // a see-through cloud must not hide what is behind it
      l.points.geometry.setDrawRange(0, s.hideArm ? l.nOff : l.n);
    }
    this.engine.requestRender();
  }

  private drop(cam: string): void {
    const l = this.layers.get(cam);
    if (!l) return;
    this.group.remove(l.points);
    l.points.geometry.dispose();
    l.points.material.dispose();
    this.layers.delete(cam);
    this.engine.requestRender();
  }

  /** How many layers and points are drawn, for checks in the browser. */
  info(): { layers: number; points: number } {
    let points = 0;
    for (const l of this.layers.values()) if (l.points.visible) points += l.points.geometry.drawRange.count;
    return { layers: this.layers.size, points };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.unsub();
    for (const l of this.loading.values()) l.ctl.abort();
    this.loading.clear();
    for (const cam of [...this.layers.keys()]) this.drop(cam);
    this.group.removeFromParent();
  }
}
