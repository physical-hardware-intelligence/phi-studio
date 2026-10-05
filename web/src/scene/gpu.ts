// What a lost GPU context leaves behind. Pure over a scene graph, so tests/scene.test.ts runs it in Node.
import type * as THREE from "three";

type Buffer = THREE.BufferGeometry | THREE.Texture;
const isTexture = (v: unknown): v is THREE.Texture => (v as THREE.Texture | null)?.isTexture === true;

/** Every geometry and texture under `root`, drawn or hidden: the objects three's geometry and texture modules put a
 * dispose listener on when they upload them (WebGLGeometries.js:55, WebGLTextures.js:719). Textures are a material's
 * maps and its shader uniforms. Materials are left out: their dispose listener lives in WebGLRenderer and reads the
 * renderer's current modules, so it frees against whichever context is live. */
export function drawnBuffers(root: THREE.Object3D): Set<Buffer> {
  const out = new Set<Buffer>();
  root.traverse((o) => {
    const { geometry, material } = o as Partial<THREE.Mesh>;
    if (geometry?.isBufferGeometry) out.add(geometry);
    for (const m of material ? (Array.isArray(material) ? material : [material]) : []) {
      for (const v of Object.values(m)) if (isTexture(v)) out.add(v);
      const u = (m as Partial<THREE.ShaderMaterial>).uniforms;
      if (u) for (const { value } of Object.values(u)) if (isTexture(value)) out.add(value);
    }
  });
  return out;
}
