// Headless check of the 3D view across a lost and restored WebGL context. Opens #/scene, loses and restores the
// context with WEBGL_lose_context, leaves the page (Engine.dispose), and fails on any WebGL object deleted against a
// newer context than the one that made it: the browser's "object does not belong to this context" warning.
// It also checks the view drew again after each restore with its buffers uploaded again.
//
// Needs Playwright and its Chromium, which Studio does not depend on:
//   npm i --no-save playwright && npx playwright install chromium
// or PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs to use another copy.
// Run against a mock Studio:
//   phi-studio --mock --port 8795 --no-browser
//   node scripts/check-context-restore.mjs "http://127.0.0.1:8795/#token=..." [cycles]
const url = process.argv[2];
const cycles = Number(process.argv[3] ?? 2);
if (!url) { console.error("Give the Studio address with its token, as phi-studio prints it."); process.exit(2); }
let chromium;
try { ({ chromium } = await import(process.env.PLAYWRIGHT ?? "playwright")); } catch {
  console.error("Playwright is not installed. Run: npm i --no-save playwright && npx playwright install chromium");
  process.exit(2);
}

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const log = [];
page.on("console", (m) => { if (!/GPU stall due to ReadPixels/.test(m.text())) log.push(`[${m.type()}] ${m.text()}`); });
page.on("pageerror", (e) => log.push(`[pageerror] ${e.message}`));

// Tag each GL object with the context generation that made it; record a delete of an older one, with its stack.
await page.addInitScript(() => {
  const P = WebGL2RenderingContext.prototype;
  const gen = new WeakMap();
  const stale = (globalThis.__staleDeletes = []);
  for (const k of ["Buffer", "Texture", "Framebuffer", "Renderbuffer", "Program", "Shader", "VertexArray", "Query", "Sampler", "TransformFeedback"]) {
    const create = P[`create${k}`], del = P[`delete${k}`];
    P[`create${k}`] = function (...a) { const o = create.apply(this, a); if (o) o.__gen = gen.get(this) ?? 0; return o; };
    P[`delete${k}`] = function (o) {
      if (o && !this.isContextLost() && o.__gen !== (gen.get(this) ?? 0)) {
        stale.push(`delete${k} ${(new Error().stack ?? "").split("\n").slice(2, 7).map((l) => l.trim()).join(" < ")}`);
      }
      return del.call(this, o);
    };
  }
  addEventListener("webglcontextrestored", (e) => { const gl = e.target.getContext("webgl2"); gen.set(gl, (gen.get(gl) ?? 0) + 1); }, true);
});

const failures = [];
const memory = () => page.evaluate(() => {
  const e = globalThis.__phiScene.engine;
  return { renders: e.stats.renders, ...e.renderer.info.memory };
});
await page.goto(url);
await page.waitForTimeout(800);
await page.evaluate(() => { location.hash = "#/scene"; });
await page.waitForFunction(() => globalThis.__phiScene?.engine?.stats.renders > 0, null, { timeout: 30000 });
await page.waitForTimeout(3000); // the arm meshes load and draw
const mark = log.length;
for (let i = 0; i < cycles; i++) {
  const before = await memory();
  await page.evaluate(async () => {
    const c = document.querySelector("canvas.scene-canvas");
    const ext = c.getContext("webgl2").getExtension("WEBGL_lose_context");
    await new Promise((r) => { c.addEventListener("webglcontextlost", r, { once: true }); ext.loseContext(); });
    await new Promise((r) => setTimeout(r, 300));
    await new Promise((r) => { c.addEventListener("webglcontextrestored", r, { once: true }); ext.restoreContext(); });
  });
  await page.waitForTimeout(2500);
  const after = await memory();
  console.log(`cycle ${i + 1}: before loss ${JSON.stringify(before)}, after restore ${JSON.stringify(after)}`);
  if (after.renders <= before.renders) failures.push(`cycle ${i + 1}: the view did not draw after the restore`);
  if (after.geometries < before.geometries || after.textures < before.textures) failures.push(`cycle ${i + 1}: fewer buffers on the GPU after the restore`);
}
await page.evaluate(() => { location.hash = "#/"; });
await page.waitForFunction(() => !globalThis.__phiScene, null, { timeout: 10000 });
await page.waitForTimeout(1000);
const stale = await page.evaluate(() => globalThis.__staleDeletes);
await browser.close();

const after = log.slice(mark);
const warnings = after.filter((l) => /does not belong to this context/.test(l));
const restored = after.filter((l) => /Context Restored/.test(l)).length;
console.log("console after the first loss:");
for (const l of after) console.log(`  ${l}`);
const kinds = {};
for (const s of stale) kinds[s.split(" ")[0]] = (kinds[s.split(" ")[0]] ?? 0) + 1;
console.log(`stale deletes: ${stale.length} ${JSON.stringify(kinds)}`);
for (const s of [...new Set(stale)].slice(0, 20)) console.log(`  ${s}`);
console.log(`"does not belong to this context" warnings: ${warnings.length}`);
if (restored !== cycles) failures.push(`expected ${cycles} restores, saw ${restored}`);
if (stale.length) failures.push(`${stale.length} WebGL objects deleted against a newer context`);
if (warnings.length) failures.push(`${warnings.length} "does not belong to this context" warnings`);
if (after.some((l) => l.startsWith("[pageerror]") || l.startsWith("[error]"))) failures.push("errors in the console");
console.log(failures.length ? `FAIL\n  ${failures.join("\n  ")}` : "PASS");
process.exit(failures.length ? 1 : 0);
