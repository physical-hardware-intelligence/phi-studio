// The 3D view's canvas with its toolbar and notes. The default export, so pages load it with React.lazy and
// three.js arrives only when a 3D view is on screen.
import { ArrowLeft, Info, Maximize2, Minimize2, OctagonX, Rotate3d, RotateCw, Scan, ScanSearch, SlidersHorizontal } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { label } from "../lib/labels";
import { MODE_LAYERS, scene, type SceneMode, type Settings, useScene } from "../lib/scene";
import { studio, useStudio, useTheme } from "../lib/studio";
import { Engine, type Preset, type Theme } from "./engine";
import { PointLayers } from "./pointcloud";
import { Segmented } from "./ViewSwitch";

let live = 0; // engines alive now; the leak check reads it through globalThis.__phiScene

// Scene colours come from scene.css custom properties, which come from tokens.css, so both themes work.
const VARS: Record<keyof Theme, string> = {
  ground: "--scene-ground", grid: "--scene-grid", ghost: "--scene-ghost", trail: "--scene-trail", warn: "--scene-warn",
  danger: "--scene-danger", frustum: "--scene-frustum", selected: "--scene-selected",
};

/** Resolve each variable to a colour three can parse. A probe element makes the browser compute it; Chrome
 * reports a color-mix() result as color(srgb r g b), which three's parser does not read, so that form is converted. */
function readTheme(el: HTMLElement): Theme {
  const probe = document.createElement("span");
  probe.hidden = true;
  el.appendChild(probe);
  const out = {} as Theme;
  for (const [k, v] of Object.entries(VARS) as [keyof Theme, string][]) {
    probe.style.color = `var(${v})`;
    const c = getComputedStyle(probe).color;
    const m = c.match(/color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)/);
    out[k] = m ? `rgb(${m.slice(1, 4).map((x) => Math.round(Math.min(1, Math.max(0, parseFloat(x))) * 255)).join(", ")})` : c;
  }
  probe.remove();
  return out;
}

const PRESETS: Preset[] = ["front", "side", "top"];
const MODES: { mode: SceneMode; text: string }[] = [
  { mode: "pose", text: "Pose" }, { mode: "cameras", text: "Cameras" }, { mode: "policy", text: "Policy" },
];
// The info tooltip: what the drawing means in each mode.
const LEGEND: Record<SceneMode, (policy: boolean) => string> = {
  pose: (policy) => `Solid: follower. Ghost: ${policy ? "policy target" : "leader"}. Grid: 5 cm squares.`,
  cameras: () => "Each cone is a camera: where it sits, where it looks, and its live picture. Wrist cameras ride on their arm.",
  policy: () => "Ghost: where the policy is sending the follower. Line: the gripper's path over the last 3 s.",
};
/** left_wrist -> "Left wrist camera"; the plain wrist key -> "Wrist camera". */
const wristName = (k: string) => `${k.charAt(0).toUpperCase()}${k.slice(1).replace(/_/g, " ")} camera`;
const wristShort = (k: string) => k.replace(/^left_/, "L ").replace(/^right_/, "R ").replace(/^wrist$/, "Wrist");
const mb = (b: number) => (b / 1e6).toFixed(1);

/** onSettings: a gear at the end of the toolbar opens the page's settings (the 3D view page passes one).
 * mode: a fixed mode for a page with one job (Teleoperate: pose, Run policy: policy); the switch then hides. */
export default function Viewer({ compact = false, onSettings, mode }: { compact?: boolean; onSettings?: () => void; mode?: SceneMode }) {
  const stage = useRef<HTMLDivElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const labels = useRef<HTMLDivElement>(null);
  const engine = useRef<Engine | null>(null);
  const model = useScene((s) => s.model);
  const error = useScene((s) => s.error);
  const saved = useScene((s) => s.settings);
  const settings: Settings = useMemo(() => (mode ? { ...saved, mode, ...MODE_LAYERS[mode] } : saved), [saved, mode]);
  const editing = useScene((s) => s.editing);
  const units = useScene((s) => s.units);
  const theme = useTheme();
  const policy = useStudio((s) => s.state?.activity === "policy");
  const [progress, setProgress] = useState<[number, number] | null>(null);
  const [wrist, setWrist] = useState<string | null>(null); // the wrist camera looked through, or null
  const [wrists, setWrists] = useState<string[]>([]);
  const [full, setFull] = useState(false);
  const [glError, setGlError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [lost, setLost] = useState(false);

  useEffect(() => { scene.loadModel(); }, []);
  useEffect(() => scene.watchUnits(), []);

  // WHY layout effect: React runs a useEffect cleanup after the page's DOM is already detached. OrbitControls.dispose
  // removes its document keydown listener from canvas.getRootNode(), which is then the detached subtree, not the
  // document, so the listener stayed and kept every disposed engine alive (about 1 MB per open). A layout effect's
  // cleanup runs before React removes the nodes.
  useLayoutEffect(() => {
    if (!model || !host.current || !labels.current) return;
    let e: Engine;
    try {
      e = new Engine(host.current, model, mode ? { ...scene.snap.settings, mode, ...MODE_LAYERS[mode] } : scene.snap.settings, {
        compact,
        labels: labels.current,
        onProgress: (a, b) => setProgress([a, b]),
        onCameraEdit: (k, p) => scene.setCamera(k, p),
        onSelectCamera: (k) => scene.edit(k),
        onWristView: setWrist,
        onWrists: setWrists,
        onLoadError: setLoadError,
        onContextLost: setLost,
      });
    } catch (err) {
      setGlError(`This browser could not start WebGL: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    engine.current = e;
    const points = new PointLayers(e); // disposed with the engine (Engine.onDispose)
    live++;
    e.setTheme(readTheme(host.current));
    e.setEditing(scene.snap.editing);
    e.setUnits(scene.snap.units);
    const g = globalThis as { __phiScene?: { engine: Engine; points: PointLayers; live: () => number }; __phiSceneLast?: unknown };
    g.__phiScene = { engine: e, points, live: () => live };
    return () => {
      const left = e.dispose();
      live--;
      engine.current = null;
      g.__phiSceneLast = { left, live };
      if (g.__phiScene?.engine === e) delete g.__phiScene;
    };
  }, [model, compact]);

  useEffect(() => { engine.current?.setSettings(settings); }, [settings]);
  useEffect(() => { engine.current?.setEditing(editing); }, [editing]);
  useEffect(() => { engine.current?.setUnits(units); }, [units]);
  useEffect(() => { if (engine.current && host.current) engine.current.setTheme(readTheme(host.current)); }, [theme]);

  // Riding a wrist camera belongs to Cameras mode; leaving the mode goes back to orbiting.
  useEffect(() => { if (settings.mode !== "cameras") engine.current?.frame("home"); }, [settings.mode]);

  useEffect(() => {
    const on = () => setFull(document.fullscreenElement === stage.current);
    document.addEventListener("fullscreenchange", on);
    return () => document.removeEventListener("fullscreenchange", on);
  }, []);

  const loading = !error && !glError && !loadError && (!model || !progress || progress[0] < progress[1]);
  const autoRotate = () => {
    if (!settings.autoRotate) engine.current?.setAutoRotate(true); // start now, not after the idle wait
    scene.update({ autoRotate: !settings.autoRotate });
  };
  const fullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.current?.requestFullscreen();
  };

  return (
    <div ref={stage} className={`scene-stage ${compact ? "is-compact" : ""}`}>
      <div ref={host} className="scene-host" />
      <div ref={labels} className="scene-labels" aria-hidden />

      <div className="scene-left">
        {full && (
          // WHY: full screen covers the top bar and its Stop. This one calls the same studio.stop().
          <button type="button" className="btn stop-btn is-live" onClick={() => studio.stop()} aria-keyshortcuts="Escape">
            <OctagonX aria-hidden /> Stop
          </button>
        )}
        {!mode && <Segmented label="What the 3D view shows" value={settings.mode} options={MODES} onChange={(m) => scene.setMode(m)} />}
      </div>

      <div className="scene-tools" role="toolbar" aria-label="3D view">
        {settings.mode === "cameras" ? (
          <>
            {wrists.map((k) => (
              <button key={k} type="button" className={`btn btn-sm btn-ghost ${wrist === k ? "is-on" : ""}`} aria-pressed={wrist === k}
                onClick={() => (wrist === k ? engine.current?.frame("home") : engine.current?.rideWrist(k))}
                title={`Look through the ${wristName(k).toLowerCase()}`}>{wristShort(k)}</button>
            ))}
            <a className="btn btn-sm btn-ghost" href="#/align" title="Put each camera back where the training data was recorded">
              <ScanSearch aria-hidden />Align
            </a>
          </>
        ) : PRESETS.map((p) => (
          <button key={p} type="button" className="btn btn-sm btn-ghost" onClick={() => engine.current?.frame(p)}>{label(p)}</button>
        ))}
        <span className="scene-tools-gap" />
        <button type="button" className="btn btn-sm btn-ghost btn-icon" onClick={() => engine.current?.fit()} title="Fit the arms in view" aria-label="Fit the arms in view"><Scan /></button>
        <button type="button" className={`btn btn-sm btn-ghost btn-icon ${settings.autoRotate ? "is-on" : ""}`} aria-pressed={settings.autoRotate}
          onClick={autoRotate} title="Turn slowly after 6 s without input" aria-label="Turn slowly when idle"><Rotate3d /></button>
        <button type="button" className="btn btn-sm btn-ghost btn-icon" onClick={fullscreen}
          title={full ? "Leave full screen" : "Full screen"} aria-label={full ? "Leave full screen" : "Full screen"}>{full ? <Minimize2 /> : <Maximize2 />}</button>
        {onSettings && (
          <button type="button" className="btn btn-sm btn-ghost btn-icon" onClick={onSettings} title="Settings" aria-label="3D view settings"><SlidersHorizontal /></button>
        )}
      </div>

      {wrist && (
        <div className="scene-banner">
          <button type="button" className="btn btn-sm" onClick={() => engine.current?.frame("home")}><ArrowLeft />Back to orbit</button>
          <span title="Mount pose from the CAD; vertical field of view estimated, not measured.">{wristName(wrist)}</span>
        </div>
      )}

      {/* WHY a tooltip, not a footnote: the caveats matter, but three sentences under the arms crowd them. */}
      <span className="scene-info" tabIndex={0}
        title={`${LEGEND[settings.mode](policy)} Joint angles map one to one onto the model (directions checked on a physical arm, 2026-10-09).`}>
        <Info aria-hidden />
      </span>

      {(error || glError) && <div className="scene-overlay"><p className="scene-error">{glError ?? error}</p></div>}
      {loadError && !error && !glError && (
        <div className="scene-overlay">
          <div className="scene-progress">
            <p className="scene-error">{loadError} Check that Studio is still running.</p>
            <button type="button" className="btn btn-sm" onClick={() => engine.current?.retryLoad()}><RotateCw aria-hidden /> Try again</button>
          </div>
        </div>
      )}
      {lost && (
        <div className="scene-overlay">
          <p className="scene-error">The browser took the graphics context away from this view. It comes back on its own when the browser allows; if it does not, reload the window. Stop and Esc still work.</p>
        </div>
      )}
      {loading && (
        <div className="scene-overlay">
          <div className="scene-progress">
            <span>Loading the arm model{progress && progress[1] ? `, ${mb(progress[0])} of ${mb(progress[1])} MB` : ""}</span>
            <div className="scene-bar"><div style={{ width: `${progress && progress[1] ? (100 * progress[0]) / progress[1] : 0}%` }} /></div>
          </div>
        </div>
      )}
    </div>
  );
}
