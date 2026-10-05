// The 3D view on Teleoperate and Run policy: a panel with a Cameras / 3D / Both switch, remembered per page.
// Eager, and small: the viewer and three.js load only when the switch shows 3D.
import { lazy, Suspense } from "react";
import { Boundary } from "../components/Boundary";
import { Cameras } from "../components/Cameras";
import { scene, useScene, type ViewMode } from "../lib/scene";
import "../styles/scene.css";

const Viewer = lazy(() => import("./Viewer"));

const MODES: { mode: ViewMode; text: string }[] = [
  { mode: "cameras", text: "Cameras" }, { mode: "3d", text: "3D" }, { mode: "both", text: "Both" },
];

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { mode: T; text: string }[]; onChange: (v: T) => void; label: string;
}) {
  return (
    <div className="scene-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.mode} type="button" aria-pressed={value === o.mode} onClick={() => onChange(o.mode)}>{o.text}</button>
      ))}
    </div>
  );
}

export function SceneFallback() {
  return <div className="scene-stage is-compact"><div className="scene-overlay"><p className="scene-note">Loading the 3D view</p></div></div>;
}

export function ViewSwitch({ page }: { page: string }) {
  const mode = useScene((s) => s.settings.view[page] ?? "both");
  return (
    <>
      <section className="panel scene-panel">
        <div className="panel-head">
          <div>
            <h2 className="panel-title">3D view</h2>
            {mode === "cameras" && <span className="panel-sub">Hidden on this page</span>}
          </div>
          <Segmented label="Show" value={mode} options={MODES} onChange={(m) => scene.setView(page, m)} />
        </div>
        {mode !== "cameras" && (
          <div className="scene-panel-body">
            {/* WHY a boundary here: without it a crash in the 3D view, or its chunk gone after a rebuild, takes the
                whole page down, and with it the torque, start and disconnect controls. Cameras stay too. */}
            <Boundary what="The 3D view">
              <Suspense fallback={<SceneFallback />}><Viewer compact /></Suspense>
            </Boundary>
          </div>
        )}
      </section>
      {mode !== "3d" && <Cameras />}
    </>
  );
}
