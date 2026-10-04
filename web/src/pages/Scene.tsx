// The 3D view page: the followers at full size, a joint readout, and the view's settings.
import { RotateCcw } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { label } from "../lib/labels";
import {
  type CamPose, cameraPose, followerSlots, limitState, PRINT, type PrintColour, scene, SPACING, toRadians, useScene,
  type Vec3,
} from "../lib/scene";
import { useStudio } from "../lib/studio";
import { SceneFallback } from "../scene/ViewSwitch";

const Viewer = lazy(() => import("../scene/Viewer"));
const deg = (rad: number) => (rad * 180) / Math.PI;

export function Scene() {
  useEffect(() => { scene.loadModel(); }, []);
  return (
    <div className="page scene-page">
      <div className="work-grid">
        <div className="col">
          <section className="panel scene-panel">
            <div className="scene-panel-body is-full">
              <Suspense fallback={<SceneFallback />}><Viewer /></Suspense>
            </div>
          </section>
          <JointReadout />
        </div>
        <div className="col rail">
          <LayoutPanel />
          <ShowPanel />
          <CamerasPanel />
          <ModelPanel />
        </div>
      </div>
    </div>
  );
}

/** Each follower's joints as read, as drawn, and where that sits in the model's range. */
function JointReadout() {
  const model = useScene((s) => s.model);
  const telemetry = useStudio((s) => s.telemetry);
  const rigArms = useStudio((s) => s.rig?.arms);
  const slots = followerSlots(telemetry, rigArms);
  if (!model) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Joints</h2>
        <span className="panel-sub">Warm: within 5% of the model's range end. Red: past it.</span>
      </div>
      {!slots.length && <p className="empty">No follower on this rig.</p>}
      {slots.map((s) => {
        const pos = telemetry?.arms[s.name]?.pos;
        return (
          <table key={s.name} className="table scene-joints">
            <thead>
              <tr><th>{label(s.name)}</th><th className="r">Reading</th><th className="r">In the model</th><th>Model range</th></tr>
            </thead>
            <tbody>
              {model.joint_order.map((j) => {
                const v = pos?.[j];
                const has = typeof v === "number" && Number.isFinite(v);
                const rad = has ? toRadians(model, j, v) : 0;
                const lim = has ? limitState(model, j, rad) : null;
                const r = model.joints[j].range;
                return (
                  <tr key={j} className={lim ? `is-${lim.state}` : ""}>
                    <td>{label(j)}</td>
                    <td className="r num">{has ? (j === "gripper" ? `${v.toFixed(0)} of 100` : `${v.toFixed(1)}°`) : "no reading"}</td>
                    <td className="r num">{has ? `${deg(rad).toFixed(1)}°` : ""}</td>
                    <td>
                      <div className="scene-range" title={r ? `${deg(r[0]).toFixed(0)}° to ${deg(r[1]).toFixed(0)}°` : "no limit"}>
                        {lim && <i style={{ left: `${Math.min(100, Math.max(0, lim.frac * 100))}%` }} />}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        );
      })}
    </section>
  );
}

function LayoutPanel() {
  const model = useScene((s) => s.model);
  const settings = useScene((s) => s.settings);
  const telemetry = useStudio((s) => s.telemetry);
  const rigArms = useStudio((s) => s.rig?.arms);
  const slots = followerSlots(telemetry, rigArms);
  if (!model) return null;
  const spacing = settings.spacing ?? model.pair_spacing_m;
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="panel-title">Layout</h2></div>
      <div className="panel-body scene-form">
        {slots.length < 2 ? (
          <p className="field-hint">One follower, drawn with its base at the origin.</p>
        ) : (
          <>
            <label className="field">
              <span className="field-label">Distance between the arm bases: {Math.round(spacing * 100)} cm</span>
              <input type="range" className="scene-slider" min={SPACING.min * 100} max={SPACING.max * 100} step={1}
                value={Math.round(spacing * 100)} onChange={(e) => scene.update({ spacing: Number(e.target.value) / 100 })} />
              <span className="field-hint">
                {settings.spacing === null
                  ? `Default ${Math.round(model.pair_spacing_m * 100)} cm, from phi's simulation twin. Not measured on your desk.`
                  : "Set by you. Left arm on the left as seen from behind the arms."}
              </span>
            </label>
            {settings.spacing !== null && (
              <button type="button" className="btn btn-sm" onClick={() => scene.update({ spacing: null })}><RotateCcw />Use the default</button>
            )}
            <label className="field">
              <span className="field-label">Wrist camera on</span>
              <select className="select" value={settings.wristArm ?? slots[0].name} onChange={(e) => scene.update({ wristArm: e.target.value })}>
                {slots.map((s) => <option key={s.name} value={s.name}>{label(s.name)}</option>)}
              </select>
            </label>
          </>
        )}
      </div>
    </section>
  );
}

function Check({ on, text, hint, set }: { on: boolean; text: string; hint?: string; set: (v: boolean) => void }) {
  return (
    <label className="scene-check">
      <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} />
      <span>{text}{hint && <span className="field-hint"> {hint}</span>}</span>
    </label>
  );
}

function ShowPanel() {
  const s = useScene((x) => x.settings);
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="panel-title">Show</h2></div>
      <div className="panel-body scene-form">
        <Check on={s.ghost} text="Leader as a ghost" hint="(the policy's target while a policy runs)" set={(v) => scene.update({ ghost: v })} />
        <Check on={s.trail} text="Gripper trail" hint="(last 3 s)" set={(v) => scene.update({ trail: v })} />
        <Check on={s.frustums} text="Cameras" set={(v) => scene.update({ frustums: v })} />
        <Check on={s.video} text="Live picture on each camera" set={(v) => scene.update({ video: v })} />
        <Check on={s.autoRotate} text="Turn slowly when idle" set={(v) => scene.update({ autoRotate: v })} />
        <label className="field">
          <span className="field-label">Printed parts</span>
          <select className="select" value={s.print} onChange={(e) => scene.update({ print: e.target.value as PrintColour })}>
            {(Object.keys(PRINT) as PrintColour[]).map((k) => <option key={k} value={k}>{PRINT[k].name}</option>)}
          </select>
        </label>
      </div>
    </section>
  );
}

/** A number field that keeps what is typed until it parses, so "-" on the way to "-5" is not thrown away. */
function Num({ value, onCommit, label: text }: { value: number; onCommit: (v: number) => void; label: string }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input className="input num" inputMode="decimal" aria-label={text} value={draft ?? String(value)}
      onChange={(e) => { setDraft(e.target.value); const v = Number(e.target.value); if (e.target.value.trim() && Number.isFinite(v)) onCommit(v); }}
      onBlur={() => setDraft(null)} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
  );
}

const cm = (m: number) => Math.round(m * 1000) / 10;

function CamerasPanel() {
  const model = useScene((s) => s.model);
  const settings = useScene((s) => s.settings);
  const editing = useScene((s) => s.editing);
  const live = useStudio((s) => s.cameras);
  if (!model) return null;
  const wristKey = model.wrist_camera.key;
  const keys = [...new Set([...Object.keys(model.camera_defaults), ...Object.keys(live)])].filter((k) => k !== wristKey).sort();
  const set = (k: string, p: CamPose, field: "pos" | "target", i: number, cmValue: number) => {
    const v = [...p[field]] as Vec3;
    v[i] = cmValue / 100;
    scene.setCamera(k, { ...p, [field]: v });
  };
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="panel-title">Cameras</h2></div>
      <div className="panel-body scene-form">
        <p className="field-hint">
          Studio does not know where your cameras stand. These poses are placed by you, not measured, and only move
          the picture in this view. Centimetres, x forward, y left, z up, from the middle of the arm bases.
        </p>
        {keys.map((k) => {
          const p = cameraPose(model, settings, k);
          const mine = k in settings.cameras;
          const open = editing === k;
          return (
            <div key={k} className={`scene-cam ${open ? "is-open" : ""}`}>
              <div className="scene-cam-head">
                <span className="strong">{label(k)}</span>
                <span className={`badge ${mine ? "tone-info" : "tone-neutral"}`}>{mine ? "Placed by you" : "Starting guess"}</span>
                <button type="button" className="btn btn-sm" onClick={() => scene.edit(open ? null : k)}>{open ? "Done" : "Move"}</button>
              </div>
              {open && (
                <>
                  <p className="field-hint">Drag the arrows in the view, or type.</p>
                  {(["pos", "target"] as const).map((f) => (
                    <div key={f} className="scene-xyz">
                      <span className="field-label">{f === "pos" ? "Position" : "Looks at"}</span>
                      {[0, 1, 2].map((i) => (
                        <Num key={i} label={`${label(k)} ${f === "pos" ? "position" : "target"} ${"xyz"[i]} in cm`} value={cm(p[f][i])}
                          onCommit={(v) => set(k, p, f, i, v)} />
                      ))}
                    </div>
                  ))}
                  <div className="scene-xyz">
                    <span className="field-label">Field of view</span>
                    <Num label={`${label(k)} vertical field of view in degrees`} value={p.fovy_deg}
                      onCommit={(v) => scene.setCamera(k, { ...p, fovy_deg: Math.min(150, Math.max(5, v)) })} />
                    <span className="field-hint">degrees, vertical</span>
                  </div>
                  {mine && (
                    <button type="button" className="btn btn-sm" onClick={() => scene.setCamera(k, null)}><RotateCcw />Back to the starting guess</button>
                  )}
                </>
              )}
            </div>
          );
        })}
        <div className="scene-cam">
          <div className="scene-cam-head">
            <span className="strong">{label(wristKey)}</span>
            <span className="badge tone-neutral">On the gripper</span>
          </div>
          <p className="field-hint">{model.wrist_camera.mount} Field of view {model.wrist_camera.fovy_deg}°: {model.wrist_camera.fov.toLowerCase()}</p>
        </div>
      </div>
    </section>
  );
}

function ModelPanel() {
  const model = useScene((s) => s.model);
  if (!model) return null;
  const g = model.mapping.gripper;
  const arm = model.mapping.shoulder_pan;
  const src = model.source;
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="panel-title">Model</h2></div>
      <div className="panel-body scene-form">
        <p>
          SO-101 from TheRobotStudio's <a href={src.repo} target="_blank" rel="noreferrer">{src.repo.split("/").pop()}</a> repository,
          commit <span className="ident">{src.commit.slice(0, 7)}</span>, {src.license}. Meshes unmodified.
        </p>
        <div>
          <span className="badge tone-warn">Mapping assumed</span>
          <p className="field-hint">Not checked on a physical arm. If a drawn joint disagrees with yours, this is the first suspect.</p>
        </div>
        <ul className="scene-list">
          <li>Body joints: the reading in degrees is the model angle{arm && arm.offset === 0 ? ", no offset" : ""}.</li>
          {g && <li>Gripper: 0 to 100 is spread evenly over the model's range, {deg(g.offset).toFixed(0)}° to {deg(g.offset + 100 * g.scale).toFixed(0)}°. A placeholder in phi too.</li>}
        </ul>
        <p className="field-hint ident">{arm?.source}<br />{g?.source}</p>
      </div>
    </section>
  );
}
