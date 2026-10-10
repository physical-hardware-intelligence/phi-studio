import { ArrowLeft, ArrowRight, Camera, Check, CircleAlert, Plug, Search, Unplug, Wand2 } from "lucide-react";
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { Boundary } from "../components/Boundary";
import { ArmsGlyph } from "../components/data/bits";
import { Notices } from "../components/Notices";
import { RigSwitch } from "../components/RigSwitch";
import { configIds, defaultIds, idChoices, initialChoice, seedPorts } from "../lib/onboard";
import { advance, portKey, type Wizard } from "../lib/portfinder";
import { SLOTS, refreshRig, shortPort, slotLabel, slug, useRig, type Layout, type RigStatus } from "../lib/rig";
import { go, studio, useStudio } from "../lib/studio";

const Viewer = lazy(() => import("../scene/Viewer"));

type Step = "rig" | "arms" | "calibration" | "drive" | "cameras" | "ready";
const STEPS: { id: Step; label: string }[] = [
  { id: "rig", label: "Rig" }, { id: "arms", label: "Arms" }, { id: "calibration", label: "Calibration" },
  { id: "drive", label: "Test drive" }, { id: "cameras", label: "Cameras" }, { id: "ready", label: "Ready" },
];
const REC_FPS = 30; // recordings take the control loop's rate (worker.py loop_hz, 30 Hz)
const ROLES: Record<Layout, string[]> = { single: ["top", "front", "wrist"], bimanual: ["top", "front", "left_wrist", "right_wrist"] };

interface Cam { source: number | string; width?: number; height?: number; fps?: number; fourcc?: string | null }
interface Draft {
  name: string; layout: Layout; ports: Record<string, string>;
  ids: "existing" | "new" | "unset"; existing: { follower: string; leader: string } | null;
}
interface Probe { source: number | string; ok: boolean; width?: number; height?: number; fps?: number; fourcc?: string | null; picture?: string | null; error?: string | null }

/** Where onboarding starts: the config if there is one, else ports.local.sh (lib/onboard.ts). */
function initial(rig: RigStatus): Draft {
  const layout: Layout = rig.layout ?? (Object.keys(rig.hint.ports).some((k) => k.startsWith("left_")) ? "bimanual" : "single");
  return withLayout({ name: rig.name ?? "", layout, ports: {}, ids: "new", existing: null }, rig, layout);
}

/** A draft for `layout`: its ports and calibration ids seeded again, as a layout change needs. */
function withLayout(d: Draft, rig: RigStatus, layout: Layout): Draft {
  return { ...d, layout, ports: seedPorts(rig, layout, SLOTS[layout]), existing: defaultIds(rig, layout), ids: initialChoice(rig, layout) };
}

export function Onboard() {
  const rig = useRig();
  const [step, setStep] = useState<Step>("rig");
  const [d, setD] = useState<Draft | null>(null);
  useEffect(() => { refreshRig(); }, []);
  useEffect(() => { if (rig && !d) setD(initial(rig)); }, [rig, d]);
  if (!rig || !d) return <div className="page"><div className="empty">Loading</div></div>;
  const i = STEPS.findIndex((s) => s.id === step);
  const next = () => setStep(STEPS[Math.min(STEPS.length - 1, i + 1)].id);
  const back = () => setStep(STEPS[Math.max(0, i - 1)].id);
  return (
    <div className="page onb">
      <Notices />
      <ol className="onb-steps" aria-label="Onboarding">
        {STEPS.map((s, k) => (
          <li key={s.id} className={k < i ? "is-done" : k === i ? "is-current" : ""}>
            <button className="onb-step" onClick={() => k < i && setStep(s.id)} disabled={k >= i}>
              <span className="onb-dot">{k < i ? <Check aria-hidden /> : k + 1}</span>{s.label}
            </button>
          </li>
        ))}
      </ol>
      <section className="panel onb-card">
        {step === "rig" && <RigStep d={d} set={setD} rig={rig} onNext={next} />}
        {step === "arms" && <ArmsStep d={d} set={setD} onBack={back} onNext={next} />}
        {step === "calibration" && <CalibrationStep d={d} set={setD} rig={rig} onBack={back} onNext={next} />}
        {step === "drive" && <DriveStep onBack={back} onNext={next} />}
        {step === "cameras" && <CamerasStep d={d} rig={rig} onBack={back} onNext={next} />}
        {step === "ready" && <ReadyStep d={d} rig={rig} />}
      </section>
    </div>
  );
}

function Foot({ onBack, onNext, next = "Next", disabled, busy, extra }: {
  onBack?: () => void; onNext: () => void; next?: string; disabled?: boolean; busy?: boolean; extra?: React.ReactNode;
}) {
  return (
    <div className="onb-foot">
      {onBack && <button className="btn btn-ghost" onClick={onBack}><ArrowLeft aria-hidden />Back</button>}
      <span className="grow" />
      {extra}
      <button className="btn btn-primary" onClick={onNext} disabled={disabled || busy}>{busy ? "Saving" : next}{!busy && <ArrowRight aria-hidden />}</button>
    </div>
  );
}

function RigStep({ d, set, rig, onNext }: { d: Draft; set: (d: Draft) => void; rig: RigStatus; onNext: () => void }) {
  const ok = d.name.trim().length > 0;
  return (
    <>
      <h2 className="onb-title">Name your rig</h2>
      <input className="input onb-name" autoFocus placeholder="Bench A" value={d.name} maxLength={60}
        onChange={(e) => set({ ...d, name: e.target.value })} onKeyDown={(e) => { if (e.key === "Enter" && ok) onNext(); }} />
      <div className="onb-layouts" role="radiogroup" aria-label="Layout">
        {(["single", "bimanual"] as Layout[]).map((l) => (
          <button key={l} role="radio" aria-checked={d.layout === l} className={`onb-layout ${d.layout === l ? "is-on" : ""}`}
            onClick={() => l !== d.layout && set(withLayout(d, rig, l))}>
            <ArmsGlyph bimanual={l === "bimanual"} />
            <span className="strong">{l === "single" ? "Single arm" : "Bimanual"}</span>
            <span className="faint t-sm">{l === "single" ? "1 leader · 1 follower" : "2 leaders · 2 followers"}</span>
          </button>
        ))}
      </div>
      <Foot onNext={onNext} disabled={!ok} />
    </>
  );
}

function usePorts(active: boolean): string[] | null {
  const [ports, setPorts] = useState<string[] | null>(null);
  useEffect(() => {
    if (!active) return;
    const off = studio.onMessage("setup_ports", (m) => setPorts((m.ports as { name: string }[]).map((p) => portKey(p.name))));
    studio.send({ cmd: "setup_ports" });
    const t = window.setInterval(() => studio.send({ cmd: "setup_ports" }), 700);
    return () => { off(); window.clearInterval(t); };
  }, [active]);
  return ports;
}

function ArmsStep({ d, set, onBack, onNext }: { d: Draft; set: (d: Draft) => void; onBack: () => void; onNext: () => void }) {
  const ports = usePorts(true);
  const [wiz, setWiz] = useState<Wizard | null>(null);
  const slots = SLOTS[d.layout];
  useEffect(() => {
    if (!wiz || !ports) return;
    const w = advance(wiz, ports);
    if (w.i >= w.arms.length) {
      set({ ...d, ports: { ...d.ports, [w.arms[0]]: w.found[w.arms[0]] } });
      setWiz(null);
    } else if (w !== wiz) setWiz(w);
  }, [ports]); // eslint-disable-line react-hooks/exhaustive-deps
  const find = (slot: string) => ports && setWiz({ arms: [slot], i: 0, phase: "unplug", base: ports, extra: [], gone: null, found: {}, problem: null, saving: false });
  return (
    <>
      <h2 className="onb-title">Find each arm</h2>
      <p className="onb-sub">Unplug an arm's USB when asked, then plug it back in.</p>
      <div className={`onb-arms n-${slots.length}`}>
        {slots.map((slot) => {
          const p = d.ports[slot];
          const here = p && ports?.includes(portKey(p));
          const finding = wiz?.arms[0] === slot;
          return (
            <div key={slot} className={`onb-arm ${finding ? "is-finding" : ""}`}>
              <span className="strong">{slotLabel(slot)}</span>
              {finding ? (
                <div className="onb-find">
                  {wiz.phase === "unplug" ? <><Unplug aria-hidden className="pulse" />Unplug it</> : <><Plug aria-hidden className="pulse" />Plug it back in</>}
                  {wiz.problem && <span className="warn-text t-sm">{wiz.problem}</span>}
                  <button className="link-btn t-sm" onClick={() => setWiz(null)}>Cancel</button>
                </div>
              ) : (
                <>
                  <span className={`mono ${p ? (here ? "ok-text" : "warn-text") : "faint"}`} title={p || undefined}>
                    {p ? <>{here ? <Check aria-hidden className="ico-inline" /> : <CircleAlert aria-hidden className="ico-inline" />}{shortPort(p)}</> : "not found"}
                  </span>
                  <span className="onb-arm-actions">
                    <button className="btn btn-sm" onClick={() => find(slot)} disabled={!ports || !!wiz}><Search aria-hidden />Find</button>
                    {ports && ports.length > 0 && (
                      <select className="select select-sm" value={p ?? ""} aria-label={`${slotLabel(slot)} port`}
                        onChange={(e) => set({ ...d, ports: { ...d.ports, [slot]: e.target.value } })}>
                        <option value="">Choose</option>
                        {[...new Set([...(p ? [portKey(p)] : []), ...ports])].map((x) => <option key={x} value={x}>{shortPort(x)}</option>)}
                      </select>
                    )}
                  </span>
                </>
              )}
            </div>
          );
        })}
      </div>
      <p className="faint t-sm">{ports === null ? "Looking for USB arms" : ports.length ? `${ports.length} USB ${ports.length === 1 ? "device" : "devices"} plugged in` : "No USB arms plugged in. You can find them later."}</p>
      <Foot onBack={onBack} onNext={onNext} />
    </>
  );
}

function CalibrationStep({ d, set, rig, onBack, onNext }: { d: Draft; set: (d: Draft) => void; rig: RigStatus; onBack: () => void; onNext: () => void }) {
  const control = useStudio((s) => s.control);
  const [busy, setBusy] = useState(false);
  const has = (role: "follower" | "leader", id: string) => rig.known[role].some((k) => k.id === id);
  const fresh = { follower: `${slug(d.name)}_follower`, leader: `${slug(d.name)}_leader` };
  const choices = idChoices(rig, d.layout);
  const picked = d.existing ?? { follower: "", leader: "" };
  const ids = d.ids === "existing" ? picked : fresh;
  const ready = d.ids === "new" || (d.ids === "existing" && !!picked.follower && !!picked.leader);
  const cfg = configIds(rig);
  const files = (x: { follower: string; leader: string }) => SLOTS[d.layout].map((slot) => {
    const role = slot.endsWith("leader") ? "leader" : "follower";
    const side = slot.startsWith("left_") ? "_left" : slot.startsWith("right_") ? "_right" : "";
    return { slot, ok: !!x[role] && has(role, `${x[role]}${side}`) };
  });
  const save = () => {
    setBusy(true);
    const off1 = studio.onMessage("rig_saved", () => { off1(); off2(); setBusy(false); onNext(); });
    const off2 = studio.onMessage("error", (m) => { if (m.cmd === "rig_write") { off1(); off2(); setBusy(false); } });
    const ports = Object.fromEntries(Object.entries(d.ports).filter(([k, v]) => v && SLOTS[d.layout].includes(k)));
    if (!studio.send({ cmd: "rig_write", rig: { name: d.name, layout: d.layout, ids, ports } })) { off1(); off2(); setBusy(false); }
  };
  const Chips = ({ x }: { x: { follower: string; leader: string } }) => (
    <span className="onb-chips">
      {files(x).map((f) => <span key={f.slot} className={`chip ${f.ok ? "sev-ok" : "sev-info"}`}>{f.ok ? "✓" : "–"} {slotLabel(f.slot)}</span>)}
    </span>
  );
  const pick = (role: "follower" | "leader", id: string) => set({ ...d, ids: "existing", existing: { ...picked, [role]: id } });
  return (
    <>
      <h2 className="onb-title">Calibration</h2>
      {d.ids === "unset" && (
        <p className="onb-warn">
          Studio found no calibration files in <span className="mono">{rig.calibration_root}</span>, yet this rig uses{" "}
          <span className="mono">{cfg.follower} · {cfg.leader}</span>. If the arms are calibrated, Studio is looking in the wrong
          folder: check HF_LEROBOT_CALIBRATION. Pick New only to calibrate the arms again.
        </p>
      )}
      <div className="onb-choices" role="radiogroup" aria-label="Calibration">
        {choices.follower.length > 0 && (
          <button role="radio" aria-checked={d.ids === "existing"} className={`onb-choice ${d.ids === "existing" ? "is-on" : ""}`}
            onClick={() => set({ ...d, ids: "existing" })}>
            <span className="strong">Use a calibration on this Mac</span>
            <span className="mono faint">{picked.follower && picked.leader ? `${picked.follower} · ${picked.leader}` : "Pick one per arm below"}</span>
            <Chips x={picked} />
          </button>
        )}
        <button role="radio" aria-checked={d.ids === "new"} className={`onb-choice ${d.ids === "new" ? "is-on" : ""}`}
          onClick={() => set({ ...d, ids: "new" })}>
          <span className="strong">New calibration for this rig</span>
          <span className="mono faint">{fresh.follower} · {fresh.leader}</span>
          <Chips x={fresh} />
        </button>
      </div>
      {d.ids === "existing" && (
        <div className="onb-picks">
          {(["follower", "leader"] as const).map((role) => (
            <label key={role} className="onb-pick">
              <span>{role === "follower" ? "Follower" : "Leader"}</span>
              <select className="select" value={picked[role]} onChange={(e) => pick(role, e.target.value)}>
                {!picked[role] && <option value="">Choose</option>}
                {choices[role].map((id) => <option key={id} value={id}>{id}</option>)}
              </select>
            </label>
          ))}
          <p className="faint t-sm">The file is chosen by this name and the arm by its port; they must be the same arm. After saving, Studio checks each arm's servos against its file.</p>
        </div>
      )}
      <Foot onBack={onBack} onNext={save} next="Save rig" busy={busy} disabled={!control || !ready}
        extra={<>
          <button className="btn" onClick={() => go("calibrate")} title="Every arm finds its own end stops"><Wand2 aria-hidden />Auto-calibrate</button>
          <button className="btn btn-ghost" onClick={() => go("calibrate")}>By hand</button>
        </>} />
      {!control && <p className="faint t-sm">Take control (top right) to save.</p>}
    </>
  );
}

function DriveStep({ onBack, onNext }: { onBack: () => void; onNext: () => void }) {
  const mock = useStudio((s) => s.mock);
  return (
    <>
      <h2 className="onb-title">Test drive</h2>
      <p className="onb-sub">Move a leader. Its follower should follow{mock ? " (simulated arms)" : ""}.</p>
      {/* WHY here: the test drive is how a person checks their own arms; on simulated ones it proves
          nothing about the rig just saved (2026-10-09, the switch only showed on Ready). */}
      {mock && <div className="onb-real"><span>These are simulated arms. Switch to the rig you just saved to drive yours.</span><RigSwitch primary /></div>}
      <ActionBar activity="teleop" />
      <div className="onb-view">
        <Boundary what="The 3D view"><Suspense fallback={<div className="empty">Loading the 3D view</div>}><Viewer compact /></Suspense></Boundary>
      </div>
      <Foot onBack={onBack} onNext={onNext} next="Looks right" />
    </>
  );
}

function CamerasStep({ d, rig, onBack, onNext }: { d: Draft; rig: RigStatus; onBack: () => void; onNext: () => void }) {
  const control = useStudio((s) => s.control);
  const [probing, setProbing] = useState(false);
  const [found, setFound] = useState<Probe[] | null>(null);
  const [roles, setRoles] = useState<Record<string, number | string>>(() =>
    Object.fromEntries(rig.cameras.filter((c) => c.source !== null).map((c) => [c.key, c.source as number | string])));
  const [busy, setBusy] = useState(false);
  const meta = useRef<Record<string, Probe>>({});
  useEffect(() => studio.onMessage("setup_cameras", (m) => {
    setProbing(!!m.probing);
    if (!m.probing && m.cameras) {
      const ok = (m.cameras as Probe[]).filter((c) => c.ok);
      ok.forEach((c) => { meta.current[String(c.source)] = c; });
      setFound(ok);
    }
  }), []);
  const role = useMemo(() => Object.fromEntries(Object.entries(roles).map(([r, s]) => [String(s), r])), [roles]);
  const assign = (src: number | string, r: string) => setRoles((x) => {
    const out = Object.fromEntries(Object.entries(x).filter(([k, v]) => k !== r && String(v) !== String(src)));
    return x[r] !== undefined && String(x[r]) === String(src) ? out : { ...out, [r]: src };
  });
  const save = () => {
    const cams: Record<string, Cam> = {};
    for (const [r, src] of Object.entries(roles)) {
      const m = meta.current[String(src)];
      const old = rig.cameras.find((c) => c.key === r);
      cams[r] = { source: src, width: m?.width ?? old?.width, height: m?.height ?? old?.height, fps: m?.fps ?? old?.fps, fourcc: m ? m.fourcc : undefined };
    }
    setBusy(true);
    const cur = configIds(rig);
    const ids = { follower: cur.follower ?? `${slug(d.name)}_follower`, leader: cur.leader ?? `${slug(d.name)}_leader` };
    const ports = Object.fromEntries(rig.arms.filter((a) => a.port).map((a) => [a.key, a.port as string]));
    const off1 = studio.onMessage("rig_saved", () => { off1(); off2(); setBusy(false); onNext(); });
    const off2 = studio.onMessage("error", (m) => { if (m.cmd === "rig_write") { off1(); off2(); setBusy(false); } });
    if (!studio.send({ cmd: "rig_write", rig: { name: d.name, layout: d.layout, ids, ports, cameras: cams } })) { off1(); off2(); setBusy(false); }
  };
  return (
    <>
      <h2 className="onb-title">Cameras</h2>
      <p className="onb-sub">Pick what each camera sees.</p>
      <div className="onb-cam-actions">
        <button className="btn" onClick={() => { setProbing(true); studio.send({ cmd: "setup_cameras_probe" }); }} disabled={!control || probing}>
          <Camera aria-hidden />{probing ? "Looking" : found ? "Look again" : "Find cameras"}</button>
      </div>
      {found && !found.length && <p className="faint">No camera answered.</p>}
      <div className="onb-cams">
        {(found ?? []).map((c) => (
          <figure key={String(c.source)} className={`onb-cam ${role[String(c.source)] ? "is-set" : ""}`}>
            {c.picture ? <img src={c.picture} alt={`camera ${c.source}`} /> : <div className="onb-cam-empty"><Camera aria-hidden /></div>}
            <figcaption>
              <span className="mono faint">#{String(c.source)} · {c.width}×{c.height}{c.fps ? ` · ${c.fps} fps` : ""}</span>
              {/* WHY: 2026-10-09 a side camera ran at 5 fps; at 30 Hz it would repeat each frame six times */}
              {c.fps !== undefined && c.fps !== null && c.fps < REC_FPS && (
                <span className="chip sev-warn">{c.fps} fps, recordings run at {REC_FPS}: check its light and USB port</span>
              )}
              <span className="onb-roles">
                {ROLES[d.layout].map((r) => {
                  const on = String(roles[r]) === String(c.source);
                  return <button key={r} aria-pressed={on} className={`seg-btn ${on ? "is-on" : ""}`} onClick={() => assign(c.source, r)}>{r.replace("_", " ")}</button>;
                })}
              </span>
            </figcaption>
          </figure>
        ))}
      </div>
      <Foot onBack={onBack} onNext={Object.keys(roles).length ? save : onNext} next={Object.keys(roles).length ? "Save cameras" : "Skip"} busy={busy}
        disabled={!control && Object.keys(roles).length > 0} />
    </>
  );
}

function ReadyStep({ d, rig }: { d: Draft; rig: RigStatus }) {
  const cal = rig.arms.filter((a) => a.calibrated).length;
  const mock = useStudio((s) => s.mock);
  return (
    <div className="onb-ready">
      <span className="onb-ready-mark"><Check aria-hidden /></span>
      <h2 className="onb-title">{d.name || "Your rig"} is set up</h2>
      <div className="onb-summary">
        <span><b className="num">{rig.arms.filter((a) => a.port).length}</b> of {SLOTS[d.layout].length} ports</span>
        <span><b className="num">{cal}</b> of {SLOTS[d.layout].length} calibrated</span>
        <span><b className="num">{rig.cameras.length}</b> cameras</span>
      </div>
      <div className="row-gap">
        <button className="btn" onClick={() => go("overview")}>Home</button>
        {mock && <RigSwitch primary />}
        <button className={`btn ${mock ? "" : "btn-primary"}`} onClick={() => go("teleop")}>Teleoperate<ArrowRight aria-hidden /></button>
      </div>
    </div>
  );
}
