import { ArrowLeft, ArrowRight, Camera, Check, CircleAlert, Plug, Search, SquareTerminal, Unplug, Wand2 } from "lucide-react";
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { Boundary } from "../components/Boundary";
import { CalibrationFiles } from "../components/CalibrationFiles";
import { DetectArms } from "../components/DetectArms";
import { ArmsGlyph } from "../components/data/bits";
import { Notices } from "../components/Notices";
import { RigSwitch } from "../components/RigSwitch";
import { advance, portKey, type Wizard } from "../lib/portfinder";
import { SLOTS, refreshRig, shortPort, slotLabel, slug, useRig, type Layout, type RigStatus } from "../lib/rig";
import { go, studio, useStudio } from "../lib/studio";

const Viewer = lazy(() => import("../scene/Viewer"));

type Step = "rig" | "arms" | "calibration" | "drive" | "cameras" | "ready";
const STEPS: { id: Step; label: string }[] = [
  { id: "rig", label: "Rig" }, { id: "arms", label: "Arms" }, { id: "calibration", label: "Calibration" },
  { id: "drive", label: "Test drive" }, { id: "cameras", label: "Cameras" }, { id: "ready", label: "Ready" },
];
const ROLES: Record<Layout, string[]> = { single: ["top", "front", "wrist"], bimanual: ["top", "front", "left_wrist", "right_wrist"] };

interface Cam { source: number | string; width?: number; height?: number; fps?: number }
interface Draft {
  name: string; layout: Layout; ports: Record<string, string>;
  ids: "existing" | "new"; existing: { follower: string; leader: string } | null;
}
interface Probe { source: number | string; ok: boolean; width?: number; height?: number; fps?: number; picture?: string | null; error?: string | null }

function baseId(id: string | null | undefined): string | null {
  return id ? id.replace(/_(left|right)$/, "") : null;
}

/** Where onboarding starts: the config if there is one, else ports.local.sh, else calibration files on this Mac. */
function initial(rig: RigStatus): Draft {
  const layout: Layout = rig.layout ?? (Object.keys(rig.hint.ports).some((k) => k.startsWith("left_")) ? "bimanual" : "single");
  const ports: Record<string, string> = {};
  for (const a of rig.arms) if (a.port) ports[a.key] = a.port;
  for (const [k, v] of Object.entries(rig.hint.ports)) if (!ports[k] && SLOTS[layout].includes(k)) ports[k] = v;
  const fromConfig = rig.arms.length ? {
    follower: baseId(rig.arms.find((a) => a.role === "follower")?.id) ?? "", leader: baseId(rig.arms.find((a) => a.role === "leader")?.id) ?? "",
  } : null;
  const hint = rig.hint.ids?.[layout];
  const known = layout === "bimanual" ? { f: rig.known.bi_follower[0]?.id, l: rig.known.bi_leader[0]?.id } : { f: rig.known.follower[0]?.id, l: rig.known.leader[0]?.id };
  const existing = fromConfig?.follower && fromConfig.leader ? fromConfig
    : hint?.follower && hint?.leader ? { follower: hint.follower, leader: hint.leader }
    : known.f && known.l ? { follower: known.f, leader: known.l } : null;
  return { name: rig.name ?? "", layout, ports, ids: existing ? "existing" : "new", existing };
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
        {step === "rig" && <RigStep d={d} set={setD} onNext={next} />}
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

function RigStep({ d, set, onNext }: { d: Draft; set: (d: Draft) => void; onNext: () => void }) {
  const ok = d.name.trim().length > 0;
  return (
    <>
      <h2 className="onb-title">Name your rig</h2>
      <input className="input onb-name" autoFocus placeholder="Bench A" value={d.name} maxLength={60}
        onChange={(e) => set({ ...d, name: e.target.value })} onKeyDown={(e) => { if (e.key === "Enter" && ok) onNext(); }} />
      <div className="onb-layouts" role="radiogroup" aria-label="Layout">
        {(["single", "bimanual"] as Layout[]).map((l) => (
          <button key={l} role="radio" aria-checked={d.layout === l} className={`onb-layout ${d.layout === l ? "is-on" : ""}`}
            onClick={() => set({ ...d, layout: l })}>
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
      <p className="onb-sub">Detect arms places every arm whose calibration matches. For any other arm, press Find, then unplug its USB and plug it back in.</p>
      <DetectArms prefer={d.existing} onPlaced={(p) => set({
        ...d, layout: p.layout ?? d.layout, ports: { ...d.ports, ...p.ports },
        ...(p.ids ? { ids: "existing" as const, existing: p.ids } : {}),
      })} />
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
      <details className="onb-calfiles">
        <summary className="strong">Calibration files on this Mac</summary>
        <CalibrationFiles compact />
      </details>
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
  const ids = d.ids === "existing" && d.existing ? d.existing : fresh;
  const files = (x: { follower: string; leader: string }) => SLOTS[d.layout].map((slot) => {
    const role = slot.endsWith("leader") ? "leader" : "follower";
    const side = slot.startsWith("left_") ? "_left" : slot.startsWith("right_") ? "_right" : "";
    return { slot, ok: has(role, `${x[role]}${side}`) };
  });
  const save = () => {
    setBusy(true);
    const off1 = studio.onMessage("rig_saved", () => { off1(); off2(); setBusy(false); onNext(); });
    const off2 = studio.onMessage("error", (m) => { if (m.cmd === "rig_write") { off1(); off2(); setBusy(false); } });
    const ports = Object.fromEntries(Object.entries(d.ports).filter(([k, v]) => v && SLOTS[d.layout].includes(k)));
    if (!studio.send({ cmd: "rig_write", rig: { name: d.name, layout: d.layout, ids, ports } })) { off1(); off2(); setBusy(false); }
  };
  const Choice = ({ value, x, title }: { value: "existing" | "new"; x: { follower: string; leader: string }; title: string }) => (
    <button role="radio" aria-checked={d.ids === value} className={`onb-choice ${d.ids === value ? "is-on" : ""}`} onClick={() => set({ ...d, ids: value })}>
      <span className="strong">{title}</span>
      <span className="mono faint">{x.follower} · {x.leader}</span>
      <span className="onb-chips">
        {files(x).map((f) => <span key={f.slot} className={`chip ${f.ok ? "sev-ok" : "sev-info"}`}>{f.ok ? "✓" : "–"} {slotLabel(f.slot)}</span>)}
      </span>
    </button>
  );
  return (
    <>
      <h2 className="onb-title">Calibration</h2>
      <div className="onb-choices" role="radiogroup" aria-label="Calibration">
        {d.existing && <Choice value="existing" x={d.existing} title="Keep this Mac's calibration" />}
        <Choice value="new" x={fresh} title="New calibration for this rig" />
      </div>
      <Foot onBack={onBack} onNext={save} next="Save rig" busy={busy} disabled={!control}
        extra={<>
          <button className="btn" onClick={() => go("calibrate")} title="LeRobot's lerobot-calibrate, one arm at a time, in Studio's terminal"><SquareTerminal aria-hidden />Calibrate in the terminal</button>
          <button className="btn btn-ghost" onClick={() => go("calibrate")} title="Every arm finds its own end stops"><Wand2 aria-hidden />Auto-calibrate</button>
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
      {mock && (
        // WHY here: saving the rig does not switch Studio to it, so a test drive on the simulated arms
        // passed while the real ones were never moved.
        <p className="warn-text t-sm row-gap">
          <CircleAlert aria-hidden className="ico-inline" />These are simulated arms, not the ones you plugged in.
          <RigSwitch primary />
        </p>
      )}
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
      cams[r] = { source: src, width: m?.width ?? old?.width, height: m?.height ?? old?.height, fps: m?.fps ?? old?.fps };
    }
    setBusy(true);
    const ids = rig.arms.length ? {
      follower: baseId(rig.arms.find((a) => a.role === "follower")?.id) ?? `${slug(d.name)}_follower`,
      leader: baseId(rig.arms.find((a) => a.role === "leader")?.id) ?? `${slug(d.name)}_leader`,
    } : { follower: `${slug(d.name)}_follower`, leader: `${slug(d.name)}_leader` };
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
              <span className="mono faint">#{String(c.source)} · {c.width}×{c.height}</span>
              <span className="onb-roles">
                {ROLES[d.layout].map((r) => (
                  <button key={r} className={`seg-btn ${String(roles[r]) === String(c.source) ? "is-on" : ""}`} onClick={() => assign(c.source, r)}>{r.replace("_", " ")}</button>
                ))}
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
