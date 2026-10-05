import { ArrowLeft, ArrowRight, CircleAlert, CircleCheck, CircleX, Check, Hand, Hourglass, Pause, Play, Power, RotateCcw, Undo2, Wand2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Notices } from "../components/Notices";
import { label, labels } from "../lib/labels";
import { Segmented } from "../scene/ViewSwitch";
import {
  JOINTS, go, studio, useStudio, TICKS_PER_DEG,
  type ArmIdentity, type CalStep, type CalView, type SessionState,
} from "../lib/studio";

const CAN_START = new Set<SessionState>(["IDENTIFIED", "READY"]);
const STEPS: { key: CalStep | "check"; title: string; detail: string }[] = [
  { key: "check", title: "Check", detail: "Torque off on every arm, cable not swapped" },
  { key: "middle", title: "Set the middle", detail: "Every joint at mid-range; homing makes it read 2047" },
  { key: "ranges", title: "Record ranges", detail: "Sweep each joint end to end, except Wrist Roll" },
  { key: "review", title: "Review and save", detail: "Old against new, then write registers and file" },
];

// LeRobot's calibration, one step per screen (so_follower.py calibrate). The worker owns every step;
// this page shows where it is and sends the next command.
export function Calibrate() {
  const cal = useStudio((s) => s.telemetry?.calibration ?? null);
  const autos = useStudio((s) => s.telemetry?.autocal ?? null);
  const done = useStudio((s) => s.calibrated);
  const [hidden, setHidden] = useState<number | null>(null);
  return (
    <div className="page calibrate">
      <Notices />
      {done && hidden !== done.at && !cal && (
        <div className="notice tone-ok" role="status">
          <CircleCheck className="notice-icon" aria-hidden />
          <div className="notice-body">
            <div className="notice-title">Calibrated {done.arm}</div>
            <div className="notice-fix">
              {done.path ? <>Registers written and saved to <span className="mono">{done.path}</span>.</>
                : "Registers written. No file was saved: this Studio has no calibration directory."}
              {" "}Every arm was read again.
            </div>
          </div>
          <button className="btn btn-ghost btn-sm notice-x" onClick={() => setHidden(done.at)} aria-label="Dismiss"><X aria-hidden /></button>
        </div>
      )}
      <div className="cal-grid">
        <ArmPicker active={autos ? autos.map((r) => r.arm) : cal ? [cal.arm] : []} />
        {autos ? <AutoWizard runs={autos} /> : cal ? <Wizard cal={cal} /> : <HowItWorks />}
      </div>
    </div>
  );
}

function blockedReason(a: ArmIdentity, st: SessionState | undefined, control: boolean, holding: string[]): string | null {
  if (!control) return "Another window has control";
  if (st === "CALIBRATING") return "Another calibration is running";
  if (st === "FAULT") return "Clear the fault first";
  if (holding.length) return `Turn torque off first: ${labels(holding)} ${holding.length === 1 ? "holds" : "hold"} torque`;
  if (!st || !CAN_START.has(st)) return "Turn torque off first";
  if (a.exact && !a.ok) return "Fix the swapped cables first";
  return null;
}

function ArmPicker({ active }: { active: string[] }) {
  const arms = useStudio((s) => s.identity);
  const st = useStudio((s) => s.state?.state);
  const control = useStudio((s) => s.control);
  const tele = useStudio((s) => s.telemetry);
  const holding = Object.entries(tele?.arms ?? {}).filter(([, a]) => a.torque).map(([n]) => n);
  const autoWhy = arms.map((a) => blockedReason(a, st, control, holding)).find((w) => w) ?? null;

  return (
    <section className="panel cal-arms">
      <div className="panel-head">
        <h2 className="panel-title">Arms</h2>
        {arms.length > 0 && !active.length && (
          <button className="btn btn-sm btn-primary" disabled={autoWhy !== null} title={autoWhy ?? "Every arm finds its own end stops"}
            onClick={() => studio.send({ cmd: "autocal_start" })}>
            <Wand2 aria-hidden /> Auto-calibrate
          </button>
        )}
      </div>
      {st === "DISCONNECTED" || !st ? (
        <div className="empty empty-action">
          <p>Connect to read each arm's calibration from its servos.</p>
          <button className="btn btn-primary" onClick={() => studio.send({ cmd: "connect" })} disabled={!control}>
            <Power aria-hidden /> Connect
          </button>
        </div>
      ) : arms.length === 0 ? (
        <p className="empty">Reading arms.</p>
      ) : (
        <ul className="pick-list">
          {arms.map((a) => {
            const why = blockedReason(a, st, control, holding);
            const isActive = active.includes(a.name);
            return (
              <li key={a.name} className={`pick-row ${isActive ? "is-active" : ""}`}>
                <div className="pick-text">
                  <div className="pick-name">
                    <span className="ellipsis">{label(a.name)}</span>
                    <span className="pick-role">{a.role === "leader" ? "Leader" : "Follower"}</span>
                  </div>
                  <CalBadge a={a} />
                </div>
                {isActive ? (
                  <span className="badge tone-info">In progress</span>
                ) : (
                  <button className="btn btn-sm btn-ghost" disabled={why !== null} title={why ?? undefined}
                    onClick={() => studio.send({ cmd: "cal_start", arm: a.name })}>
                    By hand
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {arms.length > 0 && !control && st !== "CALIBRATING" && (
        <div className="panel-foot"><span className="text-warn">View only: another window has control.</span></div>
      )}
      {arms.length > 0 && control && st && !CAN_START.has(st) && st !== "CALIBRATING" && (
        <div className="panel-foot panel-foot-action">
          <span>{st === "FAULT" ? "Clear the fault, then calibrate." : "Calibration needs torque off on every arm."}</span>
          {st !== "FAULT" && <button className="btn btn-sm" onClick={() => go("teleop")}>Go to Teleoperate</button>}
        </div>
      )}
    </section>
  );
}

// One line under the arm's name: does the calibration on its servos match the file Studio expects?
function CalBadge({ a }: { a: ArmIdentity }) {
  const [tone, Icon, text] =
    a.ok ? ["ok", CircleCheck, `Matches ${a.expected}.json`]
    : a.exact ? ["danger", CircleX, `Holds ${a.match}.json: cables swapped?`]
    : a.match === null ? ["warn", CircleAlert, "No calibration file"]
    : ["warn", CircleAlert, `${a.max_deg?.toFixed(1)}° off ${a.expected}.json`];
  return <div className={`pick-meta text-${tone}`}><Icon aria-hidden /><span className="ellipsis">{text}</span></div>;
}

function HowItWorks() {
  const rig = useStudio((s) => s.rig);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">How calibration works</h2>
          <p className="panel-sub">LeRobot's steps, the same ones its calibrate command runs</p>
        </div>
      </div>
      <ol className="steps">
        {STEPS.map((s, i) => (
          <li key={s.key} className="step is-open">
            <span className="step-mark" aria-hidden>{i + 1}</span>
            <div className="step-text">
              <div className="step-title">{s.title}</div>
              <div className="step-detail">{s.detail}</div>
            </div>
            <div />
          </li>
        ))}
      </ol>
      <p className="panel-foot">
        Cancel at any step and Studio writes the arm's old registers back.
        {rig?.mock && rig.cal_dir && <> Mock rig: files save to <span className="mono">{rig.cal_dir}</span>, never to LeRobot's own folder.</>}
      </p>
    </section>
  );
}

function Wizard({ cal }: { cal: CalView }) {
  const control = useStudio((s) => s.control);
  const mock = useStudio((s) => s.mock);
  const rig = useStudio((s) => s.rig);
  const expected = useStudio((s) => s.identity.find((a) => a.name === cal.arm)?.expected ?? cal.arm);
  const [sweeping, setSweeping] = useState(false);
  const at = STEPS.findIndex((s) => s.key === cal.step);
  const send = (cmd: string) => () => studio.send({ cmd });

  // The mock's scripted hand keeps moving a limp arm: let go of it when the recording ends.
  const sweep = (on: boolean) => { studio.send({ cmd: "inject", arm: cal.arm, kind: on ? "hand" : "still" }); setSweeping(on); };
  useEffect(() => { if (cal.step !== "ranges" && sweeping) sweep(false); }, [cal.step]); // eslint-disable-line react-hooks/exhaustive-deps
  const cancel = () => { if (sweeping) sweep(false); studio.send({ cmd: "cal_cancel" }); };

  const unmoved = Object.entries(cal.joints).filter(([, v]) => !v.fixed && v.min === v.max).map(([j]) => j);
  // WHY the arm's own file: LeRobot keeps followers and leaders in different folders (robots/so_follower,
  // teleoperators/so_leader), so cal_dir/<id>.json is not where the file goes.
  const file = rig?.arms.find((a) => a.name === cal.arm)?.file ?? `${expected}.json`;
  const path = rig?.cal_dir ? `${rig.cal_dir}/${file}` : null;

  return (
    <section className="panel wizard">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Calibrating {cal.arm}</h2>
          <p className="panel-sub">{cal.role === "leader" ? "Leader" : "Follower"}, torque off. Support it by hand.</p>
        </div>
        <button className="btn btn-ghost" onClick={cancel} disabled={!control}><RotateCcw aria-hidden /> Cancel and restore</button>
      </div>
      <ol className="stepper" aria-label="Calibration steps">
        {STEPS.map((s, i) => (
          <li key={s.key} className={`stepper-item ${i < at ? "is-done" : i === at ? "is-current" : ""}`} aria-current={i === at ? "step" : undefined}>
            <span className="step-mark" aria-hidden>{i < at ? <Check /> : i + 1}</span>
            <span className="stepper-title">{s.title}</span>
          </li>
        ))}
      </ol>
      <hr className="divider" />

      {cal.step === "middle" && (
        <div className="wizard-body">
          <div className="instr">
            <p className="instr-lead">Move every joint of {cal.arm} by hand to roughly the middle of its range, and hold it there.</p>
            <p className="muted">Studio then writes each servo's homing offset so that this pose reads 2047, half of one turn.
              Ranges are recorded around it in the next step.</p>
          </div>
          <div className="wizard-actions">
            <button className="btn btn-primary" onClick={send("cal_middle")} disabled={!control}>Set middle</button>
          </div>
        </div>
      )}

      {cal.step === "ranges" && (
        <div className="wizard-body">
          <div className="instr">
            <p className="instr-lead">Move each joint slowly through its full range, end to end. Leave wrist roll: its range is a full turn.</p>
          </div>
          <RangeBars cal={cal} />
          <div className="wizard-actions">
            {mock && (
              <button className="btn" onClick={() => sweep(!sweeping)} disabled={!control}>
                <Hand aria-hidden /> {sweeping ? "Stop the hand" : "Sweep it by hand (mock)"}
              </button>
            )}
            <span className="wizard-hint">{unmoved.length ? `Not moved yet: ${unmoved.map(label).join(", ")}` : "Every joint has moved."}</span>
            <button className="btn btn-primary" onClick={send("cal_finish")} disabled={!control || unmoved.length > 0}
              title={unmoved.length ? "Move every joint first" : undefined}>
              Finish recording
            </button>
          </div>
        </div>
      )}

      {cal.step === "review" && cal.new && (
        <div className="wizard-body">
          <ReviewTable cal={cal} />
          <div className="wizard-actions">
            <span className="wizard-hint">
              Save writes these registers to {label(cal.arm)}{path ? <> and saves <span className="mono">{path}</span></> : " (no file: Studio has no calibration directory)"}.
            </span>
            <button className="btn btn-primary" onClick={send("cal_save")} disabled={!control}>Save calibration</button>
          </div>
        </div>
      )}
    </section>
  );
}

function RangeBars({ cal }: { cal: CalView }) {
  return (
    <ul className="ranges" aria-label="Recorded ranges">
      {JOINTS.map((j) => {
        const v = cal.joints[j];
        if (!v) return null;
        const pos = (x: number) => `${(Math.min(4095, Math.max(0, x)) / 4095) * 100}%`;
        const span = (v.max - v.min) / TICKS_PER_DEG;
        return (
          <li key={j} className={`range-row ${v.fixed ? "is-fixed" : v.min === v.max ? "is-still" : ""}`}>
            <span className="range-joint">{label(j)}</span>
            <div className="range-track" aria-hidden>
              <span className="range-mid" />
              <span className="range-fill" style={{ left: pos(v.min), width: `calc(${pos(v.max)} - ${pos(v.min)})` }} />
              <span className="range-pos" style={{ left: pos(v.pos) }} />
            </div>
            <span className="range-nums mono">{v.min}<span className="faint"> / </span><b>{v.pos}</b><span className="faint"> / </span>{v.max}</span>
            <span className="range-span num">{v.fixed ? "Full turn" : v.min === v.max ? "Not moved" : `${span.toFixed(0)}°`}</span>
          </li>
        );
      })}
    </ul>
  );
}

function ReviewTable({ cal }: { cal: CalView }) {
  const cell = (o: number, n: number) => (
    <td className="r mono">
      {o === n ? <span className="faint">{n}</span> : <><span className="faint">{o} → </span><b>{n}</b></>}
    </td>
  );
  return (
    <div className="table-scroll">
      <table className="table review">
        <thead>
          <tr><th>Joint</th><th className="r">Homing offset</th><th className="r">Min</th><th className="r">Max</th><th className="r">Range</th></tr>
        </thead>
        <tbody>
          {JOINTS.map((j) => {
            const o = cal.old[j], n = cal.new![j];
            return (
              <tr key={j}>
                <td className="strong">{label(j)}</td>
                {cell(o.homing_offset, n.homing_offset)}
                {cell(o.range_min, n.range_min)}
                {cell(o.range_max, n.range_max)}
                <td className="r num">{((n.range_max - n.range_min) / TICKS_PER_DEG).toFixed(0)}°</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// -- auto-calibration (autocal.py through the worker) ----------------------------------------------
const AUTO_STEPS: { key: CalStep; title: string }[] = [
  { key: "auto-middle", title: "Middle pose" }, { key: "auto", title: "Sweep" }, { key: "auto-review", title: "Save" },
];
const SCOPES: { mode: "all" | "gripper"; text: string }[] = [{ mode: "all", text: "All joints" }, { mode: "gripper", text: "Gripper only" }];
const PHASE = { out: [ArrowRight, "far stop"], back: [ArrowLeft, "near stop"], home: [Undo2, "back"] } as const;
const SWEEP_DEFAULT = 350; // autocal.SWEEP_LIMIT_DEFAULT, of 1000

/** The SO-101 from the side in the middle pose: the model's zero pose (so101_new_calib), its body positions from the
 * FK scaled into the picture. Upper arm up, forearm and gripper level, pointing forward. */
function MiddlePose() {
  const pts = "20,110 33,90 42,72 52,36 95,34 115,34 147,37";
  return (
    <svg className="mid-pose" viewBox="0 0 160 120" role="img" aria-label="Middle pose: upper arm straight up, forearm and gripper level, pointing forward">
      <rect x="6" y="108" width="40" height="6" rx="2" className="mp-base" />
      <polyline points={pts} className="mp-arm" />
      {pts.split(" ").slice(1, 6).map((p) => { const [x, y] = p.split(",").map(Number); return <circle key={p} cx={x} cy={y} r="3.2" className="mp-joint" />; })}
      <path d="M147,37 l9,-5 M147,37 l9,5" className="mp-jaw" />
    </svg>
  );
}

function AutoWizard({ runs }: { runs: CalView[] }) {
  const control = useStudio((s) => s.control);
  const step: CalStep = runs.every((r) => r.step === "auto-middle") ? "auto-middle"
    : runs.some((r) => r.step === "auto") ? "auto" : "auto-review";
  const holding = step !== "auto-middle";
  const at = AUTO_STEPS.findIndex((s) => s.key === step);
  const paused = runs.find((r) => r.auto?.state === "paused");
  return (
    <section className="panel wizard">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Auto-calibration</h2>
          <p className="panel-sub">{runs.length} {runs.length === 1 ? "arm" : "arms"}{holding ? " · holding torque" : ""}</p>
        </div>
        <button className="btn btn-ghost" onClick={() => studio.send({ cmd: "autocal_cancel" })} disabled={!control}
          title={holding ? "Torque goes off: support the arms" : undefined}>
          <RotateCcw aria-hidden /> {holding ? "Cancel (arms go limp)" : "Cancel"}
        </button>
      </div>
      <ol className="stepper" aria-label="Auto-calibration steps">
        {AUTO_STEPS.map((s, i) => (
          <li key={s.key} className={`stepper-item ${i < at ? "is-done" : i === at ? "is-current" : ""}`} aria-current={i === at ? "step" : undefined}>
            <span className="step-mark" aria-hidden>{i < at ? <Check /> : i + 1}</span>
            <span className="stepper-title">{s.title}</span>
          </li>
        ))}
      </ol>
      <hr className="divider" />
      {step === "auto-middle" && <AutoMiddle runs={runs} />}
      {step === "auto" && (
        <div className="wizard-body">
          {paused && (
            <div className="auto-paused">
              <Pause aria-hidden /><span>{paused.auto?.why ?? "Paused"}</span>
              <button className="btn btn-sm btn-primary" disabled={!control} onClick={() => studio.send({ cmd: "autocal_resume" })}><Play aria-hidden />Resume</button>
            </div>
          )}
          <div className="auto-arms">{runs.map((r) => <AutoArm key={r.arm} run={r} />)}</div>
          <p className="wizard-hint">Stop (Esc) pauses every arm where it is.</p>
        </div>
      )}
      {step === "auto-review" && <AutoReview runs={runs} />}
    </section>
  );
}

function AutoMiddle({ runs }: { runs: CalView[] }) {
  const control = useStudio((s) => s.control);
  const [keep, setKeep] = useState<string[]>(runs.map((r) => r.arm));
  const [scope, setScope] = useState<"all" | "gripper">("all");
  const [torque, setTorque] = useState(SWEEP_DEFAULT);
  const toggle = (arm: string) => setKeep(keep.includes(arm) ? keep.filter((a) => a !== arm) : [...keep, arm]);
  const go = () => studio.send({
    cmd: "autocal_go", arms: keep, joints: scope === "gripper" ? ["gripper"] : undefined,
    torque: torque === SWEEP_DEFAULT ? undefined : torque,
  });
  return (
    <div className="wizard-body">
      <div className="auto-middle">
        <MiddlePose />
        <div className="instr">
          <p className="instr-lead">Put each arm in the middle pose, then Start: the arms hold themselves.</p>
          <ul className="auto-tips">
            <li>Roughly is enough</li>
            <li>Wrist roll takes its zero here: gripper not twisted</li>
            <li>Clear the workspace: the arms move</li>
          </ul>
        </div>
      </div>
      <div className="auto-opts">
        <div className="auto-chips" role="group" aria-label="Arms">
          {runs.map((r) => (
            <button key={r.arm} type="button" className={`auto-chip ${keep.includes(r.arm) ? "is-on" : ""}`} aria-pressed={keep.includes(r.arm)}
              onClick={() => toggle(r.arm)}>{keep.includes(r.arm) && <Check aria-hidden className="ico-inline" />}{label(r.arm)}</button>
          ))}
        </div>
        <Segmented label="Joints" value={scope} options={SCOPES} onChange={setScope} />
        <details className="auto-adv">
          <summary>Advanced</summary>
          <label className="auto-torque">
            <span>Sweep torque</span>
            <input type="range" min={150} max={600} step={10} value={torque} onChange={(e) => setTorque(Number(e.target.value))} />
            <span className="num">{Math.round(torque / 10)}%</span>
          </label>
          <p className="faint t-sm">How hard a joint may press into its stops. The gripper never goes over 20%.</p>
        </details>
      </div>
      <div className="wizard-actions">
        <span className="wizard-hint">{scope === "gripper" ? "A first, careful run: one joint per arm." : "About two minutes; the shoulder pans take turns."}</span>
        <button className="btn btn-primary" disabled={!control || !keep.length} onClick={go}><Wand2 aria-hidden /> Start</button>
      </div>
    </div>
  );
}

function AutoArm({ run }: { run: CalView }) {
  const a = run.auto;
  const order = a?.joints ?? [];
  const pos = (x: number) => `${(Math.min(4095, Math.max(0, x)) / 4095) * 100}%`;
  const state = run.step === "auto-failed" ? ["danger", "Failed"] : run.step === "auto-review" ? ["ok", "Done"]
    : a?.state === "paused" ? ["warn", "Paused"] : a?.joint ? ["info", label(a.joint)] : ["neutral", "Waiting"];
  return (
    <div className="auto-arm">
      <div className="auto-arm-head">
        <span className="strong">{label(run.arm)}</span>
        <span className={`badge tone-${state[0]}`}>{state[1]}</span>
      </div>
      <ul className="auto-joints">
        {order.map((j) => {
          const f = a?.found[j];
          const now = a?.joint === j && run.step === "auto";
          const v = run.joints[j];
          const [Icon, what] = now && a?.waiting ? [Hourglass, "waits its turn"] : now && a?.phase ? PHASE[a.phase] : [null, ""];
          return (
            <li key={j} className={`aj ${f ? "is-done" : now ? "is-now" : ""}`}>
              <span className="aj-name">{label(j)}</span>
              <span className="aj-track" aria-hidden>
                {f && <span className="aj-fill" style={{ left: pos(f.lo), width: `calc(${pos(f.hi)} - ${pos(f.lo)})` }} />}
                {v && (now || f) && <span className="aj-pos" style={{ left: pos(v.pos) }} />}
              </span>
              <span className="aj-val num">{f ? `${f.deg.toFixed(0)}°` : Icon ? <><Icon aria-hidden className="ico-inline" />{what}</> : ""}</span>
            </li>
          );
        })}
      </ul>
      {a?.state === "failed" && <p className="text-danger t-sm">{a.why}</p>}
      {Object.entries(a?.notes ?? {}).map(([j, n]) => <p key={j} className="text-warn t-sm">{label(j)}: {n}</p>)}
    </div>
  );
}

function AutoReview({ runs }: { runs: CalView[] }) {
  const control = useStudio((s) => s.control);
  const failed = runs.filter((r) => r.step === "auto-failed").map((r) => r.arm);
  return (
    <div className="wizard-body">
      <div className="auto-arms">{runs.map((r) => <AutoArm key={r.arm} run={r} />)}</div>
      <details className="auto-adv">
        <summary>Registers, old and new</summary>
        {runs.filter((r) => r.new).map((r) => <div key={r.arm}><p className="strong t-sm">{label(r.arm)}</p><ReviewTable cal={r} /></div>)}
      </details>
      <div className="wizard-actions">
        <span className="wizard-hint">
          Support the arms: Save turns their torque off.{failed.length ? ` ${labels(failed)} ${failed.length === 1 ? "keeps its" : "keep their"} old calibration.` : ""}
        </span>
        <button className="btn btn-primary" disabled={!control} onClick={() => studio.send({ cmd: "autocal_save" })}>Save</button>
      </div>
    </div>
  );
}
