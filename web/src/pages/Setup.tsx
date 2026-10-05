import {
  ArrowLeft, ArrowRight, BookOpen, CircleAlert, CircleCheck, ExternalLink, FileText, LoaderCircle, Plug, ScanSearch,
  Square, TriangleAlert, Unplug,
} from "lucide-react";
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { CommandBlock } from "../components/CommandBlock";
import { Notices } from "../components/Notices";
import { label, labels } from "../lib/labels";
import { portKey, setup, useSetup, type AlignSession } from "../lib/setup";
import { go, studio, useSection, useStudio, type LeRobotArm, type LeRobotView, type Route } from "../lib/studio";
import "../styles/setup.css";

// The whole path from a new rig to a scored policy, one step at a time, in the order of LeRobot 0.6.0's SO-101
// and imitation-learning docs. Commands come from robot-config.yaml (src/phi_studio/rigspec.py, checked against
// LeRobot's own CLI parsers by tests/test_rigspec.py). Ports, camera numbers and camera align are done here and
// saved to robot-config.yaml in place, comments kept, with a backup (src/phi_studio/setup_api.py).

const DOCS = "https://huggingface.co/docs/lerobot/v0.6.0";
type StepId = "install" | "ports" | "motors" | "calibrate" | "teleop" | "cameras" | "align" | "record" | "dataset" | "train" | "policy" | "evaluate";
const STEPS: { id: StepId; title: string; doc?: string }[] = [
  { id: "install", title: "Install and connect", doc: `${DOCS}/so101#install-lerobot-` },
  { id: "ports", title: "Find ports", doc: `${DOCS}/so101#1-find-the-usb-ports-associated-with-each-arm` },
  { id: "motors", title: "Set motor ids", doc: `${DOCS}/so101#2-set-the-motors-ids-and-baudrates` },
  { id: "calibrate", title: "Calibrate", doc: `${DOCS}/so101#calibrate` },
  { id: "teleop", title: "Teleoperate", doc: `${DOCS}/il_robots#teleoperate` },
  { id: "cameras", title: "Cameras", doc: `${DOCS}/cameras#find-your-camera` },
  { id: "align", title: "Align cameras" },
  { id: "record", title: "Record", doc: `${DOCS}/il_robots#record-a-dataset` },
  { id: "dataset", title: "Check the dataset", doc: `${DOCS}/il_robots#visualize-a-dataset` },
  { id: "train", title: "Train", doc: `${DOCS}/il_robots#train-a-policy` },
  { id: "policy", title: "Run a policy", doc: `${DOCS}/il_robots#run-inference-and-evaluate-your-policy` },
  { id: "evaluate", title: "Evaluate" },
];
type Status = "ok" | "todo" | "warn" | null; // null: Studio cannot tell, so it does not pretend to

const WEAK = 0.3; // see AlignCard
const camName = (key: string) => `${label(key.replace(/^observation\.images\./, ""))} camera`;
const isSource = (v: unknown): v is number | string => (typeof v === "number" && Number.isInteger(v)) || (typeof v === "string" && /^\/dev\/video\d+$/.test(v));

export function Setup() {
  const link = useStudio((s) => s.link);
  const index = useStudio((s) => s.files.index);
  useEffect(() => { if (link === "open") studio.loadFiles(); }, [link]);
  const lr = index?.lerobot ?? null;
  const section = useSection();
  const step: StepId = STEPS.some((s) => s.id === section) ? (section as StepId) : "install";
  const status = useStatuses(lr);
  useEffect(() => { document.querySelector(".content")?.scrollTo({ top: 0 }); }, [step]);

  return (
    <div className="page su">
      <nav className="su-rail" aria-label="Set up steps">
        <div className="field-label">Steps</div>
        <ol>
          {STEPS.map((s, i) => (
            <li key={s.id}>
              <a href={`#/setup/${s.id}`} className={`su-rail-item ${s.id === step ? "is-active" : ""}`}
                aria-current={s.id === step ? "step" : undefined}
                onClick={(e) => { e.preventDefault(); go("setup", s.id); }}>
                <StepMark n={i + 1} status={status[s.id]} />
                <span>{s.title}</span>
              </a>
            </li>
          ))}
        </ol>
      </nav>
      <div className="su-body">
        <Notices />
        {!index ? (
          <section className="panel"><p className="empty">Reading robot-config.yaml</p></section>
        ) : !lr ? (
          <section className="panel">
            <p className="empty">No robot-config.yaml found. Start Studio with --rig-dir set to the folder that has it.</p>
          </section>
        ) : lr.error ? (
          <section className="panel">
            <div className="panel-head"><h2 className="panel-title">Rig</h2><OpenConfig lr={lr} /></div>
            <p className="empty text-danger">robot-config.yaml does not parse: {lr.error}</p>
          </section>
        ) : (
          <Step id={step} lr={lr} status={status} />
        )}
      </div>
    </div>
  );
}

function StepMark({ n, status }: { n: number; status: Status }) {
  if (status === "ok") return <span className="su-mark is-ok" title="Done"><CircleCheck aria-hidden /></span>;
  if (status === "warn") return <span className="su-mark is-warn" title="Needs a look"><CircleAlert aria-hidden /></span>;
  return <span className={`su-mark num ${status === "todo" ? "is-todo" : ""}`} aria-hidden>{n}</span>;
}

/** What Studio can actually tell about each step. Motor ids, teleoperation and recording leave nothing it can
 * read back, so those show no mark rather than a guess. */
function useStatuses(lr: LeRobotView | null): Record<StepId, Status> {
  const ports = useSetup((s) => s.ports);
  const align = useSetup((s) => s.align);
  const ticks = useSetup((s) => s.ticks);
  return useMemo(() => {
    const out = Object.fromEntries(STEPS.map((s) => [s.id, null])) as Record<StepId, Status>;
    if (!lr || lr.error) return out;
    out.install = lr.problems.length ? "warn" : "ok";
    const here = new Set(ports?.ports.map((p) => portKey(p.name)) ?? []);
    out.ports = lr.arms.some((a) => !a.port) ? "todo"
      : !ports ? null : lr.arms.every((a) => here.has(portKey(a.port!))) ? "ok" : "warn";
    out.calibrate = lr.arms.every((a) => a.calibrated) ? "ok" : "todo";
    const opencv = lr.cameras.filter((c) => c.type === "opencv");
    if (opencv.length) out.cameras = opencv.every((c) => isSource(c.source)) ? "ok" : "todo";
    if (align && align.assignment.length && align.assignment.every((a) => ticks[String(a.live)]?.aligned)) out.align = "ok";
    return out;
  }, [lr, ports, align, ticks]);
}

function Step({ id, lr, status }: { id: StepId; lr: LeRobotView; status: Record<StepId, Status> }) {
  const i = STEPS.findIndex((s) => s.id === id);
  const s = STEPS[i];
  const body: Record<StepId, ReactNode> = {
    install: <InstallStep lr={lr} status={status} />,
    ports: <PortsStep lr={lr} />,
    motors: <MotorsStep lr={lr} />,
    calibrate: <CalibrateStep lr={lr} />,
    teleop: <TeleopStep lr={lr} />,
    cameras: <CamerasStep lr={lr} />,
    align: <AlignStep lr={lr} />,
    record: <RecordStep lr={lr} />,
    dataset: <DatasetStep lr={lr} />,
    train: <LinkStep route="train" text="Train a policy on your dataset, on Northeastern's Explorer cluster or on this Mac. The Train page checks the cluster first, submits the job, charts the loss as it trains, and copies the checkpoints back." button="Open Train" />,
    policy: <LinkStep route="models" text="Bring in a trained policy: type its Hugging Face repo id or search the Hub on the Models page. It reads the model's config first to check that its cameras and joints fit this rig, then downloads it and builds the command to run it on the arms." button="Open Models" also={{ route: "policy", button: "Open Run policy" }} />,
    evaluate: <LinkStep route="evaluate" text="Score a policy over many tries of one task. The Evaluate page records each try as a success or a failure and reports the success rate with a 95% interval, so two policies can be compared fairly." button="Open Evaluate" />,
  };
  return (
    <section className="panel su-step" aria-labelledby="su-step-title">
      <div className="panel-head">
        <div>
          <div className="su-kicker">Step {i + 1} of {STEPS.length}</div>
          <h2 className="panel-title" id="su-step-title">{s.title}</h2>
        </div>
        {s.doc && (
          <a className="btn btn-sm btn-ghost" href={s.doc} target="_blank" rel="noreferrer">
            <BookOpen aria-hidden /> LeRobot docs <ExternalLink aria-hidden />
          </a>
        )}
      </div>
      <div className="panel-body su-content">{body[id]}</div>
      <div className="su-nav">
        {i > 0 ? (
          <button className="btn btn-sm btn-ghost" onClick={() => go("setup", STEPS[i - 1].id)}><ArrowLeft aria-hidden /> {STEPS[i - 1].title}</button>
        ) : <span />}
        {i < STEPS.length - 1 && (
          <button className="btn btn-sm" onClick={() => go("setup", STEPS[i + 1].id)}>{STEPS[i + 1].title} <ArrowRight aria-hidden /></button>
        )}
      </div>
    </section>
  );
}

function OpenConfig({ lr }: { lr: LeRobotView }) {
  return (
    <button className="btn btn-sm" onClick={() => { studio.openFile(lr.file.path, null, lr.file.root); go("files"); }}>
      <FileText aria-hidden /> Open robot-config.yaml
    </button>
  );
}

function StepCommands({ lr, step, intro }: { lr: LeRobotView; step: string; intro?: string }) {
  const cmds = lr.commands.filter((c) => c.step === step);
  if (!cmds.length) return null;
  return (
    <div className="su-cmds">
      {intro && <p className="muted">{intro}</p>}
      <ol className="cmd-list">
        {cmds.map((c, i) => (
          <li key={c.id} className="cmd-item">
            <span className="cmd-n num" aria-hidden>{i + 1}</span>
            <div className="cmd-body">
              <div className="cmd-title">{c.title}</div>
              <p className="cmd-why">{c.why}</p>
              <CommandBlock cmd={c.cmd} run />
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Callout({ tone, children }: { tone: "warn" | "info" | "ok" | "danger"; children: ReactNode }) {
  const Icon = tone === "ok" ? CircleCheck : tone === "info" ? CircleAlert : TriangleAlert;
  return <div className={`su-callout tone-${tone}`}><Icon aria-hidden /><div>{children}</div></div>;
}

function NeedControl() {
  return (
    <span className="su-needs">
      Saving needs control of Studio.{" "}
      <button className="btn btn-sm" onClick={() => studio.send({ cmd: "take_control" })}>Take control</button>
    </span>
  );
}

// -- 1 install ---------------------------------------------------------------------------------------
function InstallStep({ lr, status }: { lr: LeRobotView; status: Record<StepId, Status> }) {
  const rig = useStudio((s) => s.rig);
  const n = lr.arms.length;
  return (
    <>
      {rig?.mock && (
        <p className="setup-note">
          Studio is running a mock {rig.bimanual ? "bimanual" : "single-arm"} rig for its own pages. These steps use the real
          rig in robot-config.yaml.
        </p>
      )}
      <ul className="su-list">
        <li>{n} SO-101 arms, built: {labels(lr.arms.map((a) => a.key))}.</li>
        <li>Each arm on its own power supply, and a USB cable from each arm's driver board to this Mac.</li>
        <li>LeRobot 0.6.0 with the Feetech motor SDK, in the env Studio runs in: <code className="ident">pip install -e ".[feetech]"</code> in the LeRobot folder.</li>
        <li>For recording: the cameras plugged in, and a Hugging Face login (step 8).</li>
      </ul>
      <RigTable lr={lr} />
      <div className="su-overview">
        <div className="field-label">Progress</div>
        <ol>
          {STEPS.slice(1).map((s, i) => (
            <li key={s.id}>
              <a href={`#/setup/${s.id}`} onClick={(e) => { e.preventDefault(); go("setup", s.id); }}>
                <StepMark n={i + 2} status={status[s.id]} /> {s.title}
              </a>
            </li>
          ))}
        </ol>
      </div>
    </>
  );
}

function RigTable({ lr }: { lr: LeRobotView }) {
  return (
    <div className="su-rig">
      <div className="su-rig-head">
        <div className="setup-title">
          <span className="strong">Rig in robot-config.yaml</span>
          <span className="badge tone-neutral">{lr.bimanual ? "Bimanual" : "Single arm"}</span>
        </div>
        <OpenConfig lr={lr} />
      </div>
      <div className="table-scroll">
        <table className="table setup-arms">
          <thead><tr><th>Arm</th><th>Port</th><th>Calibration file</th><th>Units</th><th>Largest move per step</th></tr></thead>
          <tbody>{lr.arms.map((a) => <ArmRow key={a.key} a={a} />)}</tbody>
        </table>
      </div>
      {lr.problems.length > 0 && (
        <ul className="setup-problems">
          {lr.problems.map((p) => <li key={p}><TriangleAlert aria-hidden /><span>{p}</span></li>)}
        </ul>
      )}
    </div>
  );
}

function ArmRow({ a }: { a: LeRobotArm }) {
  const cap = a.max_relative_target;
  return (
    <tr>
      <td>
        <div className="strong">{label(a.key)}</div>
        <code className="ident setup-type" title="LeRobot type">{a.type}</code>
      </td>
      <td>{a.port ? <code className="ident setup-port" title={a.port}>{a.port}</code> : <span className="text-warn">Not set</span>}</td>
      <td>
        {!a.id ? <span className="text-warn">No id, so LeRobot would save None.json</span> : (
          <span className={`setup-cal ${a.calibrated ? "text-ok" : "text-warn"}`} title={a.calibration ?? undefined}>
            {a.calibrated ? <CircleCheck aria-hidden /> : <CircleAlert aria-hidden />}
            <code className="ident">{a.id}.json</code>
            {!a.calibrated && <span>missing</span>}
          </span>
        )}
      </td>
      <td>{a.use_degrees ? "Degrees" : "-100 to 100"}</td>
      <td className="num">
        {a.role === "leader" ? <span className="faint">Leader, not limited</span>
          : cap === null ? <span className="text-warn">No limit</span>
          : typeof cap === "number" ? `${cap}` : "Per joint"}
      </td>
    </tr>
  );
}

// -- 2 ports -----------------------------------------------------------------------------------------
function PortsStep({ lr }: { lr: LeRobotView }) {
  const link = useStudio((s) => s.link);
  const control = useStudio((s) => s.control);
  const ports = useSetup((s) => s.ports);
  const wizard = useSetup((s) => s.wizard);
  const saved = useSetup((s) => s.saved);
  // WHY poll at 2 Hz: the finder reacts to a cable pulled out; half a second feels immediate and costs one
  // USB listing per tick. Only while this step is open and the tab is visible.
  useEffect(() => {
    if (link !== "open") return;
    setup.refreshPorts();
    const t = window.setInterval(() => { if (!document.hidden) setup.refreshPorts(); }, 500);
    return () => window.clearInterval(t);
  }, [link]);

  const names = ports?.ports.map((p) => p.name) ?? [];
  const byName = new Map(ports?.ports.map((p) => [portKey(p.name), p]) ?? []);
  const armOn = new Map(lr.arms.filter((a) => a.port).map((a) => [portKey(a.port!), a.key]));
  const others = names.filter((n) => !armOn.has(portKey(n)));
  const why = !control ? "control" : ports?.busy ? `The terminal is running ${ports.busy}, which holds the ports. Stop it first.`
    : !ports ? "Reading the USB ports" : names.length < lr.arms.length
      ? `${names.length} USB serial ${names.length === 1 ? "port is" : "ports are"} plugged in, and this rig has ${lr.arms.length} arms. Plug in and power every arm.`
      : "";

  return (
    <>
      <p className="muted">
        Each arm's driver board shows up as a USB serial port. Studio finds which is which by watching the one that goes
        away when you unplug an arm, the same way lerobot-find-port does, then saves them to robot-config.yaml.
      </p>
      {ports?.moved.map((m) => (
        <Callout key={m.arm} tone="warn">
          <div>
            <span className="strong">{label(m.arm)}</span> is plugged in as <code className="ident">{m.new}</code>, and
            robot-config.yaml still says <code className="ident">{m.old}</code>. Studio knows the board by its USB serial
            number ({m.serial}).
          </div>
          <button className="btn btn-sm btn-primary" disabled={!control || !!ports.busy}
            title={!control ? "Take control first" : ports.busy ? `The terminal is running ${ports.busy}` : undefined}
            onClick={() => setup.savePorts({ [m.arm]: m.new })}>Use {m.new}</button>
        </Callout>
      ))}

      <div className="table-scroll">
        <table className="table su-ports">
          <thead><tr><th>Arm</th><th>Port in robot-config.yaml</th><th>Now</th><th>USB serial</th></tr></thead>
          <tbody>
            {lr.arms.map((a) => {
              const p = a.port ? byName.get(portKey(a.port)) : undefined;
              return (
                <tr key={a.key}>
                  <td className="strong">{label(a.key)}</td>
                  <td>{a.port ? <code className="ident">{a.port}</code> : <span className="text-warn">Not set</span>}</td>
                  <td>
                    {!ports ? <span className="faint">Checking</span>
                      : !a.port ? <span className="faint">Not set</span>
                      : p ? <span className="text-ok su-inline"><Plug aria-hidden /> Plugged in</span>
                      : <span className="text-warn su-inline"><Unplug aria-hidden /> Not plugged in</span>}
                  </td>
                  <td className="faint">{p?.serial ?? (p ? "None reported" : "")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {others.length > 0 && (
        <p className="faint su-small">Other USB serial ports plugged in: {others.map((o) => <code key={o} className="ident su-gap">{o}</code>)}</p>
      )}

      <div className="su-card">
        {!wizard ? (
          <div className="su-wiz-start">
            <div>
              <div className="strong">Find every arm's port</div>
              <p className="muted">Keep every arm plugged in and powered. Studio then asks you to unplug one arm at a time and
                plug it back in. When the last one is found it saves all of them.</p>
            </div>
            {why === "control" ? <NeedControl /> : (
              <div className="su-act">
                <button className="btn btn-primary" disabled={!!why} onClick={() => setup.startWizard(lr.arms.map((a) => a.key))}>
                  <ScanSearch aria-hidden /> Find ports
                </button>
                {why && <span className="su-why">{why}</span>}
              </div>
            )}
          </div>
        ) : <Wizard lr={lr} names={names} />}
      </div>

      {saved?.what === "ports" && !wizard && (
        <Callout tone="ok">
          Saved {labels(saved.keys)} to robot-config.yaml.
          {saved.backup && <> The old file is kept at <code className="ident">{saved.backup}</code>.</>}
          {saved.no_serial && saved.no_serial.length > 0 && (
            <> {labels(saved.no_serial)} {saved.no_serial.length === 1 ? "reports" : "report"} no USB serial number, so Studio
              cannot follow {saved.no_serial.length === 1 ? "it" : "them"} to a new port name. Run the finder again if a port changes.</>
          )}
        </Callout>
      )}

      <StepCommands lr={lr} step="ports" intro="The same thing from the terminal, with LeRobot's own tool:" />
    </>
  );
}

function Wizard({ lr, names }: { lr: LeRobotView; names: string[] }) {
  const w = useSetup((s) => s.wizard)!;
  const done = w.i >= w.arms.length;
  const arm = done ? null : w.arms[w.i];
  const taken = new Set(Object.values(w.found));
  const left = names.filter((n) => !taken.has(n));
  const last = !done && w.i === w.arms.length - 1 && w.phase === "unplug";
  return (
    <div className="su-wiz">
      <ol className="su-wiz-arms">
        {w.arms.map((k, j) => (
          <li key={k} className={j < w.i ? "is-done" : j === w.i ? "is-now" : ""}>
            <span className="su-wiz-dot" aria-hidden>{j < w.i ? <CircleCheck /> : j + 1}</span>
            <span className="strong">{label(k)}</span>
            {w.found[k] && <code className="ident faint">{w.found[k]}</code>}
          </li>
        ))}
      </ol>
      <div className="su-wiz-now" aria-live="polite">
        {done ? (
          w.saving ? <><LoaderCircle className="spin" aria-hidden /> Saving to robot-config.yaml</>
            : <><span>Found every arm. Saving did not finish.</span><button className="btn btn-sm" onClick={() => setup.retrySave()}>Save again</button></>
        ) : w.phase === "unplug" ? (
          <><Unplug aria-hidden /> <span>Unplug <span className="strong">{label(arm!)}</span>'s USB cable from the Mac.</span></>
        ) : (
          <><Plug aria-hidden /> <span><code className="ident">{w.gone}</code> went away. Plug <span className="strong">{label(arm!)}</span> back in.</span></>
        )}
      </div>
      {w.problem && <p className="text-warn su-small">{w.problem}</p>}
      <div className="su-act">
        {last && left.length === 1 && (
          <button className="btn btn-sm" onClick={() => setup.useLeftover(left[0])}>
            Skip: {label(arm!)} is the one left, {left[0]}
          </button>
        )}
        <button className="btn btn-sm btn-ghost" onClick={() => setup.cancelWizard()}>Cancel</button>
        {lr.arms.length > 1 && w.i > 0 && !done && (
          <button className="btn btn-sm btn-ghost" onClick={() => setup.startWizard(w.arms)}>Start over</button>
        )}
      </div>
    </div>
  );
}

// -- 3 motors, 4 calibrate, 5 teleop ------------------------------------------------------------------
function MotorsStep({ lr }: { lr: LeRobotView }) {
  return (
    <>
      <p className="muted">
        Brand-new motors all answer to id 1, so the board cannot tell them apart. Once per arm, LeRobot gives each of
        the six motors its own id and the bus speed (baud rate), one motor at a time, starting with the gripper: only
        that motor is plugged into the board, not chained to the others. The motor stores both in its own memory
        (EEPROM), so this is done once.
      </p>
      <p className="muted">Some kits sold assembled ship with the ids already set. If yours did, skip this step.</p>
      <StepCommands lr={lr} step="motors" />
    </>
  );
}

function CalibrateStep({ lr }: { lr: LeRobotView }) {
  const missing = lr.arms.filter((a) => !a.calibrated);
  return (
    <>
      {missing.length ? (
        <Callout tone="warn">{labels(missing.map((a) => a.key))} {missing.length === 1 ? "has" : "have"} no calibration file yet.</Callout>
      ) : <Callout tone="ok">Every arm has its calibration file.</Callout>}
      <p className="muted">
        Calibration records each joint's middle and its range, so leader and follower read the same pose as the same
        numbers. LeRobot saves it as id.json. Positions are measured against it, so calibrating again shifts what a
        trained policy sees.
      </p>
      <StepCommands lr={lr} step="calibrate" />
    </>
  );
}

function TeleopStep({ lr }: { lr: LeRobotView }) {
  return (
    <>
      <p className="muted">Move the leader and the follower copies it. Check every joint and the gripper before recording.</p>
      <StepCommands lr={lr} step="teleop" />
      <div className="su-act">
        <button className="btn btn-sm" onClick={() => go("teleop")}>Open Teleoperate</button>
        <button className="btn btn-sm" onClick={() => go("scene")}>Open 3D view</button>
      </div>
    </>
  );
}

// -- 6 cameras ---------------------------------------------------------------------------------------
function CamerasStep({ lr }: { lr: LeRobotView }) {
  const control = useStudio((s) => s.control);
  const cams = useSetup((s) => s.cameras);
  const saved = useSetup((s) => s.saved);
  const opencv = lr.cameras.filter((c) => c.type === "opencv");
  const other = lr.cameras.filter((c) => c.type !== "opencv");
  const initial = () => Object.fromEntries(opencv.filter((c) => isSource(c.source)).map((c) => [c.feature, String(c.source)]));
  const [pick, setPick] = useState<Record<string, string>>(initial);
  const answered = cams.list?.filter((c) => c.ok) ?? [];
  // WHY count saved numbers too: a camera left on "Not set" keeps its saved number, and the server
  // refuses two cameras on one device.
  const effective = opencv.map((c) => pick[c.feature] || (isSource(c.source) ? String(c.source) : "")).filter(Boolean);
  const dup = effective.length !== new Set(effective).size;
  const changed = opencv.some((c) => (pick[c.feature] ?? "") !== (isSource(c.source) ? String(c.source) : ""));

  const save = () => {
    const out: Record<string, number | string> = {};
    for (const [f, v] of Object.entries(pick)) if (v !== "") out[f] = /^\d+$/.test(v) ? Number(v) : v;
    setup.saveCameras(out);
  };

  return (
    <>
      <p className="muted">
        macOS numbers cameras 0, 1, 2 in the order it finds them, and the order can change after a replug or a restart.
        Find them, pick which picture is which camera, and save. Do this at the start of each session.
      </p>
      <div className="su-act">
        {control ? (
          <button className="btn btn-primary" disabled={cams.probing} onClick={() => setup.probe()}>
            {cams.probing ? <LoaderCircle className="spin" aria-hidden /> : <ScanSearch aria-hidden />}
            {cams.probing ? "Opening each camera" : cams.list ? "Find cameras again" : "Find cameras"}
          </button>
        ) : <NeedControl />}
        <span className="su-why">Each camera opens for about a second. Stop any lerobot command first; it may hold them.</span>
      </div>

      {cams.list && (
        cams.list.length === 0 ? <p className="text-warn">No camera answered on numbers 0 to 5.</p> : (
          <div className="su-cam-grid">
            {cams.list.map((c) => {
              const uses = opencv.filter((o) => pick[o.feature] === String(c.source));
              return (
                <figure key={String(c.source)} className={`su-cam ${c.ok ? "" : "is-off"}`}>
                  {c.picture ? <img src={c.picture} alt={`Camera ${c.source}`} /> : <div className="su-cam-none">{c.error ?? "No picture"}</div>}
                  <figcaption>
                    <span className="strong">Camera {String(c.source)}</span>
                    {c.ok && c.width && <span className="faint num">{c.width} x {c.height}{c.fps ? `, ${Math.round(c.fps)} fps` : ""}</span>}
                    {uses.map((u) => <span key={u.feature} className="badge tone-info">{camName(u.feature)}</span>)}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        )
      )}

      {opencv.length > 0 && (
        <div className="su-card">
          <div className="strong">Which picture is which camera</div>
          <div className="su-pick">
            {opencv.map((c) => {
              const cur = pick[c.feature] ?? "";
              const listed = answered.some((a) => String(a.source) === cur);
              return (
                <label key={c.feature} className="field">
                  <span className="field-label" title={c.feature}>
                    {camName(c.feature)}{c.hardware && <span className="faint"> ({c.hardware})</span>}
                  </span>
                  <select className="select" value={cur} onChange={(e) => setPick({ ...pick, [c.feature]: e.target.value })}>
                    <option value="">Not set</option>
                    {cur && !listed && <option value={cur}>Camera {cur} (saved; {cams.list ? "not found now" : "not checked yet"})</option>}
                    {answered.map((a) => <option key={String(a.source)} value={String(a.source)}>Camera {String(a.source)}</option>)}
                  </select>
                </label>
              );
            })}
          </div>
          {dup && <p className="text-warn su-small">Two cameras point at the same picture. Give each its own.</p>}
          <div className="su-act">
            {control ? (
              <button className="btn btn-primary" disabled={dup || !changed || !Object.values(pick).some(Boolean)} onClick={save}>
                Save to robot-config.yaml
              </button>
            ) : <NeedControl />}
            {!changed && <span className="su-why">Matches robot-config.yaml.</span>}
          </div>
          {saved?.what === "cameras" && (
            <Callout tone="ok">Saved.{saved.backup && <> The old file is kept at <code className="ident">{saved.backup}</code>.</>}</Callout>
          )}
        </div>
      )}
      {other.length > 0 && (
        <p className="faint su-small">
          {labels(other.map((c) => c.key))}: {other.length === 1 ? "a" : ""} {String(other[0].type)} camera{other.length > 1 ? "s" : ""},
          picked by serial number in robot-config.yaml, which does not change between sessions.
        </p>
      )}
      <StepCommands lr={lr} step="cameras" intro="From the terminal:" />
    </>
  );
}

// -- 7 align -----------------------------------------------------------------------------------------
/** Also the Align page's body (pages/Align.tsx), without the intro and starting on `initialRoot`. */
export function AlignStep({ lr, intro = true, initialRoot = "" }: { lr: LeRobotView; intro?: boolean; initialRoot?: string }) {
  const link = useStudio((s) => s.link);
  const control = useStudio((s) => s.control);
  const datasets = useSetup((s) => s.datasets);
  const session = useSetup((s) => s.align);
  const status = useSetup((s) => s.alignStatus);
  const stopped = useSetup((s) => s.alignStopped);
  const [root, setRoot] = useState(initialRoot);
  const [episode, setEpisode] = useState("0");
  useEffect(() => { if (link === "open") setup.loadDatasets(); }, [link]);
  const ds = datasets?.find((d) => d.root === root) ?? datasets?.[0];
  const ep = Number(episode);
  const epOk = /^\d+$/.test(episode) && (!ds || ep < ds.episodes);

  return (
    <>
      {intro && (
        <p className="muted">
          A policy learns from pictures taken from fixed camera positions. If a camera moved since the dataset was
          recorded, the policy sees a scene it never trained on. Studio compares each live camera with a still frame of
          the dataset and says which way to move it, about three times a second, until it lines up.
        </p>
      )}
      {!session ? (
        <div className="su-card">
          {datasets === null ? <p className="faint">Looking for datasets on this Mac</p>
            : datasets.length === 0 ? <p className="text-warn">No LeRobot datasets on this Mac. Record one first (step 8).</p> : (
              <div className="su-align-form">
                <label className="field">
                  <span className="field-label">Dataset to match</span>
                  <select className="select" value={ds?.root ?? ""} onChange={(e) => { setRoot(e.target.value); setEpisode("0"); }}>
                    {datasets.map((d) => (
                      <option key={d.root} value={d.root}>{d.repo_id || d.name} ({d.episodes} episodes, {d.cameras.length} cameras)</option>
                    ))}
                  </select>
                </label>
                <label className="field su-ep">
                  <span className="field-label">Episode</span>
                  <input className="input num" value={episode} inputMode="numeric" onChange={(e) => setEpisode(e.target.value.trim())} />
                </label>
                {control ? (
                  <button className="btn btn-primary" disabled={!ds || !epOk || !!status} onClick={() => ds && setup.startAlign(ds.root, ep)}>
                    {status ? <LoaderCircle className="spin" aria-hidden /> : <ScanSearch aria-hidden />} Start
                  </button>
                ) : <NeedControl />}
              </div>
            )}
          {ds?.note && <p className="text-warn su-small">{ds.note}</p>}
          {!epOk && ds && <p className="text-warn su-small">This dataset has episodes 0 to {ds.episodes - 1}.</p>}
          {status && <p className="faint su-small" aria-live="polite">{status}</p>}
          {stopped && stopped !== "Stopped" && <p className="faint su-small">{stopped}</p>}
        </div>
      ) : <AlignLive lr={lr} session={session} />}
      <details className="su-why-box">
        <summary>Why move the camera instead of shifting the picture in software</summary>
        <p>
          A shift in software crops pixels off one edge, so the policy sees less of the scene. It also fixes one depth
          only: when a camera moves, near objects shift more across the picture than far ones (parallax), so no single
          shift makes the table, the arm and the background all line up. Moving the camera back fixes every depth.
        </p>
      </details>
    </>
  );
}

function AlignLive({ lr, session }: { lr: LeRobotView; session: AlignSession }) {
  const control = useStudio((s) => s.control);
  // A window showing the session keeps it alive; with none, the server frees the cameras (ALIGN_IDLE_S).
  useEffect(() => {
    const ping = () => { if (!document.hidden) studio.send({ cmd: "align_alive" }); };
    ping();
    const t = window.setInterval(ping, 30_000);
    document.addEventListener("visibilitychange", ping);
    return () => { window.clearInterval(t); document.removeEventListener("visibilitychange", ping); };
  }, []);
  const opencv = new Set(lr.cameras.filter((c) => c.type === "opencv").map((c) => c.feature));
  const toSave = Object.fromEntries(session.assignment.filter((a) => session.config[a.key] && opencv.has(a.key)).map((a) => [a.key, a.live]));
  const noConfig = session.assignment.filter((a) => !session.config[a.key]).map((a) => camName(a.key));
  const current = Object.fromEntries(lr.cameras.map((c) => [c.feature, c.source]));
  const same = Object.entries(toSave).every(([k, v]) => String(current[k]) === String(v));
  return (
    <div className="su-align">
      <div className="su-act su-align-head">
        <span className="muted">
          Matching episode {session.episode}, frame {session.frame}, of <code className="ident">{session.root.split("/").slice(-2).join("/")}</code>
        </span>
        <span className="term-spacer" />
        {control && (
          <button className="btn btn-sm btn-primary" disabled={!Object.keys(toSave).length || same} onClick={() => setup.saveCameras(toSave)}
            title={same ? "robot-config.yaml already has these numbers" : undefined}>
            Save camera numbers
          </button>
        )}
        <button className="btn btn-sm" disabled={!control} onClick={() => setup.stopAlign()}><Square aria-hidden /> Stop</button>
      </div>
      {session.unsure && session.why && <Callout tone="warn">{session.why} Check each pairing below and change it if it is wrong.</Callout>}
      {session.unmatched_refs.length > 0 && (
        <p className="text-warn su-small">No live camera matched {session.unmatched_refs.map(camName).join(", ")}. Plug it in, then start again.</p>
      )}
      {noConfig.length > 0 && (
        <p className="faint su-small">{noConfig.join(", ")}: robot-config.yaml has no camera with that dataset name, so its number is not saved.</p>
      )}
      <div className="su-align-grid">
        {session.assignment.map((a) => <AlignCard key={String(a.live)} live={a.live} k={a.key} session={session} />)}
      </div>
    </div>
  );
}

function AlignCard({ live, k, session }: { live: number | string; k: string; session: AlignSession }) {
  const control = useStudio((s) => s.control);
  const tick = useSetup((s) => s.ticks[String(live)]);
  const [mix, setMix] = useState(50);
  const [w, h] = tick?.size ?? [640, 480];
  const has = tick && !tick.error && tick.dx !== undefined && tick.dy !== undefined;
  // WHY a weak tier above align.py's own low cutoff (0.1): in the demo the wrong camera scored 0.10 to 0.14 and
  // the right one 0.98, so an offset read from a match under WEAK is a guess. WEAK is a judgement, not measured
  // on real cameras yet.
  const weak = has && !tick.low_match && (tick.response ?? 0) < WEAK;
  const tone = !has ? "neutral" : tick.aligned ? "ok" : tick.low_match || weak ? "warn" : "info";
  // The picture must move by (-dx, -dy) to land on the reference (align.py measure); the arrow draws that, at
  // true scale in live-image pixels.
  const ex = has ? w / 2 - tick.dx! : w / 2, ey = has ? h / 2 - tick.dy! : h / 2;
  const physical = session.physical[k];
  return (
    <figure className={`su-al tone-${tone}`}>
      <div className="su-al-stage" style={{ aspectRatio: `${w} / ${h}` }}>
        {session.references[k] && <img className="su-al-img" src={session.references[k]} alt={`Dataset picture, ${camName(k)}`} />}
        {tick?.picture && <img className="su-al-img" src={tick.picture} alt={`Live camera ${String(live)}`} style={{ opacity: mix / 100 }} />}
        <svg className="su-al-svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden>
          <defs>
            <marker id={`su-head-${String(live)}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" className="su-al-head" />
            </marker>
          </defs>
          <line x1={w / 2} y1={0} x2={w / 2} y2={h} className="su-al-cross" vectorEffect="non-scaling-stroke" />
          <line x1={0} y1={h / 2} x2={w} y2={h / 2} className="su-al-cross" vectorEffect="non-scaling-stroke" />
          {has && !tick.aligned && !tick.low_match && (
            <line x1={w / 2} y1={h / 2} x2={ex} y2={ey} className="su-al-arrow" vectorEffect="non-scaling-stroke" markerEnd={`url(#su-head-${String(live)})`} />
          )}
        </svg>
        {tick?.error && <div className="su-al-err">{tick.error}</div>}
      </div>
      <figcaption>
        <div className="su-al-row">
          <span className="strong">Camera {String(live)}</span>
          <span className="faint">shows</span>
          <select className="select su-al-select" value={k} disabled={!control} onChange={(e) => setup.assign(live, e.target.value)}
            aria-label={`Which dataset camera camera ${String(live)} shows`}>
            {Object.keys(session.references).map((r) => <option key={r} value={r}>{camName(r)}</option>)}
          </select>
        </div>
        {physical && physical !== k.replace(/^observation\.images\./, "") && (
          <p className="faint su-small">This dataset's {camName(k)} recorded the {physical} camera.</p>
        )}
        <div className={`su-al-hint tone-${tone}`} aria-live="polite">
          {has ? <Dial dx={tick.dx!} dy={tick.dy!} aligned={!!tick.aligned} low={!!tick.low_match} /> : <span className="dot" />}
          <span>{tick?.error ? "No picture" : has ? tick.hint : "Waiting for a picture"}</span>
        </div>
        {weak && <p className="text-warn su-small">Weak match. Check that this is the right camera before trusting the arrow.</p>}
        {has && (
          <div className="su-al-nums faint num">
            The live picture sits {Math.abs(tick.dx!).toFixed(0)} px {tick.dx! > 0 ? "right" : "left"} and{" "}
            {Math.abs(tick.dy!).toFixed(0)} px {tick.dy! > 0 ? "low" : "high"} of the dataset's. Match {Math.round((tick.response ?? 0) * 100)}%.
          </div>
        )}
        {tick?.note && <p className="faint su-small">{tick.note}</p>}
        <label className="su-al-mix">
          <span className="faint su-small">Dataset</span>
          <input type="range" min={0} max={100} value={mix} onChange={(e) => setMix(Number(e.target.value))} aria-label="Mix of dataset and live picture" />
          <span className="faint su-small">Live</span>
        </label>
      </figcaption>
    </figure>
  );
}

/** Which way the picture must move, readable at a glance: the arrow points along (-dx, -dy) and grows with the
 * offset up to 40 px. The stage above draws the same move at true scale. */
function Dial({ dx, dy, aligned, low }: { dx: number; dy: number; aligned: boolean; low: boolean }) {
  const m = Math.hypot(dx, dy);
  const len = 5 + 10 * Math.min(1, m / 40);
  const ux = m ? -dx / m : 0, uy = m ? -dy / m : 0;
  const head = `su-dial-${useId().replace(/:/g, "")}`;
  return (
    <svg className="su-dial" viewBox="-20 -20 40 40" aria-hidden>
      <circle r="18" className="su-dial-ring" />
      {aligned ? <path d="M-7,0 L-2,5 L8,-6" className="su-dial-ok" />
        : low ? <text y="5" textAnchor="middle" className="su-dial-q">?</text>
        : <line x1={-ux * 4} y1={-uy * 4} x2={ux * len} y2={uy * len} className="su-dial-arrow" markerEnd={`url(#${head})`} />}
      <defs>
        <marker id={head} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="su-dial-headfill" />
        </marker>
      </defs>
    </svg>
  );
}

// -- 8 record, 9 dataset -----------------------------------------------------------------------------
function RecordStep({ lr }: { lr: LeRobotView }) {
  const n = lr.features.length;
  return (
    <>
      <p className="muted">
        Each frame of a recording holds the followers' {n} joint positions as the state, the leaders' {n} positions as
        the action, and one picture per camera. A policy trained on it expects the same arms and the same camera
        names.
      </p>
      <details className="su-why-box">
        <summary>Exact names in the dataset</summary>
        <div className="key-chips">{lr.features.map((f) => <code key={f} className="ident key-chip">{f}</code>)}</div>
        <div className="key-chips">{lr.cameras.map((c) => <code key={c.feature} className="ident key-chip">{c.feature}</code>)}</div>
      </details>
      <StepCommands lr={lr} step="record" />
    </>
  );
}

function DatasetStep({ lr }: { lr: LeRobotView }) {
  const link = useStudio((s) => s.link);
  const datasets = useSetup((s) => s.datasets);
  const typed = useSetup((s) => s.blanks);
  useEffect(() => { if (link === "open") setup.loadDatasets(); }, [link]);
  const user = typed.hf_user ?? lr.hf_user ?? "";
  const mine = (datasets ?? []).filter((d) => user && d.repo_id.startsWith(`${user}/`));
  return (
    <>
      <p className="muted">Watch an episode before training on it: a dropped camera or a frozen arm shows up here, not in the loss curve.</p>
      {mine.length > 0 && (
        <div className="su-chips">
          <span className="field-label">Recorded on this Mac</span>
          {mine.map((d) => {
            const name = d.repo_id.slice(user.length + 1);
            return (
              <button key={d.root} className={`btn btn-sm ${typed.recorded === name ? "btn-primary" : ""}`} onClick={() => setup.setBlank("recorded", name)}
                title={`${d.episodes} episodes, ${d.frames} frames`}>{name}</button>
            );
          })}
        </div>
      )}
      <StepCommands lr={lr} step="dataset" />
      <p className="muted">
        Once uploaded, a dataset also plays in the browser:{" "}
        <a href="https://huggingface.co/spaces/lerobot/visualize_dataset" target="_blank" rel="noreferrer">
          LeRobot's dataset visualizer <ExternalLink aria-hidden className="su-ext" />
        </a>
      </p>
    </>
  );
}

// -- 10 to 12 ----------------------------------------------------------------------------------------
function LinkStep({ route, text, button, also }: { route: Route; text: string; button: string; also?: { route: Route; button: string } }) {
  return (
    <>
      <p className="muted">{text}</p>
      <div className="su-act">
        <button className="btn btn-primary" onClick={() => go(route)}>{button} <ArrowRight aria-hidden /></button>
        {also && <button className="btn" onClick={() => go(also.route)}>{also.button} <ArrowRight aria-hidden /></button>}
      </div>
    </>
  );
}
