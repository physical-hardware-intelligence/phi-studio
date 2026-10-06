// Calibrate page, the way that works on every Mac: LeRobot's own lerobot-calibrate, one arm at a time, in
// Studio's terminal, filled in from robot-config.yaml. Before a run Studio backs up the arm's file and turns a
// linked file into a copy; after it, Studio reads the arm's registers and checks them against the file.
// Server side: rig_cal_* in src/phi_studio/rig_api.py; commands: rigspec.lerobot_commands.
import { CircleAlert, CircleCheck, LoaderCircle, Power, RefreshCw, Save, SquareTerminal, Unplug } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { label } from "../lib/labels";
import { studio, useStudio } from "../lib/studio";
import { shortCommand, terminal, useTerminal } from "../lib/terminal";
import { read, RunWatch, type Verdict } from "../lib/termcal";

export function TerminalCalibrate() {
  const lr = useStudio((s) => s.files.index?.lerobot ?? null);
  const link = useStudio((s) => s.link);
  useEffect(() => { if (link === "open") studio.loadFiles(); }, [link]); // the commands come with the files index
  const control = useStudio((s) => s.control);
  const mock = useStudio((s) => s.mock);
  const st = useStudio((s) => s.state?.state ?? null);
  const running = useTerminal((s) => s.running?.command ?? null);
  const termError = useTerminal((s) => s.error);
  const [res, setRes] = useState<Record<string, Verdict>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [watch, setWatch] = useState<RunWatch | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cmds = (lr?.commands ?? []).filter((c) => c.step === "calibrate" && c.arm);
  const byArm = useRef<Record<string, string>>({});
  byArm.current = Object.fromEntries(cmds.map((c) => [c.arm!, c.cmd]));
  const done = (arm: string) => setBusy((b) => ({ ...b, [arm]: false }));

  useEffect(() => {
    const offs = [
      studio.onMessage("rig_cal_prepared", (m) => {
        done(m.arm);
        const cmd = byArm.current[m.arm];
        if (!cmd) return;
        setWatch(new RunWatch(m.arm));
        terminal.run(cmd);
      }),
      studio.onMessage("rig_cal_verified", (m) => { done(m.arm); setRes((r) => ({ ...r, [m.arm]: m as Verdict })); }),
      studio.onMessage("error", (m) => {
        if (typeof m.cmd !== "string" || !m.cmd.startsWith("rig_cal_")) return;
        setBusy({}); setError([m.message, m.fix].filter(Boolean).join(" "));
      }),
    ];
    return () => { for (const off of offs) off(); };
  }, []);

  const send = (cmd: string, arm: string) => {
    setError(null);
    if (studio.send({ cmd, arm })) setBusy((b) => ({ ...b, [arm]: true }));
  };

  // The run has ended when the terminal is back at the prompt: check what it left in the motors.
  useEffect(() => {
    if (watch && watch.update(running)) { setWatch(null); send("rig_cal_verify", watch.arm); }
  }, [running, watch]);
  // The terminal refused the command (Studio still held the arms): there is no run to wait for.
  useEffect(() => { if (watch && termError && !watch.started) setWatch(null); }, [termError, watch]);

  if (mock) return null; // simulated arms have no serial port for LeRobot: By hand and Auto-calibrate below
  if (!cmds.length) {
    return (
      <section className="panel">
        <h2 className="panel-title"><SquareTerminal aria-hidden className="ico-inline" />Calibrate in the terminal</h2>
        <p className="faint t-sm">
          {lr?.error ? `robot-config.yaml cannot be read: ${lr.error}` : "Save the rig on Rig setup first: Studio fills each arm's command from robot-config.yaml."}
        </p>
      </section>
    );
  }
  const held = st !== null && st !== "DISCONNECTED";
  const other = running && !watch ? shortCommand(running) : null;
  const why = !control ? "Take control (top right) to calibrate."
    : held ? "Studio is connected to the arms, so their ports are busy."
    : other ? `The terminal is running ${other}.`
    : watch ? `Calibrating ${label(watch.arm)} in the terminal.` : null;
  const anyBusy = Object.values(busy).some(Boolean);
  const allOk = cmds.every((c) => res[c.arm!]?.exact && !res[c.arm!]?.problem);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title"><SquareTerminal aria-hidden className="ico-inline" />Calibrate in the terminal</h2>
        <button className="btn btn-sm btn-ghost" disabled={!!why || anyBusy} title="Read every arm's registers and compare each with its file. Nothing moves."
          onClick={() => { for (const c of cmds) send("rig_cal_verify", c.arm!); }}>
          <RefreshCw aria-hidden /> Check all
        </button>
      </div>
      <ol className="t-sm faint">
        <li>Pick an arm. LeRobot's calibration runs in the terminal below.</li>
        <li>If it asks about an existing file, type <span className="mono">c</span> and press Enter to calibrate again. Enter alone writes that file into the motors, so Studio holds it back unless the port and the id are the same arm in robot-config.yaml.</li>
        <li>Put every joint in the middle of its range and press Enter. Wrist roll reads 0 at the twist it has now, so give it the same twist on every arm.</li>
        <li>Move every joint except wrist roll from one end to the other, then press Enter.</li>
        <li>When the command ends, Studio checks the motors against the file.</li>
      </ol>
      {held && (
        <p className="warn-text t-sm">
          <CircleAlert aria-hidden className="ico-inline" />{why}{" "}
          <button className="btn btn-sm" disabled={!control} onClick={() => studio.send({ cmd: "disconnect" })}>
            <Unplug aria-hidden /> Disconnect
          </button>
        </p>
      )}
      {!held && why && <p className="faint t-sm">{why}</p>}
      {error && <p className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{error}</p>}
      <ul className="pick-list">
        {cmds.map((c) => {
          const arm = c.arm!;
          const v = res[arm];
          const r = v ? read(v) : null;
          return (
            <li key={arm} className={`pick-row pick-wrap ${watch?.arm === arm ? "is-active" : ""}`}>
              <div className="pick-text">
                <div className="pick-name"><span className="ellipsis">{label(arm)}</span></div>
                {r ? (
                  <span className={r.tone === "ok" ? "ok-text t-sm" : "warn-text t-sm"}>
                    {r.tone === "ok" ? <CircleCheck aria-hidden className="ico-inline" /> : <CircleAlert aria-hidden className="ico-inline" />}{r.text}
                  </span>
                ) : <span className="faint t-sm mono">{c.cmd.match(/--\w+\.id=(\S+)/)?.[1] ?? ""}.json</span>}
              </div>
              <div className="row-gap">
                {busy[arm] && <LoaderCircle className="spin" aria-hidden />}
                <button className="btn btn-sm" disabled={!!why || anyBusy} title={c.cmd}
                  onClick={() => send("rig_cal_prepare", arm)}>
                  <SquareTerminal aria-hidden /> Calibrate
                </button>
                <button className="btn btn-sm btn-ghost" disabled={!!why || anyBusy} onClick={() => send("rig_cal_verify", arm)}>Check</button>
                {r?.canSave && (
                  <button className="btn btn-sm btn-ghost" disabled={!!why || anyBusy}
                    title={`Writes ${v!.path} from this arm's registers, keeping a backup. Only if this port holds the ${label(arm)}.`}
                    onClick={() => send("rig_cal_from_motors", arm)}>
                    <Save aria-hidden /> Save from motors
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      {allOk && (
        <p className="ok-text t-sm">
          <CircleCheck aria-hidden className="ico-inline" />Every arm matches its file.{" "}
          <button className="btn btn-sm btn-primary" disabled={!control || held} onClick={() => studio.send({ cmd: "connect" })}>
            <Power aria-hidden /> Connect
          </button>
        </p>
      )}
    </section>
  );
}
