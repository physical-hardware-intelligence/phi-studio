// One arm's motor ids: which of 1..6 answer, which two motors share an id, and a search at every id and rate for the
// ones that do not. The fix for a wrong id or rate is LeRobot's lerobot-setup-motors, one motor at a time; Studio
// detects the arms again when it ends. Server side: rig_motors in rig_api.py, detect.find_motors.
import { CircleAlert, LoaderCircle, Search, SquareTerminal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { FoundArm } from "../lib/armdetect";
import { studio, useStudio } from "../lib/studio";
import { terminal, useTerminal } from "../lib/terminal";

const JOINT = ["", "shoulder pan", "shoulder lift", "elbow flex", "wrist flex", "wrist roll", "gripper"];
interface Finding { level: string; text: string; fix: string }
interface Result { findings: Finding[]; setup: { role: string; cmd: string }[] }

export function MotorIds({ arm }: { arm: FoundArm }) {
  const control = useStudio((s) => s.control);
  const running = useTerminal((s) => s.running);
  const [rate, setRate] = useState<number | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const setupRan = useRef(false);

  useEffect(() => {
    const offs = [
      studio.onMessage("rig_motors", (m) => {
        if (m.port !== arm.port) return;
        if (!m.done) { setRate(m.baud as number); return; }
        setRate(null); setErr(null); setRes({ findings: m.findings as Finding[], setup: m.setup as Result["setup"] });
      }),
      studio.onMessage("error", (m) => { if (m.cmd === "rig_motors") { setRate(null); setErr([m.message, m.fix].filter(Boolean).join(" ")); } }),
    ];
    return () => offs.forEach((off) => off());
  }, [arm.port]);

  // WHY detect again on its own: the ids lerobot-setup-motors wrote are only known once the arm is pinged again
  useEffect(() => {
    const isSetup = !!running && running.command.includes("lerobot-setup-motors") && running.command.includes(arm.port);
    if (isSetup) setupRan.current = true;
    else if (!running && setupRan.current) { setupRan.current = false; setRes(null); studio.send({ cmd: "rig_scan" }); }
  }, [running, arm.port]);

  if (arm.held_by?.length) return null; // the port is someone else's: nothing to ping
  const extra = arm.ids.filter((i) => i < 1 || i > 6);
  return (
    <div className="motor-ids">
      <span className="onb-chips" aria-label="Motor ids">
        {[1, 2, 3, 4, 5, 6].map((i) => {
          const clash = arm.clashes.includes(i);
          const ok = arm.ids.includes(i);
          return (
            <span key={i} className={`chip ${clash ? "sev-error" : ok ? "sev-ok" : "sev-warn"}`}
              title={clash ? "two motors answer at this id" : ok ? "answers" : "does not answer"}>
              {clash ? "✕✕" : ok ? "✓" : "?"} {i} {JOINT[i]}
            </span>
          );
        })}
        {extra.map((i) => <span key={i} className="chip sev-error" title="an SO-101 has ids 1 to 6">id {i}</span>)}
      </span>
      <div className="row-gap">
        <button className="btn btn-sm" disabled={!control || rate !== null}
          title="Pings every id at 1 Mbaud and the low ids at every other rate. Nothing moves, nothing is written."
          onClick={() => { if (studio.send({ cmd: "rig_motors", port: arm.port })) { setRate(1_000_000); setErr(null); } }}>
          {rate !== null ? <LoaderCircle className="spin" aria-hidden /> : <Search aria-hidden />}
          {rate !== null ? `Searching at ${rate} baud` : "Find missing motors"}
        </button>
        {rate === null && !res && <span className="faint t-sm">About 15 seconds.</span>}
      </div>
      {err && <p className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{err}</p>}
      {res && res.findings.length === 0 && <p className="ok-text t-sm">Every motor answers at its id, at 1000000 baud. Detect again.</p>}
      {res && res.findings.map((f) => (
        <p key={f.text} className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{f.text} {f.fix}</p>
      ))}
      {res && res.findings.some((f) => f.fix.startsWith("Set motor ids")) && (
        <div className="motor-setup">
          <p className="faint t-sm">
            Set motor ids runs LeRobot's lerobot-setup-motors in the terminal. It names one motor at a time, gripper first:
            plug only that motor into the board, press Enter, and it writes that motor's id and rate. Studio detects the
            arms again when it ends.
          </p>
          <div className="row-gap">
            {res.setup.map((s) => (
              <button key={s.role} className="btn btn-sm" disabled={!control} title={s.cmd} onClick={() => terminal.run(s.cmd)}>
                <SquareTerminal aria-hidden />Set motor ids{res.setup.length > 1 ? ` as a ${s.role}` : ""}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
