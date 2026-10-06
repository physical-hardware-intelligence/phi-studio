// Rig setup's Detect arms: ping every motor on every USB port, read each arm's calibration registers, and put each
// arm in its slot when its registers match a calibration file exactly. Read-only: nothing moves, nothing is written.
// Server side: rig_scan in src/phi_studio/rig_api.py and detect.py; placement: lib/armdetect.ts.
import { Activity, CircleAlert, LoaderCircle, ScanSearch } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { MotorIds } from "./MotorIds";
import { place, type FoundArm, type Ids, type Placed } from "../lib/armdetect";
import { shortPort } from "../lib/rig";
import { studio, useStudio } from "../lib/studio";

export function DetectArms({ prefer, onPlaced }: { prefer: Partial<Ids> | null; onPlaced: (p: Placed) => void }) {
  const control = useStudio((s) => s.control);
  const mock = useStudio((s) => s.mock);
  const state = useStudio((s) => s.state?.state ?? null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<{ arms: FoundArm[]; placed: Placed } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef({ prefer, onPlaced });
  latest.current = { prefer, onPlaced };
  const auto = useRef(false);

  useEffect(() => {
    const off1 = studio.onMessage("rig_scan", (m) => {
      const arms = m.arms as FoundArm[];
      const placed = place(arms, latest.current.prefer);
      setScanning(false); setError(null); setResult({ arms, placed });
      latest.current.onPlaced(placed);
    });
    const off2 = studio.onMessage("error", (m) => {
      if (m.cmd !== "rig_scan" && m.cmd !== "rig_stop") return;
      setScanning(false); setError([m.message, m.fix].filter(Boolean).join(" "));
    });
    return () => { off1(); off2(); };
  }, []);

  // WHY only when the ports are free: a connected real rig holds them and the scan would be refused.
  const free = mock || state === null || state === "DISCONNECTED";
  const scan = () => { if (studio.send({ cmd: "rig_scan" })) { setScanning(true); setError(null); } };
  useEffect(() => {
    if (!auto.current && control && free) { auto.current = true; scan(); }
  }, [control, free]);

  const p = result?.placed;
  const n = result?.arms.length ?? 0;
  const placed = p ? Object.keys(p.ports).length : 0;
  return (
    <div className="onb-detect">
      <div className="row-gap">
        <button className="btn" onClick={scan} disabled={!control || scanning} title="Pings every motor and reads each arm's calibration. Nothing moves.">
          {scanning ? <LoaderCircle className="spin" aria-hidden /> : <ScanSearch aria-hidden />}{scanning ? "Detecting" : result ? "Detect again" : "Detect arms"}
        </button>
        <span className="faint t-sm">
          {!control ? "Take control (top right) to detect arms."
            : scanning ? "Pinging motors 1 to 10 on every USB port"
            : result ? (n ? `Found ${n} ${n === 1 ? "arm" : "arms"}, placed ${placed}.` : "No arm answered. Check each arm's power and USB cable.")
            : ""}
        </span>
      </div>
      {error && <p className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{error}</p>}
      {p && p.broken.map((a) => (
        <div key={a.port} className="warn-text t-sm">
          <CircleAlert aria-hidden className="ico-inline" /><span className="mono">{shortPort(a.port)}</span>: {a.problem}
          {a.held_by?.[0] && (
            <button className="btn btn-sm" disabled={!control || scanning} title={a.held_by[0].command}
              onClick={() => { if (studio.send({ cmd: "rig_stop", pid: a.held_by![0].pid })) setScanning(true); }}>
              Stop {a.held_by[0].name}
            </button>
          )}
          {!a.held_by?.length && <MotorIds arm={a} />}
        </div>
      ))}
      {p && p.unplaced.map((a) => (
        <p key={a.port} className="faint t-sm">
          <span className="mono">{shortPort(a.port)}</span>:{" "}
          {a.role ? `a ${a.role} whose calibration names no side. Use Find to place it.`
            : a.match ? `matches no calibration file exactly (nearest ${a.match.split("/").pop()}, ${a.match_deg?.toFixed(0)} deg off). Calibrate it, or use Find.`
            : "no calibration files to compare. Use Find, then calibrate."}
        </p>
      ))}
      {result && result.arms.some((a) => !a.problem) && <MotorChecks arms={result.arms.filter((a) => !a.problem)} placed={p!} />}
      {p?.note && <p className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{p.note}</p>}
      {p?.ids && <p className="ok-text t-sm">Calibration files {p.ids.follower} and {p.ids.leader} hold these arms exactly. The next step keeps them.</p>}
    </div>
  );
}

interface MotorResult {
  id: number; ok: boolean; polls: number; bus_errors: number; first_error_s: number | null;
  min_volt: number | null; max_current: number | null; temp: number | null; status: number | null; error: string | null;
}
const JOINT = ["", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];

/** Powered check, one arm at a time: each motor holds still with torque on while Studio polls the whole bus.
 * WHY: on 2026-10-05 one motor broke its bus only while driving current, so every read-only check passed. */
function MotorChecks({ arms, placed }: { arms: FoundArm[]; placed: Placed }) {
  const control = useStudio((s) => s.control);
  const [busy, setBusy] = useState<string | null>(null);
  const [res, setRes] = useState<Record<string, MotorResult[]>>({});
  useEffect(() => {
    const off1 = studio.onMessage("rig_check", (m) => {
      const port = m.port as string;
      if (m.done) { setBusy(null); if (m.motors) setRes((r) => ({ ...r, [port]: m.motors as MotorResult[] })); return; }
      if (m.motor) setRes((r) => ({ ...r, [port]: [...(r[port] ?? []), m.motor as MotorResult] }));
      else setRes((r) => ({ ...r, [port]: [] }));
    });
    const off2 = studio.onMessage("error", (m) => { if (m.cmd === "rig_check") setBusy(null); });
    return () => { off1(); off2(); };
  }, []);
  const slot = (port: string) => Object.entries(placed.ports).find(([, v]) => v === port)?.[0]?.replace("_", " ") ?? shortPort(port);
  return (
    <div className="onb-detect">
      <p className="faint t-sm">
        Check motors powers each motor in turn for 3 seconds, holding it still, and watches the whole bus. Keep a hand
        under the arm. Run it on followers before the first teleop.
      </p>
      <div className="row-gap">
        {arms.map((a) => (
          <button key={a.port} className="btn btn-sm" disabled={!control || !!busy}
            onClick={() => { if (studio.send({ cmd: "rig_check", port: a.port })) setBusy(a.port); }}>
            {busy === a.port ? <LoaderCircle className="spin" aria-hidden /> : <Activity aria-hidden />}Check {slot(a.port)}
          </button>
        ))}
      </div>
      {Object.entries(res).map(([port, ms]) => ms.length > 0 && (
        <div key={port} className="t-sm">
          <span className="strong">{slot(port)}</span>{" "}
          {ms.map((m) => (
            <span key={m.id} className={`chip ${m.ok ? "sev-ok" : "sev-error"}`} title={
              m.error ?? `${m.bus_errors}/${m.polls} bus errors, ${m.min_volt ?? "?"} V min, current ${m.max_current ?? "?"}, ${m.temp ?? "?"} C`}>
              {m.ok ? "✓" : "✕"} {m.id} {JOINT[m.id]}
            </span>
          ))}
          {ms.filter((m) => !m.ok).map((m) => (
            <p key={m.id} className="warn-text t-sm">
              <CircleAlert aria-hidden className="ico-inline" />Motor {m.id} ({JOINT[m.id]}):{" "}
              {m.error ? m.error
                : m.bus_errors ? `the bus failed ${m.bus_errors} of ${m.polls} reads while it held torque (first after ${m.first_error_s}s, lowest ${m.min_volt} V). Reseat its two 3-pin cables and check again; if it still fails, swap the cables, then the motor.`
                : `it reports error flags ${m.status}.`}
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}
