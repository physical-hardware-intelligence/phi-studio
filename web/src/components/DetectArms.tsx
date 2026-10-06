// Rig setup's Detect arms: ping every motor on every USB port, read each arm's calibration registers, and put each
// arm in its slot when its registers match a calibration file exactly. Read-only: nothing moves, nothing is written.
// Server side: rig_scan in src/phi_studio/rig_api.py and detect.py; placement: lib/armdetect.ts.
import { CircleAlert, LoaderCircle, ScanSearch } from "lucide-react";
import { useEffect, useRef, useState } from "react";
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
        <p key={a.port} className="warn-text t-sm">
          <CircleAlert aria-hidden className="ico-inline" /><span className="mono">{shortPort(a.port)}</span>: {a.problem}
          {a.held_by?.[0] && (
            <button className="btn btn-sm" disabled={!control || scanning} title={a.held_by[0].command}
              onClick={() => { if (studio.send({ cmd: "rig_stop", pid: a.held_by![0].pid })) setScanning(true); }}>
              Stop {a.held_by[0].name}
            </button>
          )}
        </p>
      ))}
      {p && p.unplaced.map((a) => (
        <p key={a.port} className="faint t-sm">
          <span className="mono">{shortPort(a.port)}</span>:{" "}
          {a.role ? `a ${a.role} whose calibration names no side. Use Find to place it.`
            : a.match ? `matches no calibration file exactly (nearest ${a.match.split("/").pop()}, ${a.match_deg?.toFixed(0)} deg off). Calibrate it, or use Find.`
            : "no calibration files to compare. Use Find, then calibrate."}
        </p>
      ))}
      {p?.note && <p className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{p.note}</p>}
      {p?.ids && <p className="ok-text t-sm">Calibration files {p.ids.follower} and {p.ids.leader} hold these arms exactly. The next step keeps them.</p>}
    </div>
  );
}
