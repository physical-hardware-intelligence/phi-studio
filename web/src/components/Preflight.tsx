// Before a policy moves the arms: are the arms up, does each one match its own calibration, and do the cameras the
// policy reads sit where its training data was recorded. Run and Evaluate show this above Start. The arm checks
// say what to do; the worker enforces them anyway. The camera check is the one only Studio can make, so it blocks
// Start until the cameras are aligned, or the person chooses to run without.
import { Check, CircleAlert, Minus, ScanSearch } from "lucide-react";
import { useState, type ReactNode } from "react";
import { fmtAgo } from "../lib/data";
import { cameraCheck, type Tone } from "../lib/preflight";
import { useStudio, type PolicyInfo } from "../lib/studio";

interface Item { key: string; name: string; tone: Tone; state: string; action?: ReactNode }

const ARMS_UP = ["READY", "ARMED", "MOVING", "STOPPED"];

/** The three checks and what blocks Start. `skip`: the person chose to run without the camera check. */
export function usePreflight(policy: PolicyInfo | undefined) {
  const state = useStudio((s) => s.state?.state ?? "DISCONNECTED");
  const identity = useStudio((s) => s.identity);
  const mock = useStudio((s) => s.mock);
  const align = useStudio((s) => s.align);
  const [skip, setSkip] = useState(false);
  const cam = cameraCheck(policy, align, mock, Date.now(), fmtAgo);
  const bad = identity.filter((a) => !a.ok).length;
  const items: Item[] = [
    {
      key: "arms", name: "Arms",
      tone: ARMS_UP.includes(state) ? "ok" : "warn",
      state: ARMS_UP.includes(state) ? "ready" : state === "IDENTIFIED" ? "confirm them" : state === "FAULT" ? "fault" : "not connected",
      action: ARMS_UP.includes(state) ? undefined : <a className="link-btn t-sm" href="#/overview">Home</a>,
    },
    {
      key: "cal", name: "Calibration",
      tone: !identity.length ? "neutral" : bad ? "warn" : "ok",
      state: !identity.length ? "not read yet" : bad ? `${bad} ${bad === 1 ? "arm does" : "arms do"} not match` : "matches",
      action: bad ? <a className="link-btn t-sm" href="#/calibrate">Calibrate</a> : undefined,
    },
    {
      key: "cams", name: "Cameras", tone: cam.block && skip ? "neutral" : cam.tone,
      state: cam.block && skip ? "skipped for this run" : cam.state,
      action: cam.tone === "neutral" && !skip ? undefined : (
        <span className="row-gap">
          {/* the policy's own training data, when known: what its cameras must match */}
          <a className="btn btn-sm" href={policy?.dataset ? `#/align/${encodeURIComponent(policy.dataset)}` : "#/align"}><ScanSearch aria-hidden />Align</a>
          {cam.block && (
            <button type="button" className="link-btn t-sm" onClick={() => setSkip(!skip)}>{skip ? "Check again" : "Run without"}</button>
          )}
        </span>
      ),
    },
  ];
  return { items, block: skip ? null : cam.block };
}

export function Preflight({ items }: { items: Item[] }) {
  return (
    <div className="preflight" role="group" aria-label="Before you run">
      {items.map((i) => (
        <div key={i.key} className={`pf-item tone-${i.tone}`}>
          <span className="pf-mark" aria-hidden>{i.tone === "ok" ? <Check /> : i.tone === "warn" ? <CircleAlert /> : <Minus />}</span>
          <span className="pf-name">{i.name}</span>
          <span className="pf-state">{i.state}</span>
          {i.action}
        </div>
      ))}
    </div>
  );
}
