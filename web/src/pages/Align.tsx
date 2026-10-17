// Camera align on a page of its own. Before a policy moves the arms (Run, Evaluate) and before more episodes join a
// dataset, each camera goes back where that dataset was recorded. The engine is align.py with Set up's align step;
// this page gives it a home, and the server keeps its verdict for Run and Evaluate (components/Preflight.tsx).
import { Check, CircleAlert } from "lucide-react";
import { useEffect } from "react";
import { Notices } from "../components/Notices";
import { fmtAgo } from "../lib/data";
import { studio, useStudio } from "../lib/studio";
import { AlignStep } from "./Setup";

/** #/align/<dataset>: Run and Evaluate send the policy's training dataset (Preflight.tsx). */
function wanted(): string {
  const part = location.hash.replace(/^#\/?/, "").split("/").slice(1).join("/");
  try { return decodeURIComponent(part); } catch { return ""; }
}

export function Align() {
  const link = useStudio((s) => s.link);
  const index = useStudio((s) => s.files.index);
  const last = useStudio((s) => s.align);
  const mock = useStudio((s) => s.mock);
  useEffect(() => { if (link === "open") studio.loadFiles(); }, [link]);
  const lr = index?.lerobot ?? null;
  const off = last ? Object.values(last.cameras).filter((ok) => !ok).length : 0;
  return (
    <div className="page align-page">
      <Notices />
      {last && (
        <div className={`align-last tone-${last.aligned ? "ok" : "warn"}`}>
          {last.aligned ? <Check aria-hidden /> : <CircleAlert aria-hidden />}
          <span>
            {last.aligned ? "Every camera in line" : `${off} of ${Object.keys(last.cameras).length} cameras off`} with{" "}
            <span className="mono">{last.root.split("/").filter(Boolean).pop()}</span>
          </span>
          <span className="faint">{fmtAgo(last.at)}</span>
        </div>
      )}
      {mock && <p className="faint t-sm">Uses the cameras plugged into this Mac, even with simulated arms.</p>}
      {!lr ? <div className="empty">Loading</div> : <AlignStep lr={lr} intro={false} initialRoot={wanted() || last?.root || ""} />}
    </div>
  );
}
