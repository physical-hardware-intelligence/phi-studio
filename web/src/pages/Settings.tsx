import { BookOpen, ChevronRight, Flag, FolderOpen, ListChecks, ListOrdered, Pencil, Ruler } from "lucide-react";
import type { ComponentType } from "react";
import { useNotes } from "../lib/data";
import { go, useStudio, type Route } from "../lib/studio";

interface Row { route: Route; icon: ComponentType<{ "aria-hidden"?: boolean }>; title: string; sub: string }

const ROWS: Row[] = [
  { route: "onboard", icon: Pencil, title: "Rig", sub: "Name, arms, calibration, cameras" },
  { route: "setup", icon: ListOrdered, title: "Advanced setup", sub: "Every LeRobot step, one by one" },
  { route: "calibrate", icon: Ruler, title: "Calibrate by hand", sub: "Middle pose, then each joint's range" },
  { route: "checks", icon: ListChecks, title: "Checks", sub: "This Mac, the rig, Studio" },
  { route: "issues", icon: Flag, title: "Issues", sub: "Notes and excluded episodes" },
  { route: "files", icon: FolderOpen, title: "Files", sub: "Config, calibration, ports" },
  { route: "guide", icon: BookOpen, title: "Guide", sub: "How each page works" },
];

// Everything that is not the daily loop, one click away.
export function Settings() {
  const fails = useStudio((s) => s.checks.results?.filter((r) => r.status === "fail").length ?? 0);
  const open = useNotes(null, null)?.filter((n) => n.kind === "issue" && n.status === "open").length ?? 0;
  const hint: Partial<Record<Route, string>> = { ...(fails ? { checks: `${fails} failed` } : {}), ...(open ? { issues: `${open} open` } : {}) };
  return (
    <div className="page">
      <section className="panel settings">
        {ROWS.map((r) => {
          const Icon = r.icon;
          return (
            <a key={r.route} className="settings-row" href={`#/${r.route}`} onClick={(e) => { e.preventDefault(); go(r.route); }}>
              <span className="settings-icon"><Icon aria-hidden /></span>
              <span className="settings-text"><span className="strong">{r.title}</span><span className="faint t-sm">{r.sub}</span></span>
              {hint[r.route] && <span className="badge tone-warn">{hint[r.route]}</span>}
              <ChevronRight aria-hidden className="settings-go" />
            </a>
          );
        })}
      </section>
    </div>
  );
}
