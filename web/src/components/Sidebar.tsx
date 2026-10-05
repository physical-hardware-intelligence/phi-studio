import { Bot, Box, ClipboardCheck, Cpu, Database, Gamepad2, House, Moon, PackageSearch, ScanSearch, Settings, Sun } from "lucide-react";
import type { ComponentType } from "react";
import { useNotes } from "../lib/data";
import { go, setTheme, useRoute, useStudio, useTheme, type Route } from "../lib/studio";
import { PhiMark } from "./PhiMark";

interface Item { route: Route; label: string; icon: ComponentType<{ "aria-hidden"?: boolean }> }

// The daily loop only: drive the rig, look at the data, train, run, evaluate. Everything else (setup,
// calibration by hand, checks, files, guide) lives under Settings, one click away.
const RIG: Item[] = [
  { route: "overview", label: "Home", icon: House },
  { route: "teleop", label: "Teleop", icon: Gamepad2 },
  { route: "scene", label: "3D view", icon: Box },
  { route: "align", label: "Align", icon: ScanSearch },
];
const DATA: Item[] = [
  { route: "data", label: "Datasets", icon: Database },
];
const POLICIES: Item[] = [
  { route: "train", label: "Train", icon: Cpu },
  { route: "models", label: "Models", icon: PackageSearch },
  { route: "policy", label: "Run", icon: Bot },
  { route: "evaluate", label: "Evaluate", icon: ClipboardCheck },
];
const FOOT: Item = { route: "settings", label: "Settings", icon: Settings };

// WHY every route: the Claude panel names the current page, including the ones reached from Settings.
export const PAGE_LABELS: Record<Route, string> = {
  overview: "Home", teleop: "Teleop", scene: "3D view", align: "Align", data: "Datasets", issues: "Issues", train: "Train",
  models: "Models", policy: "Run", evaluate: "Evaluate", settings: "Settings", onboard: "Rig setup",
  setup: "Advanced setup", calibrate: "Calibrate", checks: "Checks", files: "Files", guide: "Guide",
};
// Pages reached from Settings keep Settings lit in the sidebar.
const UNDER_SETTINGS = new Set<Route>(["settings", "onboard", "setup", "calibrate", "checks", "files", "guide", "issues"]);

export function Sidebar() {
  const route = useRoute();
  const theme = useTheme();
  const mock = useStudio((s) => s.mock);
  const link = useStudio((s) => s.link);
  const hint = useHints();
  const active = (r: Route) => route === r || (r === "settings" && UNDER_SETTINGS.has(route)) || (r === "data" && route === "issues");

  const item = (it: Item) => {
    const Icon = it.icon;
    const h = hint[it.route];
    return (
      <a key={it.route} href={`#/${it.route}`} className={`nav-item ${active(it.route) ? "is-active" : ""}`}
        aria-current={route === it.route ? "page" : undefined} onClick={(e) => { e.preventDefault(); go(it.route); }}>
        <Icon aria-hidden />
        <span className="nav-label">{it.label}</span>
        {h && <span className={`nav-hint tone-${h.tone}`}>{h.dot ? <span className="dot" /> : null}{h.text}</span>}
      </a>
    );
  };

  return (
    <nav className="sidebar" aria-label="Studio">
      <div className="brand">
        <PhiMark className="brand-mark" size={30} />
        <div className="brand-text">
          <span className="brand-name">Phi Studio</span>
          <span className="brand-rig">{mock ? "Simulated arms" : "SO-101"}</span>
        </div>
      </div>
      <div className="nav-group">{RIG.map(item)}</div>
      <div className="nav-group">
        <div className="nav-group-label">Data</div>
        {DATA.map(item)}
      </div>
      <div className="nav-group">
        <div className="nav-group-label">Policies</div>
        {POLICIES.map(item)}
      </div>
      <div className="sidebar-foot">
        <div className="nav-group">{item(FOOT)}</div>
        <div className={`link-status tone-${link === "open" ? "ok" : link === "refused" || link === "down" ? "danger" : "warn"}`}>
          <span className="dot" />
          {link === "open" ? "Connected" : link === "refused" ? "Not authorised" : link === "down" ? "Studio is not running" : "Reconnecting"}
        </div>
        <button className="btn btn-ghost btn-sm theme-toggle" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
          {theme === "dark" ? <Sun aria-hidden /> : <Moon aria-hidden />}
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>
    </nav>
  );
}

type Hint = { text: string; tone: "ok" | "warn" | "danger" | "info"; dot?: boolean };

function useHints(): Partial<Record<Route, Hint>> {
  const state = useStudio((s) => s.state);
  const needCal = useStudio((s) => s.identity.some((a) => !a.ok));
  const fails = useStudio((s) => s.checks.results?.filter((r) => r.status === "fail").length ?? 0);
  const open = useNotes(null, null)?.filter((n) => n.kind === "issue" && n.status === "open").length ?? 0;
  const off = useStudio((s) => s.align !== null && !s.align.aligned);
  const out: Partial<Record<Route, Hint>> = {};
  if (off) out.align = { text: "Off", tone: "warn" };
  if (open) out.data = { text: `${open} open`, tone: "warn" };
  if (fails) out.settings = { text: `${fails} failed`, tone: "danger", dot: true };
  if (state?.state === "FAULT") out.overview = { text: "Fault", tone: "danger", dot: true };
  else if (needCal) out.overview = { text: "Check", tone: "warn", dot: true };
  if (state?.state === "CALIBRATING") out.overview = { text: "Calibrating", tone: "info", dot: true };
  if (state?.state === "MOVING" && state.activity === "teleop") out.teleop = { text: "Live", tone: "ok", dot: true };
  if (state?.state === "MOVING" && state.activity === "policy") out.policy = { text: "Running", tone: "ok", dot: true };
  return out;
}
