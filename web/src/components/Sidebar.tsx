import { BookOpen, Bot, Box, ClipboardCheck, Cpu, Database, Flag, FolderOpen, Gamepad2, LayoutGrid, ListChecks, ListOrdered, Moon, PackageSearch, Ruler, Sun } from "lucide-react";
import { useNotes } from "../lib/data";
import { PhiMark } from "./PhiMark";
import type { ComponentType } from "react";
import { go, setTheme, useRoute, useStudio, useTheme, type Route } from "../lib/studio";

interface Item { route: Route; label: string; icon: ComponentType<{ "aria-hidden"?: boolean }> }

const TOP: Item[] = [
  { route: "overview", label: "Overview", icon: LayoutGrid },
  { route: "checks", label: "Checks", icon: ListChecks },
];
// In pipeline order: a rig first, then data and training, then policies on the arms.
const RIG: Item[] = [
  { route: "setup", label: "Set up", icon: ListOrdered },
  { route: "calibrate", label: "Calibrate", icon: Ruler },
  { route: "teleop", label: "Teleoperate", icon: Gamepad2 },
  { route: "scene", label: "3D view", icon: Box },
];
const DATA: Item[] = [
  { route: "data", label: "Datasets", icon: Database },
  { route: "issues", label: "Issues", icon: Flag },
];
const POLICIES: Item[] = [
  { route: "train", label: "Train", icon: Cpu },
  { route: "models", label: "Models", icon: PackageSearch },
  { route: "policy", label: "Run policy", icon: Bot },
  { route: "evaluate", label: "Evaluate", icon: ClipboardCheck },
];
const REFERENCE: Item[] = [
  { route: "guide", label: "Guide", icon: BookOpen },
  { route: "files", label: "Files", icon: FolderOpen },
];

// WHY one list: the Claude panel names the current page; a second copy of the labels missed two pages.
export const PAGE_LABELS = Object.fromEntries(
  [...TOP, ...RIG, ...DATA, ...POLICIES, ...REFERENCE].map((i) => [i.route, i.label]),
) as Record<Route, string>;

// Places on the left, the work on the right (Foxglove's layout). Each item may carry one live hint drawn
// from session state, so the sidebar also answers "where is something happening".
export function Sidebar() {
  const route = useRoute();
  const theme = useTheme();
  const mock = useStudio((s) => s.mock);
  const link = useStudio((s) => s.link);
  const pairs = useStudio((s) => s.identity.filter((a) => a.role === "follower").length);
  const bimanual = useStudio((s) => s.rig?.bimanual ?? false);
  const hint = useHints();

  const item = (it: Item) => {
    const Icon = it.icon;
    const h = hint[it.route];
    return (
      <a key={it.route} href={`#/${it.route}`} className={`nav-item ${route === it.route ? "is-active" : ""}`}
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
          <span className="brand-rig">{mock ? "Mock rig" : "SO-101 rig"}{bimanual ? ", bimanual" : pairs ? ", single arm" : ""}</span>
        </div>
      </div>
      <div className="nav-group">{TOP.map(item)}</div>
      <div className="nav-group">
        <div className="nav-group-label">Rig</div>
        {RIG.map(item)}
      </div>
      <div className="nav-group">
        <div className="nav-group-label">Data</div>
        {DATA.map(item)}
      </div>
      <div className="nav-group">
        <div className="nav-group-label">Policies</div>
        {POLICIES.map(item)}
      </div>
      <div className="nav-group">
        <div className="nav-group-label">Reference</div>
        {REFERENCE.map(item)}
      </div>
      <div className="sidebar-foot">
        <div className={`link-status tone-${link === "open" ? "ok" : link === "refused" || link === "down" ? "danger" : "warn"}`}>
          <span className="dot" />
          {link === "open" ? "Connected to Studio" : link === "refused" ? "Not authorised" : link === "down" ? "Studio is not running" : "Reconnecting"}
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
  const warns = useStudio((s) => s.checks.results?.filter((r) => r.status === "warn").length ?? 0);
  const open = useNotes(null, null)?.filter((n) => n.kind === "issue" && n.status === "open").length ?? 0;
  const out: Partial<Record<Route, Hint>> = {};
  if (open) out.issues = { text: String(open), tone: "warn" };
  if (fails) out.checks = { text: `${fails} failed`, tone: "danger", dot: true };
  else if (warns) out.checks = { text: `${warns} ${warns === 1 ? "warning" : "warnings"}`, tone: "warn", dot: true };
  if (state?.state === "FAULT") out.overview = { text: "Fault", tone: "danger", dot: true };
  if (needCal) out.calibrate = { text: "Check", tone: "warn", dot: true };
  if (state?.state === "CALIBRATING") out.calibrate = { text: "Active", tone: "info", dot: true };
  if (state?.state === "MOVING" && state.activity === "teleop") out.teleop = { text: "Live", tone: "ok", dot: true };
  if (state?.state === "MOVING" && state.activity === "policy") out.policy = { text: "Running", tone: "ok", dot: true };
  return out;
}
