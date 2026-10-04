// The Set up page's state: the port finder, the camera finder, camera align, and the blanks every command
// shares. The server side is src/phi_studio/setup_api.py; its message types arrive through studio.onMessage.
import { useRef, useSyncExternalStore } from "react";
import { studio } from "./studio";

export interface UsbPort {
  name: string; // what robot-config.yaml uses: the /dev/tty. form on macOS
  device: string; tty: string | null; description: string | null;
  vid: number | null; pid: number | null; serial: string | null; manufacturer: string | null;
}
export interface Moved { arm: string; old: string; new: string; serial: string }
export interface PortsView { ports: UsbPort[]; moved: Moved[]; at: number; busy: string | null }

export interface ProbedCamera {
  source: number | string; ok: boolean; width: number | null; height: number | null; fps: number | null;
  error: string | null; picture: string | null;
}
export interface Dataset {
  name: string; repo_id: string; root: string; episodes: number; frames: number; task: string | null;
  cameras: string[]; physical: Record<string, string>; note: string | null;
}
export interface AlignSession {
  root: string; episode: number; frame: number;
  physical: Record<string, string>; // dataset key -> the camera it really shows, when the keys were swapped
  references: Record<string, string>; // dataset key -> picture
  config: Record<string, string | null>; // dataset key -> the config camera with that key
  assignment: { live: number | string; key: string }[];
  unsure: boolean; why: string | null; unmatched_refs: string[];
}
export interface AlignTick {
  live: number | string; key: string; at: number;
  dx?: number; dy?: number; response?: number; aligned?: boolean; low_match?: boolean;
  hint?: string; size?: [number, number]; note?: string | null; picture?: string; error?: string;
}
export interface Saved { what: "ports" | "cameras"; keys: string[]; backup: string | null; no_serial?: string[]; at: number }

/** The port finder: unplug each arm in turn, and the port that goes away is that arm's. */
export interface Wizard {
  arms: string[]; // arm keys, in the order asked
  i: number; // the arm being found
  phase: "unplug" | "replug";
  seen: string[]; // ports present since this arm's turn began
  gone: string | null; // the port that went away when it was unplugged
  found: Record<string, string>; // arm key -> port
  problem: string | null;
  saving: boolean;
}

export interface SetupState {
  ports: PortsView | null;
  wizard: Wizard | null;
  cameras: { probing: boolean; list: ProbedCamera[] | null; at: number | null };
  datasets: Dataset[] | null;
  align: AlignSession | null;
  ticks: Record<string, AlignTick>; // by live source
  alignStatus: string | null; // what a starting session is doing
  alignStopped: string | null; // why the last session ended
  saved: Saved | null;
  blanks: Record<string, string>; // what the person typed into command blanks, kept across reloads
}

const BLANKS_KEY = "phi-studio-blanks";

function loadBlanks(): Record<string, string> {
  try {
    const v = JSON.parse(localStorage.getItem(BLANKS_KEY) ?? "{}");
    return v && typeof v === "object" ? v : {};
  } catch { return {}; }
}

/** One step of the port finder, given the ports present now. Pure, so every case reads in one place. */
export function advance(w: Wizard, now: string[]): Wizard {
  if (w.i >= w.arms.length || w.saving) return w;
  const arm = w.arms[w.i];
  if (w.phase === "unplug") {
    // WHY keep adding to seen: a cable plugged in meanwhile is not this arm, and must not read as one that left.
    const seen = [...new Set([...w.seen, ...now])];
    const missing = seen.filter((p) => !now.includes(p));
    if (missing.length === 1) return { ...w, seen, phase: "replug", gone: missing[0], problem: null };
    if (missing.length > 1) {
      return { ...w, seen, problem: `${missing.length} ports went away at once (${missing.join(", ")}). Plug all back in except ${armLabel(arm)}.` };
    }
    return { ...w, seen, problem: null };
  }
  // replug: the new port is one that was not there while the arm was out. Usually the same name comes back.
  const before = w.seen.filter((p) => p !== w.gone);
  const back = now.filter((p) => !before.includes(p));
  if (back.length === 1) {
    const found = { ...w.found, [arm]: back[0] };
    return { ...w, found, i: w.i + 1, phase: "unplug", seen: now, gone: null, problem: null };
  }
  if (back.length > 1) return { ...w, problem: `${back.length} ports appeared at once (${back.join(", ")}). Plug in only ${armLabel(arm)}.` };
  return { ...w, problem: null };
}

const armLabel = (key: string) => key.replace(/_/g, " ");

class SetupStore {
  private listeners = new Set<() => void>();
  snap: SetupState = {
    ports: null, wizard: null, cameras: { probing: false, list: null, at: null }, datasets: null,
    align: null, ticks: {}, alignStatus: null, alignStopped: null, saved: null, blanks: loadBlanks(),
  };

  constructor() {
    studio.onMessage("setup_ports", (m) => this.onPorts(m));
    studio.onMessage("setup_saved", (m) => {
      this.set({ saved: { ...m, at: Date.now() } });
      const w = this.snap.wizard;
      if (m.what === "ports" && w?.saving) this.set({ wizard: null });
      if (m.what === "ports") this.refreshPorts();
    });
    studio.onMessage("setup_cameras", (m) => this.set({
      cameras: m.probing ? { ...this.snap.cameras, probing: true } : { probing: false, list: m.cameras, at: m.at * 1000 },
    }));
    studio.onMessage("align_datasets", (m) => this.set({ datasets: m.datasets }));
    studio.onMessage("align_status", (m) => this.set({ alignStatus: m.text, alignStopped: null }));
    studio.onMessage("align_session", (m) => {
      // WHY drop ticks for cameras that now show another dataset camera: their offsets were against the old one.
      const keep: Record<string, AlignTick> = {};
      for (const a of m.assignment as AlignSession["assignment"]) {
        const t = this.snap.ticks[String(a.live)];
        if (t && t.key === a.key) keep[String(a.live)] = t;
      }
      this.set({ align: m, alignStatus: null, alignStopped: null, ticks: keep });
    });
    studio.onMessage("align_tick", (m) => this.set({ ticks: { ...this.snap.ticks, [String(m.live)]: { ...m, at: Date.now() } } }));
    studio.onMessage("align_stopped", (m) => this.set({ align: null, ticks: {}, alignStatus: null, alignStopped: m.why }));
    // A refused request: clear what waits for its reply. The message itself shows as a notice (studio.ts).
    studio.onMessage("error", (m) => {
      if (m.cmd === "align_start") this.set({ alignStatus: null });
      if (m.cmd === "setup_cameras_probe") this.set({ cameras: { ...this.snap.cameras, probing: false } });
      if (m.cmd === "setup_ports_save" && this.snap.wizard?.saving) {
        this.set({ wizard: { ...this.snap.wizard, saving: false, problem: m.message } });
      }
    });
    // Replies that will never come once the link drops.
    studio.subscribe(() => {
      if (studio.snap.link === "open") return;
      const s = this.snap;
      if (s.cameras.probing || s.alignStatus || s.align || s.wizard?.saving) {
        this.set({
          cameras: { ...s.cameras, probing: false }, alignStatus: null, align: null, ticks: {},
          wizard: s.wizard ? { ...s.wizard, saving: false } : null,
        });
      }
    });
  }

  private set(p: Partial<SetupState>): void {
    this.snap = { ...this.snap, ...p };
    this.listeners.forEach((l) => l());
  }
  subscribe = (l: () => void): (() => void) => { this.listeners.add(l); return () => this.listeners.delete(l); };

  // -- ports -------------------------------------------------------------------------------------
  refreshPorts(): void { studio.send({ cmd: "setup_ports" }); }

  private onPorts(m: PortsView): void {
    this.set({ ports: m });
    const w = this.snap.wizard;
    if (!w || w.saving) return;
    const next = advance(w, m.ports.map((p) => p.name));
    if (next.i >= next.arms.length) this.saveWizard(next);
    else if (next !== w) this.set({ wizard: next });
  }

  startWizard(arms: string[]): void {
    const now = this.snap.ports?.ports.map((p) => p.name) ?? [];
    this.set({ wizard: { arms, i: 0, phase: "unplug", seen: now, gone: null, found: {}, problem: null, saving: false }, saved: null });
  }
  cancelWizard(): void { this.set({ wizard: null }); }

  /** The last arm's port by elimination: the one port no other arm took. */
  useLeftover(port: string): void {
    const w = this.snap.wizard;
    if (!w || w.i !== w.arms.length - 1) return;
    this.saveWizard({ ...w, found: { ...w.found, [w.arms[w.i]]: port }, i: w.arms.length });
  }

  private saveWizard(w: Wizard): void {
    if (!studio.send({ cmd: "setup_ports_save", ports: w.found })) {
      this.set({ wizard: { ...w, problem: "This window is offline, so nothing was saved. Reconnect and save again." } });
      return;
    }
    this.set({ wizard: { ...w, saving: true, problem: null } });
  }
  retrySave(): void { const w = this.snap.wizard; if (w && w.i >= w.arms.length) this.saveWizard(w); }

  savePorts(ports: Record<string, string>): boolean { return studio.send({ cmd: "setup_ports_save", ports }); }

  // -- cameras -----------------------------------------------------------------------------------
  probe(): void {
    if (studio.send({ cmd: "setup_cameras_probe" })) this.set({ cameras: { ...this.snap.cameras, probing: true } });
  }
  saveCameras(cams: Record<string, number | string>): boolean { return studio.send({ cmd: "setup_cameras_save", cameras: cams }); }

  // -- align -------------------------------------------------------------------------------------
  loadDatasets(): void { studio.send({ cmd: "align_datasets" }); }
  startAlign(root: string, episode: number): void {
    if (studio.send({ cmd: "align_start", root, episode })) this.set({ alignStatus: "Starting", alignStopped: null });
  }
  stopAlign(): void { studio.send({ cmd: "align_stop" }); }
  /** Show `key` on live camera `live`; the camera that showed it takes this one's old key. */
  assign(live: number | string, key: string): void {
    const a = this.snap.align;
    if (!a) return;
    const cur = a.assignment.find((x) => String(x.live) === String(live))?.key;
    const out: Record<string, string> = {};
    for (const x of a.assignment) {
      out[String(x.live)] = String(x.live) === String(live) ? key : x.key === key && cur ? cur : x.key;
    }
    studio.send({ cmd: "align_assign", assignment: out });
  }

  // -- command blanks ----------------------------------------------------------------------------
  setBlank(name: string, value: string): void {
    const blanks = { ...this.snap.blanks, [name]: value };
    try { localStorage.setItem(BLANKS_KEY, JSON.stringify(blanks)); } catch { /* private window: kept for this page only */ }
    this.set({ blanks });
  }
}

export const setup = new SetupStore();

export function useSetup<T>(select: (s: SetupState) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(setup.subscribe, () => {
    const value = select(setup.snap);
    if (last.current && Object.is(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}
