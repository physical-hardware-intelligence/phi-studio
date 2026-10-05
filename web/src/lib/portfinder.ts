// The port finder's state machine (pages/Setup.tsx, step 2): unplug each arm in turn, and the port that goes
// away is that arm's. Pure and import-free, so tests/portfinder.test.ts runs it under Node with no browser.

/** The port finder: unplug each arm in turn, and the port that goes away is that arm's. */
export interface Wizard {
  arms: string[]; // arm keys, in the order asked
  i: number; // the arm being found
  phase: "unplug" | "replug";
  base: string[]; // ports present when this arm's turn began: its port is one of these
  extra: string[]; // ports that appeared during the turn: not this arm, which was still plugged in
  gone: string | null; // the port that went away when it was unplugged
  found: Record<string, string>; // arm key -> port
  problem: string | null;
  saving: boolean;
}

/** One step of the port finder, given the ports present now. Pure, so every case reads in one place. */
export function advance(w: Wizard, now: string[]): Wizard {
  if (w.i >= w.arms.length || w.saving) return w;
  const arm = w.arms[w.i];
  if (w.phase === "unplug") {
    // Only a port present when the turn began can be this arm's: one that came and went meanwhile is not.
    const extra = [...new Set([...w.extra, ...now.filter((p) => !w.base.includes(p))])];
    const missing = w.base.filter((p) => !now.includes(p));
    const already = Object.entries(w.found).filter(([, p]) => missing.includes(p)).map(([k]) => armLabel(k));
    if (already.length) {
      return { ...w, extra, problem: `That is ${already.join(" and ")}'s port, found already. Plug it back in and unplug ${armLabel(arm)}.` };
    }
    if (missing.length === 1) return { ...w, extra, phase: "replug", gone: missing[0], problem: null };
    if (missing.length > 1) {
      return { ...w, extra, problem: `${missing.length} ports went away at once (${missing.join(", ")}). Plug all back in except ${armLabel(arm)}.` };
    }
    return { ...w, extra, problem: null };
  }
  // replug. WHY stop on any other port leaving: a cable moved to another socket can change its name, and
  // naming this arm then would swap two arms with no error.
  const lost = w.base.filter((p) => p !== w.gone && !now.includes(p));
  if (lost.length) return { ...w, problem: `${lost.join(", ")} went away too. Plug ${lost.length === 1 ? "it" : "them"} back in where ${lost.length === 1 ? "it was" : "they were"}.` };
  // The arm's port: usually the same name comes back, or a name that was not there before.
  const back = now.filter((p) => (p === w.gone || !w.base.includes(p)) && !w.extra.includes(p));
  if (back.length === 1) {
    const found = { ...w.found, [arm]: back[0] };
    return { ...w, found, i: w.i + 1, phase: "unplug", base: now, extra: [], gone: null, problem: null };
  }
  if (back.length > 1) return { ...w, problem: `${back.length} ports appeared at once (${back.join(", ")}). Plug in only ${armLabel(arm)}.` };
  return { ...w, problem: null };
}

/** macOS lists each USB serial port twice, as /dev/cu.X and /dev/tty.X; a config may use either. */
export const portKey = (name: string) => name.replace(/^\/dev\/cu\./, "/dev/tty.");

// WHY a copy of labels.ts's label(): an import would need a .ts extension for Node's test runner,
// which the page's tsconfig does not allow.
const armLabel = (key: string) => key.split(/[_\s]+/).filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
