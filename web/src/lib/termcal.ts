// Calibrate in the terminal, the pure part: what a check of one arm's registers against its file means, and
// when its lerobot-calibrate run has ended. No React here so tests/termcal.test.ts runs it under Node.
// Server side: rig_cal_* in src/phi_studio/rig_api.py.

export interface Verdict {
  arm: string; path: string; file: boolean; exact: boolean; max_deg: number | null; worst_joint: string | null;
  problem: string | null; unfinished: string | null; at: number;
}
export interface Reading { tone: "ok" | "warn" | "error"; text: string; canSave: boolean }

const joint = (j: string | null) => (j ?? "a joint").replace(/_/g, " ");

/** What the check found, in words, and whether writing the file from the motors is safe. */
export function read(v: Verdict): Reading {
  if (v.problem) return { tone: "error", text: v.problem, canSave: false };
  // WHY never save then: LeRobot resets every joint to 0..4095 before it records ranges, so a run stopped
  // midway leaves registers that would call a wrong zero right.
  if (v.unfinished) return { tone: "error", text: `The motors hold an unfinished calibration: ${v.unfinished}. Calibrate this arm again.`, canSave: false };
  if (v.exact) return { tone: "ok", text: "Its file matches the motors.", canSave: false };
  if (!v.file) return { tone: "warn", text: "The motors hold a calibration, but no file was saved.", canSave: true };
  return {
    tone: "warn", canSave: true,
    text: `Its file differs from the motors by ${(v.max_deg ?? 0).toFixed(1)} deg on ${joint(v.worst_joint)}. Either LeRobot did not save, or this port holds another arm.`,
  };
}

/** Tracks one lerobot-calibrate run through the terminal's foreground command: it has ended once the command
 * was seen running and the terminal is back at the prompt. */
export class RunWatch {
  private seen = false;
  readonly arm: string;
  constructor(arm: string) { this.arm = arm; } // WHY no parameter property: Node's type stripping refuses it
  /** Feed every terminal status; true exactly once, when the run has ended. */
  update(running: string | null): boolean {
    if (running && /\blerobot-calibrate\b/.test(running)) { this.seen = true; return false; }
    if (!running && this.seen) { this.seen = false; return true; }
    return false;
  }
  get started(): boolean { return this.seen; }
}
