// The camera check before a policy runs (components/Preflight.tsx), pure so tests run it under Node.
import type { AlignResult, PolicyInfo } from "./studio";

// [JUDGEMENT] An align older than this no longer says where the cameras are: rigs are rebuilt daily and a camera is
// easily bumped. Not measured; the age is shown, so the person can judge a borderline case.
export const ALIGN_FRESH_S = 4 * 3600;

export type Tone = "ok" | "warn" | "neutral";
const name = (root: string) => root.split("/").filter(Boolean).pop() ?? root;

/** What the camera check finds for this policy, and why it blocks Start (null: it does not). `now` in ms. */
export function cameraCheck(policy: PolicyInfo | undefined, align: AlignResult | null, mock: boolean, now: number,
  ago: (unix: number) => string): { tone: Tone; state: string; block: string | null } {
  if (!policy?.cameras?.length) return { tone: "neutral", state: "not read by this policy", block: null };
  if (mock) return { tone: "neutral", state: "simulated", block: null };
  if (!align) return { tone: "warn", state: "not aligned yet", block: "Align the cameras first" };
  if (policy.dataset && name(align.root) !== name(policy.dataset)) {
    return { tone: "warn", state: `aligned to ${name(align.root)}, not its training data`, block: "Align to this policy's training data" };
  }
  const n = Object.keys(align.cameras).length;
  const off = Object.values(align.cameras).filter((ok) => !ok).length;
  if (!align.aligned) return { tone: "warn", state: `${off} of ${n} off`, block: "A camera is out of line" };
  if (now / 1000 - align.at > ALIGN_FRESH_S) return { tone: "warn", state: `aligned ${ago(align.at)}`, block: "Align again: the last check is old" };
  return { tone: "ok", state: `in line · ${ago(align.at)}`, block: null };
}
