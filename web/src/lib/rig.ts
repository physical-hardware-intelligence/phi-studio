// This rig as onboarding sees it (src/phi_studio/onboard_api.py): robot-config.yaml, each arm's port and
// calibration, cameras, and what is already on this Mac to start from.
import { useSyncExternalStore } from "react";
import { studio } from "./studio";

export type Layout = "single" | "bimanual";
export interface RigArm {
  key: string; role: "leader" | "follower"; side: "left" | "right" | null; port: string | null; id: string | null;
  calibration: string | null; calibrated: boolean; calibrated_at: number | null;
}
export interface RigCamera { key: string; side: string | null; feature: string; source: number | string | null; width?: number; height?: number; fps?: number }
export interface KnownId { id: string; mtime: number }
export interface RigStatus {
  path: string; exists: boolean; name: string | null; layout: Layout | null;
  arms: RigArm[]; cameras: RigCamera[]; problems: string[]; calibration_root: string;
  known: Record<"follower" | "leader" | "bi_follower" | "bi_leader", KnownId[]>;
  hint: { ports: Record<string, string>; ids: { single?: Record<string, string>; bimanual?: Record<string, string> }; file?: string };
}

export const SLOTS: Record<Layout, string[]> = {
  single: ["leader", "follower"],
  bimanual: ["left_leader", "left_follower", "right_leader", "right_follower"],
};

let status: RigStatus | null = null;
const listeners = new Set<() => void>();
let wired = false;

function wire(): void {
  if (wired) return;
  wired = true;
  studio.onMessage("rig_status", (m) => { const { type: _t, ...rest } = m; status = rest as RigStatus; listeners.forEach((l) => l()); });
  studio.onMessage("hello", () => { studio.send({ cmd: "rig_status" }); });
}

export function refreshRig(): void { wire(); studio.send({ cmd: "rig_status" }); }

export function useRig(): RigStatus | null {
  wire();
  const s = useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => status);
  if (!s) queueMicrotask(() => { if (!status) studio.send({ cmd: "rig_status" }); });
  return s;
}

export const slotLabel = (slot: string): string =>
  slot.replace("left_", "Left ").replace("right_", "Right ").replace(/^leader$/, "Leader").replace(/^follower$/, "Follower")
    .replace("leader", "leader").replace("follower", "follower");

export const shortPort = (p: string | null | undefined): string => (p ? `…${p.replace(/^\/dev\/(tty|cu)\.(usbmodem|usbserial-?)?/, "").slice(-7)}` : "");

export function slug(name: string): string {
  const s = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return (s || "rig").slice(0, 40);
}
