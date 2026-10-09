// What onboarding pre-selects: ports and calibration ids. Pure (type imports only), so the rules run
// under Node in tests/onboard.test.ts.
import type { Layout, RigStatus } from "./rig";

export interface Ids { follower: string; leader: string }

/** Calibration ids on this Mac a layout can use, per role, newest first. Bimanual counts an id X
 *  when X_left and X_right both exist (onboard_api.known_ids). */
export function idChoices(rig: RigStatus, layout: Layout): { follower: string[]; leader: string[] } {
  const k = layout === "bimanual" ? [rig.known.bi_follower, rig.known.bi_leader] : [rig.known.follower, rig.known.leader];
  return { follower: k[0].map((x) => x.id), leader: k[1].map((x) => x.id) };
}

/** The ids the config's arms use, as onboarding writes them: a bimanual config's arms carry X_left
 *  and X_right (LeRobot adds the side), a single arm's carry the file name itself, which may end in
 *  _left or _right (phi_bi_follower_right runs the right pair alone). */
export function configIds(rig: RigStatus): Partial<Ids> {
  const id = (role: "follower" | "leader") => {
    const x = rig.arms.find((a) => a.role === role)?.id;
    return x ? (rig.layout === "bimanual" ? x.replace(/_(left|right)$/, "") : x) : undefined;
  };
  return { follower: id("follower"), leader: id("leader") };
}

/** The ids to pre-select, only ever a pair whose files are on this Mac: the config's (what this rig
 *  last ran on, when it had this layout), then ports.local.sh's (the club's), then a follower and a
 *  leader with matching names (x_follower and x_leader). Otherwise none, and the person picks.
 *  WHY no "newest of each": it can pair one arm's file with the other side's, and LeRobot writes a
 *  mismatched file into the servos (2026-10-05). 2026-10-09: onboarding offered only the stale
 *  config's ids, which had no files, and not phi_follower/phi_leader, which did. */
export function defaultIds(rig: RigStatus, layout: Layout): Ids | null {
  const c = idChoices(rig, layout);
  const ok = (x: Partial<Ids> | null | undefined): x is Ids =>
    !!x?.follower && !!x.leader && c.follower.includes(x.follower) && c.leader.includes(x.leader);
  const config = rig.layout === layout ? configIds(rig) : null;
  const hint = rig.hint.ids?.[layout];
  const named = c.follower.map((f) => ({ follower: f, leader: f.replace("follower", "leader") })).find(ok);
  for (const x of [config, hint, named]) if (ok(x)) return { follower: x.follower, leader: x.leader };
  return null;
}

/** Ports to start from for a layout's slots: the config's (when it had this layout), then
 *  ports.local.sh's. The Arms step's unplug finder confirms or replaces them. */
export function seedPorts(rig: RigStatus, layout: Layout, slots: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  if (rig.layout === layout) for (const a of rig.arms) if (a.port && slots.includes(a.key)) out[a.key] = a.port;
  for (const [k, v] of Object.entries(rig.hint.ports)) if (!out[k] && slots.includes(k)) out[k] = v;
  return out;
}
