// Detect arms, the pure part: turn rig_scan (src/phi_studio/detect.py) into onboarding's slots and calibration ids.
// No React here so tests/armdetect.test.ts runs it under Node.

export type Role = "leader" | "follower";
export type Side = "left" | "right";
export interface FoundArm {
  port: string; serial: string; ids: number[]; clashes: number[]; torque: boolean | null;
  match: string | null; match_deg: number | null; matches?: string[]; role: Role | null; problem: string | null;
}
export interface Ids { follower: string; leader: string }
export interface Placed {
  layout: "single" | "bimanual" | null;
  ports: Record<string, string>; // slot -> port, slots as in rig.ts SLOTS
  ids: Ids | null; // calibration ids whose files hold these arms' registers exactly
  unplaced: FoundArm[]; // healthy arms Studio could not place: no exact calibration match, or no side
  note: string | null; // why no ids were offered although every arm was placed
  broken: FoundArm[]; // arms with a motor problem (id clash, missing id, port error)
}

const SIDE = /_(left|right)$/;

/** The calibration ids among an arm's exact matches, for its own role: "phi_bi_left" from robots/so_follower/phi_bi_left. */
function names(a: FoundArm): string[] {
  const folder = a.role === "follower" ? "robots/so_follower/" : "teleoperators/so_leader/";
  return (a.matches ?? (a.match_deg === 0 && a.match ? [a.match] : [])).filter((m) => m.startsWith(folder)).map((m) => m.slice(folder.length));
}

/** The one side an arm's matching file names agree on, or null when they name none or both. A preferred base id
 * decides when several files match: phi_bi_left and yash_right can both hold the same registers. */
function sideOf(a: FoundArm, prefer?: string): { side: Side; base: string } | null {
  const sided = names(a).map((n) => n.match(SIDE) ? { side: n.match(SIDE)![1] as Side, base: n.replace(SIDE, "") } : null).filter((x) => x !== null);
  const pick = prefer ? sided.filter((x) => x.base === prefer) : [];
  const pool = pick.length ? pick : sided;
  const sides = new Set(pool.map((x) => x.side));
  return sides.size === 1 ? pool[0] : null;
}

/** Base ids every list shares, preferred first. */
function shared(lists: string[][], prefer?: string): string[] {
  if (!lists.length) return [];
  const all = lists.reduce((acc, l) => acc.filter((x) => l.includes(x)));
  return prefer && all.includes(prefer) ? [prefer, ...all.filter((x) => x !== prefer)] : all;
}

/** A follower id and a leader id that differ. WHY differ: Studio's worker keys calibrations by id alone
 * (worker.py _save_run, mock.py calibration_files), and onboard_api.check_answers refuses equal ids. */
function pick(f: string[], l: string[]): Ids | null {
  for (const a of f) for (const b of l) if (a !== b) return { follower: a, leader: b };
  return null;
}

/** Where each found arm goes. Only an exact register match to a calibration file places an arm: a near match is
 * another arm's file, and a guess there would drive a follower with the wrong zero. */
export function place(arms: FoundArm[], prefer?: Partial<Ids> | null): Placed {
  const broken = arms.filter((a) => a.problem);
  const healthy = arms.filter((a) => !a.problem);
  const known = healthy.filter((a) => a.role);
  const unknown = healthy.filter((a) => !a.role);
  const by = (r: Role) => known.filter((a) => a.role === r);
  const leaders = by("leader"), followers = by("follower");
  const out: Placed = { layout: null, ports: {}, ids: null, unplaced: [...unknown], broken, note: null };

  if (healthy.length === 2 && leaders.length === 1 && followers.length === 1) {
    out.layout = "single";
    out.ports = { leader: leaders[0].port, follower: followers[0].port };
    const plain = (a: FoundArm, r: Role) => shared([names(a).filter((n) => !SIDE.test(n))], prefer?.[r]);
    const f = plain(followers[0], "follower"), l = plain(leaders[0], "leader");
    out.ids = pick(f, l);
    if (!out.ids && f.length && l.length) out.note = sameIds(f[0]);
    return out;
  }

  if (healthy.length >= 3 || leaders.length === 2 || followers.length === 2) out.layout = "bimanual";
  const bases: Record<Role, string[][]> = { leader: [], follower: [] };
  for (const r of ["leader", "follower"] as Role[]) {
    const group = by(r);
    const sides = group.map((a) => ({ a, s: sideOf(a, prefer?.[r]) }));
    const ok = group.length === 2 && sides.every((x) => x.s) && sides[0].s!.side !== sides[1].s!.side;
    for (const { a, s } of sides) {
      if (ok) {
        out.ports[`${s!.side}_${r}`] = a.port;
        bases[r].push(names(a).filter((n) => n.endsWith(`_${s!.side}`)).map((n) => n.replace(SIDE, "")));
      } else out.unplaced.push(a);
    }
  }
  if (bases.follower.length === 2 && bases.leader.length === 2) {
    const f = shared(bases.follower, prefer?.follower), l = shared(bases.leader, prefer?.leader);
    out.ids = pick(f, l);
    if (!out.ids && f.length && l.length) out.note = sameIds(f[0]);
  }
  return out;
}

const sameIds = (id: string): string =>
  `The followers' and leaders' calibration files are both named ${id}. Studio needs a different id for each role, so pick "New calibration" or rename the leader files.`;
