// What the server's rig_calfiles push holds (rig_api._calfiles_view, calfiles.py), and the problems it shows.

export interface CalFile {
  rel: string; id: string; folder: string; used_by: string[]; link: string | null; real: string; error: string | null;
  unfinished: string | null; same_as: string[]; same_arm_as?: string[]; on_ports: string[]; mtime: number; junk: string | null; unused: boolean;
}
export interface SharedRow {
  rel: string; id: string; state: "new" | "same" | "differs" | "link" | "broken";
  max_deg: number | null; worst_joint: string | null; error: string | null;
  named_for?: string | null; arm_of?: string | null; arm_gap?: number | null; // calfiles.shared: name vs joint stops
}

/** The rig arm a shared file physically belongs to when that is not the arm its name says (calfiles._arm_of), else
 * null. Installing it would put that arm's calibration on the named arm. */
export function wrongArm(r: SharedRow): string | null {
  return r.named_for && r.arm_of && r.named_for !== r.arm_of ? r.arm_of : null;
}
export interface CalFilesView {
  root: string;
  files: CalFile[];
  archives: { name: string; files: string[] }[];
  config: boolean;
  shared: { source: string | null; exists: boolean; rows: SharedRow[]; candidates: string[] };
  scanned: boolean;
  done?: { action: "archived" | "restored" | "installed"; files?: string[]; restored?: string[]; copied?: string[]; archive?: string | null };
}

export interface Problem { level: "danger" | "warn"; text: string }

const name = (rel: string) => rel.split("/").pop() ?? rel;

/** The things in the folder that can put a wrong calibration into an arm, most serious first. `label` names an arm
 * (lib/labels.ts; passed in so this runs under Node's tests without the app's modules). */
export function calfileProblems(v: CalFilesView, label: (key: string) => string = (k) => k): Problem[] {
  const who = (keys: string[]) => keys.map(label).join(" and ");
  const out: Problem[] = [];
  const used = v.files.filter((f) => f.used_by.length > 0);
  // WHY by the real file, not "is a link": one physical arm under two ids (phi_bi_left.json -> phi_follower.json) is
  // intended; two arms reading one file is the fault (calfiles.shared_files does the same)
  const byReal = new Map<string, string[]>();
  for (const f of used) for (const k of f.used_by) {
    const keys = byReal.get(f.real || f.rel) ?? [];
    if (!keys.includes(k)) keys.push(k);
    byReal.set(f.real || f.rel, keys);
  }
  for (const [real, keys] of byReal) {
    if (keys.length > 1) out.push({ level: "danger", text: `${who(keys)} read one file, ${name(real)}, through a link: calibrating one arm rewrites the other's. Install the shared files, or calibrate one arm again.` });
  }
  for (const f of used) {
    // WHY only between files arms use: an unused alias (phi_follower.json = phi_bi_follower_left.json) is harmless.
    // A link and its target are one file, and one arm under two ids is one arm.
    const twin = used.filter((o) => o !== f && f.same_as.includes(o.rel) && o.folder === f.folder
      && (o.real || o.rel) !== (f.real || f.rel) && !o.used_by.some((k) => f.used_by.includes(k)));
    const sameArm = (f.same_arm_as ?? []).map((rel) => used.find((o) => o.rel === rel)).filter((o) => o !== undefined);
    if (sameArm.length && f.rel < sameArm[0]!.rel) {
      out.push({ level: "danger", text: `${who(f.used_by)}'s and ${who(sameArm[0]!.used_by)}'s files (${name(f.rel)}, ${name(sameArm[0]!.rel)}) have one arm's joint stops: one of them is the other arm's calibration. Calibrate both again with c.` });
    }
    if (twin.length && f.rel < twin[0].rel) {
      out.push({ level: "danger", text: `${who(f.used_by)} and ${who(twin[0].used_by)} use files with the same numbers (${name(f.rel)}, ${name(twin[0].rel)}). Two arms never have one calibration: one of them holds the other's.` });
    }
    if (f.error) out.push({ level: "danger", text: `${who(f.used_by)}'s file ${name(f.rel)} cannot be loaded: ${f.error}` });
    else if (f.unfinished) out.push({ level: "warn", text: `${who(f.used_by)}'s file ${name(f.rel)} is unfinished: ${f.unfinished}` });
    const elsewhere = f.on_ports.length > 1;
    if (elsewhere) out.push({ level: "danger", text: `The motors on ${f.on_ports.length} ports all hold ${name(f.rel)}: those arms share one calibration.` });
  }
  const rows = v.shared.exists ? v.shared.rows.filter((r) => r.state !== "broken") : [];
  if (v.config && rows.length > 0 && !rows.some((r) => used.some((f) => f.rel === r.rel))) {
    const ids = used.map((f) => f.id).join(", ") || "none";
    out.push({ level: "warn", text: `robot-config.yaml uses none of the shared files (it uses ${ids}). Install them, then run Detect arms and Save rig on Rig setup: Studio picks the ids whose files the motors hold.` });
  }
  for (const r of rows) {
    const arm = wrongArm(r);
    if (arm) out.push({ level: "danger", text: `Shared ${name(r.rel)} is named for the ${label(r.named_for!)}, but its joint stops are the ${label(arm)}'s on this Mac (within ${r.arm_gap} deg). One of the two sets has the arms swapped. Do not install it until you know which: calibrating each arm again with c gives its true numbers.` });
  }
  const extra = v.files.filter((f) => f.unused).length;
  if (extra > 0) out.push({ level: "warn", text: `${extra} ${extra === 1 ? "file is" : "files are"} not used by the rig. Move them aside so a wrong id cannot pick one.` });
  return out;
}
