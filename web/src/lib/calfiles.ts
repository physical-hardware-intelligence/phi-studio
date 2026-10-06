// What the server's rig_calfiles push holds (rig_api._calfiles_view, calfiles.py), and the problems it shows.

export interface CalFile {
  rel: string; id: string; folder: string; used_by: string[]; link: string | null; error: string | null;
  unfinished: string | null; same_as: string[]; on_ports: string[]; mtime: number; junk: string | null; unused: boolean;
}
export interface SharedRow {
  rel: string; id: string; state: "new" | "same" | "differs" | "link" | "broken";
  max_deg: number | null; worst_joint: string | null; error: string | null;
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
  for (const f of used) {
    if (f.link) out.push({ level: "danger", text: `${who(f.used_by)}'s file ${name(f.rel)} is a link to ${f.link}: recalibrating one arm rewrites the other's file. Install the shared file, or recalibrate this arm.` });
    // WHY only between files arms use: an unused alias (phi_follower.json = phi_bi_follower_left.json) is harmless
    const twin = used.filter((o) => o !== f && f.same_as.includes(o.rel) && o.folder === f.folder);
    if (twin.length && f.rel < twin[0].rel) {
      out.push({ level: "danger", text: `${who(f.used_by)} and ${who(twin[0].used_by)} use files with the same numbers (${name(f.rel)}, ${name(twin[0].rel)}). Two arms never have one calibration: one of them holds the other's.` });
    }
    if (f.error) out.push({ level: "danger", text: `${who(f.used_by)}'s file ${name(f.rel)} cannot be loaded: ${f.error}` });
    else if (f.unfinished) out.push({ level: "warn", text: `${who(f.used_by)}'s file ${name(f.rel)} is unfinished: ${f.unfinished}` });
    const elsewhere = f.on_ports.length > 1;
    if (elsewhere) out.push({ level: "danger", text: `The motors on ${f.on_ports.length} ports all hold ${name(f.rel)}: those arms share one calibration.` });
  }
  const extra = v.files.filter((f) => f.unused).length;
  if (extra > 0) out.push({ level: "warn", text: `${extra} ${extra === 1 ? "file is" : "files are"} not used by the rig. Move them aside so a wrong id cannot pick one.` });
  return out;
}
