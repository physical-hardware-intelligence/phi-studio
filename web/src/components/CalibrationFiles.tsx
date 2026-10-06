// The calibration folder on this Mac: which file each arm uses, the files no arm uses, links, copies, and the shared
// set in the phi repo that everyone uses. LeRobot reads one file per arm, by id; every other file is one wrong id away
// from being written into an arm's servos. Server side: rig_calfiles* in rig_api.py, calfiles.py.
import { Archive, CircleAlert, Download, LoaderCircle, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { calfileProblems, type CalFile, type CalFilesView, type SharedRow } from "../lib/calfiles";
import { label } from "../lib/labels";
import { studio, useStudio } from "../lib/studio";
import { terminal } from "../lib/terminal";

const name = (rel: string) => rel.split("/").pop() ?? rel;

export function CalibrationFiles({ compact = false }: { compact?: boolean }) {
  const control = useStudio((s) => s.control);
  const link = useStudio((s) => s.link);
  const [view, setView] = useState<CalFilesView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<Set<string>>(new Set());
  const [take, setTake] = useState<Set<string>>(new Set());
  const [source, setSource] = useState("");

  const load = (src?: string) => studio.send({ cmd: "rig_calfiles", ...(src ? { source: src } : {}) });
  useEffect(() => {
    const offs = [
      studio.onMessage("rig_calfiles", (m) => {
        const v = m as unknown as CalFilesView;
        setView(v); setBusy(false); setError(null);
        // WHY preselect: the usual job is "keep the rig's files, move the rest", and "install what differs"
        setPick(new Set(v.files.filter((f) => f.unused).map((f) => f.rel)));
        // WHY not a file the motors hold: this Mac's file then matches its arm, and the shared one may be another arm's
        const held = new Set(v.files.filter((f) => f.on_ports.length > 0).map((f) => f.rel));
        setTake(new Set(v.shared.rows.filter((r) => r.state !== "same" && r.state !== "broken" && !held.has(r.rel)).map((r) => r.rel)));
        if (v.shared.source) setSource((s) => s || v.shared.source!);
      }),
      studio.onMessage("rig_scan", () => load(source || undefined)), // which motors hold which file
      studio.onMessage("error", (m) => {
        if (typeof m.cmd === "string" && m.cmd.startsWith("rig_calfiles")) { setBusy(false); setError([m.message, m.fix].filter(Boolean).join(" ")); }
      }),
    ];
    return () => offs.forEach((off) => off());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // WHY on the link, not on mount: after a reload the page mounts before the socket opens, and a send then is lost
  useEffect(() => { if (link === "open") load(source || undefined); }, [link]); // eslint-disable-line react-hooks/exhaustive-deps

  const problems = useMemo(() => (view ? calfileProblems(view, label) : []), [view]);
  if (!view) return <p className="faint t-sm"><LoaderCircle className="spin ico-inline" aria-hidden />Reading the calibration folder</p>;

  const send = (cmd: string, extra: object) => { if (studio.send({ cmd, ...extra })) { setBusy(true); setError(null); } };
  const unused = view.files.filter((f) => f.unused);
  const toggle = (set: Set<string>, rel: string, fn: (s: Set<string>) => void) => {
    const n = new Set(set); if (n.has(rel)) n.delete(rel); else n.add(rel); fn(n);
  };
  const repo = view.shared.source?.replace(/\/configs\/calibration\/?$/, "");

  return (
    <div className="calfiles">
      <p className="faint t-sm">
        LeRobot reads one file per arm, chosen by the id: <span className="mono calfiles-path">{view.root}</span>. Files no arm uses are
        never read, but a wrong id writes one of them into an arm's motors. Keep only the rig's files.
      </p>
      {error && <p className="warn-text t-sm"><CircleAlert aria-hidden className="ico-inline" />{error}</p>}
      {problems.map((p) => (
        <p key={p.text} className={`${p.level === "danger" ? "danger-text" : "warn-text"} t-sm`}>
          <CircleAlert aria-hidden className="ico-inline" />{p.text}
        </p>
      ))}
      {view.done && <p className="ok-text t-sm">{doneText(view.done)}</p>}

      <h4 className="calfiles-h">On this Mac: {view.files.length} files, {unused.length} not used by the rig</h4>
      <div className="calfiles-rows">
          {view.files.map((f) => (
            <div className="calfiles-row" key={f.rel}>
              <span>{f.unused && <input type="checkbox" aria-label={`Move ${name(f.rel)} aside`} checked={pick.has(f.rel)}
                disabled={!control || busy} onChange={() => toggle(pick, f.rel, setPick)} />}</span>
              <span className="mono calfiles-name">{f.folder.split("/").pop()}/{name(f.rel)}</span>
              <span><FileChips f={f} /></span>
            </div>
          ))}
      </div>
      <div className="row-gap">
        <button className="btn" disabled={!control || busy || pick.size === 0}
          onClick={() => send("rig_calfiles_archive", { files: [...pick], source: source || undefined })}
          title="Moves them to calibration-archive next to the folder. Nothing is deleted.">
          {busy ? <LoaderCircle className="spin" aria-hidden /> : <Archive aria-hidden />}Move {pick.size} aside
        </button>
        {!control && <span className="faint t-sm">Take control (top right) to change files.</span>}
      </div>

      {!compact && view.archives.length > 0 && (
        <details className="calfiles-archives">
          <summary className="t-sm">{view.archives.length} archived {view.archives.length === 1 ? "set" : "sets"}</summary>
          {view.archives.map((a) => (
            <div key={a.name} className="row-gap t-sm">
              <span className="mono">{a.name}</span><span className="faint">{a.files.map(name).join(", ")}</span>
              <button className="btn btn-sm" disabled={!control || busy} onClick={() => send("rig_calfiles_restore", { name: a.name })}
                title="Puts these files back. Files there now are archived first.">
                <RotateCcw aria-hidden />Restore
              </button>
            </div>
          ))}
        </details>
      )}

      <h4 className="calfiles-h">Shared calibrations</h4>
      <p className="faint t-sm">
        The phi repo keeps one file per arm in configs/calibration, so every laptop uses the same numbers. Files it
        replaces here go to the archive.
      </p>
      <div className="row-gap calfiles-wrap">
        <input className="input mono calfiles-src" value={source} placeholder="/path/to/phi/configs/calibration"
          onChange={(e) => setSource(e.target.value)} aria-label="Shared calibration folder" />
        <button className="btn btn-sm" disabled={busy} onClick={() => load(source)}>Compare</button>
        {repo && <button className="btn btn-sm" disabled={!control} title={`git -C ${repo} pull --ff-only`}
          onClick={() => terminal.run(`git -C ${JSON.stringify(repo)} pull --ff-only`)}>Update from GitHub</button>}
      </div>
      {!view.shared.exists ? (
        <p className="faint t-sm">{view.shared.source ? `No folder at ${view.shared.source}.` : "No phi repo found."} Give the folder's path.</p>
      ) : view.shared.rows.length === 0 ? (
        <p className="faint t-sm">That folder holds no calibration files. Update the repo, then Compare.</p>
      ) : (
        <>
          <div className="calfiles-rows">
              {view.shared.rows.map((r) => (
                <div className="calfiles-row" key={r.rel}>
                  <span><input type="checkbox" aria-label={`Install ${name(r.rel)}`} checked={take.has(r.rel)}
                    disabled={!control || busy || r.state === "same" || r.state === "broken"} onChange={() => toggle(take, r.rel, setTake)} /></span>
                  <span className="mono calfiles-name">{r.rel.split("/").slice(-2).join("/")}</span>
                  <span><SharedChip r={r} />
                    {r.state !== "same" && view.files.some((f) => f.rel === r.rel && f.on_ports.length > 0) && (
                      <span className="chip sev-warn" title="Installing replaces numbers that match this arm's motors. Install it only if the shared file is for this physical arm, then ENTER at LeRobot's prompt writes it into that arm.">
                        this Mac's file matches its motors</span>)}</span>
                </div>
              ))}
          </div>
          <button className="btn" disabled={!control || busy || take.size === 0}
            onClick={() => send("rig_calfiles_install", { source, files: [...take] })}>
            <Download aria-hidden />Install {take.size}
          </button>
          <p className="faint t-sm">After installing, run Detect arms: it matches each arm to its file by what its motors hold.</p>
        </>
      )}
    </div>
  );
}

function FileChips({ f }: { f: CalFile }) {
  return (
    <span className="onb-chips">
      {f.used_by.map((k) => <span key={k} className="chip sev-ok">{label(k)} uses it</span>)}
      {f.on_ports.map((p) => <span key={p} className="chip" title={p}>motors on {p.split(".").pop()} hold it</span>)}
      {f.link && <span className="chip" title="LeRobot reads and writes the file it points at">link to {f.link}</span>}
      {f.same_as.length > 0 && <span className="chip" title={f.same_as.join("\n")}>same numbers as {f.same_as.map(name).join(", ")}</span>}
      {f.error && <span className="chip sev-error" title={f.error}>broken</span>}
      {f.unfinished && <span className="chip sev-warn" title={f.unfinished}>unfinished</span>}
      {f.junk && <span className="chip" title={f.junk}>not a calibration</span>}
      {f.unused && !f.junk && <span className="chip faint">not used</span>}
    </span>
  );
}

function SharedChip({ r }: { r: SharedRow }) {
  if (r.state === "same") return <span className="chip sev-ok">same as this Mac</span>;
  if (r.state === "new") return <span className="chip">not on this Mac</span>;
  if (r.state === "link") return <span className="chip sev-warn">here a link; installs a real file</span>;
  if (r.state === "broken") return <span className="chip sev-error" title={r.error ?? ""}>broken in the repo</span>;
  return <span className="chip sev-warn" title={r.error ?? ""}>differs{r.max_deg != null ? ` by up to ${r.max_deg}° (${r.worst_joint})` : ""}</span>;
}

function doneText(d: NonNullable<CalFilesView["done"]>): string {
  if (d.action === "archived") return `Moved ${d.files?.length ?? 0} files to archive ${d.archive}.`;
  if (d.action === "restored") return `Restored ${d.restored?.length ?? 0} files${d.archive ? `; the files they replaced are in ${d.archive}` : ""}.`;
  return `Installed ${d.copied?.length ?? 0} files${d.archive ? `; the ones they replaced are in archive ${d.archive}` : ""}.`;
}
