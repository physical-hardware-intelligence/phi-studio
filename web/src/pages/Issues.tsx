import { Ban, Check, RotateCcw, Star, Trash2 } from "lucide-react";
import { Notices } from "../components/Notices";
import { useMemo, useState } from "react";
import { deleteNote, fmtAgo, saveNote, useNotes, type Note, type NoteKind } from "../lib/data";

type Status = "open" | "resolved" | "all";

// Every note and issue across datasets, newest first. A row links to its episode at its moment.
export function Issues() {
  const all = useNotes(null, null);
  const [status, setStatus] = useState<Status>("open");
  const [kind, setKind] = useState<NoteKind | "all">("all");
  const [ds, setDs] = useState("");

  const datasets = useMemo(() => {
    const m = new Map<string, string>();
    for (const n of all ?? []) m.set(n.dataset, n.dataset_name || n.dataset);
    return [...m.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [all]);
  const rows = useMemo(() => (all ?? []).filter((n) =>
    (status === "all" || n.status === status) && (kind === "all" || n.kind === kind) && (!ds || n.dataset === ds)), [all, status, kind, ds]);
  const count = (k: NoteKind) => (all ?? []).filter((n) => n.kind === k && n.status === "open").length;

  return (
    <div className="page">
      <Notices />
      <div className="statline">
        <Stat label="Open issues" value={count("issue")} tone={count("issue") ? "warn" : undefined} />
        <Stat label="Notes" value={count("note")} />
        <Stat label="Excluded episodes" value={count("bad")} />
        <Stat label="Reference episodes" value={count("good")} />
      </div>
      <section className="panel">
        <div className="toolbar">
          <div className="seg" role="group" aria-label="Status">
            {(["open", "resolved", "all"] as Status[]).map((s) => (
              <button key={s} className={`seg-btn ${status === s ? "is-on" : ""}`} onClick={() => setStatus(s)}>{{ open: "Open", resolved: "Resolved", all: "All" }[s]}</button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Kind">
            {(["all", "issue", "note", "bad", "good"] as const).map((k) => (
              <button key={k} className={`seg-btn ${kind === k ? "is-on" : ""}`} onClick={() => setKind(k)}>
                {{ all: "Everything", issue: "Issues", note: "Notes", bad: "Excluded", good: "Reference" }[k]}</button>
            ))}
          </div>
          {datasets.length > 1 && (
            <select className="select select-sm" value={ds} onChange={(e) => setDs(e.target.value)} aria-label="Dataset">
              <option value="">Every dataset</option>
              {datasets.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
            </select>
          )}
        </div>
        {!all && <div className="empty">Loading</div>}
        {all && !rows.length && <div className="empty">Nothing here. Press N on any episode to write a note.</div>}
        <div className="issues">
          {rows.map((n) => <IssueRow key={n.id} n={n} />)}
        </div>
      </section>
    </div>
  );
}

function IssueRow({ n }: { n: Note }) {
  const where = n.episode !== null ? `#/data/${n.dataset}/${n.episode}` : `#/data/${n.dataset}`;
  const icon = n.kind === "bad" ? <Ban aria-hidden className="mark-bad" /> : n.kind === "good" ? <Star aria-hidden className="mark-good" />
    : <span className={`kind-dot kind-${n.kind}`} />;
  const text = n.text || (n.kind === "bad" ? "Left out of training" : n.kind === "good" ? "Reference episode" : "");
  return (
    <div className={`issue ${n.status === "resolved" ? "is-resolved" : ""}`}>
      <span className="issue-icon">{icon}</span>
      <a className="issue-main" href={where}>
        <span className="issue-text">{text}</span>
        <span className="issue-where">
          <span className="ident">{(n.dataset_name || n.dataset).split("/").pop()}</span>
          {n.episode !== null && <span className="num">ep {n.episode}</span>}
          {n.t0 !== null && <span className="num">{n.t0.toFixed(2)} s</span>}
          <span className="faint">{n.author} · {fmtAgo(n.created)}</span>
        </span>
      </a>
      {(n.kind === "issue" || n.kind === "note") && (
        <button className="btn btn-ghost btn-sm btn-icon" title={n.status === "open" ? "Resolve" : "Reopen"}
          onClick={() => saveNote({ ...n, status: n.status === "open" ? "resolved" : "open" })}>
          {n.status === "open" ? <Check aria-hidden /> : <RotateCcw aria-hidden />}
        </button>
      )}
      <button className="btn btn-ghost btn-sm btn-icon" title="Delete" onClick={() => deleteNote(n.id, n.dataset)}><Trash2 aria-hidden /></button>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone?: "warn" }) {
  return <div className={`stat ${tone ? "tone-warn is-toned" : ""}`}><span className="stat-v num">{value}</span><span className="stat-l">{label}</span></div>;
}
