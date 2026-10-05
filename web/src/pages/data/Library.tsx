import { Camera, RefreshCw, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArmsGlyph, HealthBar } from "../../components/data/bits";
import { fmtAgo, fmtBytes, fmtDuration, invalidate, load, useResource, type DatasetSummary } from "../../lib/data";
import { Notices } from "../../components/Notices";

type Filter = "all" | "single" | "bimanual" | "attention";

export function Library() {
  const { value, error, loading, reload } = useResource<{ datasets: DatasetSummary[]; roots: string[] }>("/api/data/datasets");
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const queued = useRef(new Set<string>());

  // Analyse what has not been analysed yet, one dataset at a time, so every row gets its health bar.
  useEffect(() => {
    const todo = (value?.datasets ?? []).filter((d) => d.episodes > 0 && !d.analysis && !queued.current.has(d.id));
    if (!todo.length) return;
    let live = true;
    (async () => {
      for (const d of todo) {
        queued.current.add(d.id);
        try { await load(`/api/data/${d.id}/analysis`); } catch { /* shown on the dataset page */ }
        if (!live) return;
      }
      invalidate("/api/data/datasets");
    })();
    return () => { live = false; };
  }, [value]);

  const rows = useMemo(() => {
    const all = value?.datasets ?? [];
    const needle = q.trim().toLowerCase();
    return all.filter((d) => {
      if (needle && !`${d.repo_id} ${d.robot_type ?? ""}`.toLowerCase().includes(needle)) return false;
      if (filter === "single") return !d.bimanual;
      if (filter === "bimanual") return d.bimanual;
      if (filter === "attention") return d.episodes === 0 || (d.analysis?.episodes_by_health.error ?? 0) > 0 || !!d.notes;
      return true;
    });
  }, [value, q, filter]);

  const total = value?.datasets ?? [];
  const hours = total.reduce((s, d) => s + d.duration_s, 0) / 3600;
  const eps = total.reduce((s, d) => s + d.episodes, 0);

  return (
    <div className="page">
      <Notices />
      <div className="statline">
        <Stat label="Datasets" value={String(total.length)} />
        <Stat label="Episodes" value={eps.toLocaleString()} />
        <Stat label="Hours" value={hours.toFixed(1)} />
        <Stat label="On disk" value={fmtBytes(total.reduce((s, d) => s + (d.size ?? 0), 0))} />
      </div>
      <section className="panel">
        <div className="toolbar">
          <label className="search">
            <Search aria-hidden />
            <input className="input" placeholder="Filter datasets" value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
          <div className="seg" role="group" aria-label="Show">
            {(["all", "single", "bimanual", "attention"] as Filter[]).map((f) => (
              <button key={f} className={`seg-btn ${filter === f ? "is-on" : ""}`} onClick={() => setFilter(f)}>
                {{ all: "All", single: "Single", bimanual: "Bimanual", attention: "Needs attention" }[f]}
              </button>
            ))}
          </div>
          <span className="grow" />
          <button className="btn btn-ghost btn-sm" onClick={() => { invalidate("/api/data/datasets"); void load("/api/data/datasets?refresh=1").then(reload); }}
            title={value ? `Looks in ${value.roots.join(", ")}` : undefined}>
            <RefreshCw aria-hidden className={loading ? "spin" : ""} /> Rescan
          </button>
        </div>
        {error && <div className="empty tone-danger">{error.message}</div>}
        {!error && !value && <div className="empty">Looking for datasets</div>}
        {value && !rows.length && <div className="empty">No dataset matches.</div>}
        {!!rows.length && (
          <div className="dsl" role="list">
            <div className="dsl-head" aria-hidden>
              <span /><span>Dataset</span><span>Health</span><span className="r">Episodes</span><span className="r">Length</span>
              <span className="r">Cams</span><span className="r">Size</span><span className="r">Changed</span>
            </div>
            {rows.map((d) => <Row key={d.id} d={d} />)}
          </div>
        )}
      </section>
    </div>
  );
}

function Row({ d }: { d: DatasetSummary }) {
  const empty = d.episodes === 0;
  const [owner, name] = d.repo_id.includes("/") ? d.repo_id.split("/", 2) : [null, d.repo_id];
  return (
    <a role="listitem" className={`dsl-row ${empty ? "is-empty" : ""}`} href={`#/data/${d.id}`}>
      <ArmsGlyph bimanual={d.bimanual} />
      <span className="dsl-name">
        <span className="dsl-title">{name}</span>
        <span className="dsl-sub">
          {owner && <span className="faint">{owner}</span>}
          <span className="faint">{d.fps} fps</span>
          <span className="faint">{d.version}</span>
          {empty && <span className="badge tone-warn">empty</span>}
          {!!d.notes && <span className="badge tone-info">{d.notes} open</span>}
        </span>
      </span>
      <span>{empty ? <span className="faint t-sm">Recording stopped before episode 1</span>
        : d.analysis ? <HealthBar counts={d.analysis.episodes_by_health} total={d.episodes} /> : <span className="faint t-sm">Analysing</span>}</span>
      <span className="r num">{d.episodes}</span>
      <span className="r num">{fmtDuration(d.duration_s)}</span>
      <span className="r num" title={d.cameras.map((c) => `${c.name} ${c.w}×${c.h}`).join("\n")}>
        <Camera aria-hidden className="ico-inline" />{d.cameras.length}
      </span>
      <span className="r num">{fmtBytes(d.size)}</span>
      <span className="r faint">{fmtAgo(d.modified)}</span>
    </a>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="stat"><span className="stat-v num">{value}</span><span className="stat-l">{label}</span></div>;
}
