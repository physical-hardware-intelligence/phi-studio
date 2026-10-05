import { ArrowLeft, Ban, Check, Copy, Star } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { EpisodeStrip, FlagChips, HealthBar, Hist, Metric, SevIcon } from "../../components/data/bits";
import { Twin } from "../../components/Twin";
import {
  FLAG_LABEL, SHORT, fmtBytes, fmtDuration, useNotes, useResource,
  type DatasetAnalysis, type DatasetDetail, type Health,
} from "../../lib/data";
import type { TwinScene } from "../../lib/twin";
import { Notices } from "../../components/Notices";

type Sort = "index" | "duration" | "idle" | "track" | "path" | "flags";
const RANK: Record<Health, number> = { ok: 0, info: 1, warn: 2, error: 3 };

export function DatasetView({ id }: { id: string }) {
  const detail = useResource<DatasetDetail>(`/api/data/${id}`);
  const analysis = useResource<DatasetAnalysis>(detail.value && detail.value.summary.episodes > 0 ? `/api/data/${id}/analysis` : null);
  const notes = useNotes(id);
  const [health, setHealth] = useState<Health | "all">("all");
  const [kind, setKind] = useState<string | null>(null);
  const [task, setTask] = useState<string>("");
  const [sort, setSort] = useState<Sort>("index");
  const [copied, setCopied] = useState(false);

  const d = detail.value;
  const a = analysis.value;
  const marks = useMemo(() => {
    const m = new Map<number, { bad: boolean; good: boolean; n: number }>();
    for (const n of notes ?? []) {
      if (n.episode === null || n.status !== "open") continue;
      const e = m.get(n.episode) ?? { bad: false, good: false, n: 0 };
      if (n.kind === "bad") e.bad = true; else if (n.kind === "good") e.good = true; else e.n += 1;
      m.set(n.episode, e);
    }
    return m;
  }, [notes]);

  const rows = useMemo(() => {
    if (!a) return [];
    let r = a.episodes.filter((e) => (health === "all" || e.health === health)
      && (!kind || e.flags.some((f) => f.kind === kind && !f.dismissed))
      && (!task || e.tasks.includes(task)));
    const key: Record<Sort, (e: DatasetAnalysis["episodes"][number]) => number> = {
      index: (e) => e.index, duration: (e) => -e.metrics.duration_s, idle: (e) => -(e.metrics.idle_start_s + e.metrics.idle_end_s),
      track: (e) => -(e.metrics.track_rms ?? 0), path: (e) => -e.metrics.path_m,
      flags: (e) => -(RANK[e.health] * 100 + e.flags.filter((f) => !f.dismissed).length),
    };
    r = [...r].sort((x, y) => key[sort](x) - key[sort](y));
    return r;
  }, [a, health, kind, task, sort]);

  const excluded = useMemo(() => [...marks.entries()].filter(([, v]) => v.bad).map(([k]) => k).sort((x, y) => x - y), [marks]);
  const keep = useMemo(() => (d ? d.episodes.map((e) => e.index).filter((i) => !excluded.includes(i)) : []), [d, excluded]);
  const trainArg = `--dataset.episodes=[${keep.join(",")}]`;

  const onWorkspace = useCallback((s: TwinScene) => {
    if (!a) return;
    for (const [arm, w] of Object.entries(a.workspace)) {
      const n = w.points.length;
      const colors = new Float32Array(n * 3);
      const maxEp = Math.max(1, ...w.episode);
      for (let i = 0; i < n; i++) {
        const t = w.episode[i] / maxEp; // early episodes blue, late episodes amber
        colors[i * 3] = 0.36 + 0.55 * t; colors[i * 3 + 1] = 0.55 + 0.12 * t; colors[i * 3 + 2] = 0.95 - 0.6 * t;
      }
      s.setCloud(w.points, colors, arm);
      if (a.rest[arm]) s.setState(arm, a.rest[arm]);
      s.setGhost(arm, null);
    }
  }, [a]);

  if (detail.error) return <div className="page"><div className="empty tone-danger">{detail.error.message}</div></div>;
  if (!d) return <div className="page"><div className="empty">Loading</div></div>;
  const s = d.summary;
  const maxDur = a ? Math.max(...a.episodes.map((e) => e.metrics.duration_s)) : 1;
  const med = (xs: number[]) => { const v = [...xs].filter(Number.isFinite).sort((p, q) => p - q); return v.length ? v[Math.floor(v.length / 2)] : NaN; };
  const counts: Partial<Record<Health, number>> = {};
  a?.episodes.forEach((e) => { counts[e.health] = (counts[e.health] ?? 0) + 1; });
  const taskColor = (t: string) => `var(--task-${(d.tasks.indexOf(t) % 8) + 1})`;

  return (
    <div className="page page-wide">
      <Notices />
      <div className="crumbs">
        <a href="#/data" className="crumb-back"><ArrowLeft aria-hidden />Datasets</a>
        <h2 className="crumb-title">{s.repo_id}</h2>
        <span className="badge tone-neutral">{s.bimanual ? "bimanual" : "single arm"}</span>
        <span className="badge tone-neutral">{s.fps} fps</span>
        <span className="badge tone-neutral">{s.version}</span>
      </div>

      <div className="metrics-row">
        <Metric label="Episodes" value={String(s.episodes)} />
        <Metric label="Length" value={fmtDuration(s.duration_s)} />
        <Metric label="Frames" value={s.frames.toLocaleString()} />
        <Metric label="Cameras" value={String(s.cameras.length)} hint={s.cameras.map((c) => `${c.name} ${c.w}×${c.h} ${c.codec ?? ""}`).join("\n")} />
        <Metric label="Tasks" value={String(s.tasks)} />
        <Metric label="Size" value={fmtBytes(s.size ?? undefined)} />
        {a && <Metric label="Follower lag" value={med(a.episodes.map((e) => e.metrics.lag_ms ?? NaN)).toFixed(0)} unit="ms"
          hint="Median delay between the leader's command and the follower's motion" />}
        {a && <Metric label="Idle at start" value={med(a.episodes.map((e) => e.metrics.idle_start_s)).toFixed(1)} unit="s"
          hint="Median still time before the first motion: a trim candidate" />}
      </div>

      {s.episodes === 0 ? (
        <div className="panel"><div className="empty">This recording stopped before its first episode was saved. Nothing to show.</div></div>
      ) : (
        <div className="split">
          <div className="col">
            <section className="panel">
              <div className="toolbar">
                <div className="seg" role="group" aria-label="Health">
                  {(["all", "error", "warn", "info", "ok"] as const).map((h) => (
                    <button key={h} className={`seg-btn ${health === h ? "is-on" : ""}`} onClick={() => setHealth(h)}>
                      {h !== "all" && <i className={`hdot h-${h}`} />}
                      {{ all: "All", error: "Errors", warn: "Warnings", info: "Minor", ok: "Clean" }[h]}
                      <span className="seg-n">{h === "all" ? a?.episodes.length ?? "" : counts[h] ?? 0}</span>
                    </button>
                  ))}
                </div>
                {d.tasks.length > 1 && (
                  <select className="select select-sm" value={task} onChange={(e) => setTask(e.target.value)} aria-label="Task">
                    <option value="">Every task</option>
                    {d.tasks.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                )}
                <span className="grow" />
                <label className="faint t-sm">Sort</label>
                <select className="select select-sm" value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort">
                  <option value="index">Index</option><option value="flags">Worst first</option>
                  <option value="duration">Longest</option><option value="idle">Most idle</option>
                  <option value="track">Tracking error</option><option value="path">Tool path</option>
                </select>
              </div>
              {kind && (
                <div className="filter-note">Showing episodes with <b>{FLAG_LABEL[kind] ?? kind}</b>
                  <button className="link-btn" onClick={() => setKind(null)}>Clear</button></div>
              )}
              {!a && <div className="empty">{analysis.error ? analysis.error.message : "Analysing every episode"}</div>}
              {a && (
                <div className="eps" role="table" aria-label="Episodes">
                  <div className="eps-head" role="row">
                    <span>#</span><span>Timeline</span><span>Task</span>
                    <span className="r" title="Follower lag behind the leader">Lag</span>
                    <span className="r" title="Tracking error after removing lag">Track</span>
                    <span className="r" title="Tool path length">Path</span>
                    <span>Flags</span><span />
                  </div>
                  {rows.map((e) => {
                    const m = marks.get(e.index);
                    return (
                      <a key={e.index} role="row" className={`eps-row h-${e.health} ${m?.bad ? "is-bad" : ""}`} href={`#/data/${id}/${e.index}`}>
                        <span className="num mono">{e.index}</span>
                        <EpisodeStrip duration={e.metrics.duration_s} max={maxDur} idleStart={e.metrics.idle_start_s}
                          idleEnd={e.metrics.idle_end_s} grasps={e.grasps_t ?? []} health={e.health} />
                        <span className="eps-task" title={e.tasks.join("; ")}>
                          <i className="task-dot" style={{ background: taskColor(e.tasks[0] ?? "") }} />{e.tasks[0] ?? ""}
                        </span>
                        <span className="r num">{e.metrics.lag_ms !== null ? `${e.metrics.lag_ms.toFixed(0)}` : "–"}<span className="u">ms</span></span>
                        <span className="r num">{e.metrics.track_rms !== null ? e.metrics.track_rms.toFixed(1) : "–"}<span className="u">°</span></span>
                        <span className="r num">{e.metrics.path_m.toFixed(2)}<span className="u">m</span></span>
                        <FlagChips flags={e.flags} max={3} />
                        <span className="eps-marks">
                          {m?.bad && <Ban aria-label="excluded from training" className="mark-bad" />}
                          {m?.good && <Star aria-label="reference episode" className="mark-good" />}
                          {!!m?.n && <span className="badge tone-info">{m.n}</span>}
                        </span>
                      </a>
                    );
                  })}
                  {!rows.length && <div className="empty">No episode matches.</div>}
                </div>
              )}
            </section>

            {a && (
              <section className="panel">
                <div className="panel-head"><h3 className="panel-title">Joint coverage</h3>
                  <span className="panel-sub">Every frame. Fill measured, line commanded, band the model's range, ticks p1 · median · p99</span></div>
                <div className="hists">
                  {Object.entries(a.distributions).map(([k, h]) => (
                    <div key={k} className="hist-cell">
                      <div className="hist-head"><span className="strong">{k.startsWith("left_") ? "L " : k.startsWith("right_") ? "R " : ""}{SHORT[k.replace(/^(left|right)_/, "")] ?? k}</span>
                        <span className="faint num">{h.min.toFixed(0)} … {h.max.toFixed(0)}{k.endsWith("gripper") ? "" : "°"}</span></div>
                      <Hist edges={h.edges} a={h.state} b={h.action} model={h.model} q={h.q} />
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>

          <div className="col rail">
            <section className="panel">
              <div className="panel-head"><h3 className="panel-title">Health</h3>
                {a && <HealthBar counts={counts} total={a.episodes.length} />}</div>
              <div className="panel-body tight">
                {a?.health.map((h, i) => <div key={i} className="hrow"><SevIcon s={h.severity} /><span>{h.text}</span></div>)}
                {a && !a.health.length && <div className="hrow"><Check aria-hidden className="sev sev-ok" /><span>Files and metadata check out</span></div>}
                {a && Object.entries(a.flag_counts).sort((p, q) => q[1] - p[1]).map(([k, n]) => (
                  <button key={k} className={`hrow hrow-btn ${kind === k ? "is-on" : ""}`} onClick={() => setKind(kind === k ? null : k)}>
                    <span className="hrow-k">{FLAG_LABEL[k] ?? k}</span><span className="num faint">{n}</span>
                  </button>
                ))}
              </div>
            </section>

            {a && (
              <section className="panel">
                <div className="panel-head"><h3 className="panel-title">Workspace</h3><span className="panel-sub">Tool positions, early → late</span></div>
                <div className="ws-twin"><Twin arms={s.bimanual ? ["left", "right"] : [""]} onReady={onWorkspace} legend={null} /></div>
              </section>
            )}

            {d.tasks.length > 0 && a && (
              <section className="panel">
                <div className="panel-head"><h3 className="panel-title">Tasks</h3></div>
                <div className="panel-body tight">
                  {Object.entries(a.tasks).sort((p, q) => q[1] - p[1]).map(([t, n]) => (
                    <button key={t} className={`taskbar ${task === t ? "is-on" : ""}`} onClick={() => setTask(task === t ? "" : t)} title={t}>
                      <span className="taskbar-fill" style={{ width: `${(n / s.episodes) * 100}%`, background: taskColor(t) }} />
                      <span className="taskbar-t">{t}</span><span className="num">{n}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            <section className="panel">
              <div className="panel-head"><h3 className="panel-title">Train on</h3>
                <span className="panel-sub">{keep.length} of {s.episodes}</span></div>
              <div className="panel-body tight">
                <p className="t-sm muted">Episodes you mark <Ban aria-hidden className="ico-inline mark-bad" /> bad drop out of this list.</p>
                <div className="copyline">
                  <code className="mono">{excluded.length ? trainArg : "--dataset.episodes=all"}</code>
                  <button className="btn btn-ghost btn-sm btn-icon" disabled={!excluded.length}
                    onClick={() => { void navigator.clipboard.writeText(trainArg); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
                    title="Copy the lerobot-train argument">{copied ? <Check aria-hidden /> : <Copy aria-hidden />}</button>
                </div>
              </div>
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
