import { ArrowLeft, Ban, ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward, Star, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FlagChips, Metric, SevIcon } from "../../components/data/bits";
import { Notices } from "../../components/Notices";
import { CameraStrip } from "../../components/data/CameraStrip";
import { Signals, type SignalRow } from "../../components/data/Signals";
import { Timeline } from "../../components/data/Timeline";
import { Twin } from "../../components/Twin";
import {
  SHORT, deleteNote, dismissFlag, fmtAgo, fmtClock, saveNote, useNotes, useResource,
  type EpisodePayload, type Flag, type Note, type NoteKind,
} from "../../lib/data";
import { Player, usePlayhead } from "../../lib/player";
import type { TwinScene } from "../../lib/twin";

type Mode = "position" | "velocity" | "tracking";

export function Inspector({ id, episode }: { id: string; episode: number }) {
  const res = useResource<EpisodePayload>(`/api/data/${id}/episode/${episode}`);
  const p = res.value;
  const player = useMemo(() => new Player(), []);
  useEffect(() => () => player.dispose(), [player]);
  const [mode, setMode] = useState<Mode>("position");
  const scene = useRef<TwinScene | null>(null);
  const notes = useNotes(id, episode);
  const composer = useRef<HTMLTextAreaElement>(null);

  const cols = useMemo(() => {
    if (!p) return null;
    const n = p.t.length, d = p.names.length;
    const col = (m: number[][] | undefined, c: number) => {
      const out = new Float32Array(n);
      if (m) for (let i = 0; i < n; i++) out[i] = m[i]?.[c] ?? NaN;
      return out;
    };
    const s = p.analysis.series;
    return {
      n, d,
      state: Array.from({ length: d }, (_, c) => col(p.state, c)),
      action: Array.from({ length: d }, (_, c) => col(p.action, c)),
      vel: Array.from({ length: d }, (_, c) => col(s?.vel, c)),
      track: Array.from({ length: d }, (_, c) => col(s?.track, c)),
      err: Array.from({ length: d }, (_, c) => col(s?.err, c)),
    };
  }, [p]);

  useEffect(() => {
    if (!p) return;
    player.pause();
    player.configure(p.t.length, p.dataset.fps);
    player.seek(0);
  }, [p, player]);

  // Drive the twin from the playhead.
  const bindTwin = useCallback(() => {
    const s = scene.current;
    if (!s || !p || !cols) return;
    for (const arm of p.arms) {
      s.setTrail(arm.name, p.analysis.series?.tcp[arm.name] ?? null);
    }
    const draw = () => {
      const k = player.frame();
      for (const arm of p.arms) {
        const idx = arm.index;
        s.setState(arm.name, idx.map((c) => (c === null ? NaN : cols.state[c][k])));
        s.setGhost(arm.name, idx.map((c) => (c === null ? NaN : cols.action[c][k])));
        s.setTrailCursor(arm.name, k);
      }
    };
    draw();
    return player.on(draw);
  }, [p, cols, player]);
  const unbind = useRef<(() => void) | undefined>(undefined);
  useEffect(() => { unbind.current?.(); unbind.current = bindTwin(); return () => unbind.current?.(); }, [bindTwin]);
  const onTwin = useCallback((s: TwinScene) => { scene.current = s; unbind.current?.(); unbind.current = bindTwin(); }, [bindTwin]);

  const rows: SignalRow[] = useMemo(() => {
    if (!p || !cols) return [];
    const out: SignalRow[] = [];
    for (const arm of p.arms) {
      arm.joints.forEach((j, ji) => {
        const c = arm.index[ji];
        if (c === null) return;
        const grip = j === "gripper";
        const group = p.arms.length > 1 ? (arm.name === "left" ? "L" : "R") : undefined;
        const base = { key: `${arm.name}_${j}`, label: SHORT[j] ?? j, group };
        if (mode === "position") {
          out.push({ ...base, unit: grip ? "%" : "°", a: cols.state[c], b: cols.action[c], heat: grip ? undefined : cols.track[c],
            limit: p.limits[j] as [number, number] | undefined, minSpan: grip ? 6 : 4 });
        } else if (mode === "velocity") {
          out.push({ ...base, unit: grip ? "%/s" : "°/s", a: cols.vel[c], sym: true, minSpan: 20 });
        } else {
          out.push({ ...base, unit: grip ? "%" : "°", a: cols.track[c], b: cols.err[c], sym: true, minSpan: 4 });
        }
      });
    }
    return out;
  }, [p, cols, mode]);

  const go = useCallback((k: number) => {
    if (!p) return;
    const next = Math.max(0, Math.min(p.dataset.episodes - 1, k));
    if (next !== episode) location.hash = `#/data/${id}/${next}`;
  }, [p, id, episode]);

  const marks = useMemo(() => ({
    bad: (notes ?? []).find((n) => n.kind === "bad" && n.status === "open") ?? null,
    good: (notes ?? []).find((n) => n.kind === "good" && n.status === "open") ?? null,
  }), [notes]);
  const toggleMark = useCallback((kind: "bad" | "good") => {
    if (!p) return;
    const existing = marks[kind];
    if (existing) deleteNote(existing.id, id);
    else saveNote({ dataset: id, dataset_name: p.dataset.repo_id, episode, kind, severity: kind === "bad" ? "warn" : "info", text: "" });
  }, [marks, p, id, episode]);

  // Keys. Ignored while typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
      const fps = player.fps;
      switch (e.key) {
        case " ": e.preventDefault(); (t as HTMLElement).blur?.(); player.toggle(); break;
        case "ArrowRight": e.preventDefault(); player.step(e.shiftKey ? Math.round(fps) : 1); break;
        case "ArrowLeft": e.preventDefault(); player.step(e.shiftKey ? -Math.round(fps) : -1); break;
        case "Home": player.pause(); player.seekFrame(0); break;
        case "End": player.pause(); player.seekFrame(player.n - 1); break;
        case "]": go(episode + 1); break;
        case "[": go(episode - 1); break;
        case "n": case "N": e.preventDefault(); composer.current?.focus(); break;
        case "b": case "B": toggleMark("bad"); break;
        case "g": case "G": toggleMark("good"); break;
        case "1": setMode("position"); break;
        case "2": setMode("velocity"); break;
        case "3": setMode("tracking"); break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player, go, episode, toggleMark]);

  if (res.error) return <div className="page"><div className="empty tone-danger">{res.error.message}</div></div>;
  if (!p || !cols) return <div className="page"><div className="empty">Loading episode {episode}</div></div>;
  const a = p.analysis;
  const m = a.metrics;
  const armNames = p.arms.map((x) => x.name);
  const live = a.flags.filter((f) => !f.dismissed);

  return (
    <div className="page page-wide insp">
      <Notices />
      <div className="crumbs">
        <a href={`#/data/${id}`} className="crumb-back"><ArrowLeft aria-hidden />{p.dataset.repo_id.split("/").pop()}</a>
        <div className="ep-nav">
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => go(episode - 1)} disabled={episode <= 0} title="Previous episode ([)"><ChevronLeft aria-hidden /></button>
          <span className="ep-num"><span className="faint">Episode</span> <b className="num">{episode}</b><span className="faint num"> / {p.dataset.episodes - 1}</span></span>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => go(episode + 1)} disabled={episode >= p.dataset.episodes - 1} title="Next episode (])"><ChevronRight aria-hidden /></button>
        </div>
        <span className="ep-task" title={p.episode.tasks.join("; ")}>{p.episode.tasks[0] ?? ""}</span>
        <span className="grow" />
        <FlagChips flags={a.flags} max={3} />
        <button className={`btn btn-sm mark ${marks.good ? "is-good" : ""}`} onClick={() => toggleMark("good")} title="Reference episode (G)">
          <Star aria-hidden />{marks.good ? "Reference" : "Good"}</button>
        <button className={`btn btn-sm mark ${marks.bad ? "is-bad" : ""}`} onClick={() => toggleMark("bad")} title="Leave out of training (B)">
          <Ban aria-hidden />{marks.bad ? "Excluded" : "Bad"}</button>
      </div>

      <div className="insp-grid">
        <div className="insp-main">
          <div className={`media cams-${Object.keys(p.episode.videos).length}`}>
            <div className="media-twin">
              <Twin arms={armNames} onReady={onTwin} />
            </div>
            <CameraStrip dataset={id} videos={p.episode.videos} player={player} fps={p.dataset.fps} />
          </div>

          <Transport player={player} fps={p.dataset.fps} />
          <Timeline player={player} duration={p.t.length / p.dataset.fps} idle={a.idle} closed={a.closed} arms={armNames.length > 1 ? armNames : []}
            events={a.events} flags={a.flags} notes={notes ?? []} />

          <section className="panel sig-panel">
            <div className="toolbar">
              <div className="seg" role="tablist" aria-label="Signal">
                {(["position", "velocity", "tracking"] as Mode[]).map((x, i) => (
                  <button key={x} role="tab" aria-selected={mode === x} className={`seg-btn ${mode === x ? "is-on" : ""}`} onClick={() => setMode(x)}>
                    {{ position: "Position", velocity: "Velocity", tracking: "Tracking" }[x]}<span className="kbd">{i + 1}</span>
                  </button>
                ))}
              </div>
              <span className="grow" />
              <span className="legend">
                {mode === "position" && <><span><i className="ln ln-a" />measured</span><span><i className="ln ln-b" />commanded</span><span><i className="ln ln-heat" />tracking error</span></>}
                {mode === "velocity" && <span><i className="ln ln-a" />measured velocity</span>}
                {mode === "tracking" && <><span><i className="ln ln-a" />after lag</span><span><i className="ln ln-b" />raw</span></>}
              </span>
            </div>
            <Signals rows={rows} n={cols.n} fps={p.dataset.fps} player={player} shade={a.idle}
              marks={a.events.map((e) => ({ t: e.t, kind: e.kind }))}
              aLabel={mode === "tracking" ? "after lag" : "measured"} bLabel={mode === "tracking" ? "raw" : "commanded"} />
          </section>
        </div>

        <aside className="insp-rail">
          <section className="panel">
            <div className="metric-grid">
              <Metric label="Length" value={m.duration_s.toFixed(1)} unit="s" />
              <Metric label="Grasps" value={String(m.grasps)} />
              <Metric label="Idle start" value={m.idle_start_s.toFixed(1)} unit="s" tone={m.idle_start_s >= 1.5 ? "warn" : undefined} />
              <Metric label="Idle end" value={m.idle_end_s.toFixed(1)} unit="s" tone={m.idle_end_s >= 1.5 ? "warn" : undefined} />
              <Metric label="Lag" value={m.lag_ms !== null ? m.lag_ms.toFixed(0) : "–"} unit="ms" hint="Follower behind leader, median over joints" />
              <Metric label="Tracking" value={m.track_rms !== null ? m.track_rms.toFixed(2) : "–"} unit="°" hint="RMS error after removing the lag" />
              <Metric label="Tool path" value={m.path_m.toFixed(2)} unit="m" />
              <Metric label="Smoothness" value={m.sparc !== null ? m.sparc.toFixed(2) : "–"} hint="SPARC of tool speed: closer to 0 is smoother" />
            </div>
          </section>
          <section className="panel">
            <div className="panel-head"><h3 className="panel-title">Flags</h3><span className="panel-sub">{live.length || "None"}</span></div>
            <div className="flag-list">
              {a.flags.map((f, i) => <FlagRow key={i} f={f} onSeek={(t) => { player.pause(); player.seek(t); }}
                onDismiss={() => dismissFlag(id, episode, f, !!f.dismissed)} />)}
              {!a.flags.length && <div className="empty">Nothing unusual in this episode.</div>}
            </div>
          </section>
          <NotesPanel id={id} name={p.dataset.repo_id} episode={episode} player={player} notes={notes} composer={composer} />
        </aside>
      </div>
    </div>
  );
}

function Transport({ player, fps }: { player: Player; fps: number }) {
  const s = usePlayhead(player);
  return (
    <div className="transport">
      <button className="btn btn-ghost btn-sm btn-icon" onClick={() => player.step(-1)} title="Back one frame (←)"><SkipBack aria-hidden /></button>
      <button className="btn btn-sm btn-icon play-btn" onClick={() => player.toggle()} title="Play / pause (Space)">
        {s.playing ? <Pause aria-hidden /> : <Play aria-hidden />}</button>
      <button className="btn btn-ghost btn-sm btn-icon" onClick={() => player.step(1)} title="Forward one frame (→)"><SkipForward aria-hidden /></button>
      <span className="clock num">{fmtClock(s.frame / fps)}<span className="faint"> / {fmtClock(player.duration)}</span></span>
      <span className="faint num t-sm">frame {s.frame}</span>
      <span className="grow" />
      <div className="seg" role="group" aria-label="Speed">
        {[0.25, 0.5, 1, 2].map((r) => (
          <button key={r} className={`seg-btn ${s.rate === r ? "is-on" : ""}`} onClick={() => player.setRate(r)}>{r}×</button>
        ))}
      </div>
    </div>
  );
}

function PlayheadTime({ player }: { player: Player }) {
  const s = usePlayhead(player);
  return <span className="num">{(s.frame / player.fps).toFixed(2)} s</span>;
}

function FlagRow({ f, onSeek, onDismiss }: { f: Flag; onSeek: (t: number) => void; onDismiss: () => void }) {
  return (
    <div className={`flag-row ${f.dismissed ? "is-dismissed" : ""}`}>
      <SevIcon s={f.severity} />
      <button className="flag-text" onClick={() => f.t0 !== null && onSeek(f.t0)} disabled={f.t0 === null}>
        {f.text}{f.t0 !== null && <span className="faint num"> · {f.t0.toFixed(1)} s</span>}
      </button>
      <button className="btn btn-ghost btn-sm btn-icon" onClick={onDismiss} title={f.dismissed ? "Restore this flag" : "Not a problem: hide on every Mac"}>
        <X aria-hidden /></button>
    </div>
  );
}

function NotesPanel({ id, name, episode, player, notes, composer }: {
  id: string; name: string; episode: number; player: Player; notes: Note[] | null; composer: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const [text, setText] = useState("");
  const [kind, setKind] = useState<NoteKind>("note");
  const [pin, setPin] = useState(true);
  const submit = () => {
    if (!text.trim()) return;
    const t = player.frame() / player.fps; // read at the moment of saving; no re-render per frame
    if (saveNote({ dataset: id, dataset_name: name, episode, kind, severity: kind === "issue" ? "warn" : "info",
      text: text.trim(), t0: pin ? +t.toFixed(3) : null, t1: null })) setText("");
  };
  const list = (notes ?? []).filter((n) => n.kind === "note" || n.kind === "issue");
  return (
    <section className="panel">
      <div className="panel-head"><h3 className="panel-title">Notes</h3><span className="kbd">N</span></div>
      <div className="note-compose">
        <textarea ref={composer} className="textarea" rows={2} placeholder="What did you notice?" value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(); } }} />
        <div className="note-actions">
          <div className="seg" role="group" aria-label="Kind">
            {(["note", "issue"] as NoteKind[]).map((k) => (
              <button key={k} className={`seg-btn ${kind === k ? "is-on" : ""}`} onClick={() => setKind(k)}>{k === "note" ? "Note" : "Issue"}</button>
            ))}
          </div>
          <label className="pin"><input type="checkbox" checked={pin} onChange={(e) => setPin(e.target.checked)} /> at <PlayheadTime player={player} /></label>
          <span className="grow" />
          <button className="btn btn-primary btn-sm" onClick={submit} disabled={!text.trim()} title="Save (⌘↵)">Save</button>
        </div>
      </div>
      <div className="note-list">
        {list.map((n) => (
          <div key={n.id} className={`note kind-${n.kind}`}>
            <div className="note-meta">
              <span className={`badge ${n.kind === "issue" ? "tone-warn" : "tone-neutral"}`}>{n.kind}</span>
              {n.t0 !== null && <button className="link-btn num" onClick={() => { player.pause(); player.seek(n.t0 ?? 0); }}>{n.t0.toFixed(2)} s</button>}
              <span className="faint">{n.author} · {fmtAgo(n.created)}</span>
              <span className="grow" />
              <button className="btn btn-ghost btn-sm btn-icon" onClick={() => deleteNote(n.id, id)} title="Delete"><Trash2 aria-hidden /></button>
            </div>
            <p className="note-text">{n.text}</p>
          </div>
        ))}
        {notes && !list.length && <div className="empty">No notes yet.</div>}
      </div>
    </section>
  );
}
