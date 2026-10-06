// Record: episodes of teleop into a LeRobot dataset (recorder.py through the worker). The cameras and the 3D view
// stay live the whole time; the rail holds the recording. Keys as lerobot-record's: → or Space ends the take early
// (kept), ← or Backspace records it again, Esc stops everything.
import { ArrowLeft, ArrowRight, Circle, CircleAlert, CircleCheck, CircleX, Square, Volume2, VolumeX } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { Notices } from "../components/Notices";
import { invalidate, useResource, type DatasetSummary } from "../lib/data";
import { studio, useStudio, type RecView, type TakeHealth } from "../lib/studio";
import { ViewSwitch } from "../scene/ViewSwitch";

// What the operator last chose, kept while they move between pages.
const memory = { name: "phi/", task: "", episodes: 10, episode_s: 30, reset_s: 10 };
const MUTE_KEY = "phi-studio-rec-mute";

export function Record() {
  return (
    <div className="page record">
      <ActionBar activity="teleop" />
      <Notices />
      <div className="work-grid">
        <div className="col">
          <ViewSwitch page="record" />
        </div>
        <div className="col rail">
          <RecordPanel />
        </div>
      </div>
    </div>
  );
}

function RecordPanel() {
  const rec = useStudio((s) => s.telemetry?.recording ?? null);
  const live = rec && rec.phase !== "done";
  return (
    <section className="panel rec-panel">
      <div className="panel-head">
        <h2 className="panel-title">Record</h2>
        <Mute />
      </div>
      {live ? <Live rec={rec} /> : <Setup last={rec} />}
      {rec && rec.takes.length > 0 && <Takes rec={rec} />}
    </section>
  );
}

// -- before ------------------------------------------------------------------------------------------
function Setup({ last }: { last: RecView | null }) {
  const st = useStudio((s) => s.state);
  const control = useStudio((s) => s.control);
  const { value } = useResource<{ datasets: DatasetSummary[] }>("/api/data/datasets");
  const [f, setF] = useState(memory);
  const set = (p: Partial<typeof memory>) => { Object.assign(memory, p); setF({ ...memory }); };
  const existing = (value?.datasets ?? []).map((d) => d.repo_id).filter((r) => r.includes("/"));
  const lr = useStudio((s) => s.files.index?.lerobot ?? null);
  const mock = useStudio((s) => s.mock);
  const link = useStudio((s) => s.link);
  useEffect(() => { if (link === "open") studio.loadFiles(); }, [link]);
  // WHY the Hugging Face user: a dataset under someone else's owner cannot be uploaded later.
  useEffect(() => { if (lr?.hf_user && memory.name === "phi/") set({ name: `${lr.hf_user}/` }); }, [lr?.hf_user]);
  // Cameras robot-config.yaml names with no device: the worker opens none of them, so none is recorded.
  const leftOut = mock ? [] : (lr?.cameras ?? []).filter((c) => c.source == null).map((c) => c.feature.replace("observation.images.", ""));
  const append = existing.includes(f.name.trim());
  const teleop = st?.state === "MOVING" && st.activity === "teleop";
  const nameOk = /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(f.name.trim());
  const why = !control ? "Take control first"
    : !teleop ? "Start teleop first (above)"
    : !nameOk ? "Name it owner/name, for example phi/cube_pick"
    : !f.task.trim() ? "Say the task in a few words"
    : null;
  useEffect(() => { if (last?.finished) invalidate("/api/data/datasets"); }, [last?.finished]);
  const start = () => studio.send({
    cmd: "rec_start", repo_id: f.name.trim(), task: f.task.trim(), episodes: f.episodes,
    episode_s: f.episode_s, reset_s: f.reset_s, resume: append,
  });
  return (
    <div className="panel-body form">
      {last && <Summary rec={last} />}
      <label className="field">
        <span className="field-label">Dataset</span>
        <input className="input mono" value={f.name} list="rec-datasets" spellCheck={false}
          onChange={(e) => set({ name: e.target.value })} />
        <datalist id="rec-datasets">{existing.map((r) => <option key={r} value={r} />)}</datalist>
        <span className="field-hint">{append ? "Adds episodes to this dataset." : "New. A date and time are added to the name."}</span>
      </label>
      <label className="field">
        <span className="field-label">Task</span>
        <input className="input" value={f.task} maxLength={300} placeholder="Pick up the red cube and place it in the box"
          onChange={(e) => set({ task: e.target.value })} />
      </label>
      <div className="rec-nums">
        <Num label="Episodes" value={f.episodes} min={1} max={500} onChange={(v) => set({ episodes: v })} />
        <Num label="Length" unit="s" value={f.episode_s} min={1} max={600} onChange={(v) => set({ episode_s: v })} />
        <Num label="Reset" unit="s" value={f.reset_s} min={0} max={300} onChange={(v) => set({ reset_s: v })} />
      </div>
      {leftOut.length > 0 && (
        <p className="warn-text t-sm">
          <CircleAlert aria-hidden className="ico-inline" />Not recorded: {leftOut.join(", ")}. robot-config.yaml gives{" "}
          {leftOut.length === 1 ? "it" : "them"} no device. <a className="link-btn" href="#/setup">Find cameras on Setup</a>
        </p>
      )}
      <div className="form-actions">
        {why && <span className="field-hint">{why}</span>}
        <button className="btn btn-primary rec-go" disabled={why !== null} onClick={start}><Circle aria-hidden className="rec-dot" />Record</button>
      </div>
    </div>
  );
}

function Num({ label, unit, value, min, max, onChange }: { label: string; unit?: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <div className={unit ? "input-suffix" : undefined}>
        <input className="input num" type="number" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        {unit && <span>{unit}</span>}
      </div>
    </label>
  );
}

function Summary({ rec }: { rec: RecView }) {
  const name = rec.root.split("/").slice(-2).join("/");
  if (!rec.finished) return <p className="rec-summary faint">Finishing the dataset</p>;
  return (
    <div className={`rec-summary tone-${rec.error ? "warn" : "ok"}`}>
      {rec.error ? <CircleAlert aria-hidden /> : <CircleCheck aria-hidden />}
      <span>{rec.saved} {rec.saved === 1 ? "episode" : "episodes"} saved to <span className="mono">{name}</span>{rec.error ? `. ${rec.error}` : ""}</span>
      <a className="link-btn t-sm" href="#/data">Datasets</a>
    </div>
  );
}

// -- during ------------------------------------------------------------------------------------------
const PHASE: Record<RecView["phase"], [string, string]> = {
  warmup: ["Get ready", "info"], record: ["Recording", "danger"], reset: ["Reset", "warn"], paused: ["Paused", "warn"], done: ["Done", "ok"],
};

function Live({ rec }: { rec: RecView }) {
  const control = useStudio((s) => s.control);
  const send = (cmd: string) => () => studio.send({ cmd });
  useKeys(rec, control);
  useCues(rec);
  const left = rec.limit !== null ? Math.max(0, rec.limit - rec.t) : 0;
  // Elapsed from frames recorded, not the wall clock: the dataset's own time (frame_index / fps).
  const fps = rec.limit ? rec.target / rec.limit : 30;
  const elapsed = rec.frames / fps;
  const pct = rec.phase === "record" ? (rec.frames / rec.target) * 100 : rec.limit ? (rec.t / rec.limit) * 100 : 0;
  const [label, tone] = PHASE[rec.phase];
  const total = rec.takes.reduce((a, t) => a + (t.seconds ?? 0), 0);
  return (
    <div className="panel-body rec-live">
      <div className="rec-head">
        <span className={`rec-phase tone-${tone}`}>{rec.phase === "record" && <span className="rec-blink" aria-hidden />}{label}</span>
        <span className="faint num">Episode {Math.min(rec.episode + 1, rec.of)} of {rec.of}</span>
      </div>
      <div className="rec-clock num" aria-live="off">
        {fmt(rec.phase === "record" ? elapsed : left)}
        {rec.phase === "record" && rec.limit !== null && <span className="faint"> / {fmt(rec.limit)}</span>}
        <span className="rec-clock-what faint">{rec.phase === "record" ? "elapsed" : rec.phase === "paused" ? "" : "left"}</span>
      </div>
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}>
        <span style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
      {rec.phase === "paused" && <p className="text-warn t-sm">{rec.why}</p>}
      {rec.phase === "warmup" && (
        <p className="faint t-sm">{rec.ready ? "Hands on the leaders. The first take starts on its own." : "Starting the recorder."}</p>
      )}
      {rec.phase === "reset" && <p className="faint t-sm">Reset the scene. The next take starts on its own.</p>}
      <div className="rec-buttons">
        <button className="btn btn-primary" disabled={!control} onClick={send("rec_next")}>
          {rec.phase === "record" ? "End early" : rec.phase === "paused" ? "Resume" : "Start now"}<ArrowRight aria-hidden /><kbd>→</kbd>
        </button>
        <button className="btn" disabled={!control || !(rec.phase === "record" || rec.phase === "reset")} onClick={send("rec_redo")}>
          <ArrowLeft aria-hidden />Re-record<kbd>←</kbd>
        </button>
        <button className="btn btn-ghost" disabled={!control} onClick={send("rec_stop")} title="Keeps every saved episode; the take in progress is dropped">
          <Square aria-hidden />Stop
        </button>
      </div>
      <div className="rec-health faint t-sm num">
        <span>{rec.saved} saved</span>
        {total > 0 && <span>{fmt(total)} recorded</span>}
        {rec.dropped > 0 && <span className="text-danger">{rec.dropped} frames dropped</span>}
        {rec.late > 0 && <span className="text-warn">{rec.late} late frames</span>}
        {rec.error && <span className="text-danger">{rec.error}</span>}
      </div>
    </div>
  );
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** lerobot-record's keys: → / Space end early (or start now, or resume), ← / Backspace record again. Esc is Studio's
 * Stop everywhere (lib/studio.ts), which also ends the recording. Not while typing in a field. */
function useKeys(rec: RecView, control: boolean) {
  const phase = useRef(rec.phase);
  phase.current = rec.phase;
  useEffect(() => {
    if (!control) return;
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName))) return;
      if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); studio.send({ cmd: "rec_next" }); }
      else if (e.key === "ArrowLeft" || e.key === "Backspace") {
        if (phase.current === "record" || phase.current === "reset") { e.preventDefault(); studio.send({ cmd: "rec_redo" }); }
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [control]);
}

// -- audio cues: a high tone when a take starts, a low one when it ends, a tick for each of the last 3 seconds ---------
let audio: AudioContext | null = null;
function tone(freq: number, ms: number): void {
  try {
    if (localStorage.getItem(MUTE_KEY) === "1") return;
  } catch { /* storage blocked: play */ }
  try {
    audio ??= new AudioContext();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.08, audio.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + ms / 1000);
    o.connect(g).connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + ms / 1000);
  } catch { /* no audio: the screen still says it */ }
}

function useCues(rec: RecView) {
  const last = useRef<{ phase: string; sec: number }>({ phase: rec.phase, sec: -1 });
  useEffect(() => {
    const p = last.current;
    if (rec.phase !== p.phase) {
      if (rec.phase === "record") tone(880, 180);
      else if (p.phase === "record") tone(440, 260);
      p.phase = rec.phase;
    }
    if (rec.phase === "record" && rec.limit !== null) {
      const left = Math.ceil(rec.limit - rec.frames / (rec.target / rec.limit));
      if (left <= 3 && left >= 1 && left !== p.sec) tone(660, 70);
      p.sec = left;
    }
  }, [rec.phase, rec.frames, rec.limit, rec.target]);
}

function Mute() {
  const [muted, setMuted] = useState(() => { try { return localStorage.getItem(MUTE_KEY) === "1"; } catch { return false; } });
  const flip = () => { const m = !muted; setMuted(m); try { localStorage.setItem(MUTE_KEY, m ? "1" : "0"); } catch { /* fine */ } };
  return (
    <button className="btn btn-ghost btn-sm btn-icon" onClick={flip} aria-pressed={muted} title={muted ? "Sound off" : "Sound on"} aria-label="Sound cues">
      {muted ? <VolumeX aria-hidden /> : <Volume2 aria-hidden />}
    </button>
  );
}

// -- each take's health, checked during the reset ----------------------------------------------------
function Takes({ rec }: { rec: RecView }) {
  return (
    <div className="rec-takes">
      <div className="field-label">Takes</div>
      <ol className="rec-take-list">
        {rec.takes.map((t) => <Take key={t.episode} t={t} />)}
      </ol>
    </div>
  );
}

function Take({ t }: { t: TakeHealth }) {
  const Icon = t.health === "error" ? CircleX : t.health === "warn" ? CircleAlert : CircleCheck;
  const worst = t.flags.find((f) => f.severity === "error") ?? t.flags.find((f) => f.severity === "warn");
  return (
    <li className={`rec-take tone-${t.health === "info" ? "ok" : t.health === "error" ? "danger" : t.health}`} title={t.flags.map((f) => f.text).join("\n") || "No flags"}>
      <Icon aria-hidden />
      <span className="num">{t.episode + 1}</span>
      <span className="num">{fmt(t.seconds ?? 0)}</span>
      <span className="faint t-sm ellipsis">{worst ? worst.text : `${t.frames} frames`}</span>
    </li>
  );
}
