// The episode's timeline: flags, grasps, idle stretches and notes on one track, with the playhead.
// Positions are percentages of the episode, so it needs no resize handling.
import { useEffect, useMemo, useRef } from "react";
import type { Flag, GripEvent, Note } from "../../lib/data";
import type { Player } from "../../lib/player";

interface Props {
  player: Player;
  duration: number;
  idle: [number, number][];
  closed: { t0: number; t1: number; arm: string | null }[];
  arms: string[];
  events: GripEvent[];
  flags: Flag[];
  notes: Note[];
  onFlag?: (f: Flag) => void;
  onNote?: (n: Note) => void;
}

function ticks(duration: number): number[] {
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60];
  const step = steps.find((s) => duration / s <= 12) ?? 120;
  const out = [];
  for (let t = 0; t <= duration + 1e-9; t += step) out.push(+t.toFixed(3));
  return out;
}

export function Timeline({ player, duration, idle, closed, arms, events, flags, notes, onFlag, onNote }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const pct = (t: number) => `${(Math.max(0, Math.min(duration, t)) / duration) * 100}%`;
  const width = (t0: number, t1: number) => `${(Math.max(0, t1 - t0) / duration) * 100}%`;
  const marks = useMemo(() => ticks(duration), [duration]);

  useEffect(() => player.on((p) => {
    if (head.current) head.current.style.left = pct(p.frame() / p.fps);
  }), [player, duration]);

  useEffect(() => {
    const el = track.current;
    if (!el) return;
    let drag = false;
    const at = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      return ((e.clientX - r.left) / r.width) * duration;
    };
    const down = (e: PointerEvent) => {
      if ((e.target as HTMLElement).closest("button")) return; // a marker click is not a scrub
      drag = true; el.setPointerCapture(e.pointerId); player.pause(); player.seek(at(e));
    };
    const move = (e: PointerEvent) => { if (drag) player.seek(at(e)); };
    const up = () => { drag = false; };
    el.addEventListener("pointerdown", down); el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
    return () => { el.removeEventListener("pointerdown", down); el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
  }, [player, duration]);

  const timed = flags.filter((f) => f.t0 !== null && !f.dismissed);
  const lanes = arms.length ? arms : [""];
  return (
    <div className="tl">
      <div className="tl-ruler">
        {marks.map((t) => <span key={t} className="tl-tick" style={{ left: pct(t) }}>{t % 1 ? t.toFixed(1) : t}s</span>)}
      </div>
      <div className="tl-track" ref={track}>
        {idle.map(([a, b], i) => <div key={`i${i}`} className="tl-idle" style={{ left: pct(a), width: width(a, b) }} title={`Still ${(b - a).toFixed(1)} s`} />)}
        <div className="tl-lane tl-flags">
          {timed.map((f, i) => (
            <button key={`f${i}`} className={`tl-flag sev-${f.severity} ${f.t1 !== null && f.t1 - (f.t0 ?? 0) > duration / 200 ? "is-span" : "is-point"}`}
              style={{ left: pct(f.t0 ?? 0), width: f.t1 !== null ? width(f.t0 ?? 0, f.t1) : undefined }}
              title={f.text} onClick={() => { player.pause(); player.seek(f.t0 ?? 0); onFlag?.(f); }} />
          ))}
        </div>
        {lanes.map((arm) => (
          <div key={arm || "arm"} className="tl-lane tl-grip">
            {arms.length > 1 && <span className="tl-lane-name">{arm === "left" ? "L" : "R"}</span>}
            {closed.filter((c) => (c.arm ?? "") === arm).map((c, i) => (
              <div key={i} className="tl-closed" style={{ left: pct(c.t0), width: width(c.t0, c.t1) }} title={`Holding ${(c.t1 - c.t0).toFixed(1)} s`} />
            ))}
            {events.filter((e) => (e.arm ?? "") === arm).map((e, i) => (
              <span key={i} className={`tl-ev tl-${e.kind}`} style={{ left: pct(e.t) }} title={`${e.kind} at ${e.t.toFixed(2)} s`} />
            ))}
          </div>
        ))}
        <div className="tl-lane tl-notes">
          {notes.filter((n) => n.t0 !== null).map((n) => (
            <button key={n.id} className={`tl-note kind-${n.kind}`} style={{ left: pct(n.t0 ?? 0), width: n.t1 !== null ? width(n.t0 ?? 0, n.t1) : undefined }}
              title={n.text || n.kind} onClick={() => { player.pause(); player.seek(n.t0 ?? 0); onNote?.(n); }} />
          ))}
        </div>
        <div className="tl-head" ref={head} style={{ left: pct(player.frame() / player.fps) }} />
      </div>
    </div>
  );
}
