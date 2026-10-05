// One playhead for an episode: videos, the 3D twin, plots and the timeline all read it.
// While playing, the master clock is the first camera's displayed frame (requestVideoFrameCallback), so a
// plot cursor never runs ahead of the picture; with no video it is a frame clock on requestAnimationFrame.
import { useEffect, useRef, useState } from "react";

type Listener = (p: Player) => void;

export class Player {
  t = 0;
  playing = false;
  rate = 1;
  duration = 0;
  fps = 30;
  n = 0;
  private listeners = new Set<Listener>();
  private raf = 0;
  private last = 0;
  /** Seconds since the episode start, from the master video, or null when it cannot say. */
  clock: (() => number | null) | null = null;

  configure(n: number, fps: number): void {
    this.n = n;
    this.fps = fps;
    this.duration = n / fps;
    this.t = Math.min(this.t, this.duration);
    this.emit();
  }

  /** The frame on screen: frame k covers [k/fps, (k+1)/fps). */
  frame(t = this.t): number {
    return Math.max(0, Math.min(this.n - 1, Math.floor(t * this.fps + 1e-6)));
  }

  seek(t: number): void {
    this.t = Math.max(0, Math.min(this.duration - 1 / this.fps, t));
    this.emit();
  }

  seekFrame(k: number): void { this.seek(Math.max(0, Math.min(this.n - 1, k)) / this.fps); }
  step(frames: number): void { this.pause(); this.seekFrame(this.frame() + frames); }

  play(): void {
    if (this.playing) return;
    if (this.t >= this.duration - 1.5 / this.fps) this.t = 0;
    this.playing = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
    this.emit();
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    cancelAnimationFrame(this.raf);
    this.emit();
  }

  toggle(): void { if (this.playing) this.pause(); else this.play(); }

  setRate(r: number): void { this.rate = r; this.emit(); }

  private tick = (now: number) => {
    if (!this.playing) return;
    const fromVideo = this.clock?.();
    let t = fromVideo ?? this.t + ((now - this.last) / 1000) * this.rate;
    this.last = now;
    if (t >= this.duration - 0.5 / this.fps) {
      t = this.duration - 1 / this.fps;
      this.t = t;
      this.pause();
      return;
    }
    this.t = Math.max(0, t);
    this.emit();
    this.raf = requestAnimationFrame(this.tick);
  };

  on(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void { this.listeners.forEach((l) => l(this)); }

  dispose(): void { this.pause(); this.listeners.clear(); }
}

/** Re-render on every playhead change, at most once per animation frame. For light components only. */
export function usePlayhead(p: Player): { t: number; frame: number; playing: boolean; rate: number } {
  const [s, set] = useState({ t: p.t, frame: p.frame(), playing: p.playing, rate: p.rate });
  const pending = useRef(0);
  useEffect(() => p.on((pl) => {
    if (pending.current) return;
    pending.current = requestAnimationFrame(() => {
      pending.current = 0;
      set({ t: pl.t, frame: pl.frame(), playing: pl.playing, rate: pl.rate });
    });
  }), [p]);
  useEffect(() => () => cancelAnimationFrame(pending.current), []);
  return s;
}
