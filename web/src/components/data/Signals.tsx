// Joint signals as small multiples on two canvases: the lines (drawn when data, size or theme change) and
// the cursor (drawn every frame). Min/max decimation per pixel column, so a one-frame spike is never lost.
import { useEffect, useMemo, useRef } from "react";
import type { Player } from "../../lib/player";

export interface SignalRow {
  key: string;
  label: string;
  group?: string; // "L" / "R" on a bimanual dataset
  unit: string;
  a: Float32Array; // primary: measured state, velocity, or lag-compensated error
  b?: Float32Array; // secondary: commanded action, or raw error
  heat?: Float32Array; // |lag-compensated error|, as a strip under the row
  limit?: [number, number];
  sym?: boolean; // y range symmetric about zero
  minSpan?: number;
}

interface Props {
  rows: SignalRow[];
  n: number;
  fps: number;
  player: Player;
  shade?: [number, number][]; // seconds, e.g. idle stretches
  marks?: { t: number; kind: string }[]; // grasp / release
  aLabel: string;
  bLabel?: string;
  rowH?: number;
}

const GUTTER = 148;
const HEAT_H = 3;

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function range(r: SignalRow): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const s of [r.a, r.b]) {
    if (!s) continue;
    for (let i = 0; i < s.length; i++) {
      const v = s[i];
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  if (!Number.isFinite(lo)) return [-1, 1];
  if (r.sym) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  const span = Math.max(hi - lo, r.minSpan ?? 4);
  const mid = (hi + lo) / 2;
  return [mid - span * 0.56, mid + span * 0.56];
}

export function Signals({ rows, n, fps, player, shade = [], marks = [], aLabel, bLabel, rowH = 46 }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const base = useRef<HTMLCanvasElement>(null);
  const over = useRef<HTMLCanvasElement>(null);
  const valsA = useRef<(HTMLSpanElement | null)[]>([]);
  const valsB = useRef<(HTMLSpanElement | null)[]>([]);
  const hover = useRef<number | null>(null);
  const ranges = useMemo(() => rows.map(range), [rows]);
  const height = rows.length * rowH;

  useEffect(() => {
    const el = wrap.current, cb = base.current, co = over.current;
    if (!el || !cb || !co) return;
    let width = 1;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const x = (frame: number) => (n <= 1 ? 0 : (frame / (n - 1)) * width);
    const frameAt = (px: number) => Math.round((Math.max(0, Math.min(width, px)) / width) * (n - 1));

    const drawBase = () => {
      const ctx = cb.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const cText = cssVar("--text-primary"), cAccent = cssVar("--accent"), cBorder = cssVar("--border-subtle");
      const cWarn = cssVar("--warn-solid"), cDanger = cssVar("--danger-solid"), cFaint = cssVar("--text-tertiary");
      // idle stretches, across every row
      ctx.fillStyle = cFaint;
      ctx.globalAlpha = 0.07;
      for (const [t0, t1] of shade) ctx.fillRect(x(t0 * fps), 0, Math.max(1, x(t1 * fps) - x(t0 * fps)), height);
      ctx.globalAlpha = 1;
      // grasp / release lines
      for (const m of marks) {
        ctx.strokeStyle = m.kind === "grasp" ? cText : cFaint;
        ctx.globalAlpha = m.kind === "grasp" ? 0.28 : 0.22;
        ctx.beginPath(); ctx.moveTo(Math.round(x(m.t * fps)) + 0.5, 0); ctx.lineTo(Math.round(x(m.t * fps)) + 0.5, height); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      rows.forEach((r, ri) => {
        const top = ri * rowH, bot = top + rowH - HEAT_H - 2;
        const [lo, hi] = ranges[ri];
        const y = (v: number) => bot - ((v - lo) / (hi - lo)) * (bot - top - 6) - 3;
        ctx.strokeStyle = cBorder; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, top + rowH - 0.5); ctx.lineTo(width, top + rowH - 0.5); ctx.stroke();
        if (r.sym || (lo < 0 && hi > 0)) {
          ctx.globalAlpha = 0.5; ctx.setLineDash([2, 3]);
          ctx.beginPath(); ctx.moveTo(0, Math.round(y(0)) + 0.5); ctx.lineTo(width, Math.round(y(0)) + 0.5); ctx.stroke();
          ctx.setLineDash([]); ctx.globalAlpha = 1;
        }
        if (r.limit) {
          ctx.strokeStyle = cDanger; ctx.globalAlpha = 0.45; ctx.setLineDash([4, 4]);
          for (const L of r.limit) {
            if (L > lo && L < hi) { ctx.beginPath(); ctx.moveTo(0, Math.round(y(L)) + 0.5); ctx.lineTo(width, Math.round(y(L)) + 0.5); ctx.stroke(); }
          }
          ctx.setLineDash([]); ctx.globalAlpha = 1;
        }
        const line = (s: Float32Array, color: string, w: number, alpha: number) => {
          ctx.strokeStyle = color; ctx.lineWidth = w; ctx.globalAlpha = alpha;
          ctx.beginPath();
          if (n <= width * 1.5) {
            let started = false;
            for (let i = 0; i < n; i++) {
              const v = s[i];
              if (!Number.isFinite(v)) { started = false; continue; }
              if (!started) { ctx.moveTo(x(i), y(v)); started = true; } else ctx.lineTo(x(i), y(v));
            }
          } else {
            // min/max per pixel column
            for (let px = 0; px < width; px++) {
              const f0 = Math.floor((px / width) * (n - 1)), f1 = Math.max(f0 + 1, Math.floor(((px + 1) / width) * (n - 1)));
              let mn = Infinity, mx = -Infinity;
              for (let i = f0; i < f1 && i < n; i++) { const v = s[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
              if (!Number.isFinite(mn)) continue;
              ctx.moveTo(px + 0.5, y(mx)); ctx.lineTo(px + 0.5, y(mn) + 0.01);
            }
          }
          ctx.stroke();
          ctx.globalAlpha = 1;
        };
        if (r.b) line(r.b, cAccent, 1, 0.95);
        line(r.a, cText, 1.35, 0.9);
        if (r.heat) {
          const hy = top + rowH - HEAT_H - 1;
          for (let px = 0; px < width; px++) {
            const f0 = Math.floor((px / width) * (n - 1)), f1 = Math.max(f0 + 1, Math.floor(((px + 1) / width) * (n - 1)));
            let mx = 0;
            for (let i = f0; i < f1 && i < n; i++) { const v = Math.abs(r.heat[i]); if (v > mx) mx = v; }
            if (mx < 3) continue;
            ctx.fillStyle = mx >= 8 ? cDanger : cWarn;
            ctx.globalAlpha = Math.min(1, 0.25 + (mx - 3) / 10);
            ctx.fillRect(px, hy, 1, HEAT_H);
          }
          ctx.globalAlpha = 1;
        }
      });
    };

    const drawOver = () => {
      const ctx = co.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const k = player.frame();
      ctx.strokeStyle = cssVar("--accent"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(Math.round(x(k)) + 0.5, 0); ctx.lineTo(Math.round(x(k)) + 0.5, height); ctx.stroke();
      const h = hover.current;
      if (h !== null) {
        ctx.strokeStyle = cssVar("--text-tertiary"); ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(Math.round(x(h)) + 0.5, 0); ctx.lineTo(Math.round(x(h)) + 0.5, height); ctx.stroke();
        ctx.setLineDash([]);
      }
      const at = h ?? k;
      rows.forEach((r, ri) => {
        const a = valsA.current[ri], b = valsB.current[ri];
        if (a) a.textContent = fmt(r.a[at]);
        if (b && r.b) b.textContent = fmt(r.b[at]);
      });
    };

    const size = () => {
      width = Math.max(1, el.clientWidth - GUTTER);
      for (const c of [cb, co]) {
        c.width = Math.round(width * dpr); c.height = Math.round(height * dpr);
        c.style.width = `${width}px`; c.style.height = `${height}px`;
      }
      drawBase(); drawOver();
    };
    const ro = new ResizeObserver(size);
    ro.observe(el);
    const mo = new MutationObserver(() => { drawBase(); drawOver(); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const off = player.on(drawOver);

    let dragging = false;
    const local = (e: PointerEvent) => e.clientX - co.getBoundingClientRect().left;
    const down = (e: PointerEvent) => { dragging = true; co.setPointerCapture(e.pointerId); player.pause(); player.seekFrame(frameAt(local(e))); };
    const move = (e: PointerEvent) => {
      if (dragging) player.seekFrame(frameAt(local(e)));
      hover.current = dragging ? null : frameAt(local(e));
      drawOver();
    };
    const up = () => { dragging = false; };
    const leave = () => { hover.current = null; drawOver(); };
    co.addEventListener("pointerdown", down);
    co.addEventListener("pointermove", move);
    co.addEventListener("pointerup", up);
    co.addEventListener("pointerleave", leave);
    size();
    return () => {
      ro.disconnect(); mo.disconnect(); off();
      co.removeEventListener("pointerdown", down); co.removeEventListener("pointermove", move);
      co.removeEventListener("pointerup", up); co.removeEventListener("pointerleave", leave);
    };
  }, [rows, ranges, n, fps, player, shade, marks, height, rowH]);

  return (
    <div className="sig" ref={wrap}>
      <div className="sig-gutter" style={{ width: GUTTER }}>
        {rows.map((r, i) => (
          <div key={r.key} className={`sig-label ${i > 0 && rows[i - 1].group !== r.group ? "is-group-start" : ""}`} style={{ height: rowH }}>
            <div className="sig-name">
              {r.group && <span className="sig-group">{r.group}</span>}
              <span>{r.label}</span>
            </div>
            <div className="sig-vals">
              <span className="sig-a" ref={(e) => { valsA.current[i] = e; }} title={aLabel} />
              {r.b && <span className="sig-b" ref={(e) => { valsB.current[i] = e; }} title={bLabel} />}
              <span className="sig-unit">{r.unit}</span>
            </div>
          </div>
        ))}
      </div>
      <div className="sig-plot" style={{ height }}>
        <canvas ref={base} />
        <canvas ref={over} className="sig-over" />
      </div>
    </div>
  );
}

function fmt(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}
