// Small visual pieces shared by the data pages: a glyph, a health bar, an episode strip, flag chips and a
// histogram. Each says one thing with a picture instead of a sentence.
import { AlertOctagon, AlertTriangle, Info } from "lucide-react";
import type { Flag, Health, Severity } from "../../lib/data";
import { FLAG_LABEL } from "../../lib/data";

export function ArmsGlyph({ bimanual }: { bimanual: boolean }) {
  // One or two stylised arms: base, upper arm, forearm, gripper.
  const arm = (x: number) => (
    <g transform={`translate(${x} 0)`}>
      <path d="M1 17h8" /><path d="M5 17v-5l3.5-5 3 3" /><circle cx="5" cy="12" r="0.9" />
    </g>
  );
  return (
    <svg className="glyph" viewBox={bimanual ? "0 0 26 20" : "0 0 14 20"} aria-label={bimanual ? "bimanual" : "single arm"} role="img">
      {arm(0)}{bimanual && arm(12)}
    </svg>
  );
}

const ORDER: Health[] = ["ok", "info", "warn", "error"];
export function HealthBar({ counts, total }: { counts: Partial<Record<Health, number>>; total: number }) {
  if (!total) return null;
  const title = ORDER.filter((h) => counts[h]).map((h) => `${counts[h]} ${({ ok: "clean", info: "minor", warn: "warning", error: "error" } as const)[h]}`).join(" · ");
  return (
    <span className="hbar" title={title} role="img" aria-label={title}>
      {ORDER.map((h) => (counts[h] ? <i key={h} className={`hbar-${h}`} style={{ flexGrow: counts[h] }} /> : null))}
    </span>
  );
}

/** An episode as a bar: length to scale, still stretches at the ends hatched, each grasp a tick. */
export function EpisodeStrip({ duration, max, idleStart, idleEnd, grasps, health }: {
  duration: number; max: number; idleStart: number; idleEnd: number; grasps: number[]; health: Health;
}) {
  const w = Math.max(4, (duration / max) * 100);
  return (
    <span className="estrip" aria-hidden>
      <span className={`estrip-bar h-${health}`} style={{ width: `${w}%` }}>
        {idleStart > 0 && <i className="estrip-idle" style={{ left: 0, width: `${(idleStart / duration) * 100}%` }} />}
        {idleEnd > 0 && <i className="estrip-idle" style={{ right: 0, width: `${(idleEnd / duration) * 100}%` }} />}
        {grasps.map((t, i) => <i key={i} className="estrip-grasp" style={{ left: `${(t / duration) * 100}%` }} />)}
      </span>
    </span>
  );
}

export function SevIcon({ s }: { s: Severity }) {
  return s === "error" ? <AlertOctagon aria-hidden className="sev sev-error" />
    : s === "warn" ? <AlertTriangle aria-hidden className="sev sev-warn" /> : <Info aria-hidden className="sev sev-info" />;
}

export function FlagChips({ flags, max = 4 }: { flags: Flag[]; max?: number }) {
  const live = flags.filter((f) => !f.dismissed);
  const by = new Map<string, { n: number; s: Severity; text: string[] }>();
  const rank = { info: 1, warn: 2, error: 3 };
  for (const f of live) {
    const e = by.get(f.kind) ?? { n: 0, s: "info" as Severity, text: [] };
    e.n += 1;
    if (rank[f.severity] > rank[e.s]) e.s = f.severity;
    e.text.push(f.text);
    by.set(f.kind, e);
  }
  const items = [...by.entries()].sort((a, b) => rank[b[1].s] - rank[a[1].s]);
  return (
    <span className="chips">
      {items.slice(0, max).map(([k, e]) => (
        <span key={k} className={`chip sev-${e.s}`} title={e.text.join("\n")}>
          {FLAG_LABEL[k] ?? k}{e.n > 1 ? ` ×${e.n}` : ""}
        </span>
      ))}
      {items.length > max && <span className="chip faint">+{items.length - max}</span>}
    </span>
  );
}

/** A small histogram: measured (fill) and commanded (outline), the model's range as a band. */
export function Hist({ edges, a, b, model, q, height = 44 }: {
  edges: number[]; a: number[]; b?: number[]; model?: [number, number]; q?: number[]; height?: number;
}) {
  const n = a.length;
  const lo = edges[0], hi = edges[edges.length - 1];
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;
  const peak = Math.max(1, ...a, ...(b ?? []));
  const bw = 100 / n;
  const path = (b ?? []).map((v, i) => `${i ? "L" : "M"}${(i * bw).toFixed(2)},${(height - (v / peak) * height).toFixed(2)}H${((i + 1) * bw).toFixed(2)}`).join("");
  return (
    <svg className="hist" viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" role="img">
      {model && <rect className="hist-model" x={x(model[0])} width={x(model[1]) - x(model[0])} y={0} height={height} />}
      {a.map((v, i) => v ? <rect key={i} className="hist-a" x={i * bw + 0.08} width={bw - 0.16} y={height - (v / peak) * height} height={(v / peak) * height} /> : null)}
      {b && <path className="hist-b" d={path} />}
      {q && q.map((v, i) => <line key={i} className={i === 1 ? "hist-med" : "hist-q"} x1={x(v)} x2={x(v)} y1={0} y2={height} />)}
    </svg>
  );
}

export function Metric({ label, value, unit, hint, tone }: { label: string; value: string; unit?: string; hint?: string; tone?: "warn" | "danger" | "ok" }) {
  return (
    <div className={`metric ${tone ? `tone-${tone} is-toned` : ""}`} title={hint}>
      <span className="metric-l">{label}</span>
      <span className="metric-v num">{value}{unit && <span className="metric-u">{unit}</span>}</span>
    </div>
  );
}
