// The Φ lab mark, drawn exactly as the lab site's assets/phi-mark-ink.svg (physical-hardware-intelligence.github.io):
// an open ring, the stem, two sensor eyes. currentColor, so it is ink on light and near-white on dark.
export function PhiMark({ size = 30, className, title }: { size?: number; className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 500 500" width={size} height={size} className={className}
      role={title ? "img" : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      <g fill="none" stroke="currentColor" strokeWidth={30} strokeLinecap="butt">
        <path d="M 279.5 118 A 136 136 0 0 1 279.5 382" />
        <path d="M 220.5 118 A 136 136 0 0 0 220.5 382" />
      </g>
      <rect x={239} y={56} width={22} height={388} rx={11} fill="currentColor" />
      <circle cx={193} cy={254} r={21} fill="currentColor" />
      <circle cx={307} cy={254} r={21} fill="currentColor" />
    </svg>
  );
}
