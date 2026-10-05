// Telemetry timing for the 3D view: a ring of recent samples per arm, and the map from the worker's clock to this
// page's. Import-free so web/tests can run it under Node (npm test).

const RING = 16;

/** Telemetry for one arm, buffered so the view can draw it between updates at display rate. */
export class Track {
  private readonly t = new Float64Array(RING);
  private readonly q: Float64Array[] = Array.from({ length: RING }, () => new Float64Array(6));
  private n = 0;
  private head = 0;
  lastArrival = 0;

  /** `t`: the sample's time on this page's clock (ms). `arrival`: when it came (performance.now()). */
  push(t: number, q: ArrayLike<number>, arrival: number): void {
    // WHY: samples must stay in time order for sample() to search them. One older than the newest means the
    // worker's clock went back (a new worker), so the old samples belong to another timeline.
    if (this.n && t < this.t[(this.head + this.n - 1) % RING]) this.clear();
    const i = (this.head + this.n) % RING;
    if (this.n === RING) this.head = (this.head + 1) % RING; else this.n++;
    this.t[i] = t;
    this.q[i].set(q);
    this.lastArrival = arrival;
  }

  /** Joint angles at `t` (local ms), linear between the two samples around it; the newest sample when `t` is past
   * it (no extrapolation: a late update must not overshoot). False when there is no sample at all. */
  sample(t: number, out: Float64Array): boolean {
    if (!this.n) return false;
    const at = (k: number) => (this.head + k) % RING;
    let k = this.n - 1;
    if (t >= this.t[at(k)]) { out.set(this.q[at(k)]); return true; }
    while (k > 0 && this.t[at(k - 1)] > t) k--;
    if (k === 0) { out.set(this.q[at(0)]); return true; }
    const a = at(k - 1), b = at(k);
    const u = (t - this.t[a]) / Math.max(1e-6, this.t[b] - this.t[a]);
    for (let j = 0; j < 6; j++) out[j] = this.q[a][j] + (this.q[b][j] - this.q[a][j]) * u;
    return true;
  }

  clear(): void { this.n = 0; this.head = 0; }
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Maps the worker's clock onto this page's and picks how far behind live to draw. The offset is the smallest
 * arrival - send gap seen lately (the least-delayed message); the delay covers one update interval plus the
 * 90th-percentile lateness, so the next sample has almost always arrived when the view needs it. */
export class Clock {
  private readonly gaps: number[] = [];
  private readonly intervals: number[] = [];
  private lastT: number | null = null;
  offset = 0;
  delay = 60;
  jumped = false; // the last add() found the worker's clock had gone back and started over

  /** `tWorker`: the telemetry's t (s, the worker's time.monotonic). `now`: arrival (ms). Returns the local time (ms). */
  add(tWorker: number, now: number): number {
    // WHY start over: a new worker process restarts the clock the telemetry carries, and the old smallest gap
    // would map every new sample far into the past. 1 s of slack ignores reordering, which a pipe never does.
    this.jumped = this.lastT !== null && tWorker < this.lastT - 1;
    if (this.jumped) this.reset();
    const gap = now - tWorker * 1000;
    this.gaps.push(gap);
    if (this.gaps.length > 90) this.gaps.shift();
    if (this.lastT !== null && tWorker > this.lastT) {
      this.intervals.push((tWorker - this.lastT) * 1000);
      if (this.intervals.length > 30) this.intervals.shift();
    }
    this.lastT = tWorker;
    this.offset = Math.min(...this.gaps);
    const late = this.gaps.map((g) => g - this.offset).sort((a, b) => a - b);
    const iv = [...this.intervals].sort((a, b) => a - b);
    const interval = iv.length ? iv[iv.length >> 1] : 33;
    this.delay = clamp(interval + late[Math.floor(late.length * 0.9)] + 4, 16, 250);
    return tWorker * 1000 + this.offset;
  }

  reset(): void { this.gaps.length = 0; this.intervals.length = 0; this.lastT = null; }
}
