"""What an episode's numbers say: motion, tracking, grasps, idle time, smoothness, and the problems
they
reveal. Pure numpy, so every rule is unit-tested and the same on every Mac.

Every threshold below was set from our own recordings, not guessed (measured 2026-10-04 on
phi_so101_cubes_cylinder_lang_v1, 30 fps single arm, and lehome top_long_merged, 20 fps bimanual):
  * an encoder step is 360/4095 = 0.088 deg, and a joint at rest reads exactly 0 deg/s
  * state speed p99 is 70-90 deg/s on one arm, up to 170 deg/s on the bimanual folding data
  * action - state reaches 10 deg (single) to 24 deg (bimanual) at p99, and almost all of it is lag:
    the follower trails the leader by ~100 ms, so error ~ speed x lag. Raw error therefore cannot
    tell a
    stuck follower from a fast move; the tracking rule compares state with the action one lag
    earlier.

Units are LeRobot's: degrees for the five arm joints, 0..100 for the gripper (low = closed).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from phi_studio import kinematics

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
VERSION = 3  # bump when a rule changes, so cached analyses are recomputed

IDLE_SPEED = 3.0  # deg/s, smoothed, fastest arm joint: below this the arm is still
IDLE_MIN_S = 0.5  # a still stretch shorter than this is a pause, not idle
IDLE_FLAG_S = 1.5  # idle at the start or end worth trimming
JUMP_DEG_S = 600.0  # one-frame change no STS3215 can make (no-load max ~270 deg/s at 12 V)
TRACK_DEG = 12.0  # lag-compensated follower error that means it is not following
TRACK_MIN_S = 0.25
OUTRUN_DEG_S = 60.0  # follower this fast while behind: it is speed-limited, not blocked
GRIP_SPAN_MIN = 8.0  # gripper must travel this much in an episode to count grasps
SQUEEZE = 8.0  # follower held more open than commanded, while closed: pushing on an object
SQUEEZE_MIN_S = 1.5
SHORT_S = 3.0
MAX_LAG_S = 0.5


# -- signal tools
# -----------------------------------------------------------------------------------
def savgol_coeffs(window: int, order: int, deriv: int, dt: float) -> np.ndarray:
    """Savitzky-Golay filter taps for the deriv-th derivative (least-squares polynomial fit)."""
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    a = np.vander(x, order + 1, increasing=True)
    coeffs = np.linalg.pinv(a)[deriv] * math.factorial(deriv) / dt**deriv
    return coeffs[::-1]


def smooth_deriv(x: np.ndarray, fps: float, deriv: int, window_s: float = 0.27) -> np.ndarray:
    """deriv-th time derivative along axis 0, Savitzky-Golay (order 3), edges padded with end
    values."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    window = max(5, int(round(window_s * fps)) | 1)
    if n < window:
        window = n if n % 2 else n - 1
    if window < 5:
        g = x
        for _ in range(deriv):
            g = np.gradient(g, 1.0 / fps, axis=0)
        return g
    taps = savgol_coeffs(window, 3, deriv, 1.0 / fps)
    half = window // 2
    pad = np.concatenate([np.repeat(x[:1], half, 0), x, np.repeat(x[-1:], half, 0)], 0)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(pad[:, j], taps, mode="valid")
    return out


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True, at least min_len long."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True) if e - s >= min_len]


def estimate_lag(lead: np.ndarray, follow: np.ndarray, max_lag: int) -> tuple[float, float]:
    """How many frames `follow` trails `lead` (both velocity signals), with parabolic sub-frame
    refinement,
    and the normalised correlation at that lag. (nan, 0) when either barely moves."""
    a = lead - lead.mean()
    b = follow - follow.mean()
    if np.sqrt((a * a).mean()) < 1.0 or np.sqrt((b * b).mean()) < 1.0:  # < 1 deg/s RMS: no signal
        return float("nan"), 0.0
    n = len(a)
    max_lag = min(max_lag, n // 3)
    cs = []
    for k in range(0, max_lag + 1):
        x, y = a[: n - k], b[k:]
        cs.append(float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    c = np.asarray(cs)
    k = int(c.argmax())
    lag = float(k)
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if den < 0:
            lag = k + 0.5 * (c[k - 1] - c[k + 1]) / den
    return lag, float(c[k])


def sparc(
    speed: np.ndarray, fps: float, fc: float = 10.0, amp_th: float = 0.05, pad: int = 4
) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015). Closer to 0 is
    smoother;
    typical reaching movements score -1.5 to -3. nan for a profile with no movement."""
    s = np.asarray(speed, dtype=np.float64)
    if s.max() < 1e-6 or len(s) < 4:
        return float("nan")
    nfft = int(2 ** (math.ceil(math.log2(len(s))) + pad))
    f = np.arange(nfft) * fps / nfft
    mag = np.abs(np.fft.fft(s, nfft))
    mag = mag / mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    above = np.flatnonzero(mag >= amp_th)
    if len(above) == 0:
        return float("nan")
    f, mag = f[: above[-1] + 1], mag[: above[-1] + 1]
    df = np.diff(f) / (f[-1] - f[0] if f[-1] > f[0] else 1.0)
    return float(-np.sum(np.sqrt(df**2 + np.diff(mag) ** 2)))


def gripper_events(g: np.ndarray, fps: float) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Grasp and release events from one gripper's state (0..100, low = closed), with hysteresis at
    30%
    and 70% of this episode's own travel. Returns (events, closed index ranges)."""
    lo5, hi95 = np.percentile(g, [5, 95])
    if hi95 - lo5 < GRIP_SPAN_MIN:
        return [], []
    lo, hi = lo5 + 0.3 * (hi95 - lo5), lo5 + 0.7 * (hi95 - lo5)
    closed = bool(g[0] < (lo + hi) / 2)
    events, ranges, start = [], [], 0 if closed else None
    for i, v in enumerate(g):
        if not closed and v < lo:
            closed, start = True, i
            events.append({"t": i / fps, "frame": i, "kind": "grasp"})
        elif closed and v > hi:
            closed = False
            events.append({"t": i / fps, "frame": i, "kind": "release"})
            if start is not None:
                ranges.append((start, i))
            start = None
    if closed and start is not None:
        ranges.append((start, len(g)))
    # A gripper that starts closed (at rest) has not grasped anything yet: drop that first range.
    if ranges and ranges[0][0] == 0:
        ranges = ranges[1:]
    return events, ranges


def _round_or_none(x: float, nd: int) -> float | None:
    return round(x, nd) if math.isfinite(x) else None


# -- one episode
# ------------------------------------------------------------------------------------
def analyse_episode(
    frames: dict[str, np.ndarray],
    fps: float,
    arms: list[dict[str, Any]],
    tasks: list[str] | None = None,
    videos: dict[str, Any] | None = None,
    series: bool = True,
) -> dict[str, Any]:
    """Metrics, events and flags for one episode. With series=True, also the per-frame derived
    signals
    the inspector plots (velocity, lag-compensated tracking error, tool position)."""
    state = frames.get("state")
    action = frames.get("action")
    if state is None:
        state = action
    if action is None:
        action = state
    assert state is not None and action is not None
    n, dims = state.shape
    dur = n / fps
    out_flags: list[dict[str, Any]] = []

    def flag(
        kind: str,
        severity: str,
        text: str,
        t0: float | None = None,
        t1: float | None = None,
        arm: str | None = None,
        joint: str | None = None,
        value: float | None = None,
    ) -> None:
        out_flags.append(
            {
                "kind": kind,
                "severity": severity,
                "text": text,
                "t0": None if t0 is None else round(t0, 3),
                "t1": None if t1 is None else round(t1, 3),
                "arm": arm or None,
                "joint": joint,
                "value": None if value is None else round(value, 3),
            }
        )

    vel = smooth_deriv(state, fps, 1)
    vel_act = smooth_deriv(action, fps, 1)
    acc = smooth_deriv(state, fps, 2)
    jerk = smooth_deriv(state, fps, 3)
    raw_step = np.abs(np.diff(state, axis=0)) * fps if n > 1 else np.zeros((0, dims))

    # Idle: every arm joint of every arm still (grippers excluded: an idle hand may still twitch).
    arm_cols = [
        i
        for a in arms
        for j, i in zip(a["joints"], a["index"], strict=True)
        if i is not None and j != "gripper"
    ]
    speed_max = np.abs(vel[:, arm_cols]).max(1) if arm_cols else np.zeros(n)
    idle_runs = runs(speed_max < IDLE_SPEED, int(IDLE_MIN_S * fps))
    idle_frames = sum(e - s for s, e in idle_runs)
    idle_start = (idle_runs[0][1] / fps) if idle_runs and idle_runs[0][0] == 0 else 0.0
    idle_end = ((n - idle_runs[-1][0]) / fps) if idle_runs and idle_runs[-1][1] == n else 0.0
    if idle_start >= IDLE_FLAG_S and idle_start < dur:
        flag(
            "idle_start",
            "info",
            f"Still for {idle_start:.1f} s at the start",
            0.0,
            idle_start,
            value=idle_start,
        )
    if idle_end >= IDLE_FLAG_S and idle_end < dur:
        flag(
            "idle_end",
            "info",
            f"Still for {idle_end:.1f} s at the end",
            dur - idle_end,
            dur,
            value=idle_end,
        )
    if idle_start >= dur - 1e-9:
        flag("no_motion", "error", "The arm never moves", 0.0, dur)
    if dur < SHORT_S:
        flag("short", "warn", f"Only {dur:.1f} s long", 0.0, dur, value=dur)

    per_joint: dict[str, dict[str, Any]] = {}
    per_arm: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    closed_segments: list[dict[str, Any]] = []
    resid = np.zeros_like(state)
    max_lag = int(MAX_LAG_S * fps)
    tcp_all: dict[str, np.ndarray] = {}

    for a in arms:
        aname = a["name"]
        for jname, col in zip(a["joints"], a["index"], strict=True):
            if col is None:
                continue
            key = f"{aname}_{jname}" if aname else jname
            lag, corr = estimate_lag(vel_act[:, col], vel[:, col], max_lag)
            if math.isfinite(lag):
                # state(t) against action(t - lag), by linear interpolation of the action.
                t = np.arange(n, dtype=np.float64)
                shifted = np.interp(t - lag, t, action[:, col])
                resid[:, col] = state[:, col] - shifted
            else:
                resid[:, col] = state[:, col] - action[:, col]
            rj = resid[:, col]
            per_joint[key] = {
                "range": round(float(np.ptp(state[:, col])), 2),
                "speed_peak": round(float(np.abs(vel[:, col]).max()), 1),
                "lag_ms": None if not math.isfinite(lag) else round(lag / fps * 1000, 1),
                "lag_corr": round(corr, 3),
                "err_rms": round(float(np.sqrt(np.mean((action[:, col] - state[:, col]) ** 2))), 2),
                "track_rms": round(float(np.sqrt(np.mean(rj**2))), 2),
                "track_max": round(float(np.abs(rj).max()), 2),
                "jerk_rms": round(float(np.sqrt(np.mean(jerk[:, col] ** 2))), 0),
            }
            # A frozen sensor: the follower never changes while the leader moves it a lot.
            if np.ptp(state[:, col]) == 0 and np.ptp(action[:, col]) > 5:
                flag(
                    "frozen",
                    "error",
                    f"{jname.replace('_', ' ')} never changes while commanded "
                    f"{np.ptp(action[:, col]):.0f}°",
                    0.0,
                    dur,
                    aname,
                    jname,
                    float(np.ptp(action[:, col])),
                )
            # One-frame jumps no servo can make: a read glitch or a wrap.
            if len(raw_step):
                jumps = np.flatnonzero(raw_step[:, col] > JUMP_DEG_S)
                if len(jumps):
                    i = int(jumps[0])
                    unit = "%" if jname == "gripper" else "°"
                    flag(
                        "jump",
                        "error",
                        f"{jname.replace('_', ' ')} jumps "
                        f"{raw_step[i, col] / fps:.0f}{unit} in one "
                        f"frame{'' if len(jumps) == 1 else f' ({len(jumps)}×)'}",
                        i / fps,
                        (i + 1) / fps,
                        aname,
                        jname,
                        float(raw_step[i, col] / fps),
                    )
            if jname != "gripper":
                bad = runs(np.abs(rj) > TRACK_DEG, max(1, int(TRACK_MIN_S * fps)))
                if bad:
                    s0, e0 = max(bad, key=lambda r: np.abs(rj[r[0] : r[1]]).max())
                    peak = float(np.abs(rj[s0:e0]).max())
                    total = sum(e - s for s, e in bad) / fps
                    # WHY two causes: a follower at speed fell behind a leader
                    # moved faster than the servo
                    # can turn (the recorded action is one the arm could not
                    # do); a slow follower is held
                    # back by something (a collision, an object, an overloaded joint).
                    follower_speed = float(np.median(np.abs(vel[s0:e0, col])))
                    what = jname.replace("_", " ")
                    text = (
                        f"Leader outran the follower by {peak:.0f}° on {what} ({total:.1f} s)"
                        if follower_speed > OUTRUN_DEG_S
                        else f"Follower held {peak:.0f}° off its leader on {what} ({total:.1f} s)"
                    )
                    flag(
                        "tracking",
                        "error" if peak > 2 * TRACK_DEG else "warn",
                        text,
                        s0 / fps,
                        e0 / fps,
                        aname,
                        jname,
                        peak,
                    )

        # Gripper: grasps, releases, squeeze.
        gcol = a["index"][5] if len(a["index"]) > 5 else None
        grasps = 0
        closed_s = 0.0
        if gcol is not None:
            ev, closed = gripper_events(state[:, gcol], fps)
            for e in ev:
                events.append({**e, "arm": aname or None})
            grasps = sum(1 for e in ev if e["kind"] == "grasp")
            for s0, e0 in closed:
                closed_s += (e0 - s0) / fps
                sq = state[s0:e0, gcol] - action[s0:e0, gcol]
                closed_segments.append(
                    {
                        "t0": round(s0 / fps, 3),
                        "t1": round(e0 / fps, 3),
                        "arm": aname or None,
                        "squeeze": round(float(np.median(sq)), 2),
                    }
                )
                hard = runs(sq > SQUEEZE, int(SQUEEZE_MIN_S * fps))
                if hard:
                    hs, he = max(hard, key=lambda r: r[1] - r[0])
                    flag(
                        "squeeze",
                        "warn",
                        f"Gripper pushing {np.median(sq[hs:he]):.0f}% past the object for "
                        f"{(he - hs) / fps:.1f} s",
                        (s0 + hs) / fps,
                        (s0 + he) / fps,
                        aname,
                        "gripper",
                        float(np.median(sq[hs:he])),
                    )

        # Tool path from the model's kinematics.
        tool = None
        if a["so101"]:
            q = state[:, a["index"]]
            tool = kinematics.tcp(q)
            tcp_all[aname] = tool
            sp = np.linalg.norm(smooth_deriv(tool, fps, 1), axis=1)
            moving = np.linalg.norm(np.diff(tool, axis=0), axis=1)
            per_arm[aname] = {
                "path_m": round(float(moving.sum()), 4),
                "tcp_speed_peak": round(float(sp.max()), 4),
                "tcp_speed_mean": round(float(sp.mean()), 4),
                "sparc": _round_or_none(sparc(sp, fps), 3),
                "grasps": grasps,
                "closed_s": round(closed_s, 2),
                "reach_m": round(float(np.linalg.norm(tool[:, :2], axis=1).max()), 4),
            }
        else:
            per_arm[aname] = {"grasps": grasps, "closed_s": round(closed_s, 2)}

    task_text = " ".join(tasks or []).lower()
    if (
        any(w in task_text for w in ("pick", "grasp", "place", "fold"))
        and arms
        and all(per_arm.get(a["name"], {}).get("grasps", 0) == 0 for a in arms)
    ):
        flag("no_grasp", "warn", "The gripper never closes on anything")

    for cam, v in (videos or {}).items():
        span = float(v["to"]) - float(v["from"])
        if abs(span - dur) > 1.5 / fps:
            flag(
                "video",
                "warn",
                f"{cam.removeprefix('observation.images.')} video is {span:.2f} s, data is "
                f"{dur:.2f} s",
                None,
                None,
                value=span - dur,
            )

    lags = [
        j["lag_ms"] for j in per_joint.values() if j["lag_ms"] is not None and j["lag_corr"] > 0.6
    ]
    metrics = {
        "duration_s": round(dur, 3),
        "frames": n,
        "idle_frac": round(idle_frames / n, 4) if n else 0.0,
        "idle_start_s": round(idle_start, 2),
        "idle_end_s": round(idle_end, 2),
        "lag_ms": round(float(np.median(lags)), 1) if lags else None,
        "track_rms": round(float(np.sqrt(np.mean(resid[:, arm_cols] ** 2))), 3)
        if arm_cols
        else None,
        "speed_peak": round(float(speed_max.max()), 1),
        "path_m": round(sum(v.get("path_m", 0.0) for v in per_arm.values()), 4),
        "grasps": sum(v.get("grasps", 0) for v in per_arm.values()),
        "sparc": (
            round(float(np.mean(finite_sparc)), 3)
            if (
                finite_sparc := [v["sparc"] for v in per_arm.values() if v.get("sparc") is not None]
            )
            else None
        ),
    }
    sev = {"error": 3, "warn": 2, "info": 1}
    health = max((sev[f["severity"]] for f in out_flags), default=0)
    out: dict[str, Any] = {
        "version": VERSION,
        "metrics": metrics,
        "joints": per_joint,
        "arms": per_arm,
        "events": sorted(events, key=lambda e: e["t"]),
        "closed": closed_segments,
        "idle": [[round(s / fps, 3), round(e / fps, 3)] for s, e in idle_runs],
        "flags": out_flags,
        "health": ["ok", "info", "warn", "error"][health],
    }
    if series:
        r2 = lambda x: np.round(x, 2).tolist()  # noqa: E731
        out["series"] = {
            "vel": r2(vel),
            "acc": r2(acc),
            "track": r2(resid),
            "err": r2(action - state),
            "tcp": {k: np.round(v, 4).tolist() for k, v in tcp_all.items()},
        }
    return out


# -- a whole dataset
# --------------------------------------------------------------------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not mad or not math.isfinite(mad):
        return np.zeros_like(x)
    return (x - med) / mad


def analyse_dataset(ds: Any, progress: Any = None) -> dict[str, Any]:
    """Per-episode metrics and flags, dataset-wide distributions, outliers and health, for one
    Dataset."""
    eps = ds.episodes()
    arms = ds.arms()
    fps = ds.fps
    health: list[dict[str, Any]] = []
    if not eps:
        health.append(
            {
                "severity": "error",
                "kind": "empty",
                "text": "No episodes: the recording stopped before its first episode was saved",
            }
        )
        return {
            "version": VERSION,
            "episodes": [],
            "health": health,
            "distributions": {},
            "workspace": {},
            "rest": {},
            "lengths": [],
            "tasks": {},
            "flag_counts": {},
        }

    rows = []
    for k, e in enumerate(eps):
        fr = ds.frames(e["index"])
        a = analyse_episode(fr, fps, arms, e["tasks"], e["videos"], series=False)
        rows.append(
            {
                "index": e["index"],
                "length": e["length"],
                "tasks": e["tasks"],
                **a,
                "grasps_t": [round(ev["t"], 2) for ev in a["events"] if ev["kind"] == "grasp"],
            }
        )
        if progress:
            progress(k + 1, len(eps))

    # Outliers among this dataset's own episodes (robust z over median and MAD).
    def col(key: str) -> np.ndarray:
        return np.array(
            [r["metrics"].get(key) if r["metrics"].get(key) is not None else np.nan for r in rows],
            dtype=np.float64,
        )

    # (metric, label, unit, sides, smallest difference from the median worth a flag). WHY
    # one-sided for
    # idle and tracking: less of either is better, never a problem. WHY a floor: a tight
    # metric has a tiny
    # MAD, and 0.4 degrees more tracking error is not worth anyone's attention.
    for key, label, unit, sides, floor in (
        ("duration_s", "length", "s", 2, 3.0),
        ("path_m", "tool path", "m", 2, 0.3),
        ("idle_frac", "idle time", "", 1, 0.15),
        ("track_rms", "tracking error", "°", 1, 0.5),
    ):
        x = col(key)
        z = robust_z(x)
        med = float(np.nanmedian(x))
        for r, zi, xi in zip(rows, z, x, strict=True):
            if not math.isfinite(zi) or abs(xi - med) < floor:
                continue
            if (zi > 3.5) or (sides == 2 and zi < -3.5):
                v = r["metrics"][key]
                shown = f"{v * 100:.0f}%" if key == "idle_frac" else f"{v:.2f} {unit}".strip()
                mshown = f"{med * 100:.0f}%" if key == "idle_frac" else f"{med:.2f} {unit}".strip()
                r["flags"].append(
                    {
                        "kind": "outlier",
                        "severity": "warn",
                        "t0": None,
                        "t1": None,
                        "arm": None,
                        "joint": None,
                        "value": round(float(zi), 2),
                        "text": f"Unusual {label}: {shown} vs median {mshown}",
                    }
                )
                if r["health"] in ("ok", "info"):
                    r["health"] = "warn"

    flag_counts: dict[str, int] = {}
    for r in rows:
        for f in r["flags"]:
            flag_counts[f["kind"]] = flag_counts.get(f["kind"], 0) + 1

    # Dataset-wide distributions from every frame.
    allf = ds.all_frames()
    dist: dict[str, Any] = {}
    lim = kinematics.limits_deg()
    for a in arms:
        for jname, colx in zip(a["joints"], a["index"], strict=True):
            if colx is None:
                continue
            key = f"{a['name']}_{jname}" if a["name"] else jname
            st = allf["state"][:, colx] if "state" in allf else allf["action"][:, colx]
            ac = allf["action"][:, colx] if "action" in allf else st
            lo = min(float(st.min()), float(ac.min()), lim.get(jname, (0, 0))[0])
            hi = max(float(st.max()), float(ac.max()), lim.get(jname, (0, 0))[1])
            edges = np.linspace(lo, hi, 49)
            dist[key] = {
                "edges": np.round(edges, 2).tolist(),
                "state": np.histogram(st, edges)[0].tolist(),
                "action": np.histogram(ac, edges)[0].tolist(),
                "q": np.round(np.percentile(st, [1, 50, 99]), 2).tolist(),
                "min": round(float(st.min()), 2),
                "max": round(float(st.max()), 2),
                "model": list(lim.get(jname, (lo, hi))),
            }

    # Workspace: tool positions over the whole dataset, at most ~6000 points per arm.
    work: dict[str, Any] = {}
    st_all = allf["state"] if "state" in allf else allf["action"]
    for a in arms:
        if not a["so101"]:
            continue
        tool = kinematics.tcp(st_all[:, a["index"]])
        step = max(1, len(tool) // 6000)
        sub = tool[::step]
        work[a["name"]] = {
            "points": np.round(sub, 4).tolist(),
            "episode": allf["episode_index"][::step].tolist(),
            "min": np.round(tool.min(0), 4).tolist(),
            "max": np.round(tool.max(0), 4).tolist(),
        }

    # The pose each arm typically starts from (median of every episode's first frame), to draw
    # the arm
    # among its own workspace.
    rest: dict[str, list[float]] = {}
    first = allf["frame_index"] == 0
    for a in arms:
        if a["so101"] and first.any():
            rest[a["name"]] = np.round(np.median(st_all[first][:, a["index"]], axis=0), 2).tolist()

    tasks: dict[str, int] = {}
    for e in eps:
        for t in e["tasks"]:
            tasks[t] = tasks.get(t, 0) + 1

    total = sum(e["length"] for e in eps)
    if int(ds.info.get("total_frames", total)) != total:
        health.append(
            {
                "severity": "warn",
                "kind": "frames",
                "text": f"info.json says {ds.info.get('total_frames')} frames, "
                f"the episodes hold {total}",
            }
        )
    missing = set()
    for e in eps:
        for cam, v in e["videos"].items():
            p = ds.video_path(cam, v["chunk"], v["file"])
            if not p.is_file():
                missing.add(str(p.relative_to(ds.root)))
    if missing:
        health.append(
            {
                "severity": "error",
                "kind": "video_missing",
                "text": f"{len(missing)} video file(s) missing, e.g. {sorted(missing)[0]}",
            }
        )
    bad = sum(1 for r in rows if r["health"] == "error")
    if bad:
        health.append(
            {"severity": "error", "kind": "episodes", "text": f"{bad} episode(s) with errors"}
        )

    return {
        "version": VERSION,
        "episodes": [
            {
                k: r[k]
                for k in (
                    "index",
                    "length",
                    "tasks",
                    "metrics",
                    "arms",
                    "flags",
                    "health",
                    "grasps_t",
                )
            }
            for r in rows
        ],
        "lengths": [round(e["length"] / fps, 3) for e in eps],
        "distributions": dist,
        "workspace": work,
        "rest": rest,
        "tasks": tasks,
        "flag_counts": flag_counts,
        "health": health,
    }
