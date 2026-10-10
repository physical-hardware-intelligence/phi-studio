"""Evals: run policies on the rig under a protocol fixed in advance, score every trial, and say
what the numbers can and cannot show.

The design follows the labs that publish their protocols (Φ wiki,
concepts/robot-policy-evaluation.md):

  * A task card says what success is, the milestones that earn partial credit, the failure tags,
    and the start conditions, each tagged with what it tests. Starting an eval locks its card, so
    the rubric is written before any result is seen (Kress-Gazit et al., arXiv 2409.09491). A
    changed card is a new card.
  * Trials run in bundles: one condition, every policy once, in a random order (TRI LBM, arXiv
    2507.05331; DeepMind, 2503.20020). Drift over the day and a hard condition then hit every policy
    alike. The order comes from a seed kept with the eval, so anyone can check it.
  * Blind: windows see "Policy A", not the checkpoint, until the eval ends (TRI, 1X, RoboArena).
  * Each trial gets the highest milestone it reached and tags for what went wrong. A trial spoiled
    by the rig (a servo fault, a camera that dropped out, a lost heartbeat) is voided rather than
    counted against the policy, and its slot runs again. Studio suggests the void from the servo
    and camera readings; the person decides.
  * The server owns the record and saves it after every change, so a reload or a crash loses
    nothing, and an eval left open is picked up on the next start.

Records from the first version (one policy, success or failure per episode) are still listed.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from phi_studio import evalstats as es
from phi_studio.policy import is_finite_number

wilson = es.wilson  # the first version's name; the Train and Evaluate pages import it from here

SCHEMA = 2
# Sherry Chen's SO-101 ACT rubric, as the club's phi.utils.eval_rollouts scores it, so Studio's
# numbers stay comparable with the club's and with a published result on the same hardware.
DEFAULT_RUBRIC = (("Reached the object", 0.2), ("Grasped it", 0.4),
                  ("Carried it to the target", 0.7), ("Released it", 0.8),
                  ("Object in place", 1.0))  # fmt: skip
# From the club's rollout notes (2026-08: hovering, tapping, toppled cylinder, overload) and Dyna's
# production failure modes (missed grab, drop, imprecise placement).
DEFAULT_FAILURES = ("Missed the grasp", "Slipped or dropped", "Pushed or knocked it over",
                    "Wrong object or place", "Hesitated or oscillated", "Hit something",
                    "Ran out of time", "Stopped for safety")  # fmt: skip
# What a condition tests. Camera pose and table texture moved real success the most in published
# factor studies; a held-out position is interpolation, not a new object or task.
AXES = ("train", "held-out position", "new object", "distractors", "lighting", "camera moved",
        "table or background", "new instruction", "other")  # fmt: skip
OUTCOMES = ("success", "failure")  # first-version records
MAX_NOTE = 2000
MAX_TRIALS = 2000
MAX_POLICIES = 6
MAX_REPS = 50
MAX_CONDITIONS = 40
ALIASES = "ABCDEF"
CARD_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}\Z")
COND_ID = re.compile(r"^c[1-9]\d{0,2}\Z")
# Ends of a policy run that are not the policy's doing (worker.py STOP_REASONS, session.py).
RIG_ENDS = {"fault": "The rig faulted", "disconnected": "An arm disconnected",
            "heartbeat": "The window lost its link to Studio", "window closed": "The window closed",
            "control moved": "Control moved to another window"}  # fmt: skip
# [JUDGEMENT, not measured] Below 70% of the control rate, or a 100 ms gap (three ticks at 30 Hz),
# the arm did not move on the timing the policy was trained on.
LOOP_SLOW = 0.7
LOOP_STALL_MS = 100.0
JUMP_WARN_DEG = 20.0  # [JUDGEMENT] a new chunk starting this far from the last one jolts the arm
CLIP_WARN = 0.10  # [JUDGEMENT] the step limit cutting this share of ticks changes the motion


class EvalError(ValueError):
    pass


def _text(v: Any, limit: int, missing: str, *, empty_ok: bool = False) -> str:
    if not isinstance(v, str) or (not v.strip() and not empty_ok):
        raise EvalError(missing)
    v = v.strip()
    if len(v) > limit:
        raise EvalError(f"{missing.rstrip('.')} (at most {limit} characters).")
    return v


def _num(v: Any, lo: float, hi: float, what: str) -> float:
    if not is_finite_number(v) or not lo <= v <= hi:
        raise EvalError(f"{what} must be between {lo:g} and {hi:g}.")
    return float(v)


def _int(v: Any, lo: int, hi: int, what: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise EvalError(f"{what} must be a whole number from {lo} to {hi}.")
    return v


# -- task cards ------------------------------------------------------------------------------------
def check_card(raw: Any) -> dict[str, Any]:
    """A task card from a window, checked and normalised. New conditions get the next free id;
    existing ids are kept, so a reference image stays with its condition across edits."""
    if not isinstance(raw, dict):
        raise EvalError("Studio got no task card.")
    card: dict[str, Any] = {
        "name": _text(raw.get("name"), 80, "Name the task card."),
        "task": _text(raw.get("task"), 300, "Write the task as the policy is told it."),
        "success": _text(raw.get("success"), 300, "Say what counts as success, in one sentence."),
        "limit_s": _num(raw.get("limit_s"), 1, 600, "The time limit per trial"),
    }
    rubric = raw.get("rubric")
    if not isinstance(rubric, list) or not 1 <= len(rubric) <= 8:
        raise EvalError("A rubric has 1 to 8 milestones.")
    out, last = [], 0.0
    for i, m in enumerate(rubric, 1):
        if not isinstance(m, dict):
            raise EvalError(f"Milestone {i} is not a milestone.")
        label = _text(m.get("label"), 80, f"Name milestone {i}.")
        pts = _num(m.get("points"), 0.01, 1.0, f"Milestone {i}'s credit")
        if pts <= last:
            raise EvalError("Each milestone must be worth more than the one before it.")
        out.append({"label": label, "points": round(pts, 4)})
        last = pts
    if out[-1]["points"] != 1.0:
        raise EvalError("The last milestone is success, worth 1.")
    card["rubric"] = out
    fails = raw.get("failures", [])
    if not isinstance(fails, list) or len(fails) > 16:
        raise EvalError("At most 16 failure tags.")
    tags = [_text(f, 60, "A failure tag cannot be empty.") for f in fails]
    if len(set(tags)) != len(tags):
        raise EvalError("Each failure tag must be different.")
    card["failures"] = tags
    conds = raw.get("conditions")
    if not isinstance(conds, list) or not 1 <= len(conds) <= MAX_CONDITIONS:
        raise EvalError(f"A card has 1 to {MAX_CONDITIONS} start conditions.")
    used = {c.get("id") for c in conds if isinstance(c, dict) and isinstance(c.get("id"), str)
            and COND_ID.match(c["id"])}  # fmt: skip
    nxt, labels = 1, set()
    cs: list[dict[str, Any]] = []
    for i, c in enumerate(conds, 1):
        if not isinstance(c, dict):
            raise EvalError(f"Condition {i} is not a condition.")
        label = _text(c.get("label"), 80, f"Describe condition {i} (where things start).")
        if label in labels:
            raise EvalError(f"Two conditions are both called {label!r}.")
        labels.add(label)
        axis = c.get("axis", "train")
        if axis not in AXES:
            raise EvalError(f"Condition {i}: pick what it tests from the list.")
        cid = c.get("id")
        if not (isinstance(cid, str) and COND_ID.match(cid)) or cid in {x["id"] for x in cs}:
            while f"c{nxt}" in used:
                nxt += 1
            cid = f"c{nxt}"
            used.add(cid)
        raw_refs = c.get("refs")
        refs: dict[Any, Any] = raw_refs if isinstance(raw_refs, dict) else {}
        cs.append({"id": cid, "label": label, "axis": axis, "control": c.get("control") is True,
                   "refs": {k: v for k, v in refs.items() if isinstance(k, str)
                            and isinstance(v, str)}})  # fmt: skip
    card["conditions"] = cs
    return card


def blank_card() -> dict[str, Any]:
    """A new card's starting values: the club's rubric and failure tags."""
    return {"name": "", "task": "", "success": "", "limit_s": 30.0,
            "rubric": [{"label": lb, "points": p} for lb, p in DEFAULT_RUBRIC],
            "failures": list(DEFAULT_FAILURES), "conditions": []}  # fmt: skip


# -- the schedule ----------------------------------------------------------------------------------
def make_schedule(conditions: list[dict[str, Any]], aliases: list[str], reps: int, grouped: bool,
                  rng: random.Random) -> list[dict[str, Any]]:  # fmt: skip
    """Bundles in run order, each with its own random policy order. Control conditions (ones the
    policies trained on) open the first round: if a policy fails those, suspect the rig first.
    `grouped` keeps each condition's bundles together, for scenes slow to reset."""
    controls = [c["id"] for c in conditions if c.get("control")]
    others = [c["id"] for c in conditions if not c.get("control")]
    order: list[str] = []
    if grouped:
        rest = others[:]
        rng.shuffle(rest)
        for cid in controls + rest:
            order += [cid] * reps
    else:
        for r in range(reps):
            ctl, rest = controls[:], others[:]
            rng.shuffle(ctl)
            rng.shuffle(rest)
            rnd = ctl + rest
            if r:
                rng.shuffle(rnd)
            order += rnd
    return [{"b": i + 1, "c": cid, "order": rng.sample(aliases, len(aliases))}
            for i, cid in enumerate(order)]  # fmt: skip


def _valid(t: dict[str, Any]) -> bool:
    return t.get("void") is None


def next_slot(rec: dict[str, Any]) -> dict[str, Any] | None:
    """The next (bundle, policy) with no valid trial and not skipped, in schedule order."""
    skipped = {(s["b"], s["alias"]) for s in rec.get("skipped", [])}
    done = {(t["b"], t["alias"]) for t in rec["trials"] if _valid(t)}
    voided = {(t["b"], t["alias"]) for t in rec["trials"] if not _valid(t)}
    for bun in rec.get("schedule", []):
        for alias in bun["order"]:
            key = (bun["b"], alias)
            if key not in done and key not in skipped:
                return {"b": bun["b"], "c": bun["c"], "alias": alias, "repeat": key in voided}
    return None


def policy_of(rec: dict[str, Any], alias: str) -> dict[str, Any]:
    return next(p for p in rec["policies"] if p["alias"] == alias)


# -- what the rig did during a trial ---------------------------------------------------------------
class RunHealth:
    """Servo, loop and camera readings over one policy run, gathered from the worker's telemetry
    by the server. Readings after the run ended are ignored; readings in its first second are
    ignored for the loop rate, which is still settling from the start."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.active = False
        # WHY a lock: the server feeds telemetry from its reader thread and reads the summary from
        # the event loop; a dict read while it grows raises.
        self.lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.max_load: dict[str, float] = {}
        self.max_temp: dict[str, float] = {}
        self.min_volt: dict[str, float] = {}
        self.faults: dict[str, dict[str, list[str]]] = {}
        self.min_hz: float | None = None
        self.max_p99: float | None = None
        self.cameras_lost: list[str] = []

    def feed(self, telemetry: dict[str, Any]) -> None:
        with self.lock:
            self._feed(telemetry)

    def _feed(self, telemetry: dict[str, Any]) -> None:
        p = telemetry.get("policy")
        if not isinstance(p, dict) or not isinstance(p.get("run_id"), str):
            self.active = False
            return
        if p["run_id"] != self.run_id:
            self.run_id = p["run_id"]
            self._reset()
        self.active = bool(p.get("running"))
        if not self.active:
            return
        for name, a in (telemetry.get("arms") or {}).items():
            if not isinstance(a, dict) or a.get("role") != "follower":
                continue
            for j, h in (a.get("health") or {}).items():
                if not isinstance(h, dict):
                    continue
                for store, key, pick in ((self.max_load, "load", max), (self.max_temp, "temp", max),
                                         (self.min_volt, "volt", min)):  # fmt: skip
                    raw = h.get(key)
                    if is_finite_number(raw):
                        val = abs(float(raw)) if key == "load" else float(raw)
                        store[name] = pick(store.get(name, val), val)
                fl = [str(f) for f in h.get("faults") or []]
                if fl:
                    seen = self.faults.setdefault(name, {}).setdefault(str(j), [])
                    seen += [f for f in fl if f not in seen]
        loop = telemetry.get("loop") or {}
        if is_finite_number(p.get("episode_s")) and p["episode_s"] >= 1.0:
            hz, p99 = loop.get("hz"), loop.get("p99_ms")
            if is_finite_number(hz):
                f = float(hz)
                self.min_hz = f if self.min_hz is None else min(self.min_hz, f)
            if is_finite_number(p99):
                f = float(p99)
                self.max_p99 = f if self.max_p99 is None else max(self.max_p99, f)

    def camera(self, key: str, online: Any) -> None:
        with self.lock:
            if self.active and online is False and key not in self.cameras_lost:
                self.cameras_lost.append(key)

    def summary(self, run_id: str) -> dict[str, Any]:
        with self.lock:
            return self._summary(run_id)

    def _summary(self, run_id: str) -> dict[str, Any]:
        if run_id != self.run_id:
            return {}
        def r(d: dict[str, float]) -> dict[str, float]:
            return {k: round(v, 2) for k, v in d.items()}

        return {"max_load_pct": r(self.max_load), "max_temp_c": r(self.max_temp),
                "min_volt": r(self.min_volt),
                "faults": {a: {j: list(f) for j, f in js.items()} for a, js in self.faults.items()},
                "min_hz": self.min_hz,
                "max_p99_ms": self.max_p99, "cameras_lost": list(self.cameras_lost)}  # fmt: skip


def suspect(health: dict[str, Any], metrics: dict[str, Any], hz: float) -> dict[str, list[str]]:
    """Why a trial may be the rig's fault ("void": suggest voiding it) or may not show the policy
    as trained ("warn": kept, but said)."""
    void: list[str] = []
    for arm, joints in (health.get("faults") or {}).items():
        for j, fl in joints.items():
            void.append(f"Servo fault on {arm} {j}: {', '.join(fl)}")
    for key in health.get("cameras_lost") or []:
        void.append(f"Camera {key} dropped out during the trial")
    if is_finite_number(health.get("min_hz")) and health["min_hz"] < LOOP_SLOW * hz:
        void.append(f"The control loop slowed to {health['min_hz']:.0f} Hz (target {hz:.0f})")
    if is_finite_number(health.get("max_p99_ms")) and health["max_p99_ms"] > LOOP_STALL_MS:
        void.append(f"The control loop stalled ({health['max_p99_ms']:.0f} ms ticks)")
    end = metrics.get("ended")
    if end in RIG_ENDS:
        void.append(f"{RIG_ENDS[end]} during the trial")
    warn: list[str] = []
    chunk, cmax = metrics.get("chunk"), metrics.get("chunk_ms_max")
    if is_finite_number(chunk) and is_finite_number(cmax) and float(cmax) > float(chunk) * 1e3 / hz:
        warn.append(f"Inference took {cmax:.0f} ms, longer than the {chunk:.0f}-step chunk it "
                    "plays")
    steps, clipped = metrics.get("step"), metrics.get("clipped")
    finite = is_finite_number(steps) and is_finite_number(clipped)
    if finite and steps and clipped / steps > CLIP_WARN:
        warn.append(f"The step limit cut {clipped / steps:.0%} of the policy's ticks")
    jump = metrics.get("boundary_jump_deg")
    if is_finite_number(jump) and float(jump) > JUMP_WARN_DEG:
        warn.append(f"A new action chunk started {jump:.0f}° from where the last one ended")
    return {"void": void, "warn": warn}


_METRICS = ("chunk", "chunk_ms_p50", "chunk_ms_max", "step", "clipped", "max_jump_deg",
            "boundary_jump_deg")  # fmt: skip


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: run[k] for k in _METRICS if is_finite_number(run.get(k))}
    if isinstance(run.get("ended"), str):
        out["ended"] = run["ended"]
    if isinstance(run.get("max_jump_joint"), str):
        out["max_jump_joint"] = run["max_jump_joint"]
    return out


# -- the numbers -----------------------------------------------------------------------------------
def _rate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    n, k = len(trials), sum(1 for t in trials if t["success"])
    lo, hi = es.wilson(k, n)
    mean, mlo, mhi = es.mean_ci([float(t["score"]) for t in trials]) if trials else (None, None,
                                                                                         None)
    return {"n": n, "k": k, "rate": k / n if n else None, "ci95": [lo, hi],
            "progress": None if mean is None or math.isnan(mean) else round(mean, 4),
            "progress_ci95": [mlo, mhi]}  # fmt: skip


def summarize(rec: dict[str, Any]) -> dict[str, Any]:
    """Every number the Results panel shows. Never pools across axes without showing them apart."""
    card = rec["card"]
    conds = {c["id"]: c for c in card["conditions"]}
    valid = [t for t in rec["trials"] if _valid(t)]
    aliases = [p["alias"] for p in rec["policies"]]
    per: dict[str, Any] = {}
    for a in aliases:
        ts = [t for t in valid if t["alias"] == a]
        times = sorted(t["duration_s"] for t in ts if t["success"]
                       and is_finite_number(t.get("duration_s")))  # fmt: skip
        axes = {}
        for ax in AXES:
            sub = [t for t in ts if conds.get(t["c"], {}).get("axis") == ax]
            if sub:
                axes[ax] = _rate(sub)
        per[a] = {**_rate(ts), "median_success_s": times[len(times) // 2] if times else None,
                  "failures": dict(Counter(f for t in ts for f in t.get("failures", []))),
                  "voided": sum(1 for t in rec["trials"] if t["alias"] == a and not _valid(t)),
                  "by_axis": axes,
                  "by_condition": {c: _rate([t for t in ts if t["c"] == c]) for c in conds
                                   if any(t["c"] == c for t in ts)}}  # fmt: skip
    paired = rec.get("paired", True) and bool(rec.get("schedule"))
    planned = len(rec.get("schedule", []))
    skipped = {(s["b"], s["alias"]) for s in rec.get("skipped", [])}
    complete = next_slot(rec) is None or bool(rec.get("ended_at"))
    k = len(aliases)
    n_cmp = k * (k - 1) // 2
    alpha = float(rec.get("alpha", 0.05))
    pl = es.plan(alpha / max(1, n_cmp), max(1, min(es.MAX_PAIRS, planned))) if paired else None
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = aliases[i], aliases[j]
            if not paired or pl is None:
                ra, rb = per[a], per[b]
                pairs.append({"a": a, "b": b, "paired": False,
                              "p_a_better": es.p_better_unpaired(ra["k"], ra["n"], rb["k"], rb["n"])
                              if ra["n"] and rb["n"] else None, "decision": None})  # fmt: skip
                continue
            res = {(t["b"], t["alias"]): bool(t["success"]) for t in valid}
            seq = [(res[(bun["b"], a)], res[(bun["b"], b)]) for bun in rec["schedule"]
                   if (bun["b"], a) in res and (bun["b"], b) in res]  # fmt: skip
            d = es.decide(seq, pl, complete)
            wa = sum(1 for x, y in seq if x and not y)
            wb = sum(1 for x, y in seq if y and not x)
            more = None
            if d["state"] == "continue" and seq:
                theta = (wa + 1) / (wa + wb + 2)  # shrunk toward 1/2, never 0 or 1
                need = es.pairs_needed(theta, alpha / max(1, n_cmp))
                disc = (wa + wb) / len(seq)
                if need is not None and disc > 0:
                    more = max(0, math.ceil((need - wa - wb) / disc))
            pairs.append({"a": a, "b": b, "paired": True, "bundles": len(seq), "wins_a": wa,
                          "wins_b": wb, "p_a_better": es.p_better_paired(wa, wb),
                          "decision": d, "bundles_more_about": more})  # fmt: skip
    order = sorted(aliases, key=lambda a: (-(per[a]["rate"] or 0.0), a))
    diff = [(p["a"], p["b"]) for p in pairs if p.get("decision") and
            p["decision"]["state"] in ("a", "b")]  # fmt: skip
    # Letters only where pairwise tests ran: on unpaired data every policy would share "a",
    # which reads as a tested tie when no test was made.
    lettered = es.letters(order, diff) if k >= 3 and paired else None
    return {"policies": per, "pairs": pairs, "letters": lettered,
            "order": order, "paired": paired, "alpha": alpha, "comparisons": n_cmp,
            "planned_bundles": planned, "planned_trials": planned * k,
            "valid_trials": len(valid), "voided": len(rec["trials"]) - len(valid),
            "skipped": len(skipped), "complete": complete,
            "detectable_gap": es.detectable_gap(planned) if planned else None,
            "plan": {"nominal": pl.nominal, "achieved": pl.achieved} if pl else None}  # fmt: skip


def public(rec: dict[str, Any]) -> dict[str, Any]:
    """The record as windows see it: while a blind eval runs, policies are letters only."""
    out = dict(rec)
    if rec.get("blind") and not rec.get("ended_at"):
        out["policies"] = [{"alias": p["alias"]} for p in rec["policies"]]
        # WHY: the stamps name each checkpoint; they come back when the eval ends
        out["stamps"] = {k: v for k, v in (rec.get("stamps") or {}).items() if k != "policies"}
    return out


# -- the store -------------------------------------------------------------------------------------
def _v1_ok(rec: Any) -> bool:
    """A record the first version wrote. Anything else on disk is skipped, so one bad file cannot
    stop Studio from starting or listing."""
    return (
        isinstance(rec, dict) and rec.get("schema") is None
        and isinstance(rec.get("id"), str) and isinstance(rec.get("policy"), str)
        and isinstance(rec.get("task"), str) and is_finite_number(rec.get("started_at"))
        and isinstance(rec.get("episodes"), list)
        and all(isinstance(e, dict) and e.get("outcome") in OUTCOMES for e in rec["episodes"])
    )  # fmt: skip


def _v2_ok(rec: Any) -> bool:
    try:
        return (isinstance(rec, dict) and rec.get("schema") == SCHEMA
                and isinstance(rec.get("id"), str) and isinstance(rec.get("card"), dict)
                and isinstance(rec["card"].get("rubric"), list)
                and isinstance(rec.get("policies"), list) and len(rec["policies"]) >= 1
                and all(isinstance(p.get("alias"), str) for p in rec["policies"])
                and isinstance(rec.get("trials"), list)
                and all(isinstance(t, dict) and isinstance(t.get("alias"), str)
                        and isinstance(t.get("success"), bool) for t in rec["trials"])
                and is_finite_number(rec.get("started_at")))  # fmt: skip
    except (AttributeError, TypeError):
        return False


def _v1_summary(rec: dict[str, Any]) -> dict[str, Any]:
    eps = rec["episodes"]
    k = sum(e["outcome"] == "success" for e in eps)
    lo, hi = es.wilson(k, len(eps))
    return {"schema": 1, "id": rec["id"], "name": rec["task"] or rec["policy"],
            "policies": [rec["policy"]], "started_at": rec["started_at"],
            "ended_at": rec.get("ended_at"), "n": len(eps), "k": k,
            "rate": k / len(eps) if eps else None, "ci95": [lo, hi]}  # fmt: skip


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "card"
    return s


class EvalStore:
    def __init__(self, data_dir: Path | str, now: Callable[[], float] = time.time) -> None:
        self.dir = Path(data_dir) / "evals"
        self.cards_dir = self.dir / "cards"
        self.refs_dir = self.dir / "refs"
        self.now = now
        self.current: dict[str, Any] | None = None
        for rec in self._load():  # newest first
            if rec.get("schema") == SCHEMA and not rec.get("ended_at"):
                self.current = rec
                break

    # -- files ---------------------------------------------------------------------------------
    @staticmethod
    def _write(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)  # WHY: a crash mid-write leaves the previous file, never half of one

    def _load(self) -> list[dict[str, Any]]:
        out = []
        if self.dir.is_dir():
            for p in self.dir.glob("*.json"):
                try:
                    rec = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue  # a damaged file must not stop Studio; the others still list
                if _v2_ok(rec) or _v1_ok(rec):
                    out.append(rec)
        return sorted(out, key=lambda r: r["started_at"], reverse=True)

    def _commit(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Save, then adopt. WHY this order: if the save fails, memory still matches the disk, so a
        score reported as not saved is not counted later either."""
        self._write(self.dir / f"{rec['id']}.json", rec)
        self.current = None if rec.get("ended_at") else rec
        return rec

    def _new_id(self, base: str, folder: Path) -> str:
        rid, n = base, 2
        while (folder / f"{rid}.json").exists():
            rid, n = f"{base}-{n}", n + 1
        return rid

    # -- cards -----------------------------------------------------------------------------------
    def cards(self) -> list[dict[str, Any]]:
        out = []
        if self.cards_dir.is_dir():
            for p in self.cards_dir.glob("*.json"):
                try:
                    c = json.loads(p.read_text())
                    if isinstance(c, dict) and isinstance(c.get("id"), str):
                        out.append({**check_card(c), "id": c["id"],
                                    "created_at": c.get("created_at"),
                                    "locked_at": c.get("locked_at")})  # fmt: skip
                except (OSError, ValueError):
                    continue
        return sorted(out, key=lambda c: c.get("created_at") or 0, reverse=True)

    def card(self, cid: Any) -> dict[str, Any] | None:
        if not isinstance(cid, str) or not CARD_ID.match(cid):
            return None
        return next((c for c in self.cards() if c["id"] == cid), None)

    def save_card(self, raw: Any) -> dict[str, Any]:
        card = check_card(raw)
        cid = raw.get("id") if isinstance(raw, dict) else None
        old = self.card(cid) if cid else None
        if cid and old is None:
            raise EvalError("That task card no longer exists. Make a new one.")
        if old and old.get("locked_at"):
            raise EvalError("This card is locked: an eval used it, so its rubric is fixed. "
                            "Duplicate it to change it.")
        t = self.now()
        if old is None:
            day = time.strftime("%Y%m%d", time.localtime(t))
            cid = self._new_id(f"{_slug(card['name'])}-{day}", self.cards_dir)
        card.update(id=cid, created_at=old["created_at"] if old else t, locked_at=None)
        self._write(self.cards_dir / f"{cid}.json", card)
        return card

    def duplicate_card(self, cid: Any) -> dict[str, Any]:
        old = self.card(cid)
        if old is None:
            raise EvalError("That task card no longer exists.")
        raw = {k: v for k, v in old.items() if k not in ("id", "created_at", "locked_at")}
        raw["name"] = (old["name"][:72] + " (copy)")
        raw["conditions"] = [{**c, "refs": {}} for c in old["conditions"]]  # pictures are per card
        return self.save_card(raw)

    def _lock(self, cid: str) -> None:
        c = self.card(cid)
        if c is not None and not c.get("locked_at"):
            c["locked_at"] = self.now()
            self._write(self.cards_dir / f"{cid}.json", c)

    def save_ref(self, cid: Any, cond: Any, camera: Any, jpeg: bytes) -> str:
        """A reference picture of a staged condition, shown as a ghost over the live view so each
        trial starts the same way (TRI's overlay). Allowed until the condition has one; after the
        card is locked, a picture can be added but never replaced."""
        c = self.card(cid)
        if c is None:
            raise EvalError("That task card no longer exists.")
        cnd = next((x for x in c["conditions"] if x["id"] == cond), None)
        if cnd is None:
            raise EvalError("That condition is not on the card.")
        if not isinstance(camera, str) or not re.match(r"^[A-Za-z0-9_.-]{1,40}\Z", camera):
            raise EvalError("Not a camera name.")
        if c.get("locked_at") and camera in cnd["refs"]:
            raise EvalError("This card is locked and the condition already has its picture.")
        name = f"{cnd['id']}-{camera}.jpg"
        d = self.refs_dir / c["id"]
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (name + ".tmp")
        tmp.write_bytes(jpeg)
        os.replace(tmp, d / name)
        cnd["refs"][camera] = name
        self._write(self.cards_dir / f"{c['id']}.json", c)
        return name

    def ref_path(self, cid: str, name: str) -> Path | None:
        if not CARD_ID.match(cid) or not re.match(r"^c\d{1,3}-[A-Za-z0-9_.-]{1,40}\.jpg\Z", name):
            return None
        p = self.refs_dir / cid / name
        return p if p.is_file() else None

    # -- an eval ---------------------------------------------------------------------------------
    def _need(self) -> dict[str, Any]:
        if self.current is None:
            raise EvalError("No eval is running. Start one first.")
        return self.current

    def begin(self, card_id: Any, policies: Any, reps: Any, blind: Any, alpha: Any,
              catalog: dict[str, str], *, seed: int | None = None, grouped: Any = False,
              stamps: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
        """Start an eval: lock the card, assign letters at random, draw the schedule."""
        if self.current is not None:
            raise EvalError("An eval is already running. End it before starting another.")
        card = self.card(card_id)
        if card is None:
            raise EvalError("Choose a task card.")
        if not isinstance(policies, list) or not 1 <= len(policies) <= MAX_POLICIES:
            raise EvalError(f"Choose 1 to {MAX_POLICIES} policies.")
        if len(set(map(str, policies))) != len(policies):
            raise EvalError("Each policy can be in an eval once.")
        for p in policies:
            if not isinstance(p, str) or p not in catalog:
                raise EvalError(f"{p!r} is not a policy that can run on this rig.")
        reps = _int(reps, 1, MAX_REPS, "Bundles per condition")
        alpha = _num(alpha, 0.01, 0.2, "The false-alarm rate")
        if not isinstance(blind, bool) or not isinstance(grouped, bool):
            raise EvalError("Blind and grouped are yes or no.")
        if reps * len(card["conditions"]) * len(policies) > MAX_TRIALS:
            raise EvalError(f"That is more than {MAX_TRIALS} trials. Use fewer bundles.")
        seed = seed if isinstance(seed, int) else random.SystemRandom().randrange(2**31)
        rng = random.Random(seed)
        ids = list(policies)
        rng.shuffle(ids)  # WHY: so "A" is not always the first policy picked, which would unblind
        pols = [{"alias": ALIASES[i], "id": pid, "name": catalog[pid]} for i, pid in enumerate(ids)]
        aliases = [p["alias"] for p in pols]
        t = self.now()
        rid = self._new_id(time.strftime("%Y%m%d-%H%M%S", time.localtime(t)), self.dir)
        rec = {"schema": SCHEMA, "id": rid, "card": card, "policies": pols, "blind": blind,
               "alpha": alpha, "seed": seed, "grouped": grouped, "reps": reps,
               "schedule": make_schedule(card["conditions"], aliases, reps, grouped, rng),
               "trials": [], "skipped": [], "started_at": t, "ended_at": None,
               "stamps": stamps or {}}  # fmt: skip
        self._lock(card["id"])  # first: a card is never left unlocked under an eval
        return self._commit(rec)

    @staticmethod
    def _why_not_mine(rec: dict[str, Any], slot: dict[str, Any], run: Any) -> str | None:
        """`run` is the worker's view of its newest policy run, taken by the server from telemetry.
        WHY not fields from the browser: a window could then score a run Studio never ran."""
        if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
            return "There is no policy run to score. Run the trial first."
        pol = policy_of(rec, slot["alias"])
        if (run.get("id"), run.get("task"), run.get("limit_s")) != (
                pol["id"], rec["card"]["task"], rec["card"]["limit_s"]):
            return "The last run was not this trial's policy, task or time limit. Run the trial."
        started = run.get("started_at")
        if not is_finite_number(started) or started < rec["started_at"]:
            return "The last run started before this eval. Run the trial."
        if any(t.get("run_id") == run["run_id"] for t in rec["trials"]):
            return "This run is already scored. Run the next trial."
        return None

    def awaiting(self, run: Any) -> bool:
        """True while `run` is an ended, unscored run of the running eval's next trial."""
        rec = self.current
        slot = next_slot(rec) if rec else None
        return (rec is not None and slot is not None and isinstance(run, dict)
                and not run.get("running") and self._why_not_mine(rec, slot, run) is None)

    def score(self, stage: Any, failures: Any, note: Any, run: Any, run_id: Any, *,
              health: dict[str, Any] | None = None, hz: float = 30.0,
              stamp: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
        rec = self._need()
        slot = next_slot(rec)
        if slot is None:
            raise EvalError("Every trial is done. End the eval.")
        rubric = rec["card"]["rubric"]
        stage = _int(stage, 0, len(rubric), "The milestone")
        if not isinstance(failures, list) or len(set(map(str, failures))) != len(failures) or any(
                f not in rec["card"]["failures"] for f in failures):
            raise EvalError("Failure tags must come from the card.")
        if not isinstance(note, str) or len(note) > MAX_NOTE:
            raise EvalError(f"A note must be text of at most {MAX_NOTE} characters.")
        why = self._why_not_mine(rec, slot, run)
        if why:
            raise EvalError(why)
        assert isinstance(run, dict)
        if run_id != run["run_id"]:
            raise EvalError("A newer run replaced the one shown. Score the newest run.")
        if run.get("running"):
            raise EvalError("The trial is still running. Score it when it ends.")
        if len(rec["trials"]) >= MAX_TRIALS:
            raise EvalError(f"An eval holds at most {MAX_TRIALS} trials.")
        metrics = run_metrics(run)
        dur = run.get("episode_s")
        trial = {"n": len(rec["trials"]) + 1, "b": slot["b"], "c": slot["c"],
                 "alias": slot["alias"], "run_id": run_id, "stage": stage,
                 "score": rubric[stage - 1]["points"] if stage else 0.0,
                 "success": stage == len(rubric), "failures": failures, "note": note.strip(),
                 "duration_s": dur if is_finite_number(dur) else None, "metrics": metrics,
                 "health": health or {}, "suspect": suspect(health or {}, metrics, hz),
                 "stamp": stamp or {}, "void": None, "at": self.now()}  # fmt: skip
        return self._commit({**rec, "trials": [*rec["trials"], trial]})

    def void(self, n: Any, reason: Any) -> dict[str, Any]:
        """Take a trial out of the numbers because the rig spoiled it. Its slot runs again."""
        rec = self._need()
        reason = _text(reason, 200, "Say why the trial is void.")
        i = next((i for i, t in enumerate(rec["trials"]) if t["n"] == n), None)
        if i is None:
            raise EvalError("There is no such trial.")
        if not _valid(rec["trials"][i]):
            raise EvalError("That trial is already void.")
        trials = list(rec["trials"])
        trials[i] = {**trials[i], "void": {"reason": reason, "at": self.now()}}
        return self._commit({**rec, "trials": trials})

    def skip(self, reason: Any) -> dict[str, Any]:
        """Give up on the next slot (the condition cannot be staged, the policy cannot run). Its
        bundle then counts for no comparison involving that policy."""
        rec = self._need()
        slot = next_slot(rec)
        if slot is None:
            raise EvalError("Every trial is done. End the eval.")
        reason = _text(reason, 200, "Say why the trial is skipped.")
        sk = {"b": slot["b"], "alias": slot["alias"], "reason": reason, "at": self.now()}
        return self._commit({**rec, "skipped": [*rec.get("skipped", []), sk]})

    def undo(self) -> dict[str, Any]:
        """Undo the newest change: a score, a void or a skip."""
        rec = self._need()
        events = ([("score", t["at"], t["n"]) for t in rec["trials"]]
                  + [("void", t["void"]["at"], t["n"]) for t in rec["trials"] if t["void"]]
                  + [("skip", s["at"], i)
                     for i, s in enumerate(rec.get("skipped", []))])  # fmt: skip
        if not events:
            raise EvalError("There is nothing to undo.")
        kind, _, key = max(events, key=lambda e: e[1])
        trials, skipped = list(rec["trials"]), list(rec.get("skipped", []))
        if kind == "score":
            trials = [t for t in trials if t["n"] != key]
        elif kind == "void":
            trials = [{**t, "void": None} if t["n"] == key else t for t in trials]
        else:
            del skipped[key]
        return self._commit({**rec, "trials": trials, "skipped": skipped})

    def end(self) -> dict[str, Any]:
        return self._commit({**self._need(), "ended_at": self.now()})

    # -- listing ---------------------------------------------------------------------------------
    def record(self, rid: Any) -> dict[str, Any] | None:
        if not isinstance(rid, str) or not re.match(r"^[0-9]{8}-[0-9]{6}(-club)?(-\d+)?\Z", rid):
            return None
        return next((r for r in self._load() if r["id"] == rid), None)

    def past(self, limit: int = 50) -> list[dict[str, Any]]:
        """Summaries without trials, newest first, not counting the running eval."""
        cur = self.current["id"] if self.current else None
        out = []
        for r in self._load():
            if r["id"] == cur:
                continue
            if r.get("schema") != SCHEMA:
                out.append(_v1_summary(r))
                continue
            s = summarize(r)
            names = {p["alias"]: p.get("name") or p.get("id") for p in r["policies"]}
            out.append({"schema": SCHEMA, "id": r["id"], "name": r["card"]["name"],
                        "card": r["card"].get("id"), "started_at": r["started_at"],
                        "ended_at": r.get("ended_at"), "imported": r.get("imported"),
                        "policies": [names[a] for a in s["order"]],
                        "paired": s["paired"],
                        "results": [{"policy": names[a], **{k: s["policies"][a][k]
                                     for k in ("n", "k", "rate", "ci95", "progress")},
                                     "letters": (s["letters"] or {}).get(a),
                                     "by_axis": {ax: [r["k"], r["n"]] for ax, r in
                                                 s["policies"][a]["by_axis"].items()}}
                                    for a in s["order"]]})  # fmt: skip
            if len(out) >= limit:
                break
        return out

    # -- import and export -----------------------------------------------------------------------
    def import_club(self, path: Path) -> dict[str, Any]:
        """The club's rollout_scores.csv (phi.utils.eval_rollouts) as one ended, unpaired eval.
        Unpaired: those rollouts were not run in bundles, so comparisons may carry day-to-day
        drift, and the Results panel says so."""
        rows, card, trials = read_club_csv(path)
        models = sorted({r["model"] for r in rows})
        if len(models) > MAX_POLICIES:
            models = models[:MAX_POLICIES]
        alias = {m: ALIASES[i] for i, m in enumerate(models)}
        t0 = min(r["t"] for r in rows)
        rid = self._new_id(time.strftime("%Y%m%d-%H%M%S", time.localtime(t0)) + "-club", self.dir)
        cid = self._new_id(f"club-import-{time.strftime('%Y%m%d', time.localtime(self.now()))}",
                           self.cards_dir)  # fmt: skip
        card.update(id=cid, created_at=self.now(), locked_at=self.now())
        self._write(self.cards_dir / f"{cid}.json", card)
        kept: list[dict[str, Any]] = []
        for tr in trials:
            if tr["model"] in alias:
                kept.append({**{k: v for k, v in tr.items() if k != "model"},
                             "alias": alias[tr["model"]], "n": len(kept) + 1})
        rec = {"schema": SCHEMA, "id": rid, "card": card, "paired": False, "blind": False,
               "policies": [{"alias": alias[m], "id": m, "name": m} for m in models],
               "alpha": 0.05, "seed": None, "grouped": False, "reps": None, "schedule": [],
               "trials": kept, "skipped": [], "started_at": t0,
               "ended_at": max(r["t"] for r in rows), "stamps": {},
               "imported": {"from": str(path), "rows": len(rows), "kept": len(kept)}}  # fmt: skip
        self._write(self.dir / f"{rid}.json", rec)
        return rec


# The club's columns (phi/src/phi/utils/eval_rollouts.py FIELDS) and its rubric's labels.
CLUB_FIELDS = ("timestamp", "model", "episode", "split", "object", "container", "stage", "score",
               "grasp_approach", "hesitated", "notes")  # fmt: skip
CLUB_RUBRIC = (("Reached the object", 0.2), ("Grasped the object", 0.4),
               ("Carried it to the container", 0.7), ("Released it", 0.8),
               ("Object in the container", 1.0))  # fmt: skip


def read_club_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise EvalError(f"Could not read {path}: {e}") from e
    rd = csv.DictReader(io.StringIO(text))
    if not rd.fieldnames or set(CLUB_FIELDS) - set(rd.fieldnames):
        raise EvalError("That is not the club's rollout_scores.csv (columns differ).")
    points = {p: i + 1 for i, (_, p) in enumerate(CLUB_RUBRIC)}
    rows: list[dict[str, Any]] = []
    conds: dict[tuple[str, str, str], dict[str, Any]] = {}
    trials: list[dict[str, Any]] = []
    for r in rd:
        try:
            sc = round(float(r["score"]), 4)
            stage = 0 if sc == 0 else points[sc]
            when = time.mktime(time.strptime(r["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, ValueError):
            continue  # a row Studio cannot read is left out; the import says how many it kept
        split = "train" if r.get("split") == "train" else "held-out position"
        label = f"{r['object']} · {r['container']}" + (" · trained on" if split == "train" else "")
        key = (r["object"], r["container"], split)
        if key not in conds:
            conds[key] = {"id": f"c{len(conds) + 1}", "label": label[:80], "axis": split,
                          "control": split == "train", "refs": {}}  # fmt: skip
        rows.append({"model": r["model"], "t": when})
        tags = [f"Approach: {r['grasp_approach']}"] if r.get("grasp_approach") else []
        if r.get("hesitated") == "y":
            tags.append("Hesitated or oscillated")
        trials.append({"model": r["model"], "b": None, "c": conds[key]["id"], "run_id": None,
                       "stage": stage, "score": sc, "success": stage == len(CLUB_RUBRIC),
                       "failures": [t for t in tags if t in DEFAULT_FAILURES],
                       "note": "; ".join(x for x in (r.get("notes", "").strip(),
                                                     f"episode {r['episode']}") if x),
                       "duration_s": None, "metrics": {}, "health": {}, "suspect": {},
                       "stamp": {}, "void": None, "at": when})  # fmt: skip
    if not rows:
        raise EvalError("No rows in that file could be read.")
    card = {"name": "Club cubes and cylinder (imported)",
            "task": "Pick up the object and place it in the container",
            "success": "The object ends inside the container",
            "limit_s": 60.0, "rubric": [{"label": lb, "points": p} for lb, p in CLUB_RUBRIC],
            "failures": list(DEFAULT_FAILURES), "conditions": list(conds.values())}  # fmt: skip
    return rows, card, trials


CSV_FIELDS = ("eval", "card", "trial", "bundle", "condition", "axis", "control", "policy",
              "milestone", "milestone_label", "score", "success", "failures", "void",
              "duration_s", "suspect", "note", "time")  # fmt: skip


def to_csv(rec: dict[str, Any]) -> str:
    """One row per trial, voided ones included and marked. A running blind eval shows letters."""
    pub = public(rec)
    names = {p["alias"]: p.get("name") or p.get("id") or f"Policy {p['alias']}"
             for p in pub["policies"]}  # fmt: skip
    conds = {c["id"]: c for c in rec["card"]["conditions"]}
    labels = [m["label"] for m in rec["card"]["rubric"]]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_FIELDS)
    for t in rec["trials"]:
        c = conds.get(t["c"], {})
        w.writerow([rec["id"], rec["card"]["name"], t["n"], t.get("b") or "", c.get("label", ""),
                    c.get("axis", ""), "yes" if c.get("control") else "", names[t["alias"]],
                    t["stage"], labels[t["stage"] - 1] if t["stage"] else "none",
                    t["score"], "yes" if t["success"] else "no", "; ".join(t.get("failures", [])),
                    (t["void"] or {}).get("reason", ""), t.get("duration_s") or "",
                    "; ".join((t.get("suspect") or {}).get("void", [])), t.get("note", ""),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["at"]))])  # fmt: skip
    return buf.getvalue()


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def to_markdown(rec: dict[str, Any]) -> str:
    """A report a person can paste into an experiment write-up: protocol, numbers, comparisons,
    voids, and what the numbers cannot show."""
    s = summarize(rec)
    pub = public(rec)
    names = {p["alias"]: p.get("name") or p.get("id") or f"Policy {p['alias']}"
             for p in pub["policies"]}  # fmt: skip
    card = rec["card"]
    out = [f"# Eval {rec['id']}: {card['name']}", "",
         f"- Task given to the policy: \"{card['task']}\"",
         f"- Success: {card['success']}",
         "- Rubric: " + " · ".join(f"{m['label']} {m['points']:g}" for m in card["rubric"]),
         f"- Conditions: {len(card['conditions'])} · time limit {card['limit_s']:g} s"]
    if rec.get("imported"):
        out.append(f"- Imported from {rec['imported']['from']} ({rec['imported']['kept']} of "
                 f"{rec['imported']['rows']} rows). Not run in bundles: comparisons are unpaired "
                 "and may include day-to-day drift.")  # fmt: skip
    else:
        blind = "blind" if rec["blind"] else "not blind"
        tests = (f"false-alarm rate {s['alpha']:g} split over {s['comparisons']} comparisons"
                 if s["comparisons"] else "one policy, so no comparison")
        out.append(f"- Design: {s['planned_bundles']} bundles (each condition × every policy, "
                   f"random order, seed {rec['seed']}), {blind}, {tests}")
    out += ["", "| Policy | Success | 95% interval | Mean credit | Median time to success "
            "| Voided |", "|---|---|---|---|---|---|"]
    for a in s["order"]:
        r = s["policies"][a]
        lt = f" ({s['letters'][a]})" if s["letters"] else ""
        credit = "–" if r["progress"] is None else f"{r['progress']:.2f}"
        took = "–" if r["median_success_s"] is None else f"{r['median_success_s']:.1f} s"
        out.append(f"| {names[a]}{lt} | {r['k']}/{r['n']} ({_pct(r['rate'])}) | "
                   f"{_pct(r['ci95'][0])}–{_pct(r['ci95'][1])} | {credit} | {took} | "
                   f"{r['voided']} |")
    if s["letters"]:
        out.append("\nPolicies sharing a letter were not shown to differ.")
    out += ["", "## Comparisons"] if s["pairs"] else []
    for p in s["pairs"]:
        a, b = names[p["a"]], names[p["b"]]
        pa = p["p_a_better"]
        line = f"- {a} vs {b}: P({a} better) = {'–' if pa is None else f'{pa:.2f}'}"
        d = p.get("decision")
        if d:
            st = d["state"]
            line += (f"; decided: {a if st == 'a' else b} better at bundle {d['at_bundle']}"
                     if st in ("a", "b") else "; no difference shown within the plan"
                     if st == "none" else f"; undecided after {p['bundles']} bundles"
                     + (f", about {p['bundles_more_about']} more needed at this gap"
                        if p.get("bundles_more_about") is not None else ""))  # fmt: skip
        elif not p["paired"]:
            line += " (unpaired)"
        out.append(line)
    out += ["", "## By what each condition tests"]
    for a in s["order"]:
        parts = [f"{ax} {r['k']}/{r['n']}" for ax, r in s["policies"][a]["by_axis"].items()]
        out.append(f"- {names[a]}: " + (" · ".join(parts) or "–"))
    fails: Counter[str] = Counter()
    for a in s["order"]:
        fails.update(s["policies"][a]["failures"])
    if fails:
        out += ["", "## What went wrong", *[f"- {f}: {n}" for f, n in fails.most_common()]]
    voids = [t for t in rec["trials"] if not _valid(t)]
    if voids:
        out += ["", "## Voided trials (not counted)",
              *[f"- Trial {t['n']} ({names[t['alias']]}): {t['void']['reason']}" for t in voids]]
    st = rec.get("stamps") or {}
    if st:
        out += ["", "## Rig and software at the start", "```", json.dumps(st, indent=1), "```"]
    out += ["", "## What these numbers cannot show",
          "- Success on these conditions only; a held-out position is not a new object or task.",
          "- With n trials per policy, only large gaps are detectable"
          + (f" (about {s['detectable_gap']:.0%} at {s['planned_bundles']} bundles)."
             if s.get("detectable_gap") else "."), ""]  # fmt: skip
    return "\n".join(out)
