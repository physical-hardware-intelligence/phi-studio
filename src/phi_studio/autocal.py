"""Auto-calibration: each joint finds its own end stops, so every arm's calibration comes from its
mechanics rather than from how far a person happened to push it.

LeRobot's calibration (so_follower.py calibrate): the middle pose sets the homings (half turn),
then a person moves every joint through its range and the extremes become range_min and
range_max. DEGREES then puts 0 at the middle of that range (motors_bus.py _normalize). A person
rarely reaches the stops, and never the same way twice, so two arms read different angles for the
same pose. Here the arm drives each joint gently into both stops and the stops set the range.

Per joint, one at a time, in ORDER (wrist roll has no stops; LeRobot fixes it at 0..4095):
  out    the goal creeps toward higher ticks at SPEED_DEG_S, never more than LEAD_DEG ahead of the
         joint. Stalled (the goal LEAD_DEG ahead, the joint still for STALL_S): that is the stop.
  back   the same toward lower ticks.
  home   back to where the joint started, the middle pose, so the next joint starts from it.
Every other joint holds where it is. The worker lowers the sweeping joint's torque limit
(SWEEP_LIMIT), so pressing into a stop is gentle, and writes the goals this returns each tick.

The result per joint is the stop pair. range() turns it into a calibration: the stops pulled in by
MARGIN_DEG (a goal at the limit must not keep pressing into the stop), and a homing that puts the
middle of the stops at 2047, so every arm's file has the same shape: range_min and range_max
symmetric about 2047, apart by that arm's own travel.

Guards, each ending the run with a reason instead of guessing: a direction that takes longer than
DIR_TIMEOUT_S or travels more than TRAVEL_CAP_DEG (no stop found), a joint within EDGE_TICKS of 0
or 4095 (the range would wrap: start from a better middle pose), stops under MIN_TRAVEL_DEG apart
(a stuck joint). A travel under SHORT of the model's is noted (something in the way?). Stop, a
fault or a lost heartbeat are the worker's: it pauses this and freezes the arm.

[UNVERIFIED] Every number here is a starting value for the first session on a physical arm
(2026-10-05): speeds, the lead, the stall window and the torque limits want measuring there.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

TICKS = 4096  # one turn of the STS3215's encoder
TICKS_PER_DEG = TICKS / 360.0
HALF_TURN = 2047  # LeRobot's int(max_res / 2)
ORDER = ("gripper", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan")
SPEED_DEG_S = 25.0
LEAD_DEG = 6.0
STALL_S = 0.4
STILL_TICKS = 4  # how far a stalled joint may still wander in STALL_S (encoder noise, flex)
BACKOFF_DEG = 3.0
MARGIN_DEG = 1.5
ARRIVED_TICKS = 10  # close enough to the start pose to call it home
TRAVEL_CAP_DEG = 330.0
DIR_TIMEOUT_S = 25.0
EDGE_TICKS = 100
SHORT = 0.6  # a travel under this share of the model's is suspect
MIN_TRAVEL_DEG = 20.0  # under this the joint barely moved: a stuck joint, never a range
MAX_DT_S = 0.1  # a late tick advances the goal by at most this much time
# Torque_Limit (0..1000 of the servo's maximum) for the joint being swept; the rest hold at their
# own. Under the gripper's Overload_Torque (25 %, hardware.GRIPPER_LIMITS), so finding the closed
# stop does not trip the overload protection.
SWEEP_LIMIT = {"gripper": 200}
SWEEP_LIMIT_DEFAULT = 350


def sweep_limit(joint: str, torque: int | None = None) -> int:
    """Torque_Limit for a joint being swept: `torque` when the person set one (Advanced), else the
    default; the gripper never above its own, under its overload protection."""
    own = SWEEP_LIMIT.get(joint, SWEEP_LIMIT_DEFAULT)
    if torque is None:
        return own
    return min(torque, own) if joint in SWEEP_LIMIT else torque


@dataclass
class Sweep:
    joint: str
    start: int  # raw ticks at the middle pose
    t0: float
    phase: str = "out"  # out | back | home
    goal: float = 0.0
    origin: int = 0  # where this direction began
    hi: int | None = None
    lo: int | None = None
    seen: deque[tuple[float, int]] = field(default_factory=deque)


@dataclass
class AutoCal:
    """One arm's sweep. step() takes raw positions and returns raw goals for every joint."""

    joints: list[str]
    start: dict[str, int]
    now: float
    expected_deg: dict[str, float] = field(default_factory=dict)  # model travel per joint
    state: str = "running"  # running | paused | done | failed
    why: str | None = None
    found: dict[str, tuple[int, int]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    current: Sweep | None = None
    hold: dict[str, float] = field(default_factory=dict)
    last: float = 0.0
    settled: bool = True  # not running: the swept joint already holds where it stopped

    def __post_init__(self) -> None:
        self.joints = [j for j in ORDER if j in self.joints]
        self.hold = {j: float(v) for j, v in self.start.items()}
        self.last = self.now
        self._next(self.now)

    # -- control ---------------------------------------------------------------------------------
    def _next(self, now: float) -> None:
        left = [j for j in self.joints if j not in self.found]
        if not left:
            self.current, self.state = None, "done"
            return
        j = left[0]
        p = int(round(self.hold[j]))
        self.current = Sweep(j, start=p, t0=now, goal=float(p), origin=p)

    def _fail(self, why: str) -> None:
        self.state, self.why, self.settled = "failed", why, False

    def pause(self, why: str) -> None:
        if self.state == "running":
            self.state, self.why, self.settled = "paused", why, False

    def resume(self, raw: dict[str, int], now: float) -> None:
        """Go on from where the arm is now: a paused sweep starts its direction again."""
        if self.state != "paused":
            return
        self.state, self.why, self.last = "running", None, now
        s = self.current
        if s is not None:
            p = raw[s.joint]
            s.goal, s.origin, s.t0 = float(p), p, now
            s.seen.clear()

    def step(self, raw: dict[str, int], now: float, advance: bool = True) -> dict[str, int]:
        """The goals for this tick. Not running: every joint holds still, and the swept one holds
        where it is now, not at its goal, which may lead it into a stop by up to LEAD_DEG.
        advance=False: wait (another arm's turn); the joint's direction starts when it may go."""
        dt = min(MAX_DT_S, max(0.0, now - self.last))
        self.last = now
        s = self.current
        if self.state == "running" and s is not None:
            if advance:
                self._advance(s, raw[s.joint], now, dt)
            else:
                s.t0 = now
                s.seen.clear()
        if self.state != "running" and not self.settled:
            if s is not None:
                self.hold[s.joint] = float(raw[s.joint])
            self.settled = True
        return {j: int(round(v)) for j, v in self.hold.items()}

    def _advance(self, s: Sweep, p: int, now: float, dt: float) -> None:
        if p < EDGE_TICKS or p > TICKS - 1 - EDGE_TICKS:
            self._fail(f"{s.joint} reached tick {p}, near the end of the encoder's turn: its range "
                       "would wrap. Start again from the middle pose.")  # fmt: skip
            self.hold[s.joint] = float(p)
            return
        if s.phase == "home":
            step = SPEED_DEG_S * TICKS_PER_DEG * dt
            s.goal += max(-step, min(step, s.start - s.goal))
            self.hold[s.joint] = s.goal
            if abs(p - s.start) <= ARRIVED_TICKS and abs(s.goal - s.start) < 0.5:
                assert s.lo is not None and s.hi is not None
                if (s.hi - s.lo) / TICKS_PER_DEG < MIN_TRAVEL_DEG:
                    self._fail(f"{s.joint} moved only {(s.hi - s.lo) / TICKS_PER_DEG:.0f} degrees "
                               "between its stops: is it stuck, or held?")  # fmt: skip
                    return
                self.found[s.joint] = (s.lo, s.hi)
                self._check_travel(s.joint)
                self._next(now)
            return
        d = 1 if s.phase == "out" else -1
        lead_max = LEAD_DEG * TICKS_PER_DEG
        lead = (s.goal - p) * d
        if lead < lead_max:  # never further ahead than LEAD_DEG, so the push at a stop is bounded
            s.goal += d * min(SPEED_DEG_S * TICKS_PER_DEG * dt, lead_max - lead)
        self.hold[s.joint] = s.goal
        s.seen.append((now, p))
        while s.seen and now - s.seen[0][0] > STALL_S:
            s.seen.popleft()
        span = s.seen[-1][0] - s.seen[0][0] if len(s.seen) > 1 else 0.0
        still = max(v for _, v in s.seen) - min(v for _, v in s.seen) <= STILL_TICKS
        pressing = (s.goal - p) * d >= 0.9 * lead_max
        if pressing and still and span >= 0.9 * STALL_S:
            self._found_stop(s, p, d, now)
            return
        if abs(p - s.origin) > TRAVEL_CAP_DEG * TICKS_PER_DEG:
            self._fail(f"{s.joint} travelled {TRAVEL_CAP_DEG:.0f} degrees without meeting a stop.")
        elif now - s.t0 > DIR_TIMEOUT_S:
            self._fail(f"{s.joint} found no stop in {DIR_TIMEOUT_S:.0f} s: is it blocked, or the "
                       "torque limit too low to move it?")  # fmt: skip

    def _found_stop(self, s: Sweep, p: int, d: int, now: float) -> None:
        if d > 0:
            s.hi, s.phase = p, "back"
        else:
            s.lo, s.phase = p, "home"
        # Off the stop at once, so the servo stops pressing into it.
        s.goal = p - d * BACKOFF_DEG * TICKS_PER_DEG
        self.hold[s.joint] = s.goal
        s.origin, s.t0 = p, now
        s.seen.clear()

    def _check_travel(self, joint: str) -> None:
        lo, hi = self.found[joint]
        got = (hi - lo) / TICKS_PER_DEG
        want = self.expected_deg.get(joint)
        if want and got < SHORT * want:
            self.notes[joint] = (f"travel {got:.0f} degrees, the model's is {want:.0f}: was "
                                 "something in the way?")  # fmt: skip

    # -- result ----------------------------------------------------------------------------------
    def range(self, joint: str, homing: int) -> tuple[int, int, int]:
        """(homing, range_min, range_max) for a swept joint. `homing` is the one in the registers
        during the sweep; the new one puts the middle of the stops at HALF_TURN. LeRobot's rule,
        Present_Position = Actual_Position - Homing_Offset, so moving the homing by k moves every
        reading by -k."""
        lo, hi = self.found[joint]
        margin = int(round(MARGIN_DEG * TICKS_PER_DEG))
        shift = int(round((lo + hi) / 2)) - HALF_TURN
        return homing + shift, lo + margin - shift, hi - margin - shift

    def view(self) -> dict[str, Any]:
        s = self.current
        return {
            "state": self.state, "why": self.why, "joints": self.joints,
            "joint": s.joint if s else None, "phase": s.phase if s else None,
            "found": {j: {"lo": lo, "hi": hi, "deg": round((hi - lo) / TICKS_PER_DEG, 1)}
                      for j, (lo, hi) in self.found.items()},
            "notes": dict(self.notes),
        }  # fmt: skip


def expected_travel() -> dict[str, float]:
    """The model's travel per joint in degrees (the MJCF's joint ranges), for the SHORT check."""
    try:
        from phi_studio.kinematics import model
    except Exception:  # no numpy, or no model bundle: skip the check
        return {}
    out: dict[str, float] = {}

    def walk(b: dict[str, Any]) -> None:
        j = b.get("joint")
        if j is not None:
            lo, hi = j["range"]
            out[j["name"]] = math.degrees(hi - lo)
        for c in b.get("children", []):
            walk(c)

    try:
        walk(model()["tree"])
    except (OSError, KeyError, TypeError, ValueError):
        return {}
    return out
