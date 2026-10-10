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
MARGIN_DEG (a goal at the limit must not keep pressing into the stop), except the gripper's closed
end (gripping presses it), and a homing that puts the middle of the stops at 2047, so every arm's
file has the same shape: range_min and range_max symmetric about 2047 (the gripper's within
MARGIN_DEG), apart by that arm's own travel.

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
# Unfolding an arm from rest into the middle pose, checked in simulation (scripts/plan_autocal.py:
# the SO-101 model, exact meshes, a table; every physically possible version of a real rest pose).
# The gripper swings out first (the wrist), then the base while the arm is compact, then the
# forearm, and the upper arm last: lifting the upper arm while the forearm is folded drives the
# jaw into the table (2026-10-05, on the real arm and in simulation). Then the wrist rolls to its
# middle, so every arm sweeps with the jaw the same way round: the left follower rests with its
# wrist rolled to -139 degrees, and its jaw came within 3 mm of the table in two sweeps that clear
# by 12 mm at the middle (2026-10-08). scripts/plan_autocal.py checks the program for an arm.
POSE_ORDER = ("wrist_flex", "shoulder_pan", "elbow_flex", "shoulder_lift", "wrist_roll", "gripper")
# program()'s unfold, POSE_ORDER with the wrist in two stages: (joint, degrees from its middle, None
# for the middle). Swung straight from rest to its middle, the wrist passed the shoulder bracket at
# 9.9 mm (2026-10-08, plan_autocal.py pair by pair); to WRIST_OUT_DEG first, then to its middle once
# the elbow has opened, it clears by 10 mm. The fold back visits the same positions in reverse.
WRIST_OUT_DEG = 30.0
UNFOLD: tuple[tuple[str, float | None], ...] = (
    ("wrist_flex", WRIST_OUT_DEG), ("shoulder_pan", None), ("elbow_flex", None),
    ("wrist_flex", None), ("shoulder_lift", None), ("wrist_roll", None), ("gripper", None),
)  # fmt: skip
# A gripper-only run: the wrist lifts the jaw off the table, the gripper sweeps, the wrist folds.
# Swept at rest, the left follower's jaw met the table at 63 % open (2026-10-08, wrist rolled to
# -139); lifted to WRIST_OUT_DEG it clears by 10 mm from both rest poses seen.
GRIPPER_CLEAR: tuple[tuple[str, float | None], ...] = (("wrist_flex", WRIST_OUT_DEG),)
SPEED_DEG_S = 25.0
LEAD_DEG = 6.0
# Joints that hold the arm's weight lag their goal (P control): 21 % of stall torque at worst for
# the shoulder, 18 % for the elbow (simulation, along the whole program), which is about 4.5
# degrees of lag at the 28 % a 6-degree lead gave on the real shoulder. A 10-degree lead keeps
# that lag well under the 9 degrees that read as a stop. The torque limit, not the lead, bounds
# the push at a stop.
LEAD = {"shoulder_lift": 10.0, "elbow_flex": 10.0}
STALL_S = 0.4
STILL_TICKS = 4  # how far a stalled joint may still wander in STALL_S (encoder noise, flex)
BACKOFF_DEG = 3.0
MARGIN_DEG = 1.5
ARRIVED_TICKS = 10  # close enough to the start pose to call it home
TRAVEL_CAP_DEG = 330.0
DIR_TIMEOUT_S = 25.0
EDGE_TICKS = 100
SHORT = 0.6  # a travel under this share of the model's is suspect
# A hand calibration stops short of the stops, so a sweep that finds less than the arm's previous
# range minus this was stopped early: a table, a cable, the arm's own links.
PREVIOUS_SLACK_DEG = 5.0
MIN_TRAVEL_DEG = 20.0  # under this the joint barely moved: a stuck joint, never a range
MAX_DT_S = 0.1  # a late tick advances the goal by at most this much time
# Torque_Limit (0..1000 of the servo's maximum) for the joint being swept; the rest hold at their
# own. Under the gripper's Overload_Torque (25 %, hardware.GRIPPER_LIMITS), so finding the closed
# stop does not trip the overload protection.
SWEEP_LIMIT = {"gripper": 200}  # caps: never above, whatever the person sets
# The shoulder lifts the whole arm back from its forward stop: 27 % of stall torque in its sweep
# pose (simulation, 2026-10-08), too close to 35 %. A stop is pressed for under half a second.
SWEEP_DEFAULT: dict[str, int] = {"shoulder_lift": 450}
SWEEP_LIMIT_DEFAULT = 350
POSE_LIMIT = 400  # Torque_Limit while an arm moves itself between poses
# Where the other joints go while a joint sweeps one side, in degrees from the middle of their
# ranges, when the middle pose would let the arm hit the table (simulation, as POSE_ORDER).
SWEEP_POSES: dict[tuple[str, int], dict[str, float]] = {
    # Folding the elbow from the middle pose drives the jaw into the table (at +48 degrees): lean
    # the upper arm back 70 first; the fold then ends on its own stop, the forearm on the shoulder
    # bracket, 12.4 mm clear of everything else at any wrist roll. WHY 70, not 50: at 50 the left
    # follower's jaw, its wrist rolled to -139 degrees, went 33 mm into the table (2026-10-08).
    ("elbow_flex", 1): {"shoulder_lift": -70.0},
    # Tilting the shoulder forward with the forearm level drives the jaw into the table (at +38):
    # raise the forearm straight up and tip the wrist fully up first; 30 mm clear to the stop, and
    # the lightest pose for the shoulder to lift back (27 %). WHY the wrist too: with the forearm
    # alone the gripper came within 4 mm of the table (2026-10-08).
    ("shoulder_lift", 1): {"elbow_flex": -100.0, "wrist_flex": -90.0},
}


def lead_deg(joint: str) -> float:
    return LEAD.get(joint, LEAD_DEG)


def sweep_limit(joint: str, torque: int | None = None) -> int:
    """Torque_Limit for a joint being swept: `torque` when the person set one (Advanced), else the
    default; the gripper never above its own, under its overload protection."""
    own = SWEEP_LIMIT.get(joint, SWEEP_DEFAULT.get(joint, SWEEP_LIMIT_DEFAULT))
    if torque is None:
        return own
    return min(torque, own) if joint in SWEEP_LIMIT else torque


@dataclass(frozen=True)
class Step:
    """One move of a program. pose: the joint to `to` (raw ticks); something in its way ends the
    run. rest: the same, but meeting something counts as arriving, as an arm landing at rest does.
    sweep: the joint to its stop on side `sign` (+1 high ticks, -1 low, 0 both) and home again."""

    kind: str
    joint: str
    to: int = 0
    sign: int = 0


@dataclass
class Sweep:
    joint: str
    start: int  # raw ticks at the middle pose
    t0: float
    phase: str = "out"  # pose | rest (moves between poses) | out | back | home
    goal: float = 0.0
    origin: int = 0  # where this direction began
    hi: int | None = None
    lo: int | None = None
    seen: deque[tuple[float, int]] = field(default_factory=deque)
    only: int = 0  # a half sweep: +1 the high stop only, -1 the low one; 0 both


@dataclass
class AutoCal:
    """One arm's sweep. step() takes raw positions and returns raw goals for every joint."""

    joints: list[str]
    start: dict[str, int]
    now: float
    expected_deg: dict[str, float] = field(default_factory=dict)  # model travel per joint
    previous_deg: dict[str, float] = field(default_factory=dict)  # this arm's last range per joint
    # The middle pose in raw ticks, to drive to before sweeping: the arm puts itself there when
    # its servos' calibration says where the middle is. Empty: a person set the middle pose.
    target: dict[str, int] = field(default_factory=dict)
    # The moves in order. Empty: drive to `target` in POSE_ORDER, then sweep each joint both ways.
    program: list[Step] = field(default_factory=list)
    todo: deque[Step] = field(default_factory=deque)
    partial: dict[str, dict[str, int]] = field(default_factory=dict)  # half sweeps' stops so far
    # Pause after each move of the unfold (the program's opening pose steps), so a person sees
    # each one before the next: the first runs on an arm. The sweeps and the fold run on.
    step_through: bool = False
    cur: Step | None = None  # the program step being run
    lead: int = 0  # how many opening pose steps the program has: the unfold
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
        if not self.program:
            posing = [Step("pose", j, to=self.target[j]) for j in POSE_ORDER if j in self.target]
            self.program = posing + [Step("sweep", j) for j in self.joints]
        self.todo = deque(self.program)
        self.lead = next((k for k, st in enumerate(self.program) if st.kind != "pose"),
                         len(self.program))  # fmt: skip
        self.last = self.now
        self._next(self.now)

    # -- control ---------------------------------------------------------------------------------
    def _next(self, now: float) -> None:
        ended = self.cur
        while self.todo:
            st = self.todo.popleft()
            p = int(round(self.hold[st.joint]))
            if st.kind in ("pose", "rest"):
                if abs(p - st.to) <= ARRIVED_TICKS:
                    continue  # already there
                self.current = Sweep(st.joint, start=st.to, t0=now, goal=float(p), origin=p,
                                     phase=st.kind)  # fmt: skip
            else:
                phase = "back" if st.sign < 0 else "out"
                self.current = Sweep(st.joint, start=p, t0=now, goal=float(p), origin=p,
                                     phase=phase, only=st.sign)  # fmt: skip
            self.cur = st
            unfolded = ended is not None and any(ended is u for u in self.program[: self.lead])
            if self.step_through and unfolded:
                assert ended is not None
                self.pause(f"Unfold move done: {ended.joint}. Look at the arm; Resume moves "
                           f"{st.joint} next.")  # fmt: skip
            return
        self.current, self.cur, self.state = None, None, "done"

    def _fail(self, why: str, hold: float | None = None) -> None:
        """End the run. hold: where the moving joint should hold from now on (a blocked joint backs
        off its obstacle); None: where it stands at the next tick."""
        self.state, self.why, self.settled = "failed", why, hold is not None
        if hold is not None and self.current is not None:
            self.hold[self.current.joint] = hold

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
        if s.phase in ("pose", "rest"):
            self._pose(s, p, now, dt)
            return
        if s.phase == "home":
            step = SPEED_DEG_S * TICKS_PER_DEG * dt
            s.goal += max(-step, min(step, s.start - s.goal))
            self.hold[s.joint] = s.goal
            arrived = abs(p - s.start) <= ARRIVED_TICKS
            # WHY also still: a start pressed into a stop (an arm at rest leans on one) is out of
            # reach at the sweep's lowered torque. 2026-10-05: an elbow 20 ticks short of it waited
            # there for good. Short and still is home; it holds where it stands, not pressing.
            if abs(s.goal - s.start) < 0.5 and (arrived or self._still(s, p, now)):
                if not arrived:
                    self.hold[s.joint] = float(p)
                have = self.partial.setdefault(s.joint, {})
                have |= {k: v for k, v in (("lo", s.lo), ("hi", s.hi)) if v is not None}
                if "lo" in have and "hi" in have:
                    lo, hi = have["lo"], have["hi"]
                    if (hi - lo) / TICKS_PER_DEG < MIN_TRAVEL_DEG:
                        self._fail(f"{s.joint} moved only {(hi - lo) / TICKS_PER_DEG:.0f} degrees "
                                   "between its stops: is it stuck, or held?")  # fmt: skip
                        return
                    self.found[s.joint] = (lo, hi)
                    self._check_travel(s.joint)
                self._next(now)
            return
        d = 1 if s.phase == "out" else -1
        lead_max = lead_deg(s.joint) * TICKS_PER_DEG
        lead = (s.goal - p) * d
        if lead < lead_max:  # never further ahead than its lead, so the push at a stop is bounded
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
                       "torque limit too low to move it?",
                       hold=p - d * BACKOFF_DEG * TICKS_PER_DEG)  # fmt: skip

    def _still(self, s: Sweep, p: int, now: float) -> bool:
        """The joint has stood within STILL_TICKS for STALL_S."""
        s.seen.append((now, p))
        while s.seen and now - s.seen[0][0] > STALL_S:
            s.seen.popleft()
        span = s.seen[-1][0] - s.seen[0][0] if len(s.seen) > 1 else 0.0
        return span >= 0.9 * STALL_S and max(v for _, v in s.seen) - min(v for _, v in s.seen) \
            <= STILL_TICKS  # fmt: skip

    def _pose(self, s: Sweep, p: int, now: float, dt: float) -> None:
        """Toward the middle pose at SPEED_DEG_S, never more than LEAD_DEG ahead of the joint, as
        a sweep goes, so a joint that meets something presses gently and stops: then the run ends
        and says which joint, since a sweep from a blocked pose would measure the obstacle."""
        lead_max = lead_deg(s.joint) * TICKS_PER_DEG
        d = 1 if s.start >= s.goal else -1
        lead = (s.goal - p) * d
        left = abs(s.start - s.goal)
        if left > 0 and lead < lead_max:
            s.goal += d * min(SPEED_DEG_S * TICKS_PER_DEG * dt, lead_max - lead, left)
        self.hold[s.joint] = s.goal
        still = self._still(s, p, now)
        short = (s.goal - p) * d  # how far the joint stands behind its goal
        # WHY still and short of the lead also counts: a P-controlled servo holds a load with a
        # standing error, so a heavy joint settles just short of its goal. A whole lead short is
        # something in its way.
        if abs(s.goal - s.start) < 0.5 and (abs(p - s.start) <= ARRIVED_TICKS
                                            or (still and short < 0.9 * lead_max)):  # fmt: skip
            self.hold[s.joint] = float(s.start)
            self._next(now)
            return
        if still and short >= 0.9 * lead_max:
            if s.phase == "rest":  # landed: the arm is resting on its stop or the table
                self.hold[s.joint] = float(p)
                self._next(now)
                return
            # WHY back off: holding where it stopped keeps it pressed into whatever stopped it.
            # 2026-10-05 a shoulder held a jammed arm at 70 % for two minutes and was damaged.
            self._fail(f"{s.joint} stopped {abs(s.start - p) / TICKS_PER_DEG:.0f} degrees short "
                       "of where it was going: something is in its way.",
                       hold=p - d * BACKOFF_DEG * TICKS_PER_DEG)  # fmt: skip
        elif now - s.t0 > DIR_TIMEOUT_S:
            self._fail(f"{s.joint} did not get where it was going in {DIR_TIMEOUT_S:.0f} s.",
                       hold=p - d * BACKOFF_DEG * TICKS_PER_DEG)  # fmt: skip

    def _found_stop(self, s: Sweep, p: int, d: int, now: float) -> None:
        if d > 0:
            s.hi, s.phase = p, ("home" if s.only > 0 else "back")
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
        before = self.previous_deg.get(joint)
        if want and got < SHORT * want:
            self.notes[joint] = (f"travel {got:.0f} degrees, the model's is {want:.0f}: was "
                                 "something in the way?")  # fmt: skip
        elif before and got < before - PREVIOUS_SLACK_DEG:
            self.notes[joint] = (f"travel {got:.0f} degrees, its last calibration spanned "
                                 f"{before:.0f}: something stopped it early?")  # fmt: skip

    # -- result ----------------------------------------------------------------------------------
    def range(self, joint: str, homing: int) -> tuple[int, int, int]:
        """(homing, range_min, range_max) for a swept joint. `homing` is the one in the registers
        during the sweep; the new one puts the middle of the stops at HALF_TURN. LeRobot's rule,
        Present_Position = Actual_Position - Homing_Offset, so moving the homing by k moves every
        reading by -k."""
        lo, hi = self.found[joint]
        margin = int(round(MARGIN_DEG * TICKS_PER_DEG))
        # WHY none at the gripper's closed end (range_min, its 0): gripping is pressing it closed.
        # Pulled in, 0 would leave the jaws about 2 degrees apart, too wide to pinch cloth; a hand
        # calibration squeezes it shut. LeRobot caps the gripper's torque for that (so_follower.py
        # configure: Max_Torque_Limit 500, Overload_Torque 25).
        closed = 0 if joint == "gripper" else margin
        shift = int(round((lo + hi) / 2)) - HALF_TURN
        return homing + shift, lo + closed - shift, hi - margin - shift

    def view(self) -> dict[str, Any]:
        s = self.current
        return {
            "state": self.state, "why": self.why, "joints": self.joints,
            "step": len(self.program) - len(self.todo), "steps": len(self.program),
            "joint": s.joint if s else None, "phase": s.phase if s else None,
            "found": {j: {"lo": lo, "hi": hi, "deg": round((hi - lo) / TICKS_PER_DEG, 1)}
                      for j, (lo, hi) in self.found.items()},
            "notes": dict(self.notes),
        }  # fmt: skip


def program(joints: list[str], start: dict[str, int], mid: dict[str, int],
            unfold: bool = True, trusted: set[str] | None = None) -> list[Step]:
    """The whole run for one arm, in raw ticks. unfold: from `start` into the middle pose (UNFOLD;
    GRIPPER_CLEAR when only the gripper sweeps) and, at the end, back to `start`, so the arm
    finishes resting where it began.
    Each joint in ORDER sweeps both sides; where SWEEP_POSES names a pose for a side, the other
    joints move into it first and back to the middle after. `mid`: each joint's middle in raw
    ticks; a pose's degrees are counted from it (Studio's model map: one degree, one degree).
    trusted: the joints whose middle is known (default every joint in `mid`); a sweep pose that
    would move another one is left out, since that joint's ticks mean nothing yet."""
    trusted = set(mid) if trusted is None else trusted
    swept = [j for j in ORDER if j in joints]
    # WHY a gripper-only run moves only GRIPPER_CLEAR: it is the careful first motion on an arm, and
    # needs no more than the jaw off the table.
    arm_joints = any(j != "gripper" for j in swept)
    stages = (UNFOLD if arm_joints else GRIPPER_CLEAR) if unfold else ()
    moves = [(j, mid[j] if d is None else mid[j] + round(d * TICKS_PER_DEG))
             for j, d in stages if j in start and j in mid]  # fmt: skip
    at = dict(start)
    back: list[tuple[str, int, bool]] = []  # (joint, where the move started, its first move)
    for j, to in moves:
        back.append((j, at[j], j not in {b[0] for b in back}))
        at[j] = to
    out: list[Step] = [Step("pose", j, to=to) for j, to in moves]
    for j in swept:
        sides = {s: SWEEP_POSES.get((j, s)) for s in (1, -1)}
        sides = {s: p if p and set(p) <= trusted else None for s, p in sides.items()}
        if not any(sides.values()):
            out.append(Step("sweep", j))
            continue
        for s in (1, -1):
            pose = sides[s] or {}
            out += [Step("pose", k, to=mid[k] + round(v * TICKS_PER_DEG)) for k, v in pose.items()]
            out.append(Step("sweep", j, sign=s))
            out += [Step("pose", k, to=mid[k]) for k in reversed(list(pose))]
    # A joint's last move back lands on rest ("rest": blocked there is its rest contact, no fault)
    out += [Step("rest" if first else "pose", j, to=to) for j, to, first in reversed(back)]
    return out


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
