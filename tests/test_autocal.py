"""Auto-calibration against joints with hard stops: the stops it finds, the calibration it makes,
and every way it refuses to guess."""

from __future__ import annotations

import pytest

from phi_studio import autocal as A

HZ = 30.0


class Joint:
    """A position-controlled joint between two hard stops, chasing its goal at a capped speed."""

    def __init__(self, pos: float, lo: float, hi: float, speed_deg_s: float = 120.0) -> None:
        self.pos, self.lo, self.hi = pos, lo, hi
        self.speed = speed_deg_s * A.TICKS_PER_DEG

    def move(self, goal: float, dt: float) -> None:
        step = self.speed * dt
        self.pos += max(-step, min(step, goal - self.pos))
        self.pos = min(self.hi, max(self.lo, self.pos))


def run(cal: A.AutoCal, joints: dict[str, Joint], t: float, limit_s: float = 300.0) -> float:
    dt = 1 / HZ
    while cal.state == "running" and limit_s > 0:
        raw = {j: int(round(x.pos)) for j, x in joints.items()}
        goals = cal.step(raw, t)
        for j, x in joints.items():
            x.move(goals[j], dt)
        t += dt
        limit_s -= dt
    return t


def arm(**stops: tuple[float, float, float]) -> dict[str, Joint]:
    """joint -> (start, lo, hi) in ticks; joints not given sit still at 2047 with wide stops."""
    out = {j: Joint(2047, 200, 3900) for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                              "wrist_flex", "wrist_roll", "gripper")}  # fmt: skip
    for j, (p, lo, hi) in stops.items():
        out[j] = Joint(p, lo, hi)
    return out


def test_finds_both_stops_returns_home_and_holds_the_others() -> None:
    joints = arm(gripper=(2047, 1500, 3000), elbow_flex=(2100, 1000, 3100))
    start = {j: int(x.pos) for j, x in joints.items()}
    cal = A.AutoCal(["gripper", "elbow_flex"], start, now=0.0)
    run(cal, joints, 0.0)
    assert cal.state == "done", cal.why
    lo, hi = cal.found["gripper"]
    assert abs(lo - 1500) <= A.STILL_TICKS and abs(hi - 3000) <= A.STILL_TICKS
    lo, hi = cal.found["elbow_flex"]
    assert abs(lo - 1000) <= A.STILL_TICKS and abs(hi - 3100) <= A.STILL_TICKS
    for j, x in joints.items():  # every joint back where it started: the middle pose
        assert abs(x.pos - start[j]) <= A.ARRIVED_TICKS, j
    assert list(cal.found) == ["gripper", "elbow_flex"]  # ORDER: the gripper first


def test_the_calibration_centres_the_stops_on_the_half_turn() -> None:
    joints = arm(wrist_flex=(1900, 900, 2700))
    cal = A.AutoCal(["wrist_flex"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    run(cal, joints, 0.0)
    homing, rmin, rmax = cal.range("wrist_flex", homing=-300)
    lo, hi = cal.found["wrist_flex"]
    margin = round(A.MARGIN_DEG * A.TICKS_PER_DEG)
    assert (rmin + rmax) / 2 == pytest.approx(A.HALF_TURN, abs=1)
    assert rmax - rmin == hi - lo - 2 * margin
    # Present = Actual - Homing: the stop at `hi` now reads hi - (homing_new - homing_old).
    assert hi - (homing - -300) == rmax + margin


def test_stopping_is_gentle_it_backs_off_the_stop_at_once() -> None:
    joints = arm(gripper=(2047, 1800, 2300))
    cal = A.AutoCal(["gripper"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    t, worst = 0.0, 0.0
    while cal.state == "running" and t < 60:
        raw = {j: int(round(x.pos)) for j, x in joints.items()}
        g = cal.step(raw, t)["gripper"]
        worst = max(worst, abs(g - joints["gripper"].pos) / A.TICKS_PER_DEG)
        for j, x in joints.items():
            x.move(cal.hold[j], 1 / HZ)
        t += 1 / HZ
    assert cal.state == "done"
    assert worst <= A.LEAD_DEG + 0.1  # the goal never leads the joint by more than LEAD_DEG


def test_no_stop_within_the_travel_cap_fails() -> None:
    joints = arm(shoulder_pan=(2047, -10_000, 10_000))  # no stops at all
    cal = A.AutoCal(["shoulder_pan"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    run(cal, joints, 0.0)
    assert cal.state == "failed" and "wrap" in (cal.why or "")  # it hits the encoder's end first


def test_a_range_that_would_wrap_is_refused() -> None:
    joints = arm(shoulder_lift=(3800, 1000, 4090))
    cal = A.AutoCal(["shoulder_lift"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    run(cal, joints, 0.0)
    assert cal.state == "failed" and "middle pose" in (cal.why or "")


def test_a_joint_that_cannot_move_is_refused_not_saved_as_a_zero_range() -> None:
    class Stuck(Joint):
        def move(self, goal: float, dt: float) -> None:
            pass

    joints = arm()
    joints["elbow_flex"] = Stuck(2047, 200, 3900)
    cal = A.AutoCal(["elbow_flex"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    run(cal, joints, 0.0)
    assert cal.state == "failed" and "stuck" in (cal.why or "") and not cal.found


def test_short_travel_against_the_model_is_noted() -> None:
    joints = arm(elbow_flex=(2047, 1700, 2400))  # 61 degrees where the model has 194
    cal = A.AutoCal(["elbow_flex"], {j: int(x.pos) for j, x in joints.items()}, now=0.0,
                    expected_deg={"elbow_flex": 193.7})  # fmt: skip
    run(cal, joints, 0.0)
    assert cal.state == "done" and "in the way" in cal.notes["elbow_flex"]


def test_pause_holds_and_resume_restarts_the_direction() -> None:
    joints = arm(gripper=(2047, 1500, 3000))
    cal = A.AutoCal(["gripper"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    t = run(cal, joints, 0.0, limit_s=1.0)  # part way out
    cal.pause("Stop")
    before = dict(cal.hold)
    raw = {j: int(round(x.pos)) for j, x in joints.items()}
    assert cal.step(raw, t + 1) == {j: int(round(v)) for j, v in before.items()}  # holds
    cal.resume(raw, t + 1)
    run(cal, joints, t + 1)
    assert cal.state == "done"
    assert abs(cal.found["gripper"][1] - 3000) <= A.STILL_TICKS


def test_wrist_roll_is_never_swept() -> None:
    cal = A.AutoCal(["wrist_roll", "gripper"], {j: 2047 for j in A.ORDER} | {"wrist_roll": 2047},
                    now=0.0)  # fmt: skip
    assert cal.joints == ["gripper"]


def test_model_travel_is_read_from_the_bundle() -> None:
    t = A.expected_travel()
    if not t:
        pytest.skip("no model bundle")
    assert t["elbow_flex"] == pytest.approx(193.7, abs=0.5)


def test_a_pause_against_a_stop_holds_where_the_joint_is_not_where_the_goal_pressed() -> None:
    joints = arm(gripper=(2047, 1500, 2150))  # the + stop is 9 degrees away
    cal = A.AutoCal(["gripper"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    t = 0.0
    while cal.hold["gripper"] - joints["gripper"].pos < 3 * A.TICKS_PER_DEG and t < 10:
        t = run(cal, joints, t, limit_s=1 / HZ)  # one tick at a time, until it presses
    raw = {j: int(round(x.pos)) for j, x in joints.items()}
    cal.pause("Stop")
    assert cal.step(raw, t)["gripper"] == raw["gripper"]
