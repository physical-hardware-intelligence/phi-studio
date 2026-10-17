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


def test_the_gripper_closes_all_the_way_its_closed_stop_is_not_pulled_in() -> None:
    # 2026-10-08: pulled in by MARGIN_DEG, 0 left the jaws ~2 degrees apart; a hand calibration
    # squeezes the gripper shut, and gripping cloth needs it shut.
    joints = arm(gripper=(2047, 1800, 2300))
    cal = A.AutoCal(["gripper"], {j: int(x.pos) for j, x in joints.items()}, now=0.0)
    run(cal, joints, 0.0)
    homing, rmin, rmax = cal.range("gripper", homing=0)
    lo, hi = cal.found["gripper"]
    margin = round(A.MARGIN_DEG * A.TICKS_PER_DEG)
    assert lo - homing == rmin  # 0 is the closed stop itself
    assert hi - homing == rmax + margin  # the open end keeps its margin


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


def test_an_arm_drives_itself_into_the_middle_pose_then_finds_the_same_stops() -> None:
    """From a folded rest pose (each joint near one stop), the arm unfolds into the middle pose in
    POSE_ORDER, then sweeps: the stops match a run started from the middle by hand."""
    stops = {"shoulder_lift": (900, 3200), "elbow_flex": (800, 3000), "gripper": (1500, 3000)}
    folded = arm(**{j: (lo + 40, lo, hi) for j, (lo, hi) in stops.items()})
    start = {j: int(x.pos) for j, x in folded.items()}
    target = {j: (lo + hi) // 2 for j, (lo, hi) in stops.items()}
    cal = A.AutoCal(list(stops), start, now=0.0, target=target)
    assert (cal.current.phase, cal.current.joint) == ("pose", "elbow_flex")  # the forearm out first
    posed = []
    t, dt = 0.0, 1 / HZ
    while cal.state == "running" and cal.current and cal.current.phase == "pose":
        if not posed or posed[-1] != cal.current.joint:
            posed.append(cal.current.joint)
        goals = cal.step({j: int(round(x.pos)) for j, x in folded.items()}, t)
        for j, x in folded.items():
            x.move(goals[j], dt)
        t += dt
    assert posed == ["elbow_flex", "shoulder_lift", "gripper"]
    for j in stops:
        assert abs(folded[j].pos - target[j]) <= A.ARRIVED_TICKS, j
    run(cal, folded, t)
    assert cal.state == "done", cal.why
    for j, (lo, hi) in stops.items():
        flo, fhi = cal.found[j]
        assert abs(flo - lo) <= A.STILL_TICKS and abs(fhi - hi) <= A.STILL_TICKS, j
        assert abs(folded[j].pos - target[j]) <= A.ARRIVED_TICKS, j  # home is the middle pose now


def test_a_blocked_way_to_the_middle_pose_stops_and_says_which_joint() -> None:
    joints = arm(elbow_flex=(900, 850, 1500))  # something holds the forearm at 1500
    cal = A.AutoCal(["elbow_flex"], {j: int(x.pos) for j, x in joints.items()}, now=0.0,
                    target={"elbow_flex": 1950})  # fmt: skip
    run(cal, joints, 0.0, limit_s=60)
    assert cal.state == "failed" and "short of where it was going" in cal.why
    assert not cal.found  # nothing measured from a blocked pose


def test_home_ends_when_a_start_pressed_into_a_stop_is_out_of_reach() -> None:
    """2026-10-05: an elbow that started leaning into its stop came back 20 ticks short at the
    sweep's torque and waited for good. Short and still is home."""
    joints = arm(elbow_flex=(980, 1000, 3000))  # started 20 ticks into a stop it can't reach now
    start = {j: int(x.pos) for j, x in joints.items()}
    cal = A.AutoCal(["elbow_flex"], start, now=0.0)
    run(cal, joints, 0.0, limit_s=120)
    assert cal.state == "done", cal.why
    assert abs(cal.found["elbow_flex"][0] - 1000) <= A.STILL_TICKS
    held = cal.hold["elbow_flex"]
    assert abs(held - joints["elbow_flex"].pos) <= A.STILL_TICKS  # holds where it is, not pressing


class HeavyJoint(Joint):
    """A joint that lifts the arm: its weight takes `hold` ticks of the servo's push, which grows
    with how far the goal is ahead (P control). Lowering is free. 2026-10-05: the real shoulder
    stalled with its goal 6 degrees ahead, 17 degrees up from rest."""

    def __init__(self, pos: float, lo: float, hi: float, hold: float) -> None:
        super().__init__(pos, lo, hi)
        self.hold = hold

    def move(self, goal: float, dt: float) -> None:
        err = goal - self.pos
        if err > 0:  # lifting: the weight eats part of the push
            err = max(0.0, err - self.hold)
        step = self.speed * dt
        self.pos += max(-step, min(step, err))
        self.pos = min(self.hi, max(self.lo, self.pos))


def test_the_shoulder_lifts_the_arm_into_the_middle_pose_and_sweeps_against_its_weight() -> None:
    hold = 8 * A.TICKS_PER_DEG  # more than the old 6-degree lead, less than the shoulder's
    joints = arm()
    joints["shoulder_lift"] = HeavyJoint(940, 900, 3200, hold)  # at rest, near its low stop
    start = {j: int(x.pos) for j, x in joints.items()}
    cal = A.AutoCal(["shoulder_lift"], start, now=0.0, target={"shoulder_lift": 2050})
    run(cal, joints, 0.0)
    assert cal.state == "done", cal.why
    lo, hi = cal.found["shoulder_lift"]
    assert abs(lo - 900) <= A.STILL_TICKS and abs(hi - 3200) <= A.STILL_TICKS
    assert A.lead_deg("shoulder_lift") * A.TICKS_PER_DEG > hold > A.LEAD_DEG * A.TICKS_PER_DEG


def test_a_blocked_move_backs_off_its_obstacle_and_stays_backed_off() -> None:
    """2026-10-05: a blocked shoulder held where it stopped, pressed into a jam, for two minutes.
    Now the joint's goal moves BACKOFF_DEG back from where it stopped, and stays there."""
    joints = arm(elbow_flex=(900, 850, 1500))  # something holds the forearm at 1500
    cal = A.AutoCal(["elbow_flex"], {j: int(x.pos) for j, x in joints.items()}, now=0.0,
                    target={"elbow_flex": 1950})  # fmt: skip
    t = run(cal, joints, 0.0, limit_s=60)
    assert cal.state == "failed"
    backed = 1500 - A.BACKOFF_DEG * A.TICKS_PER_DEG
    assert abs(cal.hold["elbow_flex"] - backed) < 1
    for _ in range(30):  # later ticks keep the backed-off goal, not where it pressed
        goals = cal.step({j: int(round(x.pos)) for j, x in joints.items()}, t)
        t += 1 / HZ
    assert abs(goals["elbow_flex"] - backed) <= 1


def test_step_through_pauses_after_each_unfold_move_then_runs_on() -> None:
    """The first runs on an arm: a person sees each unfold move before the next one starts. The
    sweeps run on by themselves."""
    stops = {"shoulder_lift": (900, 3200), "elbow_flex": (800, 3000)}
    joints = arm(**{j: (lo + 40, lo, hi) for j, (lo, hi) in stops.items()})
    start = {j: int(x.pos) for j, x in joints.items()}
    target = {j: (lo + hi) // 2 for j, (lo, hi) in stops.items()}
    cal = A.AutoCal(list(stops), start, now=0.0, target=target, step_through=True)
    t, pauses = 0.0, []
    while cal.state in ("running", "paused") and t < 300:
        if cal.state == "paused":
            pauses.append(cal.why)
            cal.resume({j: int(round(x.pos)) for j, x in joints.items()}, t)
        goals = cal.step({j: int(round(x.pos)) for j, x in joints.items()}, t)
        for j, x in joints.items():
            x.move(goals[j], 1 / HZ)
        t += 1 / HZ
    assert cal.state == "done", cal.why
    unfolded = [j for j in A.POSE_ORDER if j in target]
    assert len(pauses) == len(unfolded)  # one after each unfold move, none in the sweeps
    assert all(f"Unfold move done: {j}" in w for j, w in zip(unfolded, pauses, strict=True))


def test_the_wrist_unfolds_in_two_stages_and_folds_back_through_them() -> None:
    """2026-10-08, simulation pair by pair: swung straight to its middle the wrist passed the
    shoulder bracket at 9.9 mm; out to WRIST_OUT_DEG, then the middle once the elbow has opened."""
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    start = {j: 1000 for j in names}
    mid = {j: 2047 for j in start}
    out = 2047 + round(A.WRIST_OUT_DEG * A.TICKS_PER_DEG)
    steps = A.program(["elbow_flex"], start, mid, unfold=True)
    unfold = [(st.kind, st.joint, st.to) for st in steps[:7]]
    assert unfold == [("pose", "wrist_flex", out), ("pose", "shoulder_pan", 2047),
                      ("pose", "elbow_flex", 2047), ("pose", "wrist_flex", 2047),
                      ("pose", "shoulder_lift", 2047), ("pose", "wrist_roll", 2047),
                      ("pose", "gripper", 2047)]  # fmt: skip
    fold = [(st.kind, st.joint, st.to) for st in steps[-7:]]
    assert fold == [("rest", "gripper", 1000), ("rest", "wrist_roll", 1000),
                    ("rest", "shoulder_lift", 1000), ("pose", "wrist_flex", out),
                    ("rest", "elbow_flex", 1000), ("rest", "shoulder_pan", 1000),
                    ("rest", "wrist_flex", 1000)]  # fmt: skip


def test_a_gripper_only_run_lifts_the_jaw_off_the_table_first() -> None:
    """The careful first motion on an arm: the wrist lifts the jaw clear, the gripper sweeps, the
    wrist goes back. 2026-10-08: swept at rest, the jaw met the table at 63 % open."""
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    start = {j: 1000 for j in names}
    mid = {j: 2047 for j in start}
    steps = A.program(["gripper"], start, mid, unfold=True)
    out = 2047 + round(A.WRIST_OUT_DEG * A.TICKS_PER_DEG)
    assert steps == [A.Step("pose", "wrist_flex", to=out), A.Step("sweep", "gripper"),
                     A.Step("rest", "wrist_flex", to=1000)]  # fmt: skip
    # posed by hand (a new arm): it is already in the middle pose, nothing moves but the jaws
    assert A.program(["gripper"], start, mid, unfold=False) == [A.Step("sweep", "gripper")]
    full = A.program(["gripper", "elbow_flex"], start, mid, unfold=True)
    assert full[0].kind == "pose" and full[-1].kind == "rest"  # an arm sweep still unfolds, folds
