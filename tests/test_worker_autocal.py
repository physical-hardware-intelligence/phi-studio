"""Auto-calibration through the worker on the mock rig: every arm finds its own stops at once,
and every way out (Stop, a lost heartbeat, a fault, Cancel) holds the arms and never writes
registers under torque."""

from __future__ import annotations

from phi_studio import autocal
from phi_studio.identity import JointCal, load_calibration
from phi_studio.mock import TICKS_PER_DEG, FakeClock, mock_rig
from phi_studio.worker import RigWorker

HZ = 30
DT = 1 / HZ


def make(tmp_path=None, pairs: int = 1) -> tuple[RigWorker, FakeClock, list[dict]]:
    clock = FakeClock()
    rig = mock_rig(pairs=pairs, cameras=(), clock=clock)
    out: list[dict] = []
    w = RigWorker(rig, send=out.append, clock=clock, loop_hz=HZ,
                  cal_dir=tmp_path / "cal" if tmp_path else None)  # fmt: skip
    w.handle({"cmd": "connect"})
    return w, clock, out


def tick(w: RigWorker, clock: FakeClock, seconds: float, heartbeat: bool = True) -> None:
    for _ in range(int(seconds * HZ)):
        if heartbeat:
            w.handle({"cmd": "heartbeat"})
        clock.advance(DT)
        w.tick()


def errors(out: list[dict]) -> list[str]:
    return [m["message"] for m in out if m.get("type") == "error"]


def named(w: RigWorker, name: str):
    return next(a for a in w.rig.arms if a.name == name)


def finish(w: RigWorker, clock: FakeClock, limit_s: float = 400.0) -> None:
    t = 0.0
    while any(c.step == "auto" for c in w.autos) and t < limit_s:
        tick(w, clock, 1.0)
        t += 1.0
    assert w.autos and all(c.step in ("auto-review", "auto-failed") for c in w.autos), [
        c.view()["auto"] for c in w.autos
    ]


def test_a_follower_finds_its_own_stops_and_saves_a_centred_calibration(tmp_path) -> None:
    w, clock, out = make(tmp_path)
    f = named(w, "follower")
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    assert w.session.state.name == "CALIBRATING" and w.autos[0].step == "auto-middle"
    assert not f.torque  # nothing moves before the person says the arm is in the middle pose
    w.handle({"cmd": "autocal_go"})
    assert f.torque and w.autos[0].step == "auto", errors(out)
    finish(w, clock)
    found = w.autos[0].auto.found
    assert set(found) == {"gripper", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"}
    for j, (lo, hi) in found.items():  # each stop where the mock's mechanics put it
        got = (hi - lo) / TICKS_PER_DEG
        want = f.stops[j][1] - f.stops[j][0]
        assert abs(got - want) < 1.0, (j, got, want)
    assert all(v == 1000 for v in f.torque_limit.values())  # every limit back to full
    w.handle({"cmd": "autocal_save"})
    saved = next(m for m in out if m.get("type") == "calibrated")
    assert not f.torque  # released before the registers were written
    cal = load_calibration(saved["path"])
    margin = round(autocal.MARGIN_DEG * autocal.TICKS_PER_DEG)
    for j in found:
        c = cal[j]
        assert abs((c.range_min + c.range_max) / 2 - autocal.HALF_TURN) <= margin / 2 + 1, j
        lo, hi = found[j]
        inset = margin if j == "gripper" else 2 * margin  # gripper: closed end not pulled in
        assert c.range_max - c.range_min == hi - lo - inset, j
    assert (cal["wrist_roll"].range_min, cal["wrist_roll"].range_max) == (0, 4095)
    assert not w.autos and not errors(out)
    ident = [m for m in out if m.get("type") == "identity"][-1]["arms"]
    assert next(a for a in ident if a["name"] == "follower")["ok"]  # the new file matches


def test_every_arm_at_once_leaders_too_and_pans_take_turns(tmp_path) -> None:
    w, clock, out = make(tmp_path, pairs=2)
    w.handle({"cmd": "autocal_start"})  # default: every arm
    assert len(w.autos) == 4
    w.handle({"cmd": "autocal_go"})
    panning_at_once = 0
    t = 0.0
    while any(c.step == "auto" for c in w.autos) and t < 600:
        tick(w, clock, 0.25)
        t += 0.25
        moving = [
            c
            for c in w.autos
            if c.auto.state == "running"
            and c.auto.current
            and c.auto.current.joint == "shoulder_pan"
            and c.limited == "shoulder_pan"
        ]
        panning_at_once = max(panning_at_once, len(moving))
    assert all(c.step == "auto-review" for c in w.autos), errors(out)
    assert panning_at_once == 1  # never two pans sweeping together
    w.handle({"cmd": "autocal_save"})
    assert len([m for m in out if m.get("type") == "calibrated"]) == 4
    assert not any(a.torque for a in w.rig.arms) and not errors(out)


def test_the_gripper_alone_changes_only_the_gripper(tmp_path) -> None:
    w, clock, out = make(tmp_path)
    f = named(w, "follower")
    before = f.read_calibration()
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    finish(w, clock)
    w.handle({"cmd": "autocal_save"})
    after = f.read_calibration()
    assert after["gripper"] != before["gripper"]
    assert all(after[j] == before[j] for j in before if j != "gripper")


def test_a_run_with_a_note_keeps_the_old_calibration_unless_accepted(tmp_path) -> None:
    """2026-10-08: the gripper's jaw met the table at 63 % open and the run said so. Save must not
    make that 63 % the arm's 100 % unless the person says it is right."""
    for accept, saved in (([], False), (["follower"], True)):
        w, clock, out = make(tmp_path / str(saved))
        f = named(w, "follower")
        before = f.read_calibration()
        w.handle({"cmd": "autocal_start", "arms": ["follower"]})
        w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
        finish(w, clock)
        w.autos[0].auto.notes["gripper"] = "travel 84 degrees, its last calibration spanned 133"
        w.handle({"cmd": "autocal_save", "accept": accept})
        assert (f.read_calibration()["gripper"] != before["gripper"]) is saved
        assert any(m.get("type") == "calibrated" for m in out) is saved
        assert not w.autos and not any(a.torque for a in w.rig.arms) and not errors(out)


def test_save_refuses_an_accept_that_names_no_arm_being_calibrated(tmp_path) -> None:
    w, clock, out = make(tmp_path)
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    finish(w, clock)
    w.handle({"cmd": "autocal_save", "accept": ["leader"]})
    assert errors(out) and w.autos  # nothing saved, the run still waits for a decision


def test_the_swept_joint_presses_through_a_lowered_torque_limit() -> None:
    w, clock, _ = make()
    f = named(w, "follower")
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper", "elbow_flex"], "torque": 300})
    tick(w, clock, 0.5)
    assert f.torque_limit["gripper"] == autocal.sweep_limit("gripper")  # its own cap: 200
    assert f.torque_limit["elbow_flex"] == 1000  # waits its turn at full torque, holding
    c = w.autos[0]
    while c.auto.current and c.auto.current.joint == "gripper":
        tick(w, clock, 0.5)
    assert f.torque_limit["gripper"] == 1000
    # A partial run from a hand-posed middle: the shoulder's middle is unknown, so the elbow sweeps
    # from that pose (no sweep pose moves an untrusted joint).
    assert c.auto.current.joint == "elbow_flex"
    assert f.torque_limit["elbow_flex"] == 300  # the torque asked for


def test_stop_pauses_where_the_arm_is_and_resume_goes_on() -> None:
    w, clock, out = make()
    f = named(w, "follower")
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    tick(w, clock, 1.0)
    w.handle({"cmd": "stop"})
    tick(w, clock, 0.2)
    still = dict(f.pos)
    tick(w, clock, 2.0)
    assert f.pos == still and w.autos[0].auto.state == "paused" and f.torque  # holds, no drop
    w.handle({"cmd": "autocal_resume"})
    finish(w, clock)
    assert w.autos[0].step == "auto-review" and not errors(out)


def test_a_silent_window_pauses_the_sweep() -> None:
    w, clock, _ = make()
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    tick(w, clock, 0.5)
    tick(w, clock, 1.5, heartbeat=False)
    a = w.autos[0].auto
    assert a.state == "paused" and "stopped answering" in (a.why or "")


def test_go_refuses_without_heartbeats() -> None:
    w, clock, out = make()
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    clock.advance(5.0)
    w.handle({"cmd": "autocal_go"})
    assert w.autos[0].step == "auto-middle" and not named(w, "follower").torque
    assert "heartbeats" in errors(out)[-1]


def test_a_fault_holds_the_arms_and_restores_registers_only_once_torque_is_off() -> None:
    w, clock, out = make()
    f, lead = named(w, "follower"), named(w, "leader")
    old_f, old_l = f.read_calibration(), lead.read_calibration()
    w.handle({"cmd": "autocal_start"})
    w.handle({"cmd": "autocal_go"})
    tick(w, clock, 1.0)
    w.handle({"cmd": "inject", "arm": "follower", "kind": "overload", "joint": "gripper"})
    tick(w, clock, 1.0)
    assert w.session.state.name == "FAULT" and f.torque and lead.torque  # frozen, holding
    assert f.read_calibration() != old_f  # not written under torque
    w.handle({"cmd": "release"})
    assert not f.torque and not lead.torque
    assert f.read_calibration() == old_f and lead.read_calibration() == old_l


def test_cancel_releases_then_restores() -> None:
    w, clock, _ = make()
    f = named(w, "follower")
    old = f.read_calibration()
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go"})
    tick(w, clock, 1.0)
    w.handle({"cmd": "autocal_cancel"})
    assert not f.torque and f.read_calibration() == old and not w.autos
    assert w.session.state.name in ("IDENTIFIED", "READY")


def test_disconnect_mid_sweep_releases_and_restores() -> None:
    w, clock, _ = make()
    f = named(w, "follower")
    old = f.read_calibration()
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go"})
    tick(w, clock, 1.0)
    w.handle({"cmd": "disconnect"})
    assert not f.torque and f.read_calibration() == old


def test_a_wrong_start_pose_fails_and_save_restores_that_arm() -> None:
    w, clock, out = make()
    f = named(w, "follower")
    # A servo fresh from the box (0..4095): nothing says where its middle is, so a person poses it.
    f._cal = f._own_cal = {j: JointCal(c.id, 0, 0, 0, 4095) for j, c in f._cal.items()}
    old = f.read_calibration()
    lo, hi = f.stops["shoulder_lift"]
    f.pos["shoulder_lift"] = hi - 2.0  # right by a stop: half the range lies past the turn's end
    f.goal["shoulder_lift"] = f.pos["shoulder_lift"]
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["shoulder_lift"]})
    finish(w, clock)
    assert w.autos[0].step == "auto-failed" and "middle pose" in (w.autos[0].auto.why or "")
    w.handle({"cmd": "autocal_save"})  # nothing to save for it: its old registers go back
    assert f.read_calibration() == old and not f.torque


def test_by_hand_and_auto_never_overlap() -> None:
    w, _, out = make()
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "cal_start", "arm": "leader"})
    assert w.cal is None and "auto-calibration is running" in errors(out)[-1].lower()


def new_arms(tmp_path, *names: str, pairs: int = 1) -> tuple[RigWorker, FakeClock, list[dict]]:
    """A rig where `names` have no calibration file yet, as arms fresh from the box (or a new
    calibration id) on the real rig."""
    clock = FakeClock()
    rig = mock_rig(pairs=pairs, cameras=(), clock=clock)
    for a in rig.arms:
        a.calibrated = a.name not in names
    out: list[dict] = []
    w = RigWorker(rig, send=out.append, clock=clock, loop_hz=HZ, cal_dir=tmp_path / "cal")
    w.handle({"cmd": "connect"})
    return w, clock, out


def test_an_arm_with_no_calibration_file_is_not_a_dead_arm(tmp_path) -> None:
    """2026-10-05, first connect on the real rig with a new calibration id: the loop read the
    uncalibrated leader in degrees, took the refusal for a dead bus and faulted the rig."""
    w, clock, out = new_arms(tmp_path, "leader")
    tick(w, clock, 1.0)
    assert w.session.state.name == "IDENTIFIED", errors(out)
    arms = [m for m in out if m.get("type") == "telemetry"][-1]["arms"]
    assert arms["leader"]["online"] and not arms["leader"]["calibrated"]
    assert arms["leader"]["pos"] == {} and arms["follower"]["pos"]  # the others still read
    w.handle({"cmd": "confirm"})  # teleop waits for a calibration
    assert w.session.state.name == "IDENTIFIED"
    leader = named(w, "leader").calibration_id
    assert errors(out)[-1] == f"Leader has no calibration file yet ({leader}.json)"


def test_a_new_arm_auto_calibrates_gripper_first_then_reads_in_degrees(tmp_path) -> None:
    """The rig plan's first motion is one follower's gripper. With no file, the joints not swept
    read in degrees through the calibration the servos still hold."""
    w, clock, out = new_arms(tmp_path, "follower")
    f = named(w, "follower")
    before = f.read_calibration()  # the registers: what LeRobot wrote last
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    assert f.torque and not errors(out), errors(out)
    finish(w, clock)
    w.handle({"cmd": "autocal_save"})
    assert not errors(out) and f.calibrated
    saved = load_calibration(next(m for m in out if m.get("type") == "calibrated")["path"])
    assert saved["gripper"] != before["gripper"]
    assert all(saved[j] == before[j] for j in before if j != "gripper")
    tick(w, clock, 0.5)
    arms = [m for m in out if m.get("type") == "telemetry"][-1]["arms"]
    assert arms["follower"]["calibrated"] and arms["follower"]["pos"]
    ident = [m for m in out if m.get("type") == "identity"][-1]["arms"]
    assert next(a for a in ident if a["name"] == "follower")["ok"]  # its new file matches


def test_a_fault_holds_a_follower_with_no_calibration_in_raw_ticks(tmp_path) -> None:
    w, clock, out = new_arms(tmp_path, "follower")
    f = named(w, "follower")
    f.torque, w.torque["follower"] = True, True  # powered up holding, as the real servos do
    w._freeze()
    assert w.session.state.name != "FAULT" and "follower" not in w.dead, errors(out)
    assert f.torque


def test_a_calibrated_arm_recalibrates_from_its_rest_pose_to_the_same_result(tmp_path) -> None:
    """2026-10-05: an arm whose servos already hold a calibration needs no middle pose. From a
    pose leaning on its stops, a second sweep must land on the first one's calibration."""
    w, clock, out = make(tmp_path)
    f = named(w, "follower")
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    assert not w.autos[0].anywhere  # the mock's first registers do not fit its mechanics
    w.handle({"cmd": "autocal_go"})
    finish(w, clock)
    w.handle({"cmd": "autocal_save"})
    first = f.read_calibration()
    for j in ("shoulder_lift", "elbow_flex", "wrist_flex"):  # a folded rest pose, near the stops
        f.pos[j] = f.goal[j] = f.stops[j][0] + 2.0
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    assert w.autos[0].anywhere, errors(out)
    w.handle({"cmd": "autocal_go"})
    assert f.read_calibration() == first  # no homing: the registers are untouched
    finish(w, clock)
    w.handle({"cmd": "autocal_save"})
    second = f.read_calibration()
    assert not errors(out)
    for j, c in first.items():
        d = second[j]
        assert abs(d.homing_offset - c.homing_offset) <= 2, (j, c, d)
        assert abs(d.range_min - c.range_min) <= 2, (j, c, d)
        assert abs(d.range_max - c.range_max) <= 2, (j, c, d)


def test_starts_anywhere_only_with_a_calibration_that_fits_the_pose() -> None:
    from phi_studio.identity import JointCal
    from phi_studio.worker import starts_anywhere

    hand = {"elbow_flex": JointCal(3, 0, 1291, 816, 3023),
            "wrist_roll": JointCal(5, 0, 603, 0, 4095)}  # fmt: skip
    both = ["elbow_flex", "wrist_roll"]
    assert starts_anywhere(hand, {"elbow_flex": 900, "wrist_roll": 4000}, both)
    assert starts_anywhere(hand, {"elbow_flex": 700, "wrist_roll": 0}, both)  # leaning on a stop
    assert not starts_anywhere(hand, {"elbow_flex": 300, "wrist_roll": 0}, both)  # not this arm's
    new = {"elbow_flex": JointCal(3, 0, 0, 0, 4095)}  # a servo fresh from the box
    assert not starts_anywhere(new, {"elbow_flex": 2047}, ["elbow_flex"])
    edge = {"elbow_flex": JointCal(3, 0, 0, 50, 2500)}  # a stop near the wrap
    assert not starts_anywhere(edge, {"elbow_flex": 1000}, ["elbow_flex"])


def test_a_sweep_short_of_the_last_calibration_is_flagged() -> None:
    e = autocal.AutoCal(["gripper"], {"gripper": 2047}, 0.0, expected_deg={"gripper": 110.0},
                        previous_deg={"gripper": 132.9})  # fmt: skip
    e.found["gripper"] = (1547, 2547)  # 88 degrees: over the model's floor, under the last range
    e._check_travel("gripper")
    assert e.notes["gripper"] == ("travel 88 degrees, its last calibration spanned 133: something "
                                  "stopped it early?")  # fmt: skip


def test_a_leader_voltage_dip_pauses_the_sweep_and_resume_finishes(tmp_path) -> None:
    """2026-10-05: the right leader's 5 V supply sagged under its servos' cutoff while its gripper
    pressed into a stop, and the whole rig faulted. A leader carries nothing: pause, say it."""
    w, clock, out = make(tmp_path)
    lead = named(w, "leader")
    w.handle({"cmd": "autocal_start", "arms": ["leader"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    tick(w, clock, 0.5)
    lead.faults["gripper"] = 1  # Status bit 0, voltage
    tick(w, clock, 1.0)
    lead.faults.clear()
    run = w.autos[0]
    assert w.session.state.name == "CALIBRATING", errors(out)
    assert run.auto.state == "paused" and "supply dipped" in run.auto.why
    assert len([e for e in errors(out) if "supply dipped" in e]) == 1  # said once, not per read
    assert lead.torque  # holding where it is
    w.handle({"cmd": "autocal_resume"})
    finish(w, clock)
    assert run.step == "auto-review"


def test_a_follower_voltage_fault_still_faults_the_rig(tmp_path) -> None:
    w, clock, out = make(tmp_path)
    named(w, "follower").faults["gripper"] = 1
    tick(w, clock, 1.0)
    assert w.session.state.name == "FAULT"


def test_one_lost_health_read_does_not_stop_a_sweep_three_in_a_row_do(tmp_path) -> None:
    """2026-10-05: one status read got no reply while the follower's motors started, and the whole
    run faulted at step 4 of 21. Health reads are diagnostics; positions still guard the arm."""
    w, clock, out = make(tmp_path)
    f = named(w, "follower")
    real = f.read_health
    misses = {"left": 1}

    def flaky():
        if misses["left"] > 0:
            misses["left"] -= 1
            raise ConnectionError("Failed to sync read 'Present_Load': There is no status packet!")
        return real()

    f.read_health = flaky
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    tick(w, clock, 2.0)
    assert w.session.state.name == "CALIBRATING" and w.autos[0].step == "auto", errors(out)
    misses["left"] = 3
    tick(w, clock, 2.0)
    assert w.session.state.name == "FAULT"


def test_a_failed_run_keeps_the_joint_at_its_lowered_limit(tmp_path) -> None:
    """A joint that met something must not get full torque back the moment its run fails."""
    w, clock, out = make(tmp_path)
    f = named(w, "follower")
    w.handle({"cmd": "autocal_start", "arms": ["follower"]})
    w.handle({"cmd": "autocal_go", "joints": ["gripper"]})
    tick(w, clock, 0.3)
    run = w.autos[0]
    assert run.limited == "gripper" and f.torque_limit["gripper"] == autocal.sweep_limit("gripper")
    run.auto._fail("blocked", hold=float(run.pos["gripper"]))
    tick(w, clock, 1.0)
    assert run.step == "auto-failed"
    assert f.torque_limit["gripper"] == autocal.sweep_limit("gripper")  # not 1000


def test_a_held_joint_straining_for_ten_seconds_gets_its_torque_lowered(tmp_path) -> None:
    """2026-10-05: a fault froze the follower with its shoulder in a jam at 70 % for two minutes.
    Studio only holds an arm in these states, so a sustained push means something is in the way."""
    from phi_studio import worker as W

    w, clock, out = make(tmp_path)
    f = named(w, "follower")
    w.handle({"cmd": "confirm"})
    w.handle({"cmd": "arm"})  # torque on, holding: not MOVING
    assert f.torque
    lo, hi = f.stops["elbow_flex"]
    f.pos["elbow_flex"] = hi - 1.0
    f.goal["elbow_flex"] = hi + 40.0  # pressed into its stop: the mock reports a high load
    tick(w, clock, W.STRAIN_S - 2)
    assert f.torque_limit["elbow_flex"] == 1000  # not yet: a moment's push is fine
    tick(w, clock, 4.0)
    assert f.torque_limit["elbow_flex"] == W.RELAX_LIMIT
    assert any("pushed against something" in e for e in errors(out))
    w.handle({"cmd": "release"})
    tick(w, clock, 0.2)
    assert not f.torque and f.torque_limit["elbow_flex"] == 1000  # back to full once released
