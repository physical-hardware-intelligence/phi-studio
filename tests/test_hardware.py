"""The real-arm backend against a fake Feetech bus that keeps LeRobot's register semantics: units,
torque rules, health decoding, reconnects, and the rig built from robot-config.yaml."""

from __future__ import annotations

import json
from typing import Any

import pytest

from phi_studio import hardware as H
from phi_studio.identity import JointCal
from phi_studio.rig import JOINTS

pytest.importorskip("lerobot")

CAL = {j: JointCal(i + 1, 0, -100 + 10 * i, 1000, 3000) for i, j in enumerate(JOINTS)}


class FakeBus:
    """What FeetechArm uses of FeetechMotorsBus, with LeRobot's semantics: DEGREES = (raw - mid) *
    360/4095."""

    instances: list[FakeBus] = []
    fail_connect = 0

    def __init__(self, port: str, calibration: Any) -> None:
        self.port, self.calibration = port, calibration
        self.connected = False
        self.raw = {j: 2000 for j in JOINTS}
        self.torque = {j: 0 for j in JOINTS}
        self.writes: list[tuple[str, str, int]] = []
        self.goals: list[dict[str, float]] = []
        self.normalized: list[bool] = []
        self.retries: list[int] = []
        self.regs = {
            j: {
                "Present_Load": -250,
                "Present_Temperature": 41,
                "Present_Voltage": 121,
                "Status": 0,
            }
            for j in JOINTS
        }
        FakeBus.instances.append(self)

    def connect(self, handshake: bool = True) -> None:
        if FakeBus.fail_connect:
            FakeBus.fail_connect -= 1
            raise ConnectionError("no status packet")
        self.connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False

    def _deg(self, j: str, raw: int) -> float:
        if not self.calibration:  # LeRobot's _normalize
            raise RuntimeError(f"{self.port} has no calibration registered.")
        c = self.calibration[j]
        if j == "gripper":
            return (raw - c.range_min) / (c.range_max - c.range_min) * 100
        return (raw - (c.range_min + c.range_max) / 2) * 360 / 4095

    def sync_read(self, name: str, motors: Any = None, *, normalize: bool = True,
                  num_retry: int = 0) -> dict[str, Any]:  # fmt: skip
        self.retries.append(num_retry)
        if not self.connected:
            raise ConnectionError("not connected")
        if name == "Present_Position":
            return {j: self._deg(j, r) if normalize else r for j, r in self.raw.items()}
        if name == "Torque_Enable":
            return dict(self.torque)
        return {j: self.regs[j][name] for j in JOINTS}

    def sync_write(self, name: str, values: dict[str, float], *, normalize: bool = True) -> None:
        assert name == "Goal_Position"
        self.goals.append(dict(values))
        self.normalized.append(normalize)

    def write(self, name: str, motor: str, value: int, *, normalize: bool = True) -> None:
        self.writes.append((name, motor, value))

    def enable_torque(self, motors: Any = None) -> None:
        self.torque = {j: 1 for j in JOINTS}

    def disable_torque(self, motors: Any = None) -> None:
        self.torque = {j: 0 for j in JOINTS}

    def configure_motors(self) -> None:
        self.writes.append(("configure_motors", "*", 0))

    def read_calibration(self) -> dict[str, Any]:
        """The servos' registers: the arm's own file when it has one (a correctly calibrated
        arm)."""
        from lerobot.motors import MotorCalibration

        if self.calibration:
            return dict(self.calibration)
        return {
            j: MotorCalibration(c.id, 0, c.homing_offset, c.range_min, c.range_max)
            for j, c in CAL.items()
        }

    def write_calibration(self, cal: dict[str, Any], cache: bool = True) -> None:
        if cache:
            self.calibration = cal

    def set_half_turn_homings(self, motors: Any = None) -> dict[str, int]:
        """LeRobot's: reset_calibration (homing 0, limits 0..4095, and the whole cache emptied),
        then the homing that puts the present position at 2047, for `motors` (default all). The
        raw reads follow the new homing."""
        which = list(motors) if motors is not None else list(self.raw)
        self.calibration = {}
        homings = {j: r - 2047 for j, r in self.raw.items() if j in which}
        self.raw = {j: 2047 if j in which else r for j, r in self.raw.items()}
        return homings


@pytest.fixture(autouse=True)
def fresh():
    FakeBus.instances.clear()
    FakeBus.fail_connect = 0


def arm(role: str = "follower", cal=CAL) -> H.FeetechArm:
    a = H.FeetechArm(
        f"left_{role}",
        role,
        "/dev/tty.usbmodemX",
        f"rig_{role}_left",
        dict(cal) if cal else None,
        side="left",
        bus_factory=lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None),
    )
    a.connect()
    return a


def test_motor_table_is_lerobots() -> None:
    from lerobot.motors import MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    t = H.motor_table()
    assert [m.id for m in t.values()] == [1, 2, 3, 4, 5, 6] and list(t) == list(JOINTS)
    assert (
        t["gripper"].norm_mode == MotorNormMode.RANGE_0_100
        and t["elbow_flex"].norm_mode == MotorNormMode.DEGREES
    )
    FeetechMotorsBus(
        port="/dev/null", motors=t, calibration=H.to_motor_calibration(CAL)
    )  # constructs; never opened


def test_follower_configured_with_torque_left_off() -> None:
    a = arm("follower")
    bus = FakeBus.instances[-1]
    assert a.configured and not a.read_torque()
    names = {(n, m) for n, m, _ in bus.writes}
    assert ("Operating_Mode", "shoulder_pan") in names and ("P_Coefficient", "gripper") in names
    assert ("Max_Torque_Limit", "gripper", 500) in bus.writes and (
        "Protection_Current",
        "gripper",
        250,
    ) in bus.writes


def test_arm_found_holding_is_left_holding() -> None:
    class Holding(FakeBus):
        def __init__(self, *a: Any) -> None:
            super().__init__(*a)
            self.torque = {j: 1 for j in JOINTS}

    a = H.FeetechArm(
        "follower",
        "follower",
        "/dev/x",
        "f",
        dict(CAL),
        bus_factory=lambda p, c: Holding(p, H.to_motor_calibration(c)),
    )
    a.connect()
    assert not a.configured and a.read_torque()  # not released: it would drop
    assert FakeBus.instances[-1].writes == []


def test_leader_gets_lerobots_leader_configure() -> None:
    """Position mode, as so_leader.py configure, so auto-calibration can drive it; no follower
    PID or gripper limits."""
    arm("leader")
    writes = FakeBus.instances[-1].writes
    assert {m for n, m, _ in writes if n == "Operating_Mode"} == set(JOINTS)
    assert not any(n in ("P_Coefficient", "Max_Torque_Limit") for n, _, _ in writes)


def test_positions_in_lerobot_units_and_goals_pass_through() -> None:
    a = arm()
    bus = FakeBus.instances[-1]
    bus.raw["elbow_flex"] = 2000 + 4095 // 4  # a quarter turn above mid
    pos = a.read_positions()
    assert pos["elbow_flex"] == pytest.approx(90.0, abs=0.1) and pos[
        "shoulder_pan"
    ] == pytest.approx(0, abs=0.1)
    a.write_goals({"elbow_flex": 45.0})
    assert bus.goals[-1] == {"elbow_flex": 45.0}


def test_no_calibration_refuses_positions_but_reads_raw() -> None:
    a = arm(cal=None)
    with pytest.raises(RuntimeError, match="no calibration"):
        a.read_positions()
    assert a.read_raw_positions()["gripper"] == 2000


def test_health_decoded() -> None:
    a = arm()
    FakeBus.instances[-1].regs["wrist_roll"]["Status"] = 32  # overload bit
    h = a.read_health()
    assert (
        h["shoulder_pan"].load_pct == -25.0
        and h["shoulder_pan"].voltage_v == 12.1
        and h["shoulder_pan"].temperature_c == 41
    )
    assert h["wrist_roll"].faults == ["overload"]


def test_calibration_round_trip_updates_normalisation() -> None:
    a = arm()
    got = a.read_calibration()
    assert got["shoulder_pan"] == CAL["shoulder_pan"]
    new = {j: c._replace(range_min=500, range_max=3500) for j, c in CAL.items()}
    a.write_calibration(new)
    assert a.calibration == new and FakeBus.instances[-1].calibration["gripper"].range_min == 500
    assert a.set_half_turn_homings()["gripper"] == 2000 - 2047


def test_lost_bus_reconnects_after_a_pause() -> None:
    t = [100.0]
    a = H.FeetechArm(
        "f",
        "follower",
        "/dev/x",
        "f",
        dict(CAL),
        clock=lambda: t[0],
        bus_factory=lambda p, c: FakeBus(p, H.to_motor_calibration(c)),
    )
    a.connect()
    FakeBus.instances[-1].connected = False  # unplugged
    with pytest.raises(ConnectionError):
        a.read_positions()
    FakeBus.fail_connect = 1
    with pytest.raises(ConnectionError):
        a.read_positions()  # first retry fails
    with pytest.raises(ConnectionError, match="not answering"):
        a.read_positions()  # too soon to try again
    t[0] += H.RECONNECT_S + 0.1
    assert a.read_positions()["shoulder_pan"] == pytest.approx(0, abs=0.1)


def test_build_rig_from_config(tmp_path) -> None:
    root = tmp_path / "cal"
    (root / "robots" / "so_follower").mkdir(parents=True)
    (root / "teleoperators" / "so_leader").mkdir(parents=True)
    for d, name in (
        ("robots/so_follower", "phi_bi_follower_left"),
        ("robots/so_follower", "phi_bi_follower_right"),
        ("teleoperators/so_leader", "phi_bi_leader_left"),
    ):
        (root / d / f"{name}.json").write_text(json.dumps({j: c._asdict() for j, c in CAL.items()}))
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(
        "robot:\n  type: bi_so_follower\n  id: phi_bi_follower\n"
        "  left_arm_config: {port: /dev/tty.usbmodemA}\n"
        "  right_arm_config: {port: /dev/tty.usbmodemB}\n"
        "teleop:\n  type: bi_so_leader\n  id: phi_bi_leader\n"
        "  left_arm_config: {port: /dev/tty.usbmodemC}\n"
        "  right_arm_config: {port: /dev/tty.usbmodemD}\n"
    )
    FakeBus.fail_connect = 0
    rig = H.build_rig(
        cfg,
        bus_factory=lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None),
        cal_root=root,
        cameras=False,
    )
    by = {a.name: a for a in rig.arms}
    assert set(by) == {"left_follower", "right_follower", "left_leader", "right_leader"}
    assert (
        by["left_follower"].calibration_id == "phi_bi_follower_left"
        and by["left_follower"].port == "/dev/tty.usbmodemA"
    )
    assert by["right_leader"].calibration is None  # its file is missing: identity will say so
    assert set(rig.calibration_files()) == {
        "phi_bi_follower_left",
        "phi_bi_follower_right",
        "phi_bi_leader_left",
    }


def test_build_rig_needs_ports(tmp_path) -> None:
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(
        "robot: {type: so101_follower, id: f, port: ''}\n"
        "teleop: {type: so101_leader, id: l, port: /dev/x}\n"
    )
    with pytest.raises(ValueError, match="no port"):
        H.build_rig(cfg, bus_factory=lambda p, c: FakeBus(p, c), cal_root=tmp_path, cameras=False)


def test_worker_runs_on_the_hardware_backend(tmp_path) -> None:
    """The worker's own rules on FeetechArms: connect, identify, confirm, arm (goal = present),
    teleop."""
    from phi_studio.worker import RigWorker

    root = tmp_path / "cal"
    # Two arms never share a calibration: the leader's offsets differ, as on real hardware.
    leader_cal = {j: c._replace(homing_offset=c.homing_offset + 300) for j, c in CAL.items()}
    for d, name, cal in (
        ("robots/so_follower", "f", CAL),
        ("teleoperators/so_leader", "l", leader_cal),
    ):
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / f"{name}.json").write_text(json.dumps({j: c._asdict() for j, c in cal.items()}))
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(
        "robot: {type: so101_follower, id: f, port: /dev/a}\n"
        "teleop: {type: so101_leader, id: l, port: /dev/b}\n"
    )
    rig = H.build_rig(
        cfg,
        bus_factory=lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None),
        cal_root=root,
        cameras=False,
    )
    sent: list[dict[str, Any]] = []
    w = RigWorker(rig, sent.append, clock=lambda: 0.0)
    for cmd in ("connect", "confirm", "arm"):
        w.handle({"cmd": cmd})
    assert w.session.state.name == "ARMED", [m for m in sent if m.get("type") == "error"]
    follower = next(a for a in rig.arms if a.role == "follower")
    bus = follower._bus
    assert (
        bus.goals and bus.goals[0] == follower.read_positions()
    )  # holds where it is before torque
    assert follower.read_torque()


def test_motor_ids_are_lerobots() -> None:
    assert H.MOTOR_IDS == {j: m.id for j, m in H.motor_table().items()}


def test_calibrating_by_hand_on_a_real_bus_keeps_positions_flowing(tmp_path) -> None:
    """LeRobot's homing empties the bus's calibration; the worker reads every arm each tick, so
    without a stand-in the middle step faulted a real arm as dead. Middle -> move -> save."""
    from phi_studio.worker import RigWorker

    root = tmp_path / "cal"
    leader_cal = {j: c._replace(homing_offset=c.homing_offset + 300) for j, c in CAL.items()}
    for d, name, cal in (
        ("robots/so_follower", "f", CAL),
        ("teleoperators/so_leader", "l", leader_cal),
    ):
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / f"{name}.json").write_text(json.dumps({j: c._asdict() for j, c in cal.items()}))
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(
        "robot: {type: so101_follower, id: f, port: /dev/a}\n"
        "teleop: {type: so101_leader, id: l, port: /dev/b}\n"
    )
    rig = H.build_rig(
        cfg,
        bus_factory=lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None),
        cal_root=root,
        cameras=False,
    )
    sent: list[dict[str, Any]] = []
    w = RigWorker(rig, sent.append, clock=lambda: 0.0, cal_dir=tmp_path / "out")
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "cal_start", "arm": "follower"})
    w.handle({"cmd": "cal_middle"})
    w.tick()
    errors = [m for m in sent if m.get("type") == "error"]
    assert w.session.state.name == "CALIBRATING" and not errors, errors
    follower = next(a for a in rig.arms if a.role == "follower")
    # the middle pose reads 0 degrees, give or take LeRobot's half tick (mid = 2047.5)
    assert abs(follower.read_positions()["elbow_flex"]) < 0.05
    bus = follower._bus
    bus.raw = {j: 2047 + (600 if j != "wrist_roll" else 0) for j in JOINTS}
    w.tick()
    bus.raw = {j: 2047 - (500 if j != "wrist_roll" else 0) for j in JOINTS}
    w.tick()
    w.handle({"cmd": "cal_finish"})
    w.handle({"cmd": "cal_save"})
    saved = next(m for m in sent if m.get("type") == "calibrated")
    assert saved["path"] == str(root / "robots/so_follower/f.json")  # the file Identify re-reads
    assert list((root / "robots/so_follower").glob("f.json.bak-*"))  # the old file is kept
    assert w.session.state.name != "FAULT"
    new = follower.calibration
    assert new["elbow_flex"].range_min == 1547 and new["elbow_flex"].range_max == 2647
    assert new["wrist_roll"].range_min == 0 and new["wrist_roll"].range_max == 4095


def test_raw_goals_and_torque_limits_for_auto_calibration() -> None:
    a = arm()
    bus = FakeBus.instances[-1]
    a.write_raw_goals({"gripper": 2100})
    assert bus.goals[-1] == {"gripper": 2100} and bus.normalized[-1] is False  # ticks, as given
    a.set_torque_limits({"gripper": 200, "elbow_flex": 1000})
    assert ("Torque_Limit", "gripper", 200) in bus.writes
    assert ("Torque_Limit", "elbow_flex", 1000) in bus.writes


def test_homing_some_joints_keeps_the_others_calibration() -> None:
    a = arm()
    homings = a.set_half_turn_homings(["gripper"])
    assert list(homings) == ["gripper"]
    assert a.calibration["elbow_flex"] == CAL["elbow_flex"]  # untouched
    g = a.calibration["gripper"]
    assert (g.homing_offset, g.range_min, g.range_max) == (2000 - 2047, 0, 4095)
    a.read_positions()  # normalises: the bus has a calibration for every joint again
    bare = arm(cal=None)
    with pytest.raises(RuntimeError, match="home every joint"):
        bare.set_half_turn_homings(["gripper"])


def test_connect_errors_name_the_arm_the_port_and_what_to_check() -> None:
    port_gone = ConnectionError("\nCould not connect on port '/dev/x'. Make sure you are using "
                                "the correct port.\nTry running `lerobot-find-port`\n")
    assert H.explain(port_gone, "left_follower", "/dev/x") == (
        "left_follower: nothing answers on /dev/x. Is it plugged in? Rig setup finds its port.")
    table = "\n".join(f"  - {i} (expected model: 777)" for i in range(1, 7))
    unpowered = RuntimeError(f"FeetechMotorsBus motor check failed on port '/dev/x':\n"
                             f"\nMissing motor IDs:\n{table}\n")
    assert "power" in H.explain(unpowered, "leader", "/dev/x")
    one = RuntimeError("FeetechMotorsBus motor check failed on port '/dev/x':\n\nMissing motor "
                       "IDs:\n  - 4 (expected model: 777)\n")
    assert "servo 4 of 6" in H.explain(one, "leader", "/dev/x")


def test_a_failed_connect_raises_studios_words_and_keeps_lerobots_as_the_cause() -> None:
    FakeBus.fail_connect = 1
    a = H.FeetechArm("left_follower", "follower", "/dev/x", "f", dict(CAL),
                     bus_factory=lambda p, c: FakeBus(p, None))  # fmt: skip
    with pytest.raises(ConnectionError, match="nothing answers on /dev/x") as got:
        a.connect()
    assert isinstance(got.value.__cause__, ConnectionError)


def test_position_reads_retry_a_lost_packet_before_the_bus_counts_as_down() -> None:
    a = arm()
    bus = FakeBus.instances[-1]
    a.read_positions()
    a.read_raw_positions()
    assert bus.retries[-2:] == [H.READ_RETRIES, H.READ_RETRIES]


def test_the_ports_are_closed_whenever_the_rig_is_disconnected(tmp_path) -> None:
    """macOS serial ports are not exclusive: a port Studio kept open while disconnected was shared
    with lerobot-calibrate in its own terminal, and both read garbage (2026-10-05)."""
    from phi_studio.worker import RigWorker

    root = tmp_path / "cal"
    for d, name in (("robots/so_follower", "f"), ("teleoperators/so_leader", "l")):
        (root / d).mkdir(parents=True)
        (root / d / f"{name}.json").write_text(json.dumps({j: c._asdict() for j, c in CAL.items()}))
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text("robot: {type: so101_follower, id: f, port: /dev/a}\n"
                   "teleop: {type: so101_leader, id: l, port: /dev/b}\n")  # fmt: skip
    FakeBus.fail_connect = 0
    FakeBus.instances.clear()
    bus = lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None)  # noqa: E731
    rig = H.build_rig(cfg, bus_factory=bus, cal_root=root, cameras=False)
    assert FakeBus.instances and not any(b.connected for b in FakeBus.instances)  # checked, closed
    assert all(a._bus is None for a in rig.arms)
    w = RigWorker(rig, lambda m: None, clock=lambda: 0.0)
    w.handle({"cmd": "connect"})
    assert all(a._bus is not None and a._bus.connected for a in rig.arms)  # Connect reopens
    w.handle({"cmd": "disconnect"})
    assert all(a._bus is None for a in rig.arms) and not any(b.connected for b in FakeBus.instances)
    w.tick()  # a disconnected tick must not reopen them
    assert all(a._bus is None for a in rig.arms)


def _dump(cal: dict[str, JointCal]) -> str:
    return json.dumps({j: c._asdict() for j, c in cal.items()})


def _rig_without_follower_file(tmp_path: Any) -> tuple[Any, Any, list[dict[str, Any]]]:
    """A real-arm rig whose leader has its file and whose follower has none yet."""
    from phi_studio.worker import RigWorker

    root = tmp_path / "cal"
    lead = {j: c._replace(homing_offset=c.homing_offset + 300) for j, c in CAL.items()}
    (root / "teleoperators/so_leader").mkdir(parents=True)
    (root / "robots/so_follower").mkdir(parents=True)
    (root / "teleoperators/so_leader/l.json").write_text(_dump(lead))
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text("robot: {type: so101_follower, id: f, port: /dev/a}\n"
                   "teleop: {type: so101_leader, id: l, port: /dev/b}\n")  # fmt: skip
    FakeBus.fail_connect = 0
    bus = lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None)  # noqa: E731
    rig = H.build_rig(cfg, bus_factory=bus, cal_root=root, cameras=False)
    sent: list[dict[str, Any]] = []
    return root, RigWorker(rig, sent.append, clock=lambda: 0.0), sent


def test_an_arm_with_no_calibration_file_does_not_fault_the_rig(tmp_path: Any) -> None:
    """A new arm has no file until it is calibrated. Reading it faulted the whole rig on the first
    tick, so the Calibrate page could never be reached (2026-10-05 audit)."""
    from phi_studio.session import State

    _, w, sent = _rig_without_follower_file(tmp_path)
    w.handle({"cmd": "connect"})
    for _ in range(12):  # past a health poll too
        w.tick()
    assert w.session.state is State.IDENTIFIED, w.session.snapshot()
    fol = next(a for a in w.identity if a["role"] == "follower")
    assert fol["calibrated"] is False and fol["ok"] is False
    tele = [m for m in sent if m["type"] == "telemetry"][-1]["arms"]
    assert tele["follower"]["pos"] == {} and tele["leader"]["pos"] and tele["follower"]["online"]
    w.handle({"cmd": "confirm"})
    assert "has no calibration file (robots/so_follower/f.json)" in sent[-1]["message"]


def test_identify_reads_a_file_written_after_studio_started(tmp_path: Any) -> None:
    """lerobot-calibrate in Studio's terminal writes the file; Identify must see it without a
    restart, and LeRobot's bus must normalise with it."""
    root, w, _ = _rig_without_follower_file(tmp_path)
    w.handle({"cmd": "connect"})
    (root / "robots/so_follower/f.json").write_text(_dump(CAL))
    w.handle({"cmd": "identify"})
    fol = next(a for a in w.identity if a["role"] == "follower")
    assert fol["ok"] and fol["calibrated"], fol
    arm = next(a for a in w.rig.arms if a.role == "follower")
    assert arm.calibration == CAL and arm._bus.calibration == H.to_motor_calibration(CAL)
    w.tick()
    assert set(arm.read_positions()) == set(JOINTS)


def test_an_arm_is_matched_to_its_own_file_when_another_role_shares_its_name(tmp_path: Any) -> None:
    """This Mac on 2026-10-05: robots/so_follower/phi_bi_left.json and
    teleoperators/so_leader/phi_bi_left.json both exist, and calibration files are keyed by name
    alone, so the leader's file hid the follower's. The follower then matched its copy
    phi_follower.json and Identify called its cables swapped."""
    from phi_studio.worker import RigWorker

    root = tmp_path / "cal"
    other = {j: c._replace(homing_offset=c.homing_offset + 300) for j, c in CAL.items()}
    for rel, c in (("robots/so_follower/f", CAL), ("robots/so_follower/copy_of_f", CAL),
                   ("teleoperators/so_leader/f", other), ("teleoperators/so_leader/l", other)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / f"{rel}.json").write_text(_dump(c))
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text("robot: {type: so101_follower, id: f, port: /dev/a}\n"
                   "teleop: {type: so101_leader, id: l, port: /dev/b}\n")  # fmt: skip
    FakeBus.fail_connect = 0
    bus = lambda p, c: FakeBus(p, H.to_motor_calibration(c) if c else None)  # noqa: E731
    w = RigWorker(H.build_rig(cfg, bus_factory=bus, cal_root=root, cameras=False), lambda m: None,
                  clock=lambda: 0.0)  # fmt: skip
    w.handle({"cmd": "connect"})
    fol = next(a for a in w.identity if a["role"] == "follower")
    assert fol["match"] == "f" and fol["ok"], fol
