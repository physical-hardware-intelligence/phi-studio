"""Real SO-101 arms for the robot worker, over LeRobot's own Feetech bus.

FeetechArm satisfies rig.ArmBus the way MockArm does, so every rule the worker enforces (identity
before torque, goal = present on enable, the step clip, the heartbeat, freeze then release) holds
unchanged on hardware. Positions come out of LeRobot's own normalisation with the arm's calibration
file, so a number Studio shows is the number lerobot-record would store.

Copied from LeRobot 0.6.0 so a rig behaves the same here and in lerobot-teleoperate:
  * the motor table (robots/so_follower/so_follower.py:51-62): ids 1..6, sts3215, DEGREES for the
    five arm joints, RANGE_0_100 for the gripper; the leader's is the same (so_leader.py)
  * a follower's configure (so_follower.py:159-173): position mode, P=16 I=0 D=32, and the gripper's
    limits (Max_Torque_Limit 500, Protection_Current 250, Overload_Torque 25) so it cannot burn
    itself squeezing an object
Deliberately different:
  * configure runs only when every servo's torque is off, and leaves it off. LeRobot re-enables
    torque after configuring (motors_bus.torque_disabled); Studio enables torque only after the arms
    are confirmed (session.py). An arm found holding torque is left holding: releasing it would drop
    it.
  * no stdin prompts: Studio's calibration flow (worker.py) calls set_half_turn_homings,
    read_raw_positions and write_calibration itself.

Status: unit-tested against a fake bus; not yet run on a physical arm (first rig session
2026-10-05).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phi_studio import rigspec
from phi_studio.files import lerobot_calibration_dir
from phi_studio.identity import Calibration, JointCal, load_calibration
from phi_studio.rig import JOINTS, JointHealth

RECONNECT_S = 2.0  # how often a lost bus is tried again
GRIPPER_LIMITS = {"Max_Torque_Limit": 500, "Protection_Current": 250, "Overload_Torque": 25}
PID = {"P_Coefficient": 16, "I_Coefficient": 0, "D_Coefficient": 32}


def motor_table() -> dict[str, Any]:
    """LeRobot's SO-101 motors: degrees for the arm, 0..100 for the gripper (use_degrees=True)."""
    from lerobot.motors import Motor, MotorNormMode

    deg = MotorNormMode.DEGREES
    return {
        "shoulder_pan": Motor(1, "sts3215", deg),
        "shoulder_lift": Motor(2, "sts3215", deg),
        "elbow_flex": Motor(3, "sts3215", deg),
        "wrist_flex": Motor(4, "sts3215", deg),
        "wrist_roll": Motor(5, "sts3215", deg),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


def to_motor_calibration(cal: Calibration) -> dict[str, Any]:
    from lerobot.motors import MotorCalibration

    return {j: MotorCalibration(**c._asdict()) for j, c in cal.items()}


def from_motor_calibration(mc: dict[str, Any]) -> Calibration:
    return {
        j: JointCal(
            int(c.id), int(c.drive_mode), int(c.homing_offset), int(c.range_min), int(c.range_max)
        )
        for j, c in mc.items()
    }


def default_bus(port: str, calibration: Calibration | None) -> Any:
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(
        port=port,
        motors=motor_table(),
        calibration=to_motor_calibration(calibration) if calibration else None,
    )


@dataclass
class FeetechArm:
    """One arm on one USB serial bus. Only the worker's bus thread calls it."""

    name: str  # Studio's arm key: follower, leader, left_follower, ...
    role: str  # leader | follower
    port: str
    calibration_id: str  # the LeRobot id its calibration file is named after
    calibration: Calibration | None = None  # from that file, for LeRobot's normalisation
    side: str | None = None
    serial: str | None = None
    bus_factory: Callable[[str, Calibration | None], Any] = default_bus
    clock: Callable[[], float] = time.monotonic
    _bus: Any = field(default=None, repr=False)
    _next_try: float = 0.0
    configured: bool = False
    last_error: str | None = None

    # -- connection ------------------------------------------------------------------------------
    def connect(self) -> None:
        """Open the port, check every servo answers (LeRobot's handshake), configure if torque is
        off. Raises ConnectionError or RuntimeError, which the worker treats as a dead bus."""
        bus = self.bus_factory(self.port, self.calibration)
        try:
            bus.connect(handshake=True)
        except BaseException:
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass
            raise
        self._bus = bus
        self.configured = False
        if not self._torque_on():
            self._configure()

    def _ensure(self) -> Any:
        """The open bus, reconnecting a lost one at most every RECONNECT_S."""
        if self._bus is not None:
            return self._bus
        now = self.clock()
        if now < self._next_try:
            raise ConnectionError(
                f"{self.port} is not answering ({self.last_error or 'not connected'})"
            )
        self._next_try = now + RECONNECT_S
        try:
            self.connect()
        except (OSError, RuntimeError) as e:
            self.last_error = str(e)
            raise
        self.last_error = None
        return self._bus

    def _call(self, fn: Callable[[Any], Any]) -> Any:
        """Run one bus operation; a failed one drops the bus, so the next call reconnects."""
        bus = self._ensure()
        try:
            return fn(bus)
        except (OSError, RuntimeError) as e:
            if isinstance(e, ConnectionError) or "port" in str(e).lower():
                self._drop()
            raise

    def _drop(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass

    def _torque_on(self) -> bool:
        vals = self._bus.sync_read("Torque_Enable", normalize=False)
        return any(int(v) == 1 for v in vals.values())

    def _configure(self) -> None:
        """LeRobot's follower configure, with torque left off. A leader only gets its torque off."""
        bus = self._bus
        bus.disable_torque()
        if self.role == "follower":
            from lerobot.motors.feetech import OperatingMode

            bus.configure_motors()
            for j in JOINTS:
                bus.write("Operating_Mode", j, OperatingMode.POSITION.value)
                for reg, val in PID.items():
                    bus.write(reg, j, val)
            for reg, val in GRIPPER_LIMITS.items():
                bus.write(reg, "gripper", val)
        self.configured = True

    # -- rig.ArmBus --------------------------------------------------------------------------------
    def read_calibration(self) -> Calibration:
        return from_motor_calibration(self._call(lambda b: b.read_calibration()))

    def read_positions(self) -> dict[str, float]:
        if self.calibration is None:
            # WHY refuse: without a calibration LeRobot cannot normalise, and a raw
            # tick read as degrees
            # would drive a follower somewhere unrelated.
            raise RuntimeError(f"{self.name} has no calibration file ({self.calibration_id}.json)")
        return {
            j: float(v) for j, v in self._call(lambda b: b.sync_read("Present_Position")).items()
        }

    def read_raw_positions(self) -> dict[str, int]:
        return {
            j: int(v)
            for j, v in self._call(
                lambda b: b.sync_read("Present_Position", normalize=False)
            ).items()
        }

    def set_half_turn_homings(self) -> dict[str, int]:
        return {j: int(v) for j, v in self._call(lambda b: b.set_half_turn_homings()).items()}

    def write_calibration(self, cal: Calibration) -> None:
        mc = to_motor_calibration(cal)
        self._call(
            lambda b: b.write_calibration(mc)
        )  # cache=True: LeRobot normalises with it from now on
        self.calibration = dict(cal)

    def write_goals(self, goals: dict[str, float]) -> None:
        self._call(lambda b: b.sync_write("Goal_Position", goals))

    def set_torque(self, on: bool) -> None:
        self._call(lambda b: b.enable_torque() if on else b.disable_torque())

    def read_torque(self) -> bool:
        """True when any servo holds torque: one holding joint is enough to make the arm unsafe to
        move by hand or to calibrate."""
        return bool(
            self._call(
                lambda b: any(
                    int(v) == 1 for v in b.sync_read("Torque_Enable", normalize=False).values()
                )
            )
        )

    def read_health(self) -> dict[str, JointHealth]:
        def read(b: Any) -> dict[str, JointHealth]:
            pos = (
                b.sync_read("Present_Position")
                if self.calibration is not None
                else {j: float("nan") for j in JOINTS}
            )
            load = b.sync_read("Present_Load", normalize=False)  # sign-magnitude decoded: per mille
            temp = b.sync_read("Present_Temperature", normalize=False)  # deg C
            volt = b.sync_read("Present_Voltage", normalize=False)  # 0.1 V
            status = b.sync_read("Status", normalize=False)
            return {
                j: JointHealth(
                    float(pos[j]),
                    int(load[j]) / 10.0,
                    float(temp[j]),
                    int(volt[j]) / 10.0,
                    int(status[j]),
                )
                for j in JOINTS
            }

        return self._call(read)

    def close(self) -> None:
        self._drop()


@dataclass
class HardwareRig:
    arms: list[FeetechArm]
    cameras: list[Any]
    spec: rigspec.RigSpec
    cal_root: Path

    def calibration_files(self) -> dict[str, Calibration]:
        """Every calibration file under LeRobot's folder, by id: identity matches an arm's registers
        against all of them, so an arm on another arm's cable is caught (worker.py _identify)."""
        out: dict[str, Calibration] = {}
        for kind, folder in rigspec.FOLDER.values():
            d = self.cal_root / kind / folder
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                if p.name.startswith("._"):
                    continue
                try:
                    out[p.stem] = load_calibration(p)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue  # a broken file matches nothing; Checks reports it
        return out


def build_rig(
    config: Path,
    bus_factory: Callable[[str, Calibration | None], Any] = default_bus,
    cal_root: Path | None = None,
    cameras: bool = True,
) -> HardwareRig:
    """The rig robot-config.yaml describes, every bus connected (or left to reconnect on first
    use)."""
    spec = rigspec.parse(Path(config).read_text())
    root = cal_root or lerobot_calibration_dir()
    arms: list[FeetechArm] = []
    for a in spec.arms:
        if not a.port:
            raise ValueError(f"{a.key} has no port in robot-config.yaml: finish onboarding first")
        path = a.calibration_path(root)
        cal = load_calibration(path) if path and path.is_file() else None
        arm = FeetechArm(
            a.key, a.role, a.port, a.lerobot_id or a.key, cal, side=a.side, bus_factory=bus_factory
        )
        try:
            arm.connect()
        except (
            OSError,
            RuntimeError,
        ) as e:  # an unplugged arm: the worker sees it as dead, replug works
            arm.last_error = str(e)
        arms.append(arm)
    cams: list[Any] = []
    if cameras:
        from phi_studio.cameras import cameras_from_spec

        cams = cameras_from_spec(spec)
    return HardwareRig(arms, cams, spec, root)
