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
    itself squeezing an object; a leader's (so_leader.py:127-131): position mode
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
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phi_studio import rigspec
from phi_studio.files import lerobot_calibration_dir
from phi_studio.identity import Calibration, JointCal, load_calibration
from phi_studio.rig import JOINTS, JointHealth, label

RECONNECT_S = 2.0  # how often a lost bus is tried again
# WHY retries: one lost status packet on a long USB run must not fault a recording; LeRobot's
# sync_read retries in place (motors_bus.py sync_read num_retry). A bus that misses three in a row
# is down, and the worker faults it as before.
READ_RETRIES = 2
WRITE_RETRIES = 2  # torque writes: LeRobot's default is none (feetech.py:291-305)
MOTOR_IDS = {j: i + 1 for i, j in enumerate(JOINTS)}  # motor_table's ids, without importing LeRobot
FULL_TICKS = 4095  # the STS3215's last encoder tick: reset_calibration's Max_Position_Limit
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


def explain(e: BaseException, arm: str, port: str) -> str:
    """LeRobot's connect errors in Studio's words. Its text sends people to lerobot-find-port and
    prints model tables; here: which arm, which port, and the one thing to check."""
    text = str(e)
    if "motor check failed" in text:
        missing = re.findall(r"^\s*-\s*(\d+)\s*\(expected model", text, re.M)
        if len(missing) >= len(MOTOR_IDS):
            return f"{arm}: no servo answers on {port}. Is the arm's power on?"
        if missing:
            return (f"{arm}: servo {', '.join(missing)} of {len(MOTOR_IDS)} does not answer on "
                    f"{port}. Check the cable into it, and its power.")  # fmt: skip
        return f"{arm}: a servo on {port} is not the model an SO-101 uses."
    if isinstance(e, ConnectionError) or "Could not connect" in text or "open port" in text:
        return f"{arm}: nothing answers on {port}. Is it plugged in? Rig setup finds its port."
    return f"{arm} on {port}: {text.strip()}"


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
    cal_path: Path | None = None  # the file `calibration` came from; Identify re-reads it
    max_step: float | dict[str, float] | None = None  # robot-config.yaml's max_relative_target

    # -- connection ------------------------------------------------------------------------------
    def connect(self) -> None:
        """Open the port, check every servo answers (LeRobot's handshake), configure if torque is
        off. Raises ConnectionError or RuntimeError, which the worker treats as a dead bus."""
        bus = self.bus_factory(self.port, self.calibration)
        try:
            bus.connect(handshake=True)
        except BaseException as e:
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass
            if isinstance(e, (OSError, RuntimeError)):
                raise type(e)(explain(e, label(self.name), self.port)) from e
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
        """LeRobot's configure, with torque left off: position mode for both roles (so_leader.py
        configure, so auto-calibration can drive a leader too), and a follower's PID and gripper
        limits (so_follower.py configure)."""
        from lerobot.motors.feetech import OperatingMode

        bus = self._bus
        bus.disable_torque()
        bus.configure_motors()
        for j in JOINTS:
            bus.write("Operating_Mode", j, OperatingMode.POSITION.value)
            if self.role == "follower":
                for reg, val in PID.items():
                    bus.write(reg, j, val)
        if self.role == "follower":
            for reg, val in GRIPPER_LIMITS.items():
                bus.write(reg, "gripper", val)
        self.configured = True

    def reload_calibration(self) -> None:
        """Re-read this arm's file. WHY: lerobot-calibrate in Studio's terminal writes it after
        Studio loaded it once at start. LeRobot normalises with the bus's own copy, so that is
        replaced too."""
        if self.cal_path is None:
            return
        try:
            cal = load_calibration(self.cal_path) if self.cal_path.is_file() else None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            cal = None  # a broken file is no calibration: positions are not read from it
        if cal == self.calibration:
            return
        self.calibration = cal
        if self._bus is not None:
            self._bus.calibration = to_motor_calibration(cal) if cal else {}

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
            j: float(v)
            for j, v in self._call(
                lambda b: b.sync_read("Present_Position", num_retry=READ_RETRIES)
            ).items()
        }

    def read_raw_positions(self) -> dict[str, int]:
        return {
            j: int(v)
            for j, v in self._call(
                lambda b: b.sync_read("Present_Position", normalize=False, num_retry=READ_RETRIES)
            ).items()
        }

    def set_half_turn_homings(self, joints: Sequence[str] | None = None) -> dict[str, int]:
        """LeRobot's homing for the middle pose, for `joints` (default all). WHY the cache: its
        reset_calibration empties the bus's own calibration (motors_bus.py reset_calibration), so
        every normalised read would raise until the end of the calibration and the worker would
        take the arm for dead. The homed joints' registers now hold the new homings and the full
        0..4095 range, the others what they held; normalising with the same values, as the mock
        does, keeps positions flowing (degrees from the middle pose)."""
        which = list(joints) if joints is not None else list(JOINTS)
        if self.calibration is None and len(which) < len(JOINTS):
            raise RuntimeError(f"{self.name} has no calibration file: home every joint")

        def run(b: Any) -> dict[str, int]:
            homings = {j: int(v) for j, v in b.set_half_turn_homings(which).items()}
            cal = dict(self.calibration or {})
            cal |= {j: JointCal(MOTOR_IDS[j], 0, homings[j], 0, FULL_TICKS) for j in homings}
            b.calibration = to_motor_calibration(cal)
            self.calibration = cal
            return homings

        return dict(self._call(run))

    def write_calibration(self, cal: Calibration) -> None:
        mc = to_motor_calibration(cal)
        self._call(
            lambda b: b.write_calibration(mc)
        )  # cache=True: LeRobot normalises with it from now on
        self.calibration = dict(cal)

    def write_goals(self, goals: dict[str, float]) -> None:
        self._call(lambda b: b.sync_write("Goal_Position", goals))

    def write_raw_goals(self, goals: dict[str, int]) -> None:
        self._call(lambda b: b.sync_write("Goal_Position", goals, normalize=False))

    def set_torque_limits(self, limits: dict[str, int]) -> None:
        """Torque_Limit per joint, 0..1000 (a RAM register: no EEPROM wear, gone at power-off)."""

        def run(b: Any) -> None:
            for j, v in limits.items():
                b.write("Torque_Limit", j, int(v), normalize=False)

        self._call(run)

    def set_torque(self, on: bool) -> None:
        def run(b: Any) -> None:
            if on:
                b.enable_torque(num_retry=WRITE_RETRIES)
                return
            # WHY motor by motor: LeRobot's loop stops at the first motor that fails, which left
            # every motor after a flaky one powered (right follower motor 2, 2026-10-05).
            failed = []
            for j in JOINTS:
                try:
                    b.disable_torque(j, num_retry=WRITE_RETRIES)
                except (OSError, RuntimeError):
                    failed.append(j)
            if failed:
                raise RuntimeError(f"torque did not turn off on {', '.join(failed)}")

        self._call(run)

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
            n = READ_RETRIES  # WHY: one garbled health packet faulted a moving rig
            pos = (
                b.sync_read("Present_Position", num_retry=n)
                if self.calibration is not None
                else {j: float("nan") for j in JOINTS}
            )
            load = b.sync_read("Present_Load", normalize=False, num_retry=n)  # per mille
            temp = b.sync_read("Present_Temperature", normalize=False, num_retry=n)  # deg C
            volt = b.sync_read("Present_Voltage", normalize=False, num_retry=n)  # 0.1 V
            status = b.sync_read("Status", normalize=False, num_retry=n)
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

    def release_ports(self) -> None:
        """Close every arm's serial port; the next bus call reopens it (FeetechArm._ensure).
        WHY: macOS serial ports are not exclusive, so a port Studio keeps open while
        disconnected is shared with lerobot-calibrate or any script, and both read garbage
        (2026-10-05). Torque is not touched: the worker releases it before calling this."""
        for a in self.arms:
            a.close()

    def reload_calibrations(self) -> dict[str, Calibration]:
        """Every arm's own file and every file by id, read again from disk."""
        for a in self.arms:
            a.reload_calibration()
        return self.calibration_files()

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
            a.key, a.role, a.port, a.lerobot_id or a.key, cal, side=a.side,
            bus_factory=bus_factory, cal_path=path, max_step=a.max_relative_target,
        )  # fmt: skip
        try:
            arm.connect()
        except (
            OSError,
            RuntimeError,
        ) as e:  # an unplugged arm: the worker sees it as dead, replug works
            arm.last_error = str(e)
        # WHY close again: the connect above only checks and configures; the worker starts
        # disconnected, and Connect reopens the port.
        arm.close()
        arms.append(arm)
    cams: list[Any] = []
    if cameras:
        from phi_studio.cameras import cameras_from_spec

        cams = cameras_from_spec(spec)
    return HardwareRig(arms, cams, spec, root)
