"""Find the real arms on this Mac before any config exists, read-only.

Which USB serial ports answer, which motor ids each reaches, whether two motors share an id, which
calibration file each arm's servo registers match, and which arm a person is moving by hand. Set
up's Detect arms step (rig_api.py) turns that into robot-config.yaml.

Nothing here writes a register or enables torque. Every port is closed before a function returns,
so the robot worker and LeRobot commands can open it afterwards.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phi_studio import rigspec
from phi_studio.hardware import default_bus, from_motor_calibration
from phi_studio.identity import Calibration, load_calibration, match_fingerprint

BAUD = 1_000_000  # LeRobot's Feetech default (feetech.py), what the Seeed kit ships at
MODEL_NUMBER = 777  # STS3215 (LeRobot tables.py MODEL_NUMBER_TABLE)
ARM_IDS = range(1, 7)
# WHY ids to 10, not 1..6: a motor given a wrong id is found where it went (often 7..10 after a
# miscount in lerobot-setup-motors). A full 0..253 sweep takes about 4 s a port and finds the same.
SCAN_IDS = range(1, 11)


def _short(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return text[0] if text else type(e).__name__


@dataclass
class Found:
    """What one serial port reaches."""

    port: str
    serial: str
    ids: list[int]  # motor ids that answered cleanly
    clashes: list[int]  # ids two motors answer to at once
    registers: Calibration | None = None  # read only when ids are exactly 1..6
    torque: bool | None = None
    match: str | None = None  # nearest calibration file, as kind/folder/id
    match_deg: float | None = None
    error: str | None = None

    @property
    def role(self) -> str | None:
        """leader or follower when the registers match a file exactly, else unknown."""
        if self.match is None or self.match_deg is None or self.match_deg > 0.0:
            return None
        return "follower" if self.match.startswith("robots/") else "leader"

    @property
    def problem(self) -> str | None:
        if self.error:
            return self.error
        if self.clashes:
            # WHY this wording: tonight's real fault (2026-10-05). A wrong id written to one motor
            # makes two answer at once; their replies collide and the id reads as broken, while the
            # id that motor should have reads as missing.
            ids = ", ".join(map(str, self.clashes))
            return (f"Two motors answer to id {ids}: one was given another motor's id. "
                    "Unplug the other one and set this one's id back.")  # fmt: skip
        missing = [i for i in ARM_IDS if i not in self.ids]
        if missing:
            return (f"Motor id {', '.join(map(str, missing))} does not answer. Check the cable "
                    "into it, its power, or whether it was given another id.")  # fmt: skip
        extra = [i for i in self.ids if i not in ARM_IDS]
        if extra:
            return f"Unexpected motor id {', '.join(map(str, extra))}."
        return None

    def public(self) -> dict[str, Any]:
        return {"port": self.port, "serial": self.serial, "ids": self.ids,
                "clashes": self.clashes, "torque": self.torque, "match": self.match,
                "match_deg": self.match_deg, "role": self.role,
                "problem": self.problem}  # fmt: skip


def calibration_library(root: Path) -> dict[str, Calibration]:
    """Every SO-101 calibration file under LeRobot's calibration root, keyed kind/folder/id."""
    out: dict[str, Calibration] = {}
    for role in ("follower", "leader"):
        kind, folder = rigspec.FOLDER[role]
        d = root / kind / folder
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("._"):
                continue
            try:
                out[f"{kind}/{folder}/{p.stem}"] = load_calibration(p)
            except (OSError, ValueError, KeyError, TypeError):
                continue  # not a calibration file Studio can read
    return out


def ping_all(port: str, ids: Iterable[int]) -> tuple[list[int], list[int]]:
    """Ping each id once. A clean reply from an STS3215 is a motor; a garbled one (anything but a
    timeout) means two motors answered together."""
    import scservo_sdk as scs

    ph, pk = scs.PortHandler(port), scs.PacketHandler(0)
    if not ph.openPort():
        raise OSError(f"could not open {port}")
    try:
        ph.setBaudRate(BAUD)
        ok, clash = [], []
        for i in ids:
            model, comm, _ = pk.ping(ph, i)
            if comm == scs.COMM_SUCCESS and model == MODEL_NUMBER:
                ok.append(i)
            elif comm != scs.COMM_RX_TIMEOUT:
                clash.append(i)
        return ok, clash
    finally:
        ph.closePort()


def _close(bus: Any) -> None:
    try:
        bus.disconnect(disable_torque=False)
    except Exception:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass


def scan(
    ports: list[tuple[str, str]],
    cal_root: Path,
    *,
    ping: Callable[[str, Iterable[int]], tuple[list[int], list[int]]] = ping_all,
    bus_factory: Callable[[str, Calibration | None], Any] = default_bus,
    ids: Iterable[int] = SCAN_IDS,
) -> list[Found]:
    """What each (device, serial) port reaches. Writes nothing, moves nothing."""
    library = calibration_library(cal_root)
    out = []
    for device, serial in ports:
        try:
            ok, clash = ping(device, list(ids))
        except OSError as e:
            out.append(Found(device, serial, [], [], error=_short(e)))
            continue
        f = Found(device, serial, ok, clash)
        if f.problem is None:
            bus = bus_factory(device, None)
            try:
                bus.connect(handshake=False)
                f.registers = from_motor_calibration(bus.read_calibration())
                torque = bus.sync_read("Torque_Enable", normalize=False)
                f.torque = any(int(v) == 1 for v in torque.values())
            except Exception as e:
                f.error = _short(e)
            finally:
                _close(bus)
            if f.registers and library:
                best = match_fingerprint(f.registers, library)[0]
                f.match, f.match_deg = best.name, round(best.distance.max_deg, 2)
        if ok or clash or f.error:
            out.append(f)
    return out


def serial_ports() -> list[tuple[str, str]]:
    """USB serial adapters (SO-101 boards are WCH CH343 / CH340). macOS lists each one twice, as
    /dev/cu.* and /dev/tty.*; LeRobot's configs use tty, so Studio does too."""
    from serial.tools import list_ports

    out = []
    for p in list_ports.comports():
        if p.vid is None or not p.serial_number:
            continue
        out.append((p.device.replace("/dev/cu.", "/dev/tty."), p.serial_number))
    return sorted(set(out))


def watch_motion(
    found: list[Found],
    seconds: float = 3.0,
    *,
    bus_factory: Callable[[str, Calibration | None], Any] = default_bus,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, float]:
    """How far each healthy arm moved, in encoder ticks summed over its joints, while a person
    moves one arm by hand. The arm that moved most is the one they hold."""
    buses: dict[str, Any] = {}
    try:
        for f in found:
            if f.problem is None:
                b = bus_factory(f.port, None)
                b.connect(handshake=False)
                buses[f.port] = b
        first = {p: b.sync_read("Present_Position", normalize=False) for p, b in buses.items()}
        span = {p: {j: (v, v) for j, v in r.items()} for p, r in first.items()}
        end = clock() + seconds
        while clock() < end:
            for p, b in buses.items():
                for j, v in b.sync_read("Present_Position", normalize=False).items():
                    lo, hi = span[p][j]
                    span[p][j] = (min(lo, v), max(hi, v))
            sleep(0.05)
        return {p: float(sum(hi - lo for lo, hi in s.values())) for p, s in span.items()}
    finally:
        for b in buses.values():
            _close(b)
