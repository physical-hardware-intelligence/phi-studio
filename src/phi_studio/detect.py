"""Find the real arms on this Mac before any config exists, read-only.

Which USB serial ports answer, which motor ids each reaches, whether two motors share an id, which
calibration file each arm's servo registers match, and which arm a person is moving by hand. Set
up's Detect arms step (rig_api.py) turns that into robot-config.yaml.

Nothing here writes a register or enables torque. Every port is closed before a function returns,
so the robot worker and LeRobot commands can open it afterwards.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Holder:
    """A process, not Studio, with an arm's serial port open."""

    pid: int
    command: str
    paused: bool  # stopped with Ctrl-Z: it still holds the port and reads nothing

    @property
    def name(self) -> str:
        words = self.command.split()
        script = next((w for w in words[1:] if w.endswith(".py") or "lerobot" in w), None)
        return Path(script or (words[0] if words else "?")).name

    def public(self) -> dict[str, Any]:
        return {"pid": self.pid, "command": self.command, "name": self.name,
                "paused": self.paused}  # fmt: skip


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=5).stdout  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return ""


def port_holders(ports: Iterable[str], run: Callable[[list[str]], str] = _run,
                 me: int | Iterable[int] | None = None) -> dict[str, list[Holder]]:
    """Which other processes have each port open, as lsof sees it. Keys are /dev/tty.* names.
    WHY lsof: macOS serial ports are not opened exclusively, so a second opener is not refused;
    it silently shares the line and corrupts replies. `me`: pids that are Studio itself (the
    server and its robot worker), never reported."""
    mine = {os.getpid()} if me is None else ({me} if isinstance(me, int) else set(me))
    want: dict[str, str] = {}
    for p in ports:
        tty = p.replace("/dev/cu.", "/dev/tty.")
        for dev in (tty, tty.replace("/dev/tty.", "/dev/cu.")):
            if os.path.exists(dev):
                want[dev] = tty
    if not want:
        return {}
    pids: dict[int, set[str]] = {}
    pid = None
    for line in run(["lsof", "-F", "pn", *want]).splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None and line[1:] in want and pid not in mine:
            pids.setdefault(pid, set()).add(want[line[1:]])
    if not pids:
        return {}
    info: dict[int, tuple[str, str]] = {}
    listed = run(["ps", "-o", "pid=,stat=,command=", "-p", ",".join(map(str, pids))])
    for line in listed.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit():
            info[int(parts[0])] = (parts[1], parts[2] if len(parts) > 2 else "")
    out: dict[str, list[Holder]] = {}
    for holder, devs in sorted(pids.items()):
        stat, cmd = info.get(holder, ("", ""))
        for d in sorted(devs):
            out.setdefault(d, []).append(Holder(holder, cmd, stat.startswith("T")))
    return out


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
    # WHY every exact match, not just the nearest: one arm's registers can match several files
    # (phi_bi_left.json links to phi_follower.json), and only a _left/_right name tells the side.
    matches: list[str] = field(default_factory=list)
    held_by: list[Holder] = field(default_factory=list)  # other processes with this port open
    error: str | None = None

    @property
    def role(self) -> str | None:
        """leader or follower when the registers match a file exactly, else unknown."""
        if self.match is None or self.match_deg is None or self.match_deg > 0.0:
            return None
        return "follower" if self.match.startswith("robots/") else "leader"

    @property
    def problem(self) -> str | None:
        if self.held_by:
            # WHY first: tonight's real fault (2026-10-05). A script paused with Ctrl-Z kept all
            # four ports open, and LeRobot then read its leftover bytes as garbled packets.
            h = self.held_by[0]
            return (f"{h.name} (process {h.pid}{', paused' if h.paused else ''}) has this port "
                    "open. Stop it, then scan again.")  # fmt: skip
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
                "match_deg": self.match_deg, "matches": self.matches, "role": self.role,
                "held_by": [h.public() for h in self.held_by],
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
    holders: Callable[[list[str]], dict[str, list[Holder]]] = port_holders,
) -> list[Found]:
    """What each (device, serial) port reaches. Writes nothing, moves nothing."""
    library = calibration_library(cal_root)
    held = holders([d for d, _ in ports])
    out = []
    for device, serial in ports:
        if held.get(device):
            out.append(Found(device, serial, [], [], held_by=held[device]))
            continue  # WHY no ping: replies on a shared line are not to be trusted
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
                ranked = match_fingerprint(f.registers, library)
                f.match, f.match_deg = ranked[0].name, round(ranked[0].distance.max_deg, 2)
                f.matches = [m.name for m in ranked if m.distance.max_deg == 0.0]
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


# -- motor check: the one powered test here ----------------------------------------------------
# WHY this exists: on 2026-10-05 one follower motor (shoulder_lift) corrupted every packet on its
# bus whenever it drove current, so teleop died on its first read while every read-only check
# passed. Powering one motor at a time, holding still, and polling the whole bus finds it.

ADDR = {"goal": 42, "torque": 40, "pos": 56, "load": 60, "volt": 62, "temp": 63, "status": 65,
        "current": 69}  # fmt: skip  (STS3215, LeRobot tables.py STS_SMS_SERIES_CONTROL_TABLE)
WIDTH = {"goal": 2, "torque": 1, "pos": 2, "load": 2, "volt": 1, "temp": 1, "status": 1,
         "current": 2}  # fmt: skip


@dataclass
class MotorCheck:
    """One motor powered and holding still for a few seconds."""

    id: int
    polls: int = 0
    bus_errors: int = 0  # whole-bus sync reads that failed while this motor held torque
    first_error_s: float | None = None
    min_volt: float | None = None
    max_current: int | None = None
    temp: int | None = None
    status: int | None = None  # the motor's own error flags, 0 when fine
    error: str | None = None  # could not power it at all

    @property
    def ok(self) -> bool:
        return self.error is None and self.bus_errors == 0 and not self.status

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "ok": self.ok, "polls": self.polls, "bus_errors": self.bus_errors,
                "first_error_s": self.first_error_s, "min_volt": self.min_volt,
                "max_current": self.max_current, "temp": self.temp, "status": self.status,
                "error": self.error}  # fmt: skip


class Line:
    """The few Feetech bus calls the motor check needs, over scservo_sdk."""

    def __init__(self, port: str) -> None:
        import scservo_sdk as scs

        self.scs = scs
        self.ph, self.pk = scs.PortHandler(port), scs.PacketHandler(0)
        if not self.ph.openPort():
            raise OSError(f"could not open {port}")
        self.ph.setBaudRate(BAUD)
        self.group = scs.GroupSyncRead(self.ph, self.pk, ADDR["pos"], 2)
        for i in ARM_IDS:
            self.group.addParam(i)

    def read(self, mid: int, name: str) -> int | None:
        fn = self.pk.read2ByteTxRx if WIDTH[name] == 2 else self.pk.read1ByteTxRx
        v, comm, _ = fn(self.ph, mid, ADDR[name])
        return int(v) if comm == self.scs.COMM_SUCCESS else None

    def write(self, mid: int, name: str, value: int) -> bool:
        fn = self.pk.write2ByteTxRx if WIDTH[name] == 2 else self.pk.write1ByteTxRx
        comm, _ = fn(self.ph, mid, ADDR[name], value)
        return bool(comm == self.scs.COMM_SUCCESS)

    def sync_ok(self) -> bool:
        return bool(self.group.txRxPacket() == self.scs.COMM_SUCCESS)

    def close(self) -> None:
        self.ph.closePort()


def check_motor(line: Any, mid: int, seconds: float = 3.0, *,
                clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], None] = time.sleep) -> MotorCheck:
    """Hold motor `mid` where it is with torque on for `seconds`; poll it and the whole bus."""
    r = MotorCheck(mid)
    try:
        pos = line.read(mid, "pos")
        if pos is None:
            r.error = "does not answer"
            return r
        # WHY goal first: the motor holds where it already is, so nothing moves
        if not (line.write(mid, "goal", pos) and line.write(mid, "torque", 1)):
            r.error = "would not take torque"
            return r
        t0 = clock()
        while clock() - t0 < seconds:
            r.polls += 1
            if not line.sync_ok():
                r.bus_errors += 1
                if r.first_error_s is None:
                    r.first_error_s = round(clock() - t0, 2)
            v, cur, temp, st = (line.read(mid, k) for k in ("volt", "current", "temp", "status"))
            if v is not None:
                r.min_volt = v / 10 if r.min_volt is None else min(r.min_volt, v / 10)
            if cur is not None:
                r.max_current = cur if r.max_current is None else max(r.max_current, cur)
            if temp is not None:
                r.temp = temp
            if st:
                r.status = st
            sleep(0.1)
        return r
    finally:
        for _ in range(3):  # WHY 3 tries: the bus may be what is failing
            if line.write(mid, "torque", 0):
                break


def check_arm(port: str, seconds: float = 3.0, *, line_factory: Callable[[str], Any] = Line,
              on_motor: Callable[[MotorCheck], None] | None = None,
              **kw: Any) -> list[MotorCheck]:
    """Every motor of one arm, one at a time. Torque is off on all six before and after."""
    line = line_factory(port)
    try:
        for i in ARM_IDS:
            line.write(i, "torque", 0)
        out = []
        for i in ARM_IDS:
            r = check_motor(line, i, seconds, **kw)
            out.append(r)
            if on_motor:
                on_motor(r)
        return out
    finally:
        for i in ARM_IDS:
            line.write(i, "torque", 0)
        line.close()
