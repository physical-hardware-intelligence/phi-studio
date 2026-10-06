"""Detect arms, read-only: ping results, register matching and motion watching on fake buses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phi_studio.detect import Found, calibration_library, scan, watch_motion
from phi_studio.identity import JointCal
from phi_studio.rig import JOINTS


def cal(seed: int) -> dict[str, JointCal]:
    return {j: JointCal(i + 1, 0, 100 * seed + i, 800 + i, 3200 - i) for i, j in enumerate(JOINTS)}


class FakePort:
    def __init__(self) -> None:
        self.open = True

    def closePort(self) -> None:  # noqa: N802 (scservo's name)
        self.open = False


class FakeBus:
    """Enough of FeetechMotorsBus to read registers and raw positions."""

    def __init__(self, port: str, regs: dict[str, JointCal] | None = None) -> None:
        self.port = port
        self.regs = dict(regs or cal(1))
        self.raw = {j: 2047 for j in JOINTS}
        self.port_handler = FakePort()
        self.calls: list[str] = []

    def connect(self, handshake: bool = True) -> None:
        self.calls.append(f"connect:{handshake}")

    def read_calibration(self) -> dict[str, JointCal]:
        return dict(self.regs)

    def sync_read(self, name: str, normalize: bool = True) -> dict[str, Any]:
        assert not normalize, "detect reads raw registers only"
        if name == "Present_Position":
            return dict(self.raw)
        if name == "Torque_Enable":
            return {j: 0 for j in JOINTS}
        raise KeyError(name)

    def write(self, *a: Any, **k: Any) -> None:
        raise AssertionError("detect must never write a register")

    def enable_torque(self) -> None:
        raise AssertionError("detect must never enable torque")


def write_cal(root: Path, kind: str, name: str, c: dict[str, JointCal]) -> None:
    p = root / kind / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({j: v._asdict() for j, v in c.items()}))


def test_scan_reports_clashes_missing_ids_and_which_file_each_arm_holds(tmp_path: Path) -> None:
    write_cal(tmp_path, "robots/so_follower", "yash_follower", cal(1))
    write_cal(tmp_path, "teleoperators/so_leader", "phi_leader", cal(2))
    pings = {"/dev/tty.f": ([1, 2, 3, 4, 5, 6], []), "/dev/tty.l": ([1, 2, 3, 4, 5, 6], []),
             "/dev/tty.bad": ([2, 3, 4, 5], [6]), "/dev/tty.none": ([], [])}  # fmt: skip
    regs = {"/dev/tty.f": cal(1), "/dev/tty.l": cal(2)}
    opened: list[FakeBus] = []

    def make(port: str, c: Any) -> FakeBus:
        assert c is None  # no normalisation: registers only
        b = FakeBus(port, regs[port])
        opened.append(b)
        return b

    found = scan([(p, f"SN{i}") for i, p in enumerate(pings)], tmp_path,
                 ping=lambda p, ids: pings[p], bus_factory=make)  # fmt: skip
    by = {f.port: f for f in found}
    assert set(by) == {"/dev/tty.f", "/dev/tty.l", "/dev/tty.bad"}  # nothing on .none: not an arm
    f, ld, bad = by["/dev/tty.f"], by["/dev/tty.l"], by["/dev/tty.bad"]
    assert f.role == "follower" and f.match and f.match.endswith("yash_follower")
    assert ld.role == "leader" and f.torque is False
    assert bad.problem and "id 6" in bad.problem and "1 does not" not in bad.problem
    assert bad.registers is None
    assert all(not b.port_handler.open for b in opened)


def test_an_arm_whose_registers_match_no_file_has_no_role(tmp_path: Path) -> None:
    write_cal(tmp_path, "robots/so_follower", "other", cal(9))
    f = scan([("/dev/tty.x", "S")], tmp_path, ping=lambda p, ids: ([1, 2, 3, 4, 5, 6], []),
             bus_factory=lambda p, c: FakeBus(p, cal(1)))[0]  # fmt: skip
    assert f.role is None and f.match_deg and f.match_deg > 0 and f.problem is None


def test_a_missing_id_and_an_unopenable_port_are_named(tmp_path: Path) -> None:
    def ping(p: str, ids: Any) -> tuple[list[int], list[int]]:
        if p == "/dev/tty.gone":
            raise OSError("could not open /dev/tty.gone")
        return [2, 3, 4, 5, 6], []

    by = {f.port: f for f in scan([("/dev/tty.m", "A"), ("/dev/tty.gone", "B")], tmp_path,
                                  ping=ping, bus_factory=lambda p, c: FakeBus(p))}  # fmt: skip
    assert by["/dev/tty.m"].problem and "id 1 does not answer" in by["/dev/tty.m"].problem
    assert by["/dev/tty.gone"].problem == "could not open /dev/tty.gone"


def test_the_library_skips_files_it_cannot_read(tmp_path: Path) -> None:
    write_cal(tmp_path, "robots/so_follower", "good", cal(1))
    (tmp_path / "robots/so_follower/broken.json").write_text("{")
    (tmp_path / "robots/so_follower/._good.json").write_text("mac junk")
    assert list(calibration_library(tmp_path)) == ["robots/so_follower/good"]


def test_watch_motion_finds_the_arm_being_moved() -> None:
    buses: dict[str, FakeBus] = {}

    def make(port: str, c: Any) -> FakeBus:
        buses[port] = FakeBus(port)
        return buses[port]

    t = [0.0]

    def sleep(dt: float) -> None:
        t[0] += dt
        buses["/dev/tty.b"].raw["elbow_flex"] += 40

    found = [Found(p, "", [1, 2, 3, 4, 5, 6], []) for p in ("/dev/tty.a", "/dev/tty.b")]
    moved = watch_motion(found, 1.0, bus_factory=make, clock=lambda: t[0], sleep=sleep)
    assert moved["/dev/tty.b"] > 100 and moved["/dev/tty.a"] == 0
    assert all(not b.port_handler.open for b in buses.values())
