"""Detect arms, read-only: ping results, register matching and motion watching on fake buses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

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
    assert f.matches == ["robots/so_follower/yash_follower"]
    assert ld.matches[0].endswith("phi_leader")
    assert ld.role == "leader" and f.torque is False
    assert bad.problem and "id 6" in bad.problem and "1 does not" not in bad.problem
    assert bad.registers is None
    assert all(not b.port_handler.open for b in opened)


def test_an_arm_whose_registers_match_no_file_has_no_role(tmp_path: Path) -> None:
    write_cal(tmp_path, "robots/so_follower", "other", cal(9))
    f = scan([("/dev/tty.x", "S")], tmp_path, ping=lambda p, ids: ([1, 2, 3, 4, 5, 6], []),
             bus_factory=lambda p, c: FakeBus(p, cal(1)))[0]  # fmt: skip
    assert f.role is None and f.match_deg and f.match_deg > 0 and f.problem is None
    assert f.matches == []


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


LSOF = "p4242\nn/dev/tty.usbmodemA\np99\nn/dev/cu.usbmodemB\np7\nn/dev/tty.usbmodemA\n"
PS = ("4242 T    /opt/anaconda3/envs/phi/bin/python health_check/roll_live.py\n"
      "99   S+   /opt/anaconda3/envs/phi/bin/lerobot-teleoperate --robot.type=so101_follower\n")


def test_port_holders_names_each_process_and_skips_studio(monkeypatch: Any) -> None:
    from phi_studio import detect

    monkeypatch.setattr(detect.os.path, "exists", lambda p: True)
    calls: list[list[str]] = []

    def run(argv: list[str]) -> str:
        calls.append(argv)
        return LSOF if argv[0] == "lsof" else PS

    held = detect.port_holders(["/dev/tty.usbmodemA", "/dev/cu.usbmodemB"], run=run, me=7)
    assert set(held) == {"/dev/tty.usbmodemA", "/dev/tty.usbmodemB"}  # cu folded into tty
    a = held["/dev/tty.usbmodemA"]
    assert [(h.pid, h.name, h.paused) for h in a] == [(4242, "roll_live.py", True)]  # 7 is Studio
    assert held["/dev/tty.usbmodemB"][0].name == "lerobot-teleoperate"
    assert "-F" in calls[0] and "/dev/cu.usbmodemA" in calls[0]  # both names of each device


def test_a_held_port_is_reported_and_never_pinged(tmp_path: Path) -> None:
    from phi_studio.detect import Holder

    pinged: list[str] = []

    def ping(p: str, ids: Any) -> tuple[list[int], list[int]]:
        pinged.append(p)
        return [1, 2, 3, 4, 5, 6], []

    h = Holder(4242, "python health_check/roll_live.py", True)
    found = scan([("/dev/tty.a", "A"), ("/dev/tty.b", "B")], tmp_path, ping=ping,
                 bus_factory=lambda p, c: FakeBus(p),
                 holders=lambda ps: {"/dev/tty.a": [h]})  # fmt: skip
    by = {f.port: f for f in found}
    assert pinged == ["/dev/tty.b"]
    problem = by["/dev/tty.a"].problem
    assert problem and "roll_live.py (process 4242, paused)" in problem
    assert by["/dev/tty.a"].public()["held_by"][0]["paused"] is True
    assert by["/dev/tty.b"].problem is None


class FakeLine:
    """A bus where motor `bad` garbles every packet while it holds torque (2026-10-05)."""

    def __init__(self, bad: int | None = None, mute: int | None = None) -> None:
        self.bad, self.mute = bad, mute
        self.torque = {i: 0 for i in range(1, 7)}
        self.goals: dict[int, int] = {}
        self.log: list[tuple[int, str, int]] = []
        self.closed = False

    def read(self, mid: int, name: str) -> int | None:
        if mid == self.mute:
            return None
        return {"pos": 2000 + mid, "volt": 118, "current": 3, "temp": 35, "status": 0,
                "torque": self.torque[mid]}[name]  # fmt: skip

    def write(self, mid: int, name: str, value: int) -> bool:
        self.log.append((mid, name, value))
        if name == "torque":
            self.torque[mid] = value
        if name == "goal":
            self.goals[mid] = value
        return mid != self.mute

    def sync_ok(self) -> bool:
        return not (self.bad and self.torque[self.bad])

    def close(self) -> None:
        self.closed = True


def run_check(line: FakeLine) -> list[Any]:
    from phi_studio.detect import check_arm

    t = [0.0]

    def sleep(dt: float) -> None:
        t[0] += dt

    seen: list[int] = []
    out = check_arm("/dev/tty.x", 1.0, line_factory=lambda p: line, clock=lambda: t[0],
                    sleep=sleep, on_motor=lambda r: seen.append(r.id))  # fmt: skip
    assert seen == [1, 2, 3, 4, 5, 6]
    return out


def test_motor_check_finds_the_one_motor_that_breaks_the_bus_under_torque() -> None:
    line = FakeLine(bad=2)
    out = run_check(line)
    assert [r.ok for r in out] == [True, False, True, True, True, True]
    assert out[1].bus_errors == out[1].polls > 0 and out[1].first_error_s == 0.0
    assert out[0].min_volt == 11.8 and out[0].max_current == 3
    assert all(line.goals[i] == 2000 + i for i in range(1, 7))  # each held where it was
    goal_then_torque = [(m, n) for m, n, v in line.log if n in ("goal", "torque") and v != 0]
    assert goal_then_torque[:2] == [(1, "goal"), (1, "torque")]
    assert all(v == 0 for v in line.torque.values()) and line.closed


def test_a_motor_that_does_not_answer_is_named_and_never_powered() -> None:
    line = FakeLine(mute=4)
    out = run_check(line)
    assert out[3].error == "does not answer" and not out[3].ok
    assert (4, "torque", 1) not in line.log
    assert [r.ok for r in out if r.id != 4] == [True] * 5


def test_studio_and_its_worker_are_never_reported_as_holders(monkeypatch: Any) -> None:
    from phi_studio import detect

    monkeypatch.setattr(detect.os.path, "exists", lambda p: True)
    run = lambda argv: LSOF if argv[0] == "lsof" else PS  # noqa: E731
    held = detect.port_holders(["/dev/tty.usbmodemA", "/dev/tty.usbmodemB"], run=run, me={7, 4242})
    assert set(held) == {"/dev/tty.usbmodemB"}


def test_scan_fails_closed_when_lsof_cannot_run(tmp_path: Path) -> None:
    pinged: list[str] = []

    def ping(p: str, ids: Any) -> tuple[list[int], list[int]]:
        pinged.append(p)
        return [1, 2, 3, 4, 5, 6], []

    def broken(ps: list[str]) -> dict[str, Any]:
        raise OSError("lsof exited 2")

    found = scan([("/dev/tty.a", "A")], tmp_path, ping=ping, bus_factory=lambda p, c: FakeBus(p),
                 holders=broken)  # fmt: skip
    assert not pinged  # a port nobody could vouch for is never pinged
    assert found[0].problem and "Could not check" in found[0].problem


def test_lsof_that_fails_is_an_error_not_an_empty_answer(monkeypatch: Any) -> None:
    from phi_studio import detect

    class Done:
        returncode, stdout, stderr = 2, "", "lsof: bad option"

    monkeypatch.setattr(detect.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(OSError):
        detect._run(["lsof", "-F", "pc", "/dev/tty.x"])
    Done.returncode = 1  # lsof's "nothing has it open"
    assert detect._run(["lsof", "/dev/tty.x"]) == ""


class StuckLine(FakeLine):
    """Motor 3 takes torque on and ignores torque off."""

    def write(self, mid: int, name: str, value: int) -> bool:
        if mid == 3 and name == "torque" and value == 0 and self.torque[3]:
            self.log.append((mid, name, value))
            return True  # the write "succeeds" but the register stays 1
        return super().write(mid, name, value)


def test_a_motor_that_keeps_torque_after_the_check_is_reported() -> None:
    out = run_check(StuckLine())
    assert not out[2].ok and "torque may still be on" in (out[2].error or "")
    assert [r.ok for r in out if r.id != 3] == [True] * 5


def search(found: dict[int, list[int]], clash: dict[int, list[int]] | None = None) -> Any:
    from phi_studio import detect

    asked: list[tuple[int, int]] = []

    def ping(port: str, ids: Any, baud: int) -> tuple[list[int], list[int]]:
        ids = list(ids)
        asked.append((baud, len(ids)))
        return ([i for i in found.get(baud, []) if i in ids],
                [i for i in (clash or {}).get(baud, []) if i in ids])

    r = detect.find_motors("/dev/tty.x", ping=ping)
    return r, asked


def test_motor_search_pings_every_id_at_1mbaud_and_low_ids_at_other_rates() -> None:
    r, asked = search({1_000_000: [1, 2, 3, 4, 5, 6]})
    assert asked[0] == (1_000_000, 254) and all(n == 21 for _, n in asked[1:]) and len(asked) == 8
    assert r.findings() == []


def test_a_motor_never_set_up_is_found_at_its_factory_rate() -> None:
    r, _ = search({1_000_000: [1, 3, 4, 5, 6], 115_200: [1]})
    f = r.findings()
    assert len(f) == 1 and "115200 baud" in f[0]["text"] and "Set motor ids" in f[0]["fix"]


def test_a_motor_on_a_wrong_id_and_a_dead_one_are_told_apart() -> None:
    r, _ = search({1_000_000: [1, 3, 4, 5, 6, 12]})
    want = "A motor answers at id 12, which an SO-101 does not use."
    assert [x["text"] for x in r.findings()] == [want]
    r, _ = search({1_000_000: [1, 3, 4, 5, 6]})
    assert r.findings()[0]["text"] == "Nothing answers for id 2 at any id or rate."


def test_two_motors_on_one_id_are_not_reported_dead() -> None:
    r, _ = search({1_000_000: [1, 4, 5, 6]}, {1_000_000: [3]})  # motor 2 was given id 3
    texts = [x["text"] for x in r.findings()]
    assert texts == ["Two motors answer together at id 3 (1000000 baud)."]
