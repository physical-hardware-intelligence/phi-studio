"""Detect arms: assignment rules, the config it writes, the ids it keeps, the files it writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from phi_studio import configedit, detect, rigspec
from phi_studio.detect import Found, calibration_library
from phi_studio.identity import JointCal, load_calibration
from phi_studio.rig import JOINTS
from phi_studio.rig_api import (
    RigApi,
    RigError,
    _section,
    check_assignment,
    distinct_ids,
    existing_id,
)


def cal(seed: int) -> dict[str, JointCal]:
    return {j: JointCal(i + 1, 0, 100 * seed + i, 800 + i, 3200 - i) for i, j in enumerate(JOINTS)}


def found(port: str, seed: int = 1, problem: bool = False) -> Found:
    ids = [2, 3, 4, 5] if problem else [1, 2, 3, 4, 5, 6]
    return Found(port, "SN", ids, [6] if problem else [], registers=None if problem else cal(seed))


PORTS = ("/dev/tty.a", "/dev/tty.b", "/dev/tty.c", "/dev/tty.d")
FOUR = {p: found(p, i) for i, p in enumerate(PORTS)}
BI = [{"port": "/dev/tty.a", "role": "follower", "side": "left"},
      {"port": "/dev/tty.b", "role": "follower", "side": "right"},
      {"port": "/dev/tty.c", "role": "leader", "side": "left"},
      {"port": "/dev/tty.d", "role": "leader", "side": "right"}]  # fmt: skip


def test_two_pairs_with_one_left_and_one_right_each_are_accepted() -> None:
    assert check_assignment(BI, FOUR) == BI


def test_one_pair_drops_any_side() -> None:
    got = check_assignment([dict(BI[0]), dict(BI[2])], FOUR)
    assert [a["side"] for a in got] == [None, None]


@pytest.mark.parametrize("arms, why", [
    ([], "Choose a role"),
    ([{**BI[0], "role": "boss"}], "needs a role"),
    ([BI[0], BI[1], BI[2]], "1 leaders and 2 followers"),
    ([BI[0], {**BI[1], "side": "left"}, BI[2], BI[3]], "one follower must be left"),
    ([BI[0], {**BI[1], "port": "/dev/tty.a"}, BI[2], BI[3]], "chosen twice"),
    ([{**BI[0], "port": "/dev/tty.zz"}, BI[2]], "not an arm Studio found"),
    ([{**BI[0], "side": "up"}, BI[2]], "not a side"),
])  # fmt: skip
def test_an_assignment_studio_cannot_drive_is_refused(arms: list, why: str) -> None:
    with pytest.raises(RigError, match=why):
        check_assignment(arms, FOUR)


def test_an_arm_with_a_motor_problem_cannot_be_assigned() -> None:
    bad = {**FOUR, "/dev/tty.a": found("/dev/tty.a", problem=True)}
    with pytest.raises(RigError, match="Two motors answer to id 6"):
        check_assignment(BI, bad)


def test_the_sections_parse_as_the_bimanual_rig_lerobot_expects_with_a_step_limit() -> None:
    text = _section("follower", "phi_bi", BI[:2]) + _section("leader", "phi_bi", BI[2:])
    spec = rigspec.parse(text)
    assert spec.problems == () and spec.bimanual
    got = {a.key: (a.port, a.lerobot_id, a.type) for a in spec.arms}
    assert got["right_follower"] == ("/dev/tty.b", "phi_bi_right", "bi_so_follower")
    assert got["left_leader"] == ("/dev/tty.c", "phi_bi_left", "bi_so_leader")
    raw = yaml.safe_load(text)
    assert raw["robot"]["left_arm_config"]["max_relative_target"] > 0
    assert "max_relative_target" not in raw["teleop"]["left_arm_config"]
    one = rigspec.parse(_section("follower", "f", [BI[0]]) + _section("leader", "l", [BI[2]]))
    assert [a.type for a in one.arms] == ["so101_follower", "so101_leader"] and not one.bimanual


def write_cal(root: Path, rel: str, c: dict[str, JointCal]) -> Path:
    p = root / f"{rel}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({j: v._asdict() for j, v in c.items()}))
    return p


def test_existing_id_keeps_the_files_that_already_hold_these_arms(tmp_path: Path) -> None:
    # the 2026-10-05 rig: phi_bi_left is a symlink to phi_follower, which matches the left arm too
    write_cal(tmp_path, "robots/so_follower/phi_follower", cal(0))
    (tmp_path / "robots/so_follower/phi_bi_left.json").symlink_to("phi_follower.json")
    write_cal(tmp_path, "robots/so_follower/phi_bi_right", cal(1))
    write_cal(tmp_path, "teleoperators/so_leader/phi_bi_left", cal(2))
    write_cal(tmp_path, "teleoperators/so_leader/phi_bi_right", cal(9))  # wrong for tty.d
    lib = calibration_library(tmp_path)
    assert existing_id("follower", BI, FOUR, lib) == "phi_bi"  # phi_follower has no _left/_right
    assert existing_id("leader", BI, FOUR, lib) is None  # only one side matches
    single = [dict(BI[0], side=None), dict(BI[2], side=None)]
    assert existing_id("follower", single, FOUR, lib, prefer="phi_follower") == "phi_follower"


CONFIG = """# my rig
machine: mac
robot:                 # the follower
  type: so101_follower
  id: phi_follower
  port: /dev/tty.old1  # found by hand

# the leader, moved by hand
teleop:
  type: so101_leader
  id: phi_leader
  port: /dev/tty.old2

cameras:
  wrist: {type: opencv, index_or_path: TBD}
dataset: {hf_user: me}
"""


def test_replacing_sections_keeps_every_other_line_and_backs_up(tmp_path: Path) -> None:
    p = tmp_path / "robot-config.yaml"
    p.write_text(CONFIG)
    sections = {"robot": _section("follower", "phi_bi", BI[:2]),
                "teleop": _section("leader", "phi_bi", BI[2:])}  # fmt: skip
    backup = configedit.replace_sections(p, sections, tmp_path / "bk")
    assert backup.read_text() == CONFIG
    new = p.read_text()
    for kept in ("# my rig", "machine: mac", "# the leader, moved by hand", "cameras:",
                 "dataset: {hf_user: me}"):  # fmt: skip
        assert kept in new
    assert "/dev/tty.old" not in new
    assert yaml.safe_load(new)["cameras"] == yaml.safe_load(CONFIG)["cameras"]
    assert len(rigspec.parse(new).arms) == 4


def test_a_section_that_would_not_read_back_leaves_the_file_alone(tmp_path: Path) -> None:
    p = tmp_path / "robot-config.yaml"
    p.write_text(CONFIG)
    with pytest.raises(configedit.ConfigEditError):
        configedit.replace_sections(p, {"robot": "robot:\n  type: [unclosed\n"}, tmp_path / "bk")
    assert p.read_text() == CONFIG and not (tmp_path / "bk").exists()


class _Studio:
    spec: dict[str, Any] = {}

    def _note(self, *a: Any) -> None:
        pass


def test_each_arm_gets_its_own_registers_a_link_is_replaced_never_its_target(
    tmp_path: Path,
) -> None:
    api = RigApi(_Studio())  # type: ignore[arg-type]
    api.studio.spec = {"cal_root": str(tmp_path)}
    api.found = dict(FOUR)
    old = write_cal(tmp_path, "robots/so_follower/phi_bi_left", cal(9))
    target = write_cal(tmp_path, "teleoperators/so_leader/yash_leader", cal(7))
    link = tmp_path / "teleoperators/so_leader/phi_bi_right.json"
    link.symlink_to("yash_leader.json")
    out = api._write_calibrations(BI, {"follower": "phi_bi", "leader": "phi_bi"})
    by = {o["port"]: o for o in out}
    assert by["/dev/tty.a"]["action"] == "replaced" and Path(by["/dev/tty.a"]["backup"]).is_file()
    assert load_calibration(old) == FOUR["/dev/tty.a"].registers
    assert by["/dev/tty.c"]["action"] == "written"
    assert not link.is_symlink() and load_calibration(link) == FOUR["/dev/tty.d"].registers
    assert load_calibration(target) == cal(7)  # the file the link pointed at is untouched
    again = api._write_calibrations(BI, {"follower": "phi_bi", "leader": "phi_bi"})
    assert {o["action"] for o in again} == {"unchanged"}


def test_one_id_for_both_roles_gives_the_leader_its_own() -> None:
    from phi_studio.onboard_api import check_answers

    ids = distinct_ids({"follower": "phi_bi", "leader": "phi_bi"})
    assert ids == {"follower": "phi_bi", "leader": "phi_bi_leader"}
    check_answers({"name": "Bench", "layout": "bimanual", "ids": ids, "ports": {}})  # accepted
    assert distinct_ids({"follower": "a", "leader": "b"}) == {"follower": "a", "leader": "b"}


def test_stop_process_terms_then_continues_a_paused_one_and_kills_if_needed() -> None:
    import signal

    from phi_studio.rig_api import stop_process

    sent: list[int] = []
    state = {"alive": True}

    def kill(pid: int, sig: int) -> None:
        sent.append(sig)
        if sig == signal.SIGKILL:
            state["alive"] = False

    never_dies = {"alive": lambda p: state["alive"], "sleep": lambda s: None}
    assert stop_process(5, wait_s=0.3, kill=kill, **never_dies)
    assert sent == [signal.SIGINT, signal.SIGCONT, signal.SIGTERM, signal.SIGKILL]
    sent.clear()  # a process that obeys Ctrl-C (LeRobot's torque-off path) gets nothing else
    assert stop_process(5, kill=lambda p, s: sent.append(s), alive=lambda p: False,
                        sleep=lambda s: None)  # fmt: skip
    assert sent == [signal.SIGINT, signal.SIGCONT]


def test_registers_a_calibration_run_never_finished_are_named() -> None:
    from phi_studio.identity import unfinished

    assert unfinished(cal(1)) is None
    reset = {j: c._replace(homing_offset=0, range_min=0, range_max=4095) for j, c in cal(1).items()}
    assert "shoulder pan still has the factory range" in (unfinished(reset) or "")
    roll_only = cal(1) | {"wrist_roll": reset["wrist_roll"]}  # LeRobot keeps wrist roll 0..4095
    assert unfinished(roll_only) is None
    flat = cal(1) | {"gripper": cal(1)["gripper"]._replace(range_min=2000, range_max=2000)}
    assert "gripper has no range" in (unfinished(flat) or "")


def test_before_lerobot_calibrates_a_linked_file_becomes_its_own_copy(tmp_path: Path) -> None:
    """LeRobot saves with open(path, "w"), which writes through a link: recalibrating
    phi_bi_left.json rewrote phi_follower.json too."""
    from phi_studio.rig_api import _own_copy

    target = write_cal(tmp_path, "robots/so_follower/phi_follower", cal(3))
    link = tmp_path / "robots/so_follower/phi_bi_left.json"
    link.symlink_to("phi_follower.json")
    unlinked, kept = _own_copy(link, tmp_path)
    assert unlinked and not link.is_symlink() and load_calibration(link) == cal(3)
    link.write_text("{}")  # what LeRobot's save does next
    assert load_calibration(target) == cal(3)
    assert kept is not None and load_calibration(kept) == cal(3)
    none = tmp_path / "robots/so_follower/none.json"
    assert _own_copy(none, tmp_path) == (False, None)  # a new arm


class _Client:
    def __init__(self) -> None:
        self.pushed: list[dict[str, Any]] = []

    def push(self, m: dict[str, Any]) -> None:
        self.pushed.append(m)


def _cal_api(tmp_path: Path, monkeypatch: Any, regs: dict[str, JointCal]) -> RigApi:
    import phi_studio.rig_api as R

    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(_section("follower", "phi_bi", BI[:2]) + _section("leader", "phi_bi", BI[2:]))
    api = RigApi(_Studio())  # type: ignore[arg-type]
    api.studio.spec = {"cal_root": str(tmp_path / "cal"), "kind": "lerobot"}
    api.studio.last = {}  # type: ignore[attr-defined]
    monkeypatch.setattr(R, "config_target", lambda studio: cfg)

    async def idle(studio: Any) -> None:
        return None

    monkeypatch.setattr(R, "lerobot_busy", idle)
    monkeypatch.setattr(R.detect, "serial_ports", lambda: [(p, "SN") for p in PORTS])
    monkeypatch.setattr(R.detect, "scan", lambda ports, root, holders=None: [
        Found(ports[0][0], "SN", [1, 2, 3, 4, 5, 6], [], registers=regs)])
    return api


def test_verify_compares_the_arms_registers_with_its_file(tmp_path: Path, monkeypatch: Any) -> None:
    import asyncio

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    c = _Client()
    asyncio.run(api.cal_verify(c, {"arm": "left_follower"}))  # type: ignore[arg-type]
    assert c.pushed[-1]["file"] is False and c.pushed[-1]["exact"] is False
    write_cal(tmp_path / "cal", "robots/so_follower/phi_bi_left", cal(1))
    asyncio.run(api.cal_verify(c, {"arm": "left_follower"}))  # type: ignore[arg-type]
    v = c.pushed[-1]
    assert v["exact"] and v["max_deg"] == 0 and v["unfinished"] is None and v["problem"] is None


def test_save_from_motors_writes_finished_registers_and_refuses_unfinished_ones(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import asyncio

    reset = {j: c._replace(homing_offset=0, range_min=0, range_max=4095) for j, c in cal(1).items()}
    api = _cal_api(tmp_path, monkeypatch, reset)
    path = tmp_path / "cal/robots/so_follower/phi_bi_left.json"
    with pytest.raises(RigError, match="factory range"):
        asyncio.run(api.cal_from_motors(_Client(), {"arm": "left_follower"}))  # type: ignore[arg-type]
    assert not path.exists()
    api = _cal_api(tmp_path, monkeypatch, cal(1))
    c = _Client()
    asyncio.run(api.cal_from_motors(c, {"arm": "left_follower"}))  # type: ignore[arg-type]
    assert load_calibration(path) == cal(1) and c.pushed[-1]["exact"]


def test_verify_refuses_while_studio_holds_the_arms(tmp_path: Path, monkeypatch: Any) -> None:
    import asyncio

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    api.studio.last = {"state": {"state": "IDENTIFIED"}}  # type: ignore[attr-defined]
    with pytest.raises(RigError, match="worker holds their ports"):
        asyncio.run(api.cal_verify(_Client(), {"arm": "left_follower"}))  # type: ignore[arg-type]


def _holding(monkeypatch: Any, matches: list[str]) -> None:
    import phi_studio.rig_api as R

    monkeypatch.setattr(R.detect, "scan", lambda ports, root, holders=None: [
        Found(ports[0][0], "SN", [1, 2, 3, 4, 5, 6], [], registers=cal(1), matches=matches)])


def test_save_from_motors_refuses_motors_that_hold_another_arms_calibration(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import asyncio

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    _holding(monkeypatch, ["teleoperators/so_leader/phi_bi_left"])  # the left leader's file
    path = tmp_path / "cal/robots/so_follower/phi_bi_left.json"
    with pytest.raises(RigError, match="cables swapped"):
        asyncio.run(api.cal_from_motors(_Client(), {"arm": "left_follower"}))  # type: ignore[arg-type]
    assert not path.exists()


def test_a_copy_of_its_own_file_is_not_called_a_swap(tmp_path: Path, monkeypatch: Any) -> None:
    import asyncio

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    _holding(monkeypatch, ["robots/so_follower/phi_bi_left", "robots/so_follower/phi_bi_right"])
    c = _Client()
    asyncio.run(api.cal_verify(c, {"arm": "left_follower"}))  # type: ignore[arg-type]
    assert c.pushed[-1]["problem"] is None


@pytest.mark.parametrize("holders, why", [
    (lambda ports: {ports[0]: [detect.Holder(4242, "python roll_live.py", True)]}, "4242"),
    (lambda ports: (_ for _ in ()).throw(OSError("lsof failed")), "Could not check"),
])  # fmt: skip
def test_motor_check_looks_again_for_a_holder_before_powering(
    monkeypatch: Any, holders: Any, why: str
) -> None:
    import asyncio

    import phi_studio.rig_api as R

    async def idle(studio: Any) -> None:
        return None

    monkeypatch.setattr(R, "lerobot_busy", idle)
    powered: list[str] = []
    monkeypatch.setattr(R.detect, "check_arm", lambda port, s, **kw: powered.append(port))
    api = RigApi(_Studio())  # type: ignore[arg-type]
    api.studio.spec = {"kind": "lerobot"}
    api.studio.last = {}  # type: ignore[attr-defined]
    api.found = dict(FOUR)  # the scan saw no holder; one appeared since
    api._holders = holders  # type: ignore[method-assign]
    with pytest.raises(RigError, match=why):
        asyncio.run(api.check(_Client(), {"port": "/dev/tty.a"}))  # type: ignore[arg-type]
    assert not powered


def test_calibration_files_are_listed_moved_aside_restored_and_installed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import asyncio

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    root = tmp_path / "cal"
    write_cal(root, "robots/so_follower/phi_bi_left", cal(1))  # the left follower's own file
    write_cal(root, "robots/so_follower/yash_follower", cal(2))
    shared = tmp_path / "phi/configs/calibration"
    write_cal(shared, "robots/so_follower/phi_bi_right", cal(3))
    c = _Client()
    run = lambda cmd, **m: asyncio.run(getattr(api, cmd)(c, m))  # noqa: E731
    run("calfiles", source=str(shared))
    v = c.pushed[-1]
    by = {f["rel"]: f for f in v["files"]}
    assert by["robots/so_follower/phi_bi_left.json"]["used_by"] == ["left_follower"]
    assert by["robots/so_follower/yash_follower.json"]["unused"]
    assert v["shared"]["rows"][0]["state"] == "new"
    with pytest.raises(RigError, match="left_follower uses"):
        run("calfiles_archive", files=["robots/so_follower/phi_bi_left.json"])
    run("calfiles_archive", files=["robots/so_follower/yash_follower.json"])
    assert not (root / "robots/so_follower/yash_follower.json").exists()
    name = c.pushed[-1]["done"]["archive"]
    run("calfiles_restore", name=name)
    assert (root / "robots/so_follower/yash_follower.json").is_file()
    run("calfiles_install", source=str(shared), files=["robots/so_follower/phi_bi_right.json"])
    assert load_calibration(root / "robots/so_follower/phi_bi_right.json") == cal(3)
    assert c.pushed[-1]["done"]["copied"] == ["robots/so_follower/phi_bi_right.json"]


def test_calibration_files_are_not_moved_while_lerobot_runs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import asyncio

    import phi_studio.rig_api as R

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    write_cal(tmp_path / "cal", "robots/so_follower/old", cal(2))

    async def busy(studio: Any) -> str:
        return "lerobot-calibrate"

    monkeypatch.setattr(R, "lerobot_busy", busy)
    with pytest.raises(RigError, match="lerobot-calibrate"):
        msg = {"files": ["robots/so_follower/old.json"]}
        asyncio.run(api.calfiles_archive(_Client(), msg))  # type: ignore[arg-type]
    assert (tmp_path / "cal/robots/so_follower/old.json").is_file()


def test_motor_search_reports_findings_and_the_setup_command_for_the_arm(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import asyncio

    import phi_studio.rig_api as R

    api = _cal_api(tmp_path, monkeypatch, cal(1))
    api._holders = lambda ports: {}  # type: ignore[method-assign]

    def find(port: str, on_rate: Any = None) -> Any:
        on_rate(1_000_000)
        return R.detect.MotorSearch(port, found={1_000_000: [1, 3, 4, 5, 6], 115_200: [1]})

    monkeypatch.setattr(R.detect, "find_motors", find)
    c = _Client()
    asyncio.run(api.find_motors(c, {"port": "/dev/tty.a"}))  # type: ignore[arg-type]
    assert c.pushed[0]["done"] is False and c.pushed[0]["baud"] == 1_000_000
    end = c.pushed[-1]
    assert end["done"] and "115200 baud" in end["findings"][0]["text"]
    assert end["setup"] == [{"role": "follower", "cmd": "lerobot-setup-motors "
                             "--robot.type=so101_follower --robot.port=/dev/tty.a"}]
    asyncio.run(api.find_motors(c, {"port": "/dev/tty.zz"}))  # type: ignore[arg-type]
    assert [x["role"] for x in c.pushed[-1]["setup"]] == ["follower", "leader"]
