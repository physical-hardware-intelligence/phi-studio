"""Detect arms: assignment rules, the config it writes, the ids it keeps, the files it writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from phi_studio import configedit, rigspec
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
    assert sent == [signal.SIGTERM, signal.SIGCONT, signal.SIGKILL]
    sent.clear()  # a process that obeys SIGTERM is never sent SIGKILL
    assert stop_process(5, kill=lambda p, s: sent.append(s), alive=lambda p: False,
                        sleep=lambda s: None)  # fmt: skip
    assert sent == [signal.SIGTERM, signal.SIGCONT]
