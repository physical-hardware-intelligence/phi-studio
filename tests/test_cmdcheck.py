"""A LeRobot command must name each arm by its own port and its own id (cmdcheck.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from phi_studio import cmdcheck, rigspec

BI = """
robot:
  type: bi_so_follower
  id: phi_bi_follower
  left_arm_config: {port: /dev/tty.usbmodemLF}
  right_arm_config: {port: /dev/tty.usbmodemRF}
teleop:
  type: bi_so_leader
  id: phi_bi_leader
  left_arm_config: {port: /dev/tty.usbmodemLL}
  right_arm_config: {port: /dev/tty.usbmodemRL}
"""
SPEC = rigspec.parse(BI)
ROOT = Path("/cal")


def levels(cmd: str) -> list[str]:
    return [v.level for v in cmdcheck.check(cmd, SPEC, ROOT)]


def test_the_rigs_own_bimanual_command_is_ok() -> None:
    cmd = ("lerobot-teleoperate --robot.type=bi_so_follower --robot.id=phi_bi_follower "
           "--robot.left_arm_config.port=/dev/tty.usbmodemLF "
           "--robot.right_arm_config.port=/dev/tty.usbmodemRF "
           "--teleop.type=bi_so_leader --teleop.id=phi_bi_leader "
           "--teleop.left_arm_config.port=/dev/tty.usbmodemLL "
           "--teleop.right_arm_config.port=/dev/tty.usbmodemRL")
    assert levels(cmd) == ["ok"] * 4


def test_the_right_arm_run_with_the_left_arms_id_is_danger() -> None:
    # The 2026-10-05 fault: the right follower's port with the left follower's file.
    cmd = ("lerobot-calibrate --robot.type=so101_follower --robot.port /dev/cu.usbmodemRF "
           "--robot.id=phi_bi_follower_left")
    v = cmdcheck.check(cmd, SPEC, ROOT)[0]
    assert v.level == "danger" and v.arm == "right_follower"
    assert "Left Follower's id" in v.message and "phi_bi_follower_left.json" in v.message
    assert "--robot.id=phi_bi_follower_right" in v.fix


def test_a_leader_port_used_as_the_follower_is_danger() -> None:
    cmd = ("lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemLL "
           "--robot.id=x")
    assert levels(cmd) == ["danger"]


def test_a_port_robot_config_does_not_know_is_unknown() -> None:
    cmd = "lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.zz --robot.id=x"
    assert levels(cmd) == ["unknown"]


@pytest.mark.parametrize("cmd", [
    "lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/tty.zz",
    "lerobot-find-port", "ls -la", "echo 'unclosed"])
def test_commands_that_read_no_calibration_are_not_checked(cmd: str) -> None:
    assert cmdcheck.check(cmd, SPEC, ROOT) == []


def test_python_dash_m_is_read_as_the_cli() -> None:
    assert cmdcheck.program("/opt/x/bin/python -m lerobot.scripts.lerobot_teleoperate --x=1") == (
        "lerobot-teleoperate")
    assert cmdcheck.program("/opt/x/bin/lerobot-record --a=b") == "lerobot-record"


def test_the_prompt_is_matched_and_checked_by_its_id() -> None:
    line = ("Press ENTER to use provided calibration file associated with the id "
            "phi_bi_follower_right, or type 'c' and press ENTER to run calibration: ")
    m = cmdcheck.PROMPT.search(line)
    assert m and m.group(1) == "phi_bi_follower_right"
    good = ("lerobot-teleoperate --robot.type=bi_so_follower --robot.id=phi_bi_follower "
            "--robot.left_arm_config.port=/dev/tty.usbmodemLF "
            "--robot.right_arm_config.port=/dev/tty.usbmodemRF")
    assert cmdcheck.at_prompt(good, "phi_bi_follower_right", SPEC, ROOT).level == "ok"
    crossed = good.replace("LF ", "@@ ").replace("RF", "LF").replace("@@", "RF")
    v = cmdcheck.at_prompt(crossed, "phi_bi_follower_right", SPEC, ROOT)
    assert v.level == "danger" and v.arm == "left_follower"
    assert cmdcheck.at_prompt(None, "phi_bi_follower_right", SPEC, ROOT).level == "unknown"


def test_a_linked_file_is_danger(tmp_path: Path) -> None:
    d = tmp_path / "robots/so_follower"
    d.mkdir(parents=True)
    (d / "phi_bi_follower_left.json").write_text("{}")
    (d / "phi_bi_follower_right.json").symlink_to("phi_bi_follower_left.json")
    cmd = ("lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemRF "
           "--robot.id=phi_bi_follower_right")
    v = cmdcheck.check(cmd, SPEC, tmp_path)[0]
    assert v.level == "danger" and "link" in v.message
