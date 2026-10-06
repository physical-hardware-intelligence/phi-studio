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


@pytest.fixture(autouse=True)
def plugged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every /dev path the tests name is plugged in, unless a test says otherwise."""
    monkeypatch.setattr(cmdcheck, "plugged_in", lambda p: True)


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


def test_two_arms_on_one_file_through_a_link_is_danger(tmp_path: Path) -> None:
    d = tmp_path / "robots/so_follower"
    d.mkdir(parents=True)
    (d / "phi_bi_follower_left.json").write_text("{}")
    (d / "phi_bi_follower_right.json").symlink_to("phi_bi_follower_left.json")
    cmd = ("lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemRF "
           "--robot.id=phi_bi_follower_right")
    v = cmdcheck.check(cmd, SPEC, tmp_path)[0]
    assert v.level == "danger" and "link" in v.message


def test_a_port_that_is_not_plugged_in_is_refused() -> None:
    cmd = ("lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemLF "
           "--robot.id=phi_bi_follower_left")
    v = cmdcheck.check(cmd, SPEC, ROOT, exists=lambda p: False)[0]
    assert v.level == "danger" and "Nothing is plugged in" in v.message


# The phi rig's shape (~/phi/robot-config.yaml): one id for both bimanual pairs, and the original
# follower's phi_bi_left.json a link to its single-arm phi_follower.json, on purpose.
PHI = BI.replace("phi_bi_follower", "phi_bi").replace("phi_bi_leader", "phi_bi")
PHI_SPEC = rigspec.parse(PHI)


def phi_root(tmp_path: Path) -> Path:
    f = tmp_path / "robots/so_follower"
    f.mkdir(parents=True)
    (f / "phi_follower.json").write_text("{}")
    (f / "phi_bi_left.json").symlink_to("phi_follower.json")
    return tmp_path


def test_studios_own_commands_pass_on_the_phi_rig(tmp_path: Path) -> None:
    """Review finding: every link was DANGER, so Run refused the rig's own commands."""
    root = phi_root(tmp_path)
    cmds = [c["cmd"] for c in rigspec.lerobot_commands(PHI_SPEC)]
    cmds.append(rigspec.rollout_command(PHI_SPEC, "/models/act", "pick the cube up"))
    seen = 0
    for cmd in cmds:
        vs = cmdcheck.check(cmd, PHI_SPEC, root)
        assert all(v.level == "ok" for v in vs), (cmd, [v.message for v in vs])
        seen += bool(vs)
    assert seen >= 4  # calibrate, teleoperate, record, rollout were really checked
    assert cmdcheck.program(cmds[-1]) in cmdcheck.CONNECTS


def test_the_prompt_takes_the_worst_arm_that_shares_its_id(tmp_path: Path) -> None:
    """Review finding: phi_bi_left names the left follower AND the left leader; the first match
    was the follower (ok) while the leader's ports were swapped."""
    cmd = ("lerobot-teleoperate --robot.type=bi_so_follower --robot.id=phi_bi "
           "--robot.left_arm_config.port=/dev/tty.usbmodemLF "
           "--robot.right_arm_config.port=/dev/tty.usbmodemRF "
           "--teleop.type=bi_so_leader --teleop.id=phi_bi "
           "--teleop.left_arm_config.port=/dev/tty.usbmodemRL "
           "--teleop.right_arm_config.port=/dev/tty.usbmodemLL")
    assert [v.level for v in cmdcheck.check(cmd, PHI_SPEC, phi_root(tmp_path))] == [
        "ok", "ok", "danger", "danger"]
    assert cmdcheck.at_prompt(cmd, "phi_bi_left", PHI_SPEC, tmp_path).level == "danger"


def test_another_calibration_dir_is_unknown() -> None:
    cmd = ("lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemLF "
           "--robot.id=phi_bi_follower_left --robot.calibration_dir=/tmp/other")
    v = cmdcheck.check(cmd, SPEC, ROOT)[0]
    assert v.level == "unknown" and "/tmp/other" in v.message


def test_a_command_line_ps_printed_without_its_quotes_is_still_read() -> None:
    """ps -o args= drops quoting: an apostrophe in the task made shlex fail, and the check saw
    no arms at all."""
    cmd = ("/opt/env/bin/python /opt/env/bin/lerobot-record --robot.type=so101_follower "
           "--robot.port=/dev/tty.usbmodemRF --robot.id=phi_bi_follower_left "
           "--dataset.single_task=pick the robot's cube")
    assert levels(cmd) == ["danger"]


@pytest.mark.parametrize("cmd, may", [
    ("/opt/env/bin/python /opt/env/bin/lerobot-calibrate --robot.type=x", True),
    ("python3 teleop_bi.py", True),
    ("/opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/Python.framework/Versions/3.14/Resources/"
     "Python.app/Contents/MacOS/Python /tmp/fakeprompt.py --teleop.id=x", True),  # ps, 2026-10-06
    ("lerobot-rollout --robot.type=x", True),
    ("cat /tmp/run.log", False),
    ("less run.log", False),
])  # fmt: skip
def test_only_python_can_be_lerobot_asking(cmd: str, may: bool) -> None:
    assert cmdcheck.may_prompt(cmd) is may
