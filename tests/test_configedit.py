"""robot-config.yaml edits that keep every comment, blank line and key where Parv wrote it."""

from __future__ import annotations

import hashlib
import os
from functools import reduce
from pathlib import Path
from typing import Any

import pytest
import yaml

from phi_studio import configedit, rigspec
from phi_studio.configedit import ConfigEditError, camera_source_path, port_path, set_values

SINGLE = """\
# Phi rig, single arm.

robot:                       # the follower
  type: so101_follower
  id: phi_follower
  port: /dev/tty.usbmodemF   # found by unplugging
  cameras:
    front: {type: opencv, index_or_path: TBD, width: 640, height: 480, fps: 30}  # C920

teleop:
  type: so101_leader
  id: phi_leader
  port: /dev/tty.usbmodemL
"""

BIMANUAL = """\
robot:
  type: bi_so_follower
  id: bimanual_follower
  left_arm_config:
    port: /dev/tty.usbmodemFL    # left follower
    cameras:
      wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  right_arm_config:
    port: /dev/tty.usbmodemFR
  cameras:
    top: {type: intelrealsense, serial_number_or_name: TBD, width: 640, height: 480, fps: 30}

# Leaders, moved by hand.
teleop:
  type: bi_so_leader
  id: bimanual_leader
  left_arm_config:
    port: /dev/tty.usbmodemLL
  right_arm_config:
    port: /dev/tty.usbmodemLR    # right leader
"""

# The shape of this Mac's robot-config.yaml (header comments with an emoji, notes outside the rig,
# comments aligned in one column, a top-level block-style cameras: section, quoted strings, a
# list). WHY not a copy: that file is git-excluded on purpose, and this repo is public.
MAC = """\
# SO-101 rig config for one Mac. Machine-specific, git-excluded.
#
# \U0001f6a8 Ports change per USB socket and per reboot: re-find them with the unplug wizard.

machine: test-mac

env:
  # the plain `conda activate` is broken here
  activate: "source /opt/conda/etc/profile.d/conda.sh && conda activate phi"
  conda_env: phi
  repo: /home/someone/phi               # the club repo

robot:                                   # the follower arm
  type: so101_follower                   # registered CLI alias
  id: phi_follower
  port: /dev/tty.usbmodem5B7B0096441     # found with lerobot-find-port

teleop:                                  # the leader arm
  type: so101_leader
  id: phi_leader
  port: /dev/tty.usbmodem5B7B0139311     # found with lerobot-find-port

# Keys below are dataset feature names.
cameras:
  wrist:
    hardware: "32x32 mm USB module, gripper-mounted"
    type: opencv
    index_or_path: TBD                   # confirm each session
    width: 640
    height: 480
    fps: 30
  front:
    hardware: "USB webcam (clamp mount)"
    type: opencv
    index_or_path: TBD                   # confirm each session
    width: 640
    height: 480
    fps: 30

# WHY 640x480: USB bandwidth.

dataset:
  hf_user: someone
  repo_id: TBD                           # set per session
  previous:
    - "someone/old_set"                  # 53 eps
  naming_note: "old set used key `wrist_cam`"

hardware_notes:
  estop: "the follower USB cable is the kill switch"
  calibration:
    location: "~/.cache/huggingface/lerobot/calibration"
"""

# Cameras on both robot: and the top level. rigspec reads robot.cameras and ignores the other.
BOTH_CAMERA_BLOCKS = """\
robot:
  type: so101_follower
  id: f
  port: /dev/a
  cameras:
    front: {type: opencv, index_or_path: 0}
teleop:
  type: so101_leader
  id: l
  port: /dev/b
cameras:
  front: {type: opencv, index_or_path: 9}
"""

# robot.cameras holds no camera mapping, so rigspec falls back to the top-level block.
EMPTY_ROBOT_CAMERAS = BOTH_CAMERA_BLOCKS.replace("{type: opencv, index_or_path: 0}", "TBD")

REAL = Path("/Users/parvpatodia/phi/robot-config.yaml")


def _get(data: Any, path: tuple[str, ...]) -> Any:
    return reduce(lambda d, k: d[k], path, data)


def _changed(before: str, after: str) -> list[tuple[str, str]]:
    """(old, new) for each line that differs; the line count must not change."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    assert len(a) == len(b)
    return [(x, y) for x, y in zip(a, b, strict=True) if x != y]


def _rig_changes(spec: rigspec.RigSpec) -> dict[tuple[str, ...], Any]:
    """A new port for every arm and a new device for every camera, through the path helpers."""
    out: dict[tuple[str, ...], Any] = {port_path(a): f"/dev/new_{a.key}" for a in spec.arms}
    for i, c in enumerate(spec.cameras):
        out[camera_source_path(c, spec)] = i + 2
    return out


# -- editing values in place ---------------------------------------------------------------------


def test_single_arm_ports_change_and_every_other_byte_stays():
    out = set_values(SINGLE, {("robot", "port"): "/dev/tty.usbmodem5A7A0185681",
                              ("teleop", "port"): "/dev/tty.usbmodem58FA0829321"})  # fmt: skip
    # The comment keeps one space when the value grows past its column.
    assert out == SINGLE.replace(
        "  port: /dev/tty.usbmodemF   # found by unplugging\n",
        "  port: /dev/tty.usbmodem5A7A0185681 # found by unplugging\n",
    ).replace("  port: /dev/tty.usbmodemL\n", "  port: /dev/tty.usbmodem58FA0829321\n")
    assert [a.port for a in rigspec.parse(out).arms] == [
        "/dev/tty.usbmodem5A7A0185681",
        "/dev/tty.usbmodem58FA0829321",
    ]


def test_bimanual_all_four_ports():
    spec = rigspec.parse(BIMANUAL)
    new = {a.key: f"/dev/tty.usbmodem{a.key.upper()}" for a in spec.arms}
    out = set_values(BIMANUAL, {port_path(a): new[a.key] for a in spec.arms})
    assert {a.key: a.port for a in rigspec.parse(out).arms} == new
    changed = _changed(BIMANUAL, out)
    assert len(changed) == 4 and all(old.strip().startswith("port:") for old, _ in changed)
    # Trailing comments survive on the lines that had them.
    assert "port: /dev/tty.usbmodemLEFT_FOLLOWER # left follower\n" in out
    assert "port: /dev/tty.usbmodemRIGHT_LEADER # right leader\n" in out


def test_mac_shaped_config_keeps_comments_aligned():
    spec = rigspec.parse(MAC)
    out = set_values(MAC, _rig_changes(spec))
    changed = _changed(MAC, out)
    assert len(changed) == 4  # two ports, two cameras; nothing else moved
    for old, new in changed:
        assert old.index("#") == new.index("#") == 41  # the comment column holds
        assert old[old.index("#") :] == new[new.index("#") :]
    assert "    index_or_path: 2                     # confirm each session\n" in out
    after = rigspec.parse(out)
    assert [c.source for c in after.cameras] == [2, 3]
    assert [a.port for a in after.arms] == ["/dev/new_follower", "/dev/new_leader"]


@pytest.mark.skipif(not REAL.exists(), reason="this Mac's robot-config.yaml is not here")
def test_this_macs_real_config_read_only():
    raw, mtime = REAL.read_bytes(), REAL.stat().st_mtime_ns
    text = raw.decode("utf-8")
    spec = rigspec.parse(text)
    changes = _rig_changes(spec)
    out = set_values(text, changes)  # in memory only; the file is never written
    changed = _changed(text, out)
    assert len(changed) == len(changes)
    for old, new in changed:
        if "#" in old:
            assert old[old.index("#") :] == new[new.index("#") :]
    after = rigspec.parse(out)
    assert {a.key: a.port for a in after.arms} == {a.key: f"/dev/new_{a.key}" for a in spec.arms}
    assert [c.source for c in after.cameras] == [i + 2 for i in range(len(spec.cameras))]
    assert hashlib.sha256(REAL.read_bytes()).digest() == hashlib.sha256(raw).digest()
    assert REAL.stat().st_mtime_ns == mtime


def test_flow_mapping_camera_value():
    path = ("robot", "cameras", "front", "index_or_path")
    out = set_values(SINGLE, {path: 2})
    assert out == SINGLE.replace("index_or_path: TBD,", "index_or_path: 2,")
    # A comma would end the value inside {...}, so that text is quoted.
    out = set_values(SINGLE, {path: "/dev/video, 2"})
    assert "{type: opencv, index_or_path: '/dev/video, 2', width: 640," in out
    assert _get(yaml.safe_load(out), path) == "/dev/video, 2"


@pytest.mark.parametrize(
    "value, text",
    [
        ("/dev/tty.usbmodem5A7A0185681", "/dev/tty.usbmodem5A7A0185681"),
        (2, "2"),
        ("2", "'2'"),
        (True, "true"),
        ("true", "'true'"),
        (None, "null"),
        (1.5, "1.5"),
        ("", "''"),
        ("a, b", "'a, b'"),
        ("key: v", "'key: v'"),
        ("# x", "'# x'"),
        ("it's", "it's"),
    ],
)
def test_values_read_back_as_the_same_type(value, text):
    for path in (("robot", "port"), ("robot", "cameras", "front", "index_or_path")):
        out = set_values(SINGLE, {path: value})
        got = _get(yaml.safe_load(out), path)
        assert got == value and type(got) is type(value)
        assert f"{path[-1]}: {text}" in out


# -- adding keys ---------------------------------------------------------------------------------


def test_missing_key_is_added_at_the_end_of_its_block_with_sibling_indent():
    no_port = SINGLE.replace("  port: /dev/tty.usbmodemF   # found by unplugging\n", "")
    out = set_values(no_port, {("robot", "port"): "/dev/a"})
    # After robot's last line (a nested camera), before the blank line and teleop:.
    assert out == no_port.replace("fps: 30}  # C920\n", "fps: 30}  # C920\n  port: /dev/a\n")
    # Two new keys in one section keep the order asked for.
    two = set_values(no_port, {("teleop", "use_degrees"): False,
                               ("teleop", "calibration_dir"): "/c"})  # fmt: skip
    assert two == no_port + "  use_degrees: false\n  calibration_dir: /c\n"


def test_missing_key_in_a_nested_and_a_parent_block_at_the_same_line():
    text = "robot:\n  type: bi_so_follower\n  left_arm_config:\n    port: /dev/a\nteleop: {}\n"
    out = set_values(text, {("robot", "id"): "b",
                            ("robot", "left_arm_config", "use_degrees"): True})  # fmt: skip
    # The deeper key goes first, or it would land inside the new `id:` line.
    assert out == text.replace("port: /dev/a\n", "port: /dev/a\n    use_degrees: true\n  id: b\n")


def test_missing_key_in_a_flow_mapping():
    text = SINGLE.replace("index_or_path: TBD, ", "")
    out = set_values(text, {("robot", "cameras", "front", "index_or_path"): 0})
    assert "{type: opencv, width: 640, height: 480, fps: 30, index_or_path: 0}  # C920" in out
    empty = set_values("robot:\n  cameras:\n    front: {}\n",
                       {("robot", "cameras", "front", "a"): 1})  # fmt: skip
    assert empty == "robot:\n  cameras:\n    front: {a: 1}\n"


def test_missing_key_edge_layouts():
    # An empty value, with a comment after it.
    assert set_values("robot:\n  port:   # tbd\n", {("robot", "port"): "/dev/a"}) == (
        "robot:\n  port: /dev/a # tbd\n"
    )
    # No newline at the end of the file.
    assert set_values("robot:\n  type: x", {("robot", "port"): "/dev/a"}) == (
        "robot:\n  type: x\n  port: /dev/a\n"
    )
    # A block scalar as the last value: the new key goes after it and the scalar is unchanged.
    text = "robot:\n  type: x\n  notes: |\n    hi\n\nteleop:\n  type: y\n"
    out = set_values(text, {("robot", "port"): "/dev/a"})
    assert yaml.safe_load(out)["robot"] == {"type": "x", "notes": "hi\n", "port": "/dev/a"}
    # Windows line endings stay Windows line endings.
    crlf = set_values("robot:\r\n  type: x\r\n", {("robot", "port"): "/dev/a"})
    assert crlf == "robot:\r\n  type: x\r\n  port: /dev/a\r\n"


# -- refusals ------------------------------------------------------------------------------------

LEGACY = """\
robot:
  type: bi_so100_follower
  id: b
  left_arm_port: /dev/a
  right_arm_port: /dev/b
"""


@pytest.mark.parametrize(
    "text, path, value, why",
    [
        ("base: &b {port: /dev/a}\nrobot: *b\n", ("robot", "port"), "/dev/x", "anchor or alias"),
        ("robot:\n  port: &p /dev/a\nteleop:\n  port: *p\n", ("teleop", "port"), "/dev/x",
         "anchor or alias"),
        ("base: &b {type: x}\nrobot:\n  <<: *b\n  port: /dev/a\n", ("robot", "port"), "/dev/x",
         "merge key"),
        ("robot:\n  port: |\n    /dev/a\n", ("robot", "port"), "/dev/x", "more than one line"),
        ("robot:\n  port: /dev/a\n    b\n", ("robot", "port"), "/dev/x", "more than one line"),
        ('robot:\n  port: "/dev/a\n    b"\n', ("robot", "port"), "/dev/x", "more than one line"),
        ("robot:\n  port:\t/dev/a\n", ("robot", "port"), "/dev/x", "tab on line 2"),
        ("robot:\n  arms:\n    - port: /dev/a\n", ("robot", "arms", "0", "port"), "/dev/x",
         "is a list"),
        (SINGLE, ("robot", "left_arm_config", "port"), "/dev/x",
         "no robot.left_arm_config section"),
        ("teleop:\n  port: /dev/a\n", ("robot", "port"), "/dev/x", "no robot section"),
        ("robot:\n  left_arm_config: TBD\n", ("robot", "left_arm_config", "port"), "/dev/x",
         "no robot.left_arm_config section"),
        (SINGLE, ("robot", "cameras"), 1, "not a single value"),
        ("robot:\n  port: /dev/a\n  port: /dev/b\n", ("robot", "port"), "/dev/x", "appears twice"),
        ("robot: [\n", ("robot", "port"), "/dev/x", "not valid YAML"),
        ("- a\n- b\n", ("robot", "port"), "/dev/x", "not a mapping"),
        (SINGLE, ("robot", "port"), "a\nb", "one-line"),
        (SINGLE, ("robot", "port"), [1], "not a list"),
        (SINGLE, (), 1, "not a key path"),
        (LEGACY, ("robot", "left_arm_config", "port"), "/dev/x",
         "no robot.left_arm_config section"),
    ],
    ids=["alias-parent", "alias-value", "merge", "block-scalar", "multiline-plain",
         "multiline-quoted", "tab", "sequence", "missing-parent", "missing-section",
         "scalar-parent", "not-scalar", "duplicate", "bad-yaml", "not-mapping", "multiline-value",
         "list-value", "empty-path", "legacy-flat-port"],
)  # fmt: skip
def test_refuses_what_it_cannot_edit_safely(text, path, value, why):
    with pytest.raises(ConfigEditError, match=why) as err:
        set_values(text, {path: value})
    assert isinstance(err.value, ValueError)


def test_legacy_flat_ports_are_refused_through_port_path():
    arm = rigspec.parse(LEGACY).arms[0]
    with pytest.raises(ConfigEditError, match="left_arm_config"):
        set_values(LEGACY, {port_path(arm): "/dev/x"})


def test_a_result_that_reads_back_wrong_is_never_returned(monkeypatch):
    monkeypatch.setattr(configedit, "_emit", lambda value, name: "'5'")  # 5 would come back as text
    with pytest.raises(ConfigEditError, match="did not read back"):
        set_values(SINGLE, {("robot", "port"): 5})
    monkeypatch.setattr(configedit, "_emit", lambda value, name: "[")  # not YAML at all
    with pytest.raises(ConfigEditError, match="did not read back"):
        set_values(SINGLE, {("robot", "port"): "/dev/x"})


def test_overlapping_edits_are_refused(monkeypatch):
    monkeypatch.setattr(configedit, "_plan", lambda *a: (0, 5, "x"))
    with pytest.raises(ConfigEditError, match="same place"):
        set_values(SINGLE, {("robot", "port"): "/dev/x", ("teleop", "port"): "/dev/y"})


# -- writing the file ----------------------------------------------------------------------------


def test_write_values_backs_up_then_replaces_atomically(tmp_path):
    cfg = tmp_path / "robot-config.yaml"
    original = SINGLE.replace("\n", "\r\n").encode()
    cfg.write_bytes(original)
    os.chmod(cfg, 0o640)
    backups = tmp_path / "backups" / "config"
    backup = configedit.write_values(cfg, {("robot", "port"): "/dev/new"}, backups)
    assert backup.parent == backups and backup.name.startswith("robot-config.yaml.")
    assert backup.read_bytes() == original
    want = set_values(original.decode(), {("robot", "port"): "/dev/new"})
    assert cfg.read_bytes() == want.encode()
    assert b"\r\n" in cfg.read_bytes() and b"\n" not in cfg.read_bytes().replace(b"\r\n", b"")
    assert cfg.stat().st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backups", "robot-config.yaml"]
    second = configedit.write_values(cfg, {("robot", "port"): "/dev/newer"}, backups)
    assert second != backup and len(list(backups.iterdir())) == 2


def test_a_refused_edit_writes_nothing(tmp_path):
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(SINGLE)
    with pytest.raises(ConfigEditError):
        configedit.write_values(cfg, {("robot", "nope", "port"): "/dev/x"}, tmp_path / "b")
    assert cfg.read_text() == SINGLE and not (tmp_path / "b").exists()


def test_a_crash_during_replace_leaves_the_original_and_no_temp_file(tmp_path, monkeypatch):
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(SINGLE)

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(configedit.os, "replace", boom)
    with pytest.raises(OSError, match="disk gone"):
        configedit.write_values(cfg, {("robot", "port"): "/dev/x"}, tmp_path / "b")
    assert cfg.read_text() == SINGLE
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_write_values_through_a_symlink_keeps_the_link(tmp_path):
    real = tmp_path / "real.yaml"
    real.write_text(SINGLE)
    link = tmp_path / "robot-config.yaml"
    link.symlink_to(real)
    configedit.write_values(link, {("robot", "port"): "/dev/x"}, tmp_path / "b")
    assert link.is_symlink() and rigspec.parse(real.read_text()).arms[0].port == "/dev/x"


# -- paths for rigspec's objects -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [SINGLE, BIMANUAL, MAC, BOTH_CAMERA_BLOCKS, EMPTY_ROBOT_CAMERAS],
    ids=["single", "bimanual", "mac", "both-camera-blocks", "empty-robot-cameras"],
)
def test_paths_agree_with_rigspec(text):
    spec = rigspec.parse(text)
    data = yaml.safe_load(text)
    assert spec.arms and spec.cameras
    for arm in spec.arms:
        assert _get(data, port_path(arm)) == arm.port
    for cam in spec.cameras:
        path = camera_source_path(cam, spec)
        assert _get(data, path) == cam.fields[path[-1]]
    after = rigspec.parse(set_values(text, _rig_changes(spec)))
    assert {a.key: a.port for a in after.arms} == {a.key: f"/dev/new_{a.key}" for a in spec.arms}
    assert [(c.key, c.side, c.source) for c in after.cameras] == [
        (c.key, c.side, i + 2) for i, c in enumerate(spec.cameras)
    ]


def test_camera_paths_by_where_the_camera_lives():
    bi = rigspec.parse(BIMANUAL)
    assert [camera_source_path(c, bi) for c in bi.cameras] == [
        ("robot", "left_arm_config", "cameras", "wrist", "index_or_path"),
        ("robot", "cameras", "top", "serial_number_or_name"),
    ]
    mac = rigspec.parse(MAC)
    assert camera_source_path(mac.cameras[0], mac) == ("cameras", "wrist", "index_or_path")
    odd = rigspec.parse(SINGLE.replace("type: opencv", "type: zed"))
    with pytest.raises(ConfigEditError, match="type 'zed'"):
        camera_source_path(odd.cameras[0], odd)
