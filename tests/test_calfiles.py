"""The calibration folder: inventory, archive and restore, install a shared set (calfiles.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from phi_studio import calfiles, rigspec
from phi_studio.identity import JointCal, load_calibration, save_calibration
from phi_studio.rig import JOINTS

CFG = """
robot: {type: so101_follower, id: phi_follower, port: /dev/tty.F}
teleop: {type: so101_leader, id: phi_leader, port: /dev/tty.L}
"""
SPEC = rigspec.parse(CFG)
F, L = "robots/so_follower", "teleoperators/so_leader"


def cal(seed: int) -> dict[str, JointCal]:
    return {j: JointCal(i + 1, 0, 100 * seed + i, 500 + seed, 3500 - seed)
            for i, j in enumerate(JOINTS)}  # fmt: skip


def put(root: Path, rel: str, seed: int) -> Path:
    p = root / rel
    save_calibration(cal(seed), p)
    return p


def mac(base: Path) -> Path:
    """A folder like the one on Parv's Mac on 2026-10-05. WHY a subfolder: the archive is made
    next to the calibration folder, so it must stay inside this test's tmp_path."""
    root = base / "calibration"
    put(root, f"{F}/phi_follower.json", 1)
    (root / F / "phi_bi_left.json").symlink_to("phi_follower.json")
    put(root, f"{F}/yash_follower.json", 2)
    put(root, f"{F}/Ava.json", 1)  # the same numbers as phi_follower
    (root / F / "S0-101.json.bak-2026-08-06").write_text("{}")
    (root / F / "._phi_follower.json").write_bytes(b"\x00\x05")
    put(root, f"{L}/phi_leader.json", 5)
    (root / L / "broken.json").write_text("{not json")
    return root


def test_inventory_says_what_each_file_is(tmp_path: Path) -> None:
    by = {f.rel: f for f in calfiles.inventory(mac(tmp_path), SPEC,
                                                {"/dev/tty.F": [f"{F}/phi_follower"]})}  # fmt: skip
    assert by[f"{F}/phi_follower.json"].used_by == ["follower"]
    assert by[f"{F}/phi_follower.json"].on_ports == ["/dev/tty.F"]
    assert set(by[f"{F}/phi_follower.json"].same_as) == {f"{F}/Ava.json", f"{F}/phi_bi_left.json"}
    assert by[f"{F}/phi_bi_left.json"].link == "phi_follower.json"
    assert by[f"{F}/yash_follower.json"].public()["unused"]
    assert by[f"{F}/._phi_follower.json"].junk and by[f"{F}/S0-101.json.bak-2026-08-06"].junk
    assert by[f"{L}/broken.json"].error and by[f"{L}/phi_leader.json"].used_by == ["leader"]


def test_archive_moves_unused_files_and_restore_brings_them_back(tmp_path: Path) -> None:
    root = mac(tmp_path)
    rels = [f"{F}/yash_follower.json", f"{F}/phi_bi_left.json", f"{F}/._phi_follower.json"]
    dest = calfiles.archive(root, rels, SPEC, when="t1")
    assert not any((root / r).exists() or (root / r).is_symlink() for r in rels)
    assert (dest / F / "phi_bi_left.json").is_symlink()  # moved as a link, target untouched
    assert load_calibration(root / F / "phi_follower.json") == cal(1)
    again = calfiles.archive(root, [f"{F}/Ava.json"], SPEC, when="t1")
    assert again.name == "t1-2"  # never into an archive that already holds files
    assert [a["name"] for a in calfiles.archives(root)] == ["t1-2", "t1"]
    out = calfiles.restore(root, "t1")
    assert sorted(out["restored"]) == sorted(rels) and out["archived"] == []
    assert (root / F / "yash_follower.json").is_file()
    assert not (calfiles.archive_root(root) / "t1").exists()


def test_archive_refuses_the_file_an_arm_uses_and_paths_outside(tmp_path: Path) -> None:
    root = mac(tmp_path)
    with pytest.raises(ValueError, match="the follower uses"):
        calfiles.archive(root, [f"{F}/phi_follower.json"], SPEC)
    for bad in ("../x.json", f"{F}/../../x.json", "robots/other/x.json"):
        with pytest.raises(ValueError):
            calfiles.archive(root, [bad], SPEC)
    assert (root / F / "phi_follower.json").is_file()


def test_install_copies_a_shared_set_and_archives_what_it_replaces(tmp_path: Path) -> None:
    root, source = mac(tmp_path), tmp_path / "phi/configs/calibration"
    put(source, f"{F}/phi_follower.json", 1)  # the same numbers
    put(source, f"{F}/phi_bi_left.json", 1)  # replaces the link with a real file
    put(source, f"{L}/phi_leader.json", 6)  # differs
    put(source, f"{F}/phi_bi_follower_right.json", 7)  # new
    rows = {r["rel"]: r for r in calfiles.shared(source, root)}
    assert rows[f"{F}/phi_follower.json"]["state"] == "same"
    assert rows[f"{F}/phi_bi_left.json"]["state"] == "link"
    assert rows[f"{L}/phi_leader.json"]["state"] == "differs"
    assert rows[f"{L}/phi_leader.json"]["max_deg"] > 0
    assert rows[f"{F}/phi_bi_follower_right.json"]["state"] == "new"
    out = calfiles.install(source, root, list(rows), when="i1")
    assert sorted(out["archived"]) == [f"{F}/phi_bi_left.json", f"{L}/phi_leader.json"]
    assert f"{F}/phi_follower.json" not in out["copied"]
    assert load_calibration(root / L / "phi_leader.json") == cal(6)
    assert not (root / F / "phi_bi_left.json").is_symlink()
    assert load_calibration(root / F / "phi_follower.json") == cal(1)  # the link's target kept
    undo = calfiles.restore(root, "i1")  # puts the old leader file and the link back
    assert load_calibration(root / L / "phi_leader.json") == cal(5)
    assert (root / F / "phi_bi_left.json").is_symlink() and undo["archive"]


def test_the_phi_repo_is_found_as_a_shared_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    rig = tmp_path / "rig"
    (rig / "configs/calibration").mkdir(parents=True)
    assert calfiles.shared_candidates(rig) == [(rig / "configs/calibration").resolve()]
    assert calfiles.shared_candidates(None) == []


def test_archive_refuses_a_path_spelled_another_way_and_a_folder(tmp_path: Path) -> None:
    """Review finding: "./" in a rel moved the file an arm uses past the used check."""
    root = mac(tmp_path)
    for bad in (f"{F}/./phi_follower.json", f"{F}/", F, f"{F}//phi_follower.json"):
        with pytest.raises(ValueError):
            calfiles.archive(root, [bad], SPEC)
    assert (root / F / "phi_follower.json").is_file()


def test_a_links_target_is_used_by_the_arm_that_reads_it(tmp_path: Path) -> None:
    """The phi rig: phi_bi_left.json -> phi_follower.json. Moving phi_follower.json aside would
    leave the left follower's link pointing at nothing."""
    bi = rigspec.parse("""
robot: {type: bi_so_follower, id: phi_bi, left_arm_config: {port: /dev/tty.LF},
        right_arm_config: {port: /dev/tty.RF}}
""")
    root = mac(tmp_path)
    by = {f.rel: f for f in calfiles.inventory(root, bi)}
    assert by[f"{F}/phi_follower.json"].used_by == ["left_follower"]
    assert by[f"{F}/phi_bi_left.json"].real == f"{F}/phi_follower.json"
    assert calfiles.shared_files(list(by.values())) == []  # one arm, one file: fine
    with pytest.raises(ValueError, match="left_follower"):
        calfiles.archive(root, [f"{F}/phi_follower.json"], bi)
    (root / F / "phi_bi_right.json").symlink_to("phi_follower.json")  # now two arms
    files = calfiles.inventory(root, bi)
    assert calfiles.shared_files(files) == [(f"{F}/phi_follower.json",
                                             ["left_follower", "right_follower"])]  # fmt: skip



def arm(stops: list[tuple[int, int]], pose: int = 0) -> dict[str, JointCal]:
    """One physical arm calibrated in some pose: the stops (range + homing) stay where the horn
    puts them, while homing moves with the pose and the ranges move the other way."""
    out = {j: JointCal(i + 1, 0, pose + 10 * i, lo - pose - 10 * i, hi - pose - 10 * i)
           for i, (j, (lo, hi)) in enumerate(zip(JOINTS, stops, strict=True))}  # fmt: skip
    out["wrist_roll"] = JointCal(5, 0, pose, 0, 4095)  # LeRobot never narrows wrist roll
    return out


A = [(800, 3400), (800, 3200), (820, 3020), (940, 3280), (0, 4095), (2030, 3540)]
B = [(lo + 900, hi + 900) for lo, hi in A]  # another arm: its horns sit 79 deg round


def test_stop_gap_names_the_physical_arm_whatever_the_calibration_pose() -> None:
    from phi_studio.identity import SAME_ARM_DEG, angle_shift, stop_gap

    again = arm(A, pose=300)  # the same arm calibrated in another pose: registers all differ
    assert stop_gap(arm(A), again).max_deg == 0 < SAME_ARM_DEG
    assert stop_gap(arm(A), arm(B)).max_deg > SAME_ARM_DEG
    # LeRobot's zero is the middle of the recorded range (motors_bus.py:871), not the pose, so
    # the body joints read the same; wrist roll (range 0..4095) is the one zero the pose sets
    shift = angle_shift(arm(A), again).per_joint_deg
    assert {j: d for j, d in shift.items() if j != "wrist_roll"} == dict.fromkeys(
        [j for j in JOINTS if j != "wrist_roll"], 0.0)
    assert shift["wrist_roll"] == round(300 * 360 / 4096, 1)


BI = rigspec.parse("""
teleop: {type: bi_so_leader, id: phi_bi_leader, left_arm_config: {port: /dev/tty.LL},
         right_arm_config: {port: /dev/tty.RL}}
""")
LEAD = "teleoperators/so_leader"


def test_a_shared_file_with_the_other_arms_stops_is_named_for_the_wrong_arm(tmp_path: Path) -> None:
    """The phi rig on 2026-10-06: the shared left leader file had the stops of this Mac's right
    leader, and the right one the left's."""
    root, source = tmp_path / "calibration", tmp_path / "phi/configs/calibration"
    save_calibration(arm(A), root / LEAD / "phi_bi_leader_left.json")
    save_calibration(arm(B), root / LEAD / "phi_bi_leader_right.json")
    save_calibration(arm(B, pose=200), source / LEAD / "phi_bi_leader_left.json")  # swapped
    save_calibration(arm(A, pose=-150), source / LEAD / "phi_bi_leader_right.json")
    rows = {r["rel"].rsplit("/", 1)[1]: r for r in calfiles.shared(source, root, BI)}
    left = rows["phi_bi_leader_left.json"]
    assert (left["named_for"], left["arm_of"]) == ("left_leader", "right_leader")
    assert rows["phi_bi_leader_right.json"]["arm_of"] == "left_leader"
    save_calibration(arm(A, pose=90), source / LEAD / "phi_bi_leader_left.json")  # right way round
    rows = {r["rel"].rsplit("/", 1)[1]: r for r in calfiles.shared(source, root, BI)}
    left = rows["phi_bi_leader_left.json"]
    assert left["arm_of"] == left["named_for"] == "left_leader" and left["state"] == "differs"


def test_two_arms_files_with_one_arms_stops_are_found(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    save_calibration(arm(A), root / LEAD / "phi_bi_leader_left.json")
    save_calibration(arm(A, pose=250), root / LEAD / "phi_bi_leader_right.json")  # A again
    by = {f.rel: f for f in calfiles.inventory(root, BI)}
    assert by[f"{LEAD}/phi_bi_leader_left.json"].same_arm_as == [f"{LEAD}/phi_bi_leader_right.json"]
