"""The SO-101 model behind the 3D view: parsing, the LeRobot-to-MJCF map, and forward kinematics
checked against two independent references.

fixtures/so101_fk_reference.json was written by MuJoCo 3.14.0 loading the same vendored MJCF, and by
phi's hand-written ECE 4560 chain (phi simulation/so101_forward_kinematics.py), with
fixtures/make_so101_fk_reference.py run in a separate venv: MuJoCo must not be installed in Studio's
environment. That script's docstring has the exact command."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from phi_studio import robot_model as rm

REF = json.loads((Path(__file__).parent / "fixtures" / "so101_fk_reference.json").read_text())
MM = 1e-3


@pytest.fixture(scope="module")
def model() -> rm.RobotModel:
    return rm.parse_mjcf()


def test_parses_bodies_joints_and_ranges(model: rm.RobotModel) -> None:
    names = [b.name for b in model.bodies]
    assert names == ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                     "wrist_camera_mount", "wrist_camera", "moving_jaw_so101_v1"]  # fmt: skip
    assert list(model.joints) == list(rm.JOINTS)
    # Ranges straight from the file (radians, <compiler angle="radian">).
    assert model.joints["elbow_flex"].range == (-1.69, 1.69)
    gripper = model.joints["gripper"].range
    assert gripper == pytest.approx((-0.17453297762778586, 1.7453291995659765))
    assert all(j.axis == (0.0, 0.0, 1.0) and j.type == "hinge" for j in model.joints.values())
    # Quaternions keep the file's w, x, y, z order ("3.56e-16 1.23e-15 -1 -4.15e-16": y = -1).
    shoulder = model.body("shoulder")
    assert shoulder.parent == "base" and shoulder.quat[2] == pytest.approx(-1.0)
    assert all(math.isclose(sum(c * c for c in b.quat), 1.0) for b in model.bodies)


def test_default_classes_resolve(model: rm.RobotModel) -> None:
    visual = [g for g in model.geoms if g.role == "visual"]
    collision = [g for g in model.geoms if g.role == "collision"]
    # 4 base parts (upstream dropped the base's collision meshes) plus 15 per-part pairs.
    assert len(visual) == 19 and len(collision) == 15
    assert {g.group for g in visual} == {2} and {g.group for g in collision} == {3}
    assert all(g.type == "mesh" and g.mesh in model.meshes for g in model.geoms)
    servo = next(g for g in visual if g.mesh == "sts3215_03a_v1")
    assert servo.rgba == (0.1, 0.1, 0.1, 1.0)  # from its material
    plate = next(g for g in visual if g.mesh == "base_so101_v2")
    assert plate.rgba == (1.0, 0.82, 0.12, 1.0)
    assert model.meshes["wrist_camera_mount_so101_v1"].scale == (0.001, 0.001, 0.001)
    assert set(model.sites) == {"baseframe", "gripperframe"} and model.cameras == {}


def test_unsupported_orientation_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "m.xml"
    p.write_text('<mujoco><worldbody><body name="b" euler="0 0 1"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="quat only"):
        rm.parse_mjcf(p)


def test_degrees_default_converts_ranges(tmp_path: Path) -> None:
    p = tmp_path / "m.xml"
    p.write_text('<mujoco><worldbody><body name="b"><joint name="j" range="-90 45"/></body>'
                 "</worldbody></mujoco>")  # fmt: skip
    assert rm.parse_mjcf(p).joints["j"].range == pytest.approx((-math.pi / 2, math.pi / 4))


# -- the mapping -----------------------------------------------------------------------------------
def test_arm_joints_map_degrees_to_radians(model: rm.RobotModel) -> None:
    q = rm.lerobot_to_mjcf(model, {"shoulder_pan": 90.0, "elbow_flex": -45.0, "wrist_roll": 0.0})
    assert q == pytest.approx({"shoulder_pan": math.pi / 2, "elbow_flex": -math.pi / 4,
                               "wrist_roll": 0.0})  # fmt: skip


def test_gripper_maps_0_100_onto_the_model_range(model: rm.RobotModel) -> None:
    lo, hi = model.joints["gripper"].range or (0.0, 0.0)
    q = rm.lerobot_to_mjcf(model, {"gripper": 0.0})
    assert q["gripper"] == pytest.approx(lo)  # closed: -10 deg
    assert rm.lerobot_to_mjcf(model, {"gripper": 100.0})["gripper"] == pytest.approx(hi)
    half = rm.lerobot_to_mjcf(model, {"gripper": 50.0})["gripper"]
    assert math.degrees(half) == pytest.approx(45)


def test_readings_past_the_limits_are_not_clamped(model: rm.RobotModel) -> None:
    q = rm.lerobot_to_mjcf(model, {"wrist_flex": 101.7})  # phi saw 6.7 deg past the MJCF limit
    assert q["wrist_flex"] > (model.joints["wrist_flex"].range or (0, 0))[1]


def test_minus100_mode_needs_calibration(model: rm.RobotModel) -> None:
    with pytest.raises(ValueError, match="calibration range"):
        rm.lerobot_to_mjcf(model, {"elbow_flex": 10.0}, units="m100")
    cal = {"elbow_flex": {"range_min": 1000, "range_max": 3047, "drive_mode": 0}}
    # Half the range (1023.5 ticks) is 90 deg, so +100 is +90 deg.
    q = rm.lerobot_to_mjcf(model, {"elbow_flex": 100.0, "gripper": 0.0}, "m100", cal)
    assert math.degrees(q["elbow_flex"]) == pytest.approx(2047 / 2 * 360 / 4095)
    assert rm.m100_to_degrees(50, 0, 4095, drive_mode=1) == pytest.approx(-90.0)
    with pytest.raises(ValueError, match="unknown units"):
        rm.lerobot_to_mjcf(model, {"gripper": 1.0}, units="raw")


def test_mapping_is_served_once_and_says_it_is_unverified() -> None:
    m = rm.model_json()["mapping"]
    assert set(m) == set(rm.JOINTS)
    assert all("not checked" in v["status"] for v in m.values())
    lift = m["shoulder_lift"]
    assert lift["scale"] == pytest.approx(math.pi / 180) and lift["offset"] == 0


# -- forward kinematics against independent references ---------------------------------------------
def _q(rec: dict) -> dict[str, float]:
    return dict(zip(REF["joints"], rec["q"], strict=True))


@pytest.mark.parametrize("pose", sorted(REF["poses"]))
def test_fk_matches_mujoco(model: rm.RobotModel, pose: str) -> None:
    rec = REF["poses"][pose]
    world = rm.forward_kinematics(model, _q(rec))
    for name, ref in rec["bodies"].items():
        assert np.allclose(world[name][:3, 3], ref["pos"], atol=1e-6), name  # 1 um, far under 1 mm
        assert np.allclose(world[name][:3, :3], rm.quat_matrix(tuple(ref["quat"])), atol=1e-6), name
    tip = rm.site_pose(model, world, rm.TOOL_SITE)
    assert np.linalg.norm(tip[:3, 3] - rec["gripperframe"]["pos"]) < MM
    assert np.allclose(tip[:3, :3].reshape(-1), rec["gripperframe"]["mat"], atol=1e-6)


@pytest.mark.parametrize("pose", sorted(REF["poses"]))
def test_fk_matches_phis_hand_written_chain(model: rm.RobotModel, pose: str) -> None:
    rec = REF["poses"][pose]
    g = rm.forward_kinematics(model, _q(rec))["gripper"]
    assert np.linalg.norm(g[:3, 3] - rec["phi_fk_gripper"]["pos"]) < MM
    assert np.allclose(g[:3, :3].reshape(-1), rec["phi_fk_gripper"]["mat"], atol=1e-6)


def _stl(path: Path, scale: tuple[float, float, float]) -> np.ndarray:
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    assert len(b) == 84 + 50 * n, f"{path.name} is not a binary STL"
    rows = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                         count=n, offset=84)  # fmt: skip
    return rows["v"].reshape(-1, 3).astype(float) * np.array(scale)


@pytest.mark.parametrize("pose", ["zero", "random0"])
def test_every_drawn_mesh_lands_where_mujoco_puts_it(model: rm.RobotModel, pose: str) -> None:
    """The web view draws each raw STL at body pose x geom pos/quat. MuJoCo recentres meshes and
    moves the geom frame, so compare the world bounding boxes of the actual surfaces."""
    rec = REF["poses"][pose]
    world = rm.forward_kinematics(model, _q(rec))
    visual = [g for g in model.geoms if g.role == "visual"]
    assert len(visual) == len(rec["visual_boxes"])
    for g, box in zip(visual, rec["visual_boxes"], strict=True):
        assert (g.body, g.mesh) == (box["body"], box["mesh"])
        v = _stl(model.mesh_path(g.mesh or ""), model.meshes[g.mesh or ""].scale)
        t = rm.geom_pose(model, world, g)
        # WHY einsum, not @: numpy's Accelerate matmul on macOS raises spurious FP warnings here.
        w = np.einsum("ij,nj->ni", t[:3, :3], v) + t[:3, 3]
        assert np.abs(w.min(0) - box["min"]).max() < 1e-5, g.mesh  # 10 um: float32 vertices
        assert np.abs(w.max(0) - box["max"]).max() < 1e-5, g.mesh


def test_each_joint_has_its_own_driving_servo(model: rm.RobotModel) -> None:
    drives = rm.driving_servos(model)
    assert set(drives) == set(rm.JOINTS)
    assert len(set(drives.values())) == 6
    for j, gi in drives.items():
        g = model.geoms[gi]
        child = model.body(model.joints[j].body)
        assert g.body == child.parent and (g.mesh or "").startswith("sts3215")
        gap = np.linalg.norm(np.array(g.pos) - np.array(child.pos))
        assert gap == pytest.approx(0.0225, abs=0.001), j  # the layout the inference rests on


def test_wrist_camera_looks_along_the_lens(model: rm.RobotModel) -> None:
    """The camera frame looks along the camera body's +z (the lens) with image up = body +y."""
    r = rm.quat_matrix(rm.WRIST_CAMERA["quat"])  # type: ignore[arg-type]
    assert np.allclose(r @ [0, 0, -1], [0, 0, 1])  # a camera looks along its own -z
    assert np.allclose(r @ [0, 1, 0], [0, 1, 0])
    assert rm.WRIST_CAMERA["body"] in {b.name for b in model.bodies}


def test_model_json_lists_only_meshes_it_draws() -> None:
    j = rm.model_json()
    assert set(j["meshes"]) == {g["mesh"] for g in j["geoms"]}
    assert all(g["role"] == "visual" for g in j["geoms"])
    assert rm.mesh_file("sts3215_03a_v1") is not None
    for bad in ("../so101_new_calib_camera", "LICENSE", "", "sts3215_03a_v1.stl"):
        assert rm.mesh_file(bad) is None
