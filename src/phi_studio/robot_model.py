"""The SO-101 follower as Studio draws it in 3D: the MJCF parsed into plain data, forward
kinematics, and the one map from LeRobot joint readings to MJCF joint angles.

Model: assets/so101/so101_new_calib_camera.xml from TheRobotStudio/SO-ARM100, Simulation/SO101
(Apache-2.0, see NOTICE). It is so101_new_calib.xml, which onshape-to-robot generated from the
official Onshape CAD, plus two bodies upstream added for the wrist camera mount and the camera
(SO-ARM100 PR #176). Without those two bodies it is byte for byte the file phi vendors in
simulation/model. Nothing here needs MuJoCo.

Conventions, all MuJoCo's:
  * metres and radians (the file sets <compiler angle="radian">)
  * quaternions are (w, x, y, z), scalar first. three.js uses (x, y, z, w): the web side reorders
  * a body's pos and quat are relative to its parent body; a hinge turns its body about `axis`
    through `pos`, both in that body's own frame
  * z is up, and the arm reaches along +x at zero pan
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ASSET_DIR = Path(__file__).parent / "assets" / "so101"
MJCF = ASSET_DIR / "so101_new_calib_camera.xml"
SOURCE = {
    "repo": "https://github.com/TheRobotStudio/SO-ARM100",
    "path": "Simulation/SO101/so101_new_calib_camera.xml",
    "commit": "5f6d2b876a53a4872e405b991dd925556c9e38a4",  # main on 2026-10-04
    "generator": "onshape-to-robot, from the official SO-101 Onshape CAD",
    "license": "Apache-2.0",
}
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
JOINTS = (*ARM_JOINTS, GRIPPER)  # LeRobot's order, and the MJCF's
TOOL_SITE = "gripperframe"  # between the jaw tips: the point the gripper trail follows
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)  # MuJoCo's geom rgba when nothing sets one
UNSUPPORTED_ORIENTATION = ("euler", "axisangle", "xyaxes", "zaxis")

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z


@dataclass(frozen=True)
class Joint:
    name: str
    body: str
    type: str  # hinge | slide
    axis: Vec3
    pos: Vec3
    range: tuple[float, float] | None  # radians for a hinge; None when unlimited


@dataclass(frozen=True)
class Geom:
    index: int
    body: str
    name: str | None
    type: str
    mesh: str | None
    pos: Vec3
    quat: Quat
    rgba: tuple[float, float, float, float]
    material: str | None
    role: str  # visual | collision
    group: int


@dataclass(frozen=True)
class Site:
    name: str
    body: str
    pos: Vec3
    quat: Quat


@dataclass(frozen=True)
class Camera:
    name: str
    body: str
    pos: Vec3
    quat: Quat
    fovy_deg: float


@dataclass(frozen=True)
class Body:
    name: str
    parent: str | None  # None: a child of the world
    pos: Vec3
    quat: Quat
    joints: tuple[str, ...]


@dataclass(frozen=True)
class Mesh:
    name: str
    file: str  # relative to the model's meshdir
    scale: Vec3


@dataclass
class RobotModel:
    name: str
    bodies: list[Body]  # parents before children
    joints: dict[str, Joint]
    geoms: list[Geom]
    sites: dict[str, Site]
    cameras: dict[str, Camera]
    meshes: dict[str, Mesh]
    materials: dict[str, tuple[float, float, float, float]]
    mesh_dir: Path
    defaults: dict[str, Any] = field(default_factory=dict)

    def body(self, name: str) -> Body:
        return next(b for b in self.bodies if b.name == name)

    def mesh_path(self, name: str) -> Path:
        return self.mesh_dir / self.meshes[name].file


# -- parsing ---------------------------------------------------------------------------------------
def _floats(text: str, n: int) -> tuple[float, ...]:
    vals = tuple(float(x) for x in text.split())
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return vals


class _Classes:
    """MJCF default classes: each has a parent and per-tag attributes. The top-level <default> is
    the class "main"; a nested <default class="x"> inherits from the one around it."""

    def __init__(self, root: ET.Element) -> None:
        self.parent: dict[str, str | None] = {"main": None}
        self.attrs: dict[str, dict[str, dict[str, str]]] = {"main": {}}
        # WHY a loop over every top-level <default>: this file has two (its own classes, then
        # joints_properties.xml pasted in), and both define classes under main.
        for d in root.findall("default"):
            self._walk(d, "main")

    def _walk(self, el: ET.Element, name: str) -> None:
        own = self.attrs.setdefault(name, {})
        for child in el:
            if child.tag == "default":
                sub = child.get("class")
                if not sub:
                    raise ValueError("a nested <default> needs a class name")
                self.parent[sub] = name
                self._walk(child, sub)
            else:
                own.setdefault(child.tag, {}).update(child.attrib)

    def chain(self, name: str) -> list[str]:
        """main first, `name` last."""
        if name not in self.parent:
            raise ValueError(f"unknown default class {name!r}")
        out: list[str] = []
        c: str | None = name
        while c is not None:
            out.append(c)
            c = self.parent[c]
        return out[::-1]

    def resolve(self, el: ET.Element, inherited: str) -> tuple[dict[str, str], list[str]]:
        """The element's attributes with its class defaults filled in, and its class chain."""
        chain = self.chain(el.get("class") or inherited)
        out: dict[str, str] = {}
        for c in chain:
            out.update(self.attrs[c].get(el.tag, {}))
        out.update(el.attrib)
        return out, chain


def _pose(attrs: dict[str, str], what: str) -> tuple[Vec3, Quat]:
    bad = [k for k in UNSUPPORTED_ORIENTATION if k in attrs]
    if bad:  # WHY refuse: reading only quat would silently misplace a part
        raise ValueError(f"{what}: orientation given as {bad[0]}; this parser reads quat only")
    pos = _floats(attrs.get("pos", "0 0 0"), 3)
    w, x, y, z = _floats(attrs.get("quat", "1 0 0 0"), 4)
    n = math.sqrt(w * w + x * x + y * y + z * z)  # MuJoCo normalises quaternions on load
    return (pos[0], pos[1], pos[2]), (w / n, x / n, y / n, z / n)


def parse_mjcf(path: Path = MJCF) -> RobotModel:
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") == "degree"
    autolimits = (compiler.get("autolimits", "true") if compiler is not None else "true") == "true"
    mesh_dir = path.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    classes = _Classes(root)

    materials: dict[str, tuple[float, float, float, float]] = {}
    meshes: dict[str, Mesh] = {}
    for asset in root.findall("asset"):
        for m in asset.findall("material"):
            a, _ = classes.resolve(m, "main")
            r = _floats(a.get("rgba", "1 1 1 1"), 4)
            materials[a["name"]] = (r[0], r[1], r[2], r[3])
        for m in asset.findall("mesh"):
            a, _ = classes.resolve(m, "main")
            file = a["file"]
            name = a.get("name") or Path(file).stem
            s = _floats(a.get("scale", "1 1 1"), 3)
            meshes[name] = Mesh(name, file, (s[0], s[1], s[2]))

    model = RobotModel(root.get("model", "robot"), [], {}, [], {}, {}, meshes, materials, mesh_dir)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("no <worldbody>")
    for tag in ("include", "frame", "replicate", "attach"):
        if world.find(f".//{tag}") is not None:
            raise ValueError(f"<{tag}> in the worldbody is not supported")
    for body in world.findall("body"):
        _body(model, classes, body, None, "main", degrees, autolimits)
    return model


def _body(model: RobotModel, classes: _Classes, el: ET.Element, parent: str | None, cls: str,
          degrees: bool, autolimits: bool) -> None:  # fmt: skip
    name = el.get("name") or f"body{len(model.bodies)}"
    cls = el.get("childclass", cls)
    pos, quat = _pose(el.attrib, f"body {name}")
    joints: list[str] = []
    for j in el.findall("joint"):
        a, _ = classes.resolve(j, cls)
        kind = a.get("type", "hinge")
        if kind not in ("hinge", "slide"):
            raise ValueError(f"joint {a.get('name')}: type {kind} is not supported")
        rng = None
        limited = a.get("limited", "auto")
        if "range" in a and (limited == "true" or (limited == "auto" and autolimits)):
            lo, hi = _floats(a["range"], 2)
            if degrees and kind == "hinge":
                lo, hi = math.radians(lo), math.radians(hi)
            rng = (lo, hi)
        ax = np.array(_floats(a.get("axis", "0 0 1"), 3))
        ax = ax / np.linalg.norm(ax)
        jp = _floats(a.get("pos", "0 0 0"), 3)
        if float(a.get("ref", "0")) != 0.0:
            raise ValueError(f"joint {a.get('name')}: a nonzero ref is not supported")
        jn = a.get("name") or f"{name}_joint{len(joints)}"
        model.joints[jn] = Joint(jn, name, kind, (float(ax[0]), float(ax[1]), float(ax[2])),
                                 (jp[0], jp[1], jp[2]), rng)  # fmt: skip
        joints.append(jn)
    model.bodies.append(Body(name, parent, pos, quat, tuple(joints)))
    for g in el.findall("geom"):
        a, chain = classes.resolve(g, cls)
        gpos, gquat = _pose(a, f"geom in {name}")
        mat = a.get("material")
        rgba = DEFAULT_RGBA
        if "rgba" in a and _floats(a["rgba"], 4) != DEFAULT_RGBA:
            r = _floats(a["rgba"], 4)
            rgba = (r[0], r[1], r[2], r[3])
        elif mat is not None:
            rgba = model.materials[mat]
        if "visual" in chain:
            role = "visual"
        elif "collision" in chain:
            role = "collision"
        else:  # no class says: a geom that collides with nothing is only drawn
            no_contact = a.get("contype", "1") == "0" and a.get("conaffinity", "1") == "0"
            role = "visual" if no_contact else "collision"
        mesh = a.get("mesh")
        if mesh is not None and mesh not in model.meshes:
            raise ValueError(f"geom in {name} uses mesh {mesh!r}, which no <mesh> defines")
        model.geoms.append(Geom(len(model.geoms), name, a.get("name"), a.get("type", "sphere"),
                                mesh, gpos, gquat, rgba, mat, role,
                                int(a.get("group", "0"))))  # fmt: skip
    for s in el.findall("site"):
        a, _ = classes.resolve(s, cls)
        spos, squat = _pose(a, f"site in {name}")
        model.sites[a["name"]] = Site(a["name"], name, spos, squat)
    for c in el.findall("camera"):
        a, _ = classes.resolve(c, cls)
        cpos, cquat = _pose(a, f"camera in {name}")
        model.cameras[a["name"]] = Camera(a["name"], name, cpos, cquat,
                                          float(a.get("fovy", "45")))  # fmt: skip
    for child in el.findall("body"):
        _body(model, classes, child, name, cls, degrees, autolimits)


@lru_cache(maxsize=1)
def load() -> RobotModel:
    return parse_mjcf(MJCF)


# -- kinematics ------------------------------------------------------------------------------------
def quat_matrix(q: Quat) -> np.ndarray:
    """3x3 rotation of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def transform(pos: Vec3 | np.ndarray, rot: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def axis_rotation(axis: Vec3, angle: float) -> np.ndarray:
    """Rodrigues: rotation by `angle` radians about the unit vector `axis`."""
    k = np.array(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def forward_kinematics(model: RobotModel, q: Mapping[str, float]) -> dict[str, np.ndarray]:
    """World pose (4x4) of every body for joint angles `q` in MJCF radians. A joint missing from
    `q` sits at 0, MuJoCo's qpos0 for a hinge."""
    world: dict[str, np.ndarray] = {}
    for b in model.bodies:
        t = (world[b.parent] if b.parent else np.eye(4)) @ transform(b.pos, quat_matrix(b.quat))
        for jn in b.joints:
            j = model.joints[jn]
            a = float(q.get(jn, 0.0))
            if j.type == "hinge":
                p = np.array(j.pos)
                t = t @ transform(p, axis_rotation(j.axis, a)) @ transform(-p, np.eye(3))
            else:
                t = t @ transform(np.array(j.axis) * a, np.eye(3))
        world[b.name] = t
    return world


def site_pose(model: RobotModel, world: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    s = model.sites[name]
    return world[s.body] @ transform(s.pos, quat_matrix(s.quat))


def geom_pose(model: RobotModel, world: Mapping[str, np.ndarray], g: Geom) -> np.ndarray:
    """Where the mesh's own file coordinates sit in the world. MuJoCo recentres a mesh on load
    and moves the geom frame to match, so its geom_xpos differs from this; the drawn surface does
    not."""
    return world[g.body] @ transform(g.pos, quat_matrix(g.quat))


# -- LeRobot readings -> MJCF angles ---------------------------------------------------------------
# WHY these maps, and how sure they are:
#   * LeRobot's DEGREES mode reads deg = (raw - mid) * 360 / 4095, mid = the middle of the range
#     recorded at calibration (lerobot 0.6.0 motors_bus.py:870-873). The upstream README says
#     so101_new_calib puts each joint's zero at the middle of its joint range. Phi maps arm joints
#     as sim_rad = deg2rad(lerobot_deg): phi simulation/so101_mujoco_utils.py:80-84 and
#     group/sim/sim_gate.py:56-58 (offsets default to 0). Sign +1, scale pi/180, offset 0.
#   * NOT VERIFIED ON AN ARM. phi group/sim/SIM_SETUP.md:37 says the sim-to-real joint mapping
#     is not solved: the two "middles" agree only by coincidence, replaying 120 episodes put the
#     tool 13 mm under the table (median), and wrist_roll's zero is wherever the wrist sat when
#     calibration was confirmed. phi's twin scene uses wrist_roll -30 deg, fitted by eye to two
#     frames (group/sim/twin_scene.py:32-34); Studio does not, because that offset belongs to one
#     calibration file. Expect the drawn pose to be off by a few degrees per joint.
#   * Gripper: LeRobot reads 0..100 (RANGE_0_100, motors_bus.py:867-869; 0 closed, 100 open per
#     the upstream README). Phi maps it linearly onto the MJCF range, -10..100 deg, and calls that
#     a placeholder: sim_gate.py:20-23 and :60-61, so101_mujoco_utils.py:63-70. The upstream README
#     says this mapping "is not yet reflected" in the MJCF. One indirect check
#     (SIM_SETUP.md:138-139) put it 2.5 deg off. Studio uses the same placeholder.
MAP_STATUS = "assumed: not checked on a physical arm"
MAP_SOURCES = {
    "arm": "phi simulation/so101_mujoco_utils.py:80-84, group/sim/sim_gate.py:56-58",
    GRIPPER: "phi group/sim/sim_gate.py:20-23, simulation/so101_mujoco_utils.py:63-70 "
    "(a placeholder there too)",
}
TICKS_PER_REV = 4096  # STS3215; LeRobot divides by 4095, its max_res (motors_bus.py:872)


@dataclass(frozen=True)
class JointMap:
    """rad = scale * reading + offset, for a reading in LeRobot's units."""

    joint: str
    unit: str  # "degrees" | "0..100"
    scale: float
    offset: float
    status: str
    source: str

    def __call__(self, reading: float) -> float:
        return self.scale * reading + self.offset


def joint_maps(model: RobotModel) -> dict[str, JointMap]:
    out = {j: JointMap(j, "degrees", math.pi / 180, 0.0, MAP_STATUS, MAP_SOURCES["arm"])
           for j in ARM_JOINTS}  # fmt: skip
    rng = model.joints[GRIPPER].range
    assert rng is not None
    lo, hi = rng
    out[GRIPPER] = JointMap(GRIPPER, "0..100", (hi - lo) / 100.0, lo, MAP_STATUS,
                            MAP_SOURCES[GRIPPER])  # fmt: skip
    return out


def m100_to_degrees(value: float, range_min: int, range_max: int, drive_mode: int = 0) -> float:
    """A body-joint reading in LeRobot's -100..100 mode as the degrees DEGREES mode would read.
    The -100..100 mode needs the calibrated range in ticks: norm = (raw - min) / (max - min) * 200
    - 100, negated when drive_mode is set (motors_bus.py:864-866). Inverting it and applying
    DEGREES mode's formula gives deg = norm / 200 * (max - min) * 360 / 4095. The -100..100 mode
    clamps raw to the range first, so a reading past a stop cannot be recovered."""
    if range_max <= range_min:
        raise ValueError("range_max must be above range_min")
    norm = -value if drive_mode else value
    return norm / 200.0 * (range_max - range_min) * 360.0 / (TICKS_PER_REV - 1)


def lerobot_to_mjcf(
    model: RobotModel,
    readings: Mapping[str, float],
    units: str = "degrees",
    calibration: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, float]:
    """LeRobot joint readings to MJCF radians. units: "degrees" (use_degrees) or "m100" (the
    -100..100 mode, which needs each body joint's calibration: range_min, range_max, drive_mode).
    The gripper is 0..100 in both modes. Readings past the model's limits are kept, not clamped:
    the view shows what the arm reported."""
    if units not in ("degrees", "m100"):
        raise ValueError(f"unknown units {units!r}")
    maps = joint_maps(model)
    out = {}
    for j, v in readings.items():
        if j not in maps:
            continue
        if units == "m100" and j != GRIPPER:
            c = (calibration or {}).get(j)
            if c is None:
                raise ValueError(f"{j}: a -100..100 reading needs that joint's calibration range")
            v = m100_to_degrees(v, int(c["range_min"]), int(c["range_max"]),
                                int(c.get("drive_mode", 0)))  # fmt: skip
        out[j] = maps[j](v)
    return out


# -- what the web view needs -----------------------------------------------------------------------
def driving_servos(model: RobotModel) -> dict[str, int]:
    """For each joint, the visual servo geom that turns it: the STS3215 bolted in the parent body
    whose origin is nearest the joint's axis. Inferred from the layout (each one sits 22.5 mm from
    its joint's anchor), not named in the MJCF."""
    out = {}
    for j in model.joints.values():
        child = model.body(j.body)
        if child.parent is None:
            continue
        anchor = np.array(child.pos) + quat_matrix(child.quat) @ np.array(j.pos)
        servos = [g for g in model.geoms if g.body == child.parent and g.role == "visual"
                  and (g.mesh or "").startswith("sts3215")]  # fmt: skip
        if servos:
            best = min(servos, key=lambda g: float(np.linalg.norm(np.array(g.pos) - anchor)))
            out[j.name] = best.index
    return out


# The wrist camera, in the frame of the upstream CAD body "wrist_camera". The camera mesh is a
# 32 x 32 mm board with a 13 mm round lens barrel centred on it, running from z = 0 to 26.3 mm
# (read from wrist_camera_so101_v1.stl), so the lens looks along the body's +z. Image up is taken
# as the body's +y. Phi's by-eye wrist pose, fitted to real wrist frames (group/sim/twin_scene.py:
# 47-50), agrees: its view direction is 0.85 degrees from this lens axis, its up vector (gripper -y)
# projects onto the image plane as exactly this body's +y, and its position is 20 mm from the lens
# front. A camera looks along its own -z, so the frame turns 180 degrees about y: quat (w, x, y, z)
# = (0, 0, 1, 0).
WRIST_CAMERA = {
    "key": "wrist",
    "body": "wrist_camera",
    "pos": (0.0, 0.0, 0.0263),  # the lens front; the optical centre is a few mm behind, unmeasured
    "quat": (0.0, 0.0, 1.0, 0.0),
    "fovy_deg": 85.0,
    "mount": "From the upstream CAD (SO-ARM100 PR #176). "
    "That your mount matches it is not checked.",
    "fov": "Estimated by eye in phi (group/sim/twin_scene.py:50), not measured.",
}
# Front and top cameras: the user places them. These starting poses are phi's by-eye estimates for
# its own rig (group/sim/twin_scene.py:44-46), in the MJCF world frame of one arm.
CAMERA_DEFAULTS = {
    "front": {"pos": (1.05, 0.04, 0.33), "target": (0.15, 0.0, 0.06), "up": (0.0, 0.0, 1.0),
              "fovy_deg": 36.0},
    "top": {"pos": (0.23, -0.02, 0.78), "target": (0.23, -0.02, 0.0), "up": (0.0, -1.0, 0.0),
            "fovy_deg": 45.0},
}  # fmt: skip
# Two followers side by side, facing +x. Phi has no measured spacing (dual_arm_scene.py takes it
# as a required argument), so this is a plausible desk layout, not a measurement.
PAIR_SPACING_M = 0.40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def model_json() -> dict[str, Any]:
    """The model as the web view reads it. Mesh URLs carry a content hash so the browser may cache
    them for good."""
    m = load()
    used = {g.mesh for g in m.geoms if g.mesh and g.role == "visual"}
    meshes = {}
    for name in sorted(used):
        p = m.mesh_path(name)
        meshes[name] = {"url": f"/api/scene/mesh/{name}?v={_sha(p)}", "bytes": p.stat().st_size,
                        "scale": m.meshes[name].scale}  # fmt: skip
    maps = joint_maps(m)
    return {
        "model": m.name,
        "source": SOURCE,
        "conventions": {"length": "m", "angle": "rad", "quat": "w x y z", "up": "+z",
                        "forward": "+x"},  # fmt: skip
        "bodies": [{"name": b.name, "parent": b.parent, "pos": b.pos, "quat": b.quat,
                    "joints": list(b.joints)} for b in m.bodies],  # fmt: skip
        "joints": {j.name: {"body": j.body, "type": j.type, "axis": j.axis, "pos": j.pos,
                            "range": j.range} for j in m.joints.values()},  # fmt: skip
        "geoms": [{"index": g.index, "body": g.body, "mesh": g.mesh, "pos": g.pos, "quat": g.quat,
                   "rgba": g.rgba, "material": g.material, "role": g.role}
                  for g in m.geoms if g.role == "visual"],  # fmt: skip
        "sites": {s.name: {"body": s.body, "pos": s.pos, "quat": s.quat}
                  for s in m.sites.values()},  # fmt: skip
        "cameras": {c.name: {"body": c.body, "pos": c.pos, "quat": c.quat, "fovy_deg": c.fovy_deg}
                    for c in m.cameras.values()},  # fmt: skip
        "meshes": meshes,
        "joint_order": list(JOINTS),
        "mapping": {j: {"unit": x.unit, "scale": x.scale, "offset": x.offset, "status": x.status,
                        "source": x.source} for j, x in maps.items()},  # fmt: skip
        "drives": driving_servos(m),
        "tool_site": TOOL_SITE,
        "wrist_camera": WRIST_CAMERA,
        "wrist_camera_bodies": ["wrist_camera_mount", "wrist_camera"],
        "camera_defaults": CAMERA_DEFAULTS,
        "pair_spacing_m": PAIR_SPACING_M,
    }


def mesh_file(name: str) -> Path | None:
    """The STL behind a mesh name the model uses, or None. Only names from the parsed model are
    accepted, so a request cannot name any other file."""
    m = load()
    if name not in model_json()["meshes"]:
        return None
    return m.mesh_path(name)
