"""Writes tests/fixtures/so101_fk_reference.json: reference numbers for the 3D view's kinematics,
from two sources that share no code with Studio.

  mujoco         MuJoCo loading the vendored MJCF (src/phi_studio/assets/so101/
                 so101_new_calib_camera.xml) and running mj_kinematics: every body's world position
                 and quaternion, the gripperframe site, and the world box of every visual mesh at
                 two poses.
  phi_fk_gripper phi's hand-written chain, simulation/so101_forward_kinematics.py in the phi repo
                 (ECE 4560), which writes the transforms as Rz/Rx products by hand. It covers the 5
                 arm joints and ends at the `gripper` body.

Read by tests/test_robot_model.py (Python FK) and web/tests/scene.test.ts (the browser's FK).

Needs MuJoCo, which must NOT go into the phi env (it pins its own numpy and could change what the
phi env runs). Use a separate venv:

    python3.12 -m venv /tmp/mjvenv
    /tmp/mjvenv/bin/pip install mujoco==3.14.0 numpy==2.5.3
    /tmp/mjvenv/bin/python tests/fixtures/make_so101_fk_reference.py . \\
        tests/fixtures/so101_fk_reference.json /path/to/phi/simulation/so101_forward_kinematics.py

The committed fixture was made with mujoco 3.14.0 and numpy 2.5.3, and phi's
so101_forward_kinematics.py with sha256 3c04f879...cb64bc19. Running the command above reproduces it
byte for byte (checked 2026-10-04). The 4 random poses come from a fixed seed (20261004). """

import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PHI_FK = Path(sys.argv[3])
XML = REPO / "src/phi_studio/assets/so101/so101_new_calib_camera.xml"

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]

rng = np.random.default_rng(20261004)
poses = {
    "zero": np.zeros(6),
    "mock_home": np.deg2rad([0, -90, 90, 60, 0, -10 + 0.05 * 110]),
    "low_limits": lo.copy(),
    "high_limits": hi.copy(),
}
for i in range(4):
    poses[f"random{i}"] = rng.uniform(lo, hi)

spec = importlib.util.spec_from_file_location("phi_fk", PHI_FK)
assert spec is not None and spec.loader is not None
phi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phi)


def phi_gripper(qdeg: np.ndarray) -> np.ndarray:
    return (phi.get_gw1(qdeg[0]) @ phi.get_g12(qdeg[1]) @ phi.get_g23(qdeg[2])
            @ phi.get_g34(qdeg[3]) @ phi.get_g45(qdeg[4]))  # fmt: skip


out: dict = {"mujoco_version": mujoco.__version__, "joints": names, "poses": {}}
for key, q in poses.items():
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    bodies = {}
    for b in range(1, m.nbody):
        bodies[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = {
            "pos": d.xpos[b].tolist(),
            "quat": d.xquat[b].tolist(),
        }
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    rec = {
        "q": q.tolist(),
        "bodies": bodies,
        "gripperframe": {"pos": d.site_xpos[sid].tolist(), "mat": d.site_xmat[sid].tolist()},
    }
    if key in ("zero", "random0"):
        # World bounding box of every visual mesh as MuJoCo places it: its stored vertices are
        # recentred, so this also checks Studio's use of the raw STL with the geom pos/quat.
        boxes = []
        for g in range(m.ngeom):
            if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a : a + n].astype(float)
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            boxes.append(
                {
                    "geom": int(g),
                    "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
                    "mesh": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid),
                    "min": w.min(0).tolist(),
                    "max": w.max(0).tolist(),
                }
            )
        rec["visual_boxes"] = boxes
    g = phi_gripper(np.rad2deg(q[:5]))
    rec["phi_fk_gripper"] = {"pos": g[:3, 3].tolist(), "mat": g[:3, :3].reshape(-1).tolist()}
    out["poses"][key] = rec

# How far apart the two references are (both should be ~1e-8 m: phi rounds one 9 nm offset).
worst = 0.0
for rec in out["poses"].values():
    a = np.array(rec["bodies"]["gripper"]["pos"])
    b = np.array(rec["phi_fk_gripper"]["pos"])
    worst = max(worst, float(np.abs(a - b).max()))
print("mujoco vs phi_fk, gripper body position, worst over poses:", worst, "m")
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT, "poses", len(out["poses"]))
