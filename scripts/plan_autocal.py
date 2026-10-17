"""Check auto-calibration's whole program for one arm in simulation, before it runs on the arm.

The program is the one Studio runs (phi_studio.autocal: POSE_ORDER, ORDER, SWEEP_POSES): unfold from
the arm's rest pose into the middle pose, sweep every joint to both stops (moving the other joints
into a joint's sweep pose first where it has one), and fold back to rest. This walks it in 2-degree
steps on the SO-101 model and measures the true distance between the arm's parts, the base and the
table at every step.

  * Exact, not convex hulls: MuJoCo turns meshes into their hulls for contact, which fills the
    U-brackets a folded forearm nests into. Distances here come from FCL on the compiled
    triangles; MuJoCo only does the kinematics.
  * The base ships visual meshes only; they are added as collision geometry. A table sits at the
    base's lowest point.
  * Pass: 10 mm clear everywhere, pair by pair, except each joint's own end stop (the folded
    forearm on the shoulder bracket, the upper arm on the base: a sweep is meant to reach those)
    and what the arm rests on: those may stay as close as they rested, not closer, and once left
    may only be met again landing back on rest. The fold back visits the unfold's positions in
    reverse.
  * A hand calibration's middle is a few degrees off the model's, so the rest pose is checked in
    every version that is physically possible: shoulder, elbow and wrist shifted up to 12 degrees,
    keeping those where the limp arm rests on the table or itself (within 3 mm), not in it.

Needs MuJoCo and FCL, which must NOT go into the phi env (see tests/fixtures/make_so101_fk_reference.py):

    python3.12 -m venv /tmp/mjvenv
    /tmp/mjvenv/bin/pip install mujoco==3.14.0 numpy==2.5.3 python-fcl
    /tmp/mjvenv/bin/python scripts/plan_autocal.py REST_POSE.json CALIBRATION.json [SHIFT_STEP] [--joints gripper]

REST_POSE.json: the arm's rest pose in LeRobot degrees (gripper 0..100), read through its own
calibration file; CALIBRATION.json: that file (for each joint's range). SHIFT_STEP: degrees between
the shifted versions (default 4; 8 checks the same band in a quarter of the time). Exit status 0 when
every version of the rest pose passes. --joints: what the run sweeps (default all), so a
gripper-only run is checked as Studio runs it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import fcl
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from phi_studio import autocal  # noqa: E402  (after the path; imports nothing heavy)

ASSET = REPO / "src/phi_studio/assets/so101"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CHAIN = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1",
         "wrist_camera_mount", "wrist_camera"]  # fmt: skip
TOOL = {"gripper", "moving_jaw_so101_v1", "wrist_camera_mount", "wrist_camera"}
# A joint's own end stop is a contact between its links; the sweep is meant to reach it.
STOP = {("elbow_flex", 1): {("shoulder", "lower_arm")},
        ("shoulder_lift", -1): {("base", "upper_arm")},
        ("wrist_flex", 1): {("lower_arm", "gripper"), ("lower_arm", "moving_jaw_so101_v1")},
        ("wrist_flex", -1): {("lower_arm", "gripper"), ("lower_arm", "wrist_camera_mount")}}  # fmt: skip
MARGIN, STEP, BEYOND = 0.010, 2.0, 2.0  # metres clear; degrees per step; degrees past a range


def build() -> mujoco.MjModel:
    root = ET.parse(ASSET / "so101_new_calib_camera.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSET / "assets"))
    base = root.find("worldbody").find("body")
    for g in list(base.findall("geom")):
        if g.get("class") == "visual":
            ET.SubElement(base, "geom", dict(g.attrib)).set("class", "collision")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    zmin = min(_verts(m, d, i)[:, 2].min() for i in range(m.ngeom) if m.geom_bodyid[i] == base_id)
    ET.SubElement(root.find("worldbody"), "geom", {"name": "table", "type": "plane",
                  "size": "1 1 0.01", "pos": f"0 0 {zmin - 0.0005}"})  # fmt: skip
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(ET.tostring(root, encoding="unicode"))
    return mujoco.MjModel.from_xml_path(f.name)


def _verts(m: mujoco.MjModel, d: mujoco.MjData, i: int) -> np.ndarray:
    mid = m.geom_dataid[i]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]


def body(m: mujoco.MjModel, i: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])


class Scene:
    """The model with exact clearances. q is LeRobot units: degrees, gripper 0..100."""

    def __init__(self) -> None:
        self.m = m = build()
        self.d = mujoco.MjData(m)
        self.verts, self.objs, self.body = {}, {}, {}
        for i in range(m.ngeom):
            if m.geom_group[i] != 3 or m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[i]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
            bvh = fcl.BVHModel()
            bvh.beginModel(len(v), len(f))
            bvh.addSubModel(v, f)
            bvh.endModel()
            self.objs[i], self.verts[i], self.body[i] = fcl.CollisionObject(bvh, fcl.Transform()), v, body(m, i)
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.table_z = float(m.geom_pos[tid][2])
        gs = sorted(self.objs)
        self.pairs = [(a, b) for a, b in itertools.combinations(gs, 2) if self._may_touch(a, b)]
        self.on_table = [g for g in gs if self.body[g] not in ("base", "shoulder")]

    def _may_touch(self, a: int, b: int) -> bool:
        na, nb = self.body[a], self.body[b]
        if na == nb or abs(CHAIN.index(na) - CHAIN.index(nb)) <= 1:
            return False  # one part, or neighbours that share a joint
        return not ({na, nb} <= TOOL or ("wrist" in (na, nb) and TOOL & {na, nb}))

    def distances(self, q: dict[str, float]) -> dict[tuple[str, str], float]:
        """Every pair of bodies that may touch (and each body and the table): how far apart, in
        metres; 0 when meshes touch, below 0 into the table. Pairs 3 cm apart or more by their
        bounding boxes get that box gap, which is a lower bound."""
        m, d = self.m, self.d
        for j, v in q.items():
            adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            d.qpos[adr] = np.deg2rad(-10 + v / 100 * 110 if j == "gripper" else v)  # Studio's map
        mujoco.mj_kinematics(m, d)
        lo, hi = {}, {}
        for i, v in self.verts.items():
            r, p = d.geom_xmat[i].reshape(3, 3), d.geom_xpos[i]
            w = v @ r.T + p
            lo[i], hi[i] = w.min(0), w.max(0)
            self.objs[i].setTransform(fcl.Transform(r.copy(), p.copy()))
        out: dict[tuple[str, str], float] = {}

        def keep(key: tuple[str, str], x: float) -> None:
            out[key] = min(out.get(key, np.inf), x)

        for g in self.on_table:  # a mesh's distance to a plane is its lowest vertex's height
            keep((self.body[g], "table"), float(lo[g][2] - self.table_z))
        for a, b in self.pairs:
            key = (self.body[a], self.body[b])
            gap = float(np.linalg.norm(np.maximum(0, np.maximum(lo[a] - hi[b], lo[b] - hi[a]))))
            if gap >= 0.03:
                keep(key, gap)
                continue
            if fcl.collide(self.objs[a], self.objs[b], fcl.CollisionRequest(), fcl.CollisionResult()):
                keep(key, 0.0)
            else:
                keep(key, fcl.distance(self.objs[a], self.objs[b], fcl.DistanceRequest(), fcl.DistanceResult()))
        return out

    def clearance(self, q: dict[str, float]) -> tuple[float, tuple[str, str]]:
        """The closest pair (metres; 0 or less when touching) and which bodies."""
        key, x = min(self.distances(q).items(), key=lambda kv: kv[1])
        return x, key


def steps_deg(cal: dict, start: dict[str, float], joints: list[str]) -> list[tuple[str, str, float, int]]:
    """Studio's own program (autocal.program) for an arm that starts anywhere, in planner units.
    Raw ticks come from the file's ranges, as the worker's `mid` does: each joint's middle is its
    range's, so a pose of v degrees is v here; the gripper's 0..100 spans its range."""
    lo = {j: cal[j]["range_min"] for j in JOINTS}
    hi = {j: cal[j]["range_max"] for j in JOINTS}
    mid = {j: (lo[j] + hi[j]) // 2 for j in JOINTS}

    def tick(j: str, v: float) -> int:
        if j == "gripper":
            return round(lo[j] + v / 100 * (hi[j] - lo[j]))
        return mid[j] + round(v * autocal.TICKS_PER_DEG)

    def unit(j: str, t: int) -> float:
        if j == "gripper":
            return (t - lo[j]) / (hi[j] - lo[j]) * 100
        return (t - mid[j]) / autocal.TICKS_PER_DEG

    swept = [j for j in joints if j != "wrist_roll"]
    prog = autocal.program(swept, {j: tick(j, start[j]) for j in JOINTS}, mid, unfold=True)
    return [(st.kind, st.joint, unit(st.joint, st.to), st.sign) for st in prog]


def walk(sc: Scene, start: dict[str, float], half: dict[str, float],
         steps: list[tuple[str, str, float, int]]) -> tuple[bool, str, float]:
    """`steps` (steps_deg) from `start`. Returns (passed, what failed, closest clearance on the way).

    Pair by pair: what the arm rests on (closer than MARGIN at the start) may stay as close but not
    closer, and once it moves off a contact it may only come back to it landing at the end. Every
    other pair stays MARGIN apart, except a sweep's own stop. WHY by pair: one closest distance
    let a new contact (2026-10-08: the jaw on the table) hide behind an old one (upper arm on base)."""
    q = dict(start)
    rested = {p: x for p, x in sc.distances(q).items() if x < MARGIN}
    on = dict(rested)  # rest contacts not yet left
    worst = [np.inf]

    def go(j: str, to: float, what: str, stop: set | None = None, landing: bool = False) -> str | None:
        a = q[j]
        n = max(1, int(np.ceil(abs(to - a) / STEP)))
        for k in range(1, n + 1):
            q[j] = a + (to - a) * k / n
            dist = sc.distances(q)
            for p, x in dist.items():
                if x >= MARGIN:
                    on.pop(p, None)  # moved off it
                    continue
                if p in on or (landing and p in rested):
                    if x < (on.get(p) if p in on else rested[p]) - 0.002:
                        return f"{what}: {j} at {q[j]:.0f} goes deeper into {p[0]}-{p[1]}"
                    continue
                if stop and p in stop:
                    continue
                return f"{what}: {j} at {q[j]:.0f}, {p[0]}-{p[1]} {x * 1000:.1f} mm"
            free = [x for p, x in dist.items()
                    if p not in on and not (landing and p in rested) and not (stop and p in stop)]
            worst[0] = min([worst[0], *free])
            if stop and any(dist[p] < MARGIN for p in stop if p in dist):
                return None  # reached its own stop: the sweep ends here
        return None

    # WHY the whole fold back lands, not only its "rest" steps: once the upper arm is back on the
    # base, the wrist's last stage (a "pose") moves with it resting there again.
    folding = max((k for k, st in enumerate(steps) if st[0] == "sweep"), default=-1) + 1
    for k, (kind, j, to, sign) in enumerate(steps):
        if kind in ("pose", "rest") and k >= folding:
            err = go(j, to, "fold back", landing=True)
        elif kind == "pose":
            err = go(j, to, f"move {j}")
        else:  # a sweep: to each stop it is asked for, then back where it began (Sweep.start)
            began, err = q[j], None
            for side in ((sign,) if sign else (1, -1)):
                end = (100.0 if side > 0 else 0.0) if j == "gripper" else side * (half[j] + BEYOND)
                err = err or go(j, end, f"sweep {j} {'+-'[side < 0]}", STOP.get((j, side), set()))
            err = err or go(j, began, f"back after {j}")
        if err:
            return False, err, worst[0]
    return True, "", worst[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rest", type=Path, help="REST_POSE.json")
    ap.add_argument("cal", type=Path, help="CALIBRATION.json")
    ap.add_argument("shift_step", type=int, nargs="?", default=4)
    ap.add_argument("--joints", default=",".join(JOINTS), help="what the run sweeps, e.g. gripper")
    args = ap.parse_args()
    rest = json.loads(args.rest.read_text())
    cal = json.loads(args.cal.read_text())
    joints = args.joints.split(",")
    half = {j: (c["range_max"] - c["range_min"]) / 2 * 360 / 4095 for j, c in cal.items() if j != "gripper"}
    sc = Scene()
    variants = []
    for dl, de, dw in itertools.product(range(-12, 13, args.shift_step), repeat=3):
        q = dict(rest) | {"shoulder_lift": rest["shoulder_lift"] + dl, "elbow_flex": rest["elbow_flex"] + de,
                          "wrist_flex": rest["wrist_flex"] + dw}  # fmt: skip
        if abs(sc.clearance(q)[0]) <= 0.003:
            variants.append(((dl, de, dw), q))
    print(f"rest pose {rest}\n{len(variants)} physically possible versions (shifts up to 12 degrees)",
          flush=True)  # fmt: skip
    if not variants:
        print("no physically possible version: is the rest pose read through this arm's own file?")
        return 2
    fails, worst = [], np.inf
    for shift, q in variants:
        steps = steps_deg(cal, q, joints)  # per version: the fold back returns to its own rest
        ok, err, w = walk(sc, q, half, steps)
        if ok:
            worst = min(worst, w)
        else:
            fails.append((shift, err))
    print("program: " + ", ".join(f"{k} {j}" + (f" {'+-'[g < 0]}" if g else "") for k, j, _, g in steps))
    print(f"program: {len(variants) - len(fails)}/{len(variants)} versions pass; closest {worst * 1000:.1f} mm")
    for shift, err in fails[:10]:
        print(f"   FAIL shift {shift}: {err}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
