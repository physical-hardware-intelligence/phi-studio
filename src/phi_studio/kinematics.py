"""SO-101 forward kinematics from Studio's model asset (assets/so101/model.json).

The same kinematic tree the 3D view draws, so a plotted tool path and the arm on screen agree.
Vectorised
over frames: one call turns an episode's joint array into tool positions.

Units follow LeRobot (so_follower.py:51-62): five arm joints in degrees, gripper 0..100. "new_calib"
model: a joint's zero is the middle of its range, as LeRobot >= 0.5 calibrates, so degrees map to
model
radians directly (scripts/build_so101_model.py). The gripper's 0..100 spans the model's -10..100
degrees.

Accuracy: checked against MuJoCo's own kinematics on the same MJCF (tests/test_kinematics.py), to
well
under a micrometre. How closely a real arm matches depends on its calibration; that is what
auto-calibration is for.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MODEL = Path(__file__).parent / "assets" / "so101" / "model.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@lru_cache(maxsize=1)
def model() -> dict[str, Any]:
    return json.loads(MODEL.read_text())


def quat_to_mat(q: list[float] | np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n else 0.0
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def axis_angle(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotations about one unit axis by N angles -> [N, 3, 3] (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    c, s = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def to_radians(q: np.ndarray) -> np.ndarray:
    """[N, 6] LeRobot units (degrees, gripper 0..100) -> [N, 6] model radians."""
    q = np.asarray(q, dtype=np.float64)
    out = np.deg2rad(q)
    lo, hi = model()["gripper_deg"]
    out[:, 5] = np.deg2rad(lo + q[:, 5] / 100.0 * (hi - lo))
    return out


def forward(q: np.ndarray, frames: tuple[str, ...] = ("gripperframe",)) -> dict[str, np.ndarray]:
    """World positions [N, 3] (metres, base frame, z up) of the named bodies and sites, for joint
    values q [N, 6] in LeRobot units."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    if q.shape[1] != 6:
        raise ValueError(f"expected [N, 6] joint values, got {q.shape}")
    rad = to_radians(q)
    n = len(q)
    out: dict[str, np.ndarray] = {}
    want = set(frames)

    def walk(b: dict[str, Any], rot: np.ndarray, pos: np.ndarray) -> None:
        pos = pos + rot @ np.asarray(b["pos"])
        rot = rot @ quat_to_mat(b["quat"])
        j = b["joint"]
        if j is not None:
            jp = np.asarray(j["pos"])
            r = axis_angle(np.asarray(j["axis"], dtype=np.float64), rad[:, JOINTS.index(j["name"])])
            # Rotation about an axis through the joint's anchor jp (zero for every SO-101 joint).
            pos = pos + rot @ jp - (rot @ r @ jp[:, None])[..., 0]
            rot = rot @ r
        if b["name"] in want:
            out[b["name"]] = pos.copy()
        for s in b["sites"]:
            if s["name"] in want:
                out[s["name"]] = pos + rot @ np.asarray(s["pos"])
        for c in b["children"]:
            walk(c, rot, pos)

    walk(model()["tree"], np.broadcast_to(np.eye(3), (n, 3, 3)).copy(), np.zeros((n, 3)))
    missing = want - out.keys()
    if missing:
        raise KeyError(f"no body or site named {', '.join(sorted(missing))}")
    return out


def tcp(q: np.ndarray) -> np.ndarray:
    """Tool centre point [N, 3]: the model's gripperframe site, between the jaws."""
    return forward(q, (model()["tcp_site"],))[model()["tcp_site"]]


def limits_deg() -> dict[str, tuple[float, float]]:
    """Each joint's model range in LeRobot units (degrees; the gripper in 0..100)."""
    out: dict[str, tuple[float, float]] = {}

    def walk(b: dict[str, Any]) -> None:
        j = b["joint"]
        if j is not None:
            lo, hi = (float(np.rad2deg(x)) for x in j["range"])
            if j["name"] == "gripper":
                glo, ghi = model()["gripper_deg"]
                lo, hi = ((v - glo) / (ghi - glo) * 100.0 for v in (lo, hi))
            out[j["name"]] = (round(lo, 2), round(hi, 2))
        for c in b["children"]:
            walk(c)

    walk(model()["tree"])
    return out
