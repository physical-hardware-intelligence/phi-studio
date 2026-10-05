"""Environment reconstruction: a metric point cloud of the workspace from one camera picture.

Pure geometry and fitting, no model and no server, so every step is testable against a synthetic
scene (tests/test_recon.py).

The depth model (Depth Anything V2 Small, relative) predicts AFFINE-INVARIANT INVERSE DEPTH:
d = s / z + t with s and t unknown for each picture (arXiv 2406.09414, Sections 3 and 7.2).
Larger d is nearer. d is not depth and not metric. This module recovers s and t from the one
surface whose metric depth Studio knows without measuring anything: the table, the plane z = 0
the arm stands on.

    1. Each pixel's ray, from the camera pose and a pinhole model, is cut with the plane z = 0.
       Where it hits in front of the camera, near an arm base, the table's depth z_plane is known.
    2. Fit 1/z_plane = a * d + b on those pixels (a = 1/s, b = -t/s), robustly: RANSAC on pixel
       pairs, then weighted least squares on the inliers. Inliers are the table; the arm, objects
       and walls stand above it and disagree.
    3. z = 1 / (a * d + b) = s / (d - t) for every pixel, back-projected to the world.

Conventions (the 3D view's, web/src/scene/engine.ts):
  * world: the MJCF frame, metres, z up, the table is z = 0, a single arm's base at the origin
  * a camera looks down its own -z, +y is image up, +x image right (three.js); camera-to-world
    matrices are built the way three's Matrix4.lookAt builds them
  * pixels: u right, v down, (0, 0) the top-left corner of the top-left pixel
  * z: depth along the optical axis, which is how depth maps are defined. That the model's z is
    optical-axis depth rather than distance along the ray is an inference from how its training
    depth is defined, not something the paper states
  * pinhole intrinsics from the vertical field of view, square pixels, principal point at the
    image centre, NO lens distortion model
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from phi_studio import robot_model

# [JUDGEMENT] Table pixels count only within this radius of an arm base. WHY: near the arm the
# plane z = 0 is almost surely the desk the arm is screwed to; farther out a ray's hit with z = 0
# may be floor seen past the desk edge, or another desk, which would bias the fit.
WORKSPACE_RADIUS_M = 0.6
# [JUDGEMENT] A table pixel agrees with a fit when its depth is within this fraction of the
# table's depth: 3% is 24 mm at 0.8 m. WHY relative: model noise grows with distance, and a
# relative test needs no guess of the model's arbitrary output scale.
INLIER_TOLERANCE = 0.03
# [JUDGEMENT] Below this share of table pixels agreeing, the scale is not shown. WHY 0.5: a
# correctly placed synthetic desk with 40% of the table under objects gives 0.6 (tested). On a
# real frame (phi_so101_8bin_v1, episode 0, frame 60, front) [RUN], a placement whose table grid
# visibly lands on the wall gave 0.44, and a placement fitted by eye gave 0.60.
MIN_INLIER_FRACTION = 0.5
# [JUDGEMENT] At least this many table pixels, and this share of the picture. WHY: fewer leaves
# the fit to a sliver at the image edge, where the depth model is weakest.
MIN_PLANE_PIXELS = 500
MIN_PLANE_FRACTION = 0.03
# [JUDGEMENT] The farthest table pixel (5th percentile of 1/z) must be at least this many times
# the depth of the nearest (95th percentile). WHY: s and t are told apart only by how d changes
# with depth; a camera square on to the table sees it at one depth and any (s, t) on a line fits.
MIN_DEPTH_RATIO = 1.15
# [JUDGEMENT] Table nearer than this is not used. WHY: a lens a few millimetres off the desk sees
# a blur, and the model's output there means nothing; the wrist camera does this often.
MIN_TABLE_DEPTH_M = 0.03
# [JUDGEMENT] Points farther than this are dropped. WHY: the workspace is about a metre across,
# and at d - t near zero the depth runs off to infinity on noise alone.
FAR_CAP_M = 2.5
# [JUDGEMENT] Points per camera sent to the page. WHY: 80k points draw in well under a frame on
# integrated graphics, and 640x480 halved is 76,800, the whole picture at half resolution.
MAX_POINTS = 80_000
# [JUDGEMENT] "Hide points on the arm" drops points within this distance of a part's box. WHY:
# the CAD boxes fit the parts, but the drawn pose itself may be off by a few degrees per joint
# (robot_model's mapping is not checked on an arm).
ARM_MARGIN_M = 0.015
RANSAC_ITERATIONS = 256
CHECK_PLACEMENT = ("Check where the camera sits in the 3D view and its field of view. The wrist "
                   "camera is placed by the arm's joint readings.")
RANSAC_SCORE_SAMPLE = 4000  # pixels each hypothesis is scored on; the refit uses all of them
MAGIC = b"PCL1"


class ScaleRefused(ValueError):
    """No metric cloud: the scale from the table is not trustworthy. Carries the fit."""

    def __init__(self, message: str, fit: Fit | None = None) -> None:
        super().__init__(message)
        self.fit = fit


# -- cameras ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float,
                  x_stretch: float = 1.0) -> Pinhole:  # fmt: skip
        """Vertical field of view, as the 3D view stores it (three's PerspectiveCamera.fov).
        x_stretch: how much more the picture was scaled across than down (1 for square pixels)."""
        if not 1.0 <= fovy_deg <= 179.0:
            raise ValueError(f"field of view {fovy_deg} degrees is not possible for a camera")
        f = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
        return cls(width, height, f * x_stretch, f, width / 2, height / 2)


def look_at(pos: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """Camera-to-world 4x4, as three's Matrix4.lookAt: z = eye - target, x = up x z, y = z x x."""
    eye, tgt, u = (np.asarray(v, dtype=np.float64) for v in (pos, target, up))
    z = eye - tgt
    if np.linalg.norm(z) == 0:
        z = np.array([0.0, 0.0, 1.0])
    z = z / np.linalg.norm(z)
    x = np.cross(u, z)
    if np.linalg.norm(x) < 1e-12:  # up parallel to the view: nudge z, as three does
        z = z + (np.array([1e-4, 0, 0]) if abs(u[2]) == 1 else np.array([0, 0, 1e-4]))
        z = z / np.linalg.norm(z)
        x = np.cross(u, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, y, z, eye
    return out


def _rotate(v: np.ndarray, r: np.ndarray) -> np.ndarray:
    """r @ each vector in v (..., 3). WHY einsum, not `@`: on this Mac `@` (Apple Accelerate)
    printed divide-by-zero and overflow warnings on finite inputs [RUN], as align.py found too."""
    return np.einsum("ij,...j->...i", r, v)


def pixel_rays(cam: Pinhole) -> np.ndarray:
    """(h, w, 3) rays through pixel centres in the camera frame, scaled so z = -1: a point at
    optical-axis depth z is z times its ray."""
    u = np.arange(cam.width) + 0.5
    v = np.arange(cam.height) + 0.5
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - cam.cx) / cam.fx, -(vv - cam.cy) / cam.fy,
                     -np.ones_like(uu)], axis=-1)  # fmt: skip


def project(cam: Pinhole, c2w: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, ...]:
    """World points (..., 3) to pixel u, v and optical-axis depth z (positive in front)."""
    pc = _rotate(np.asarray(points, dtype=np.float64) - c2w[:3, 3], c2w[:3, :3].T)
    z = -pc[..., 2]
    return cam.cx + cam.fx * pc[..., 0] / z, cam.cy - cam.fy * pc[..., 1] / z, z


def unproject(cam: Pinhole, c2w: np.ndarray, u: np.ndarray, v: np.ndarray,
              z: np.ndarray) -> np.ndarray:  # fmt: skip
    """Pixel u, v at optical-axis depth z to world points (..., 3)."""
    z = np.asarray(z, dtype=np.float64)
    pc = np.stack([(np.asarray(u) - cam.cx) / cam.fx * z, -(np.asarray(v) - cam.cy) / cam.fy * z,
                   -z], axis=-1)  # fmt: skip
    return _rotate(pc, c2w[:3, :3]) + c2w[:3, 3]


def table_depth(cam: Pinhole, c2w: np.ndarray, bases: Iterable[Sequence[float]],
                radius: float = WORKSPACE_RADIUS_M) -> np.ndarray:  # fmt: skip
    """(h, w) optical-axis depth of the table (z = 0) per pixel; NaN where the ray misses it,
    hits it behind the camera, or hits it farther than `radius` from every arm base."""
    o = c2w[:3, 3]
    out = np.full((cam.height, cam.width), np.nan)
    if o[2] <= 0:  # at or under the table: no ray meets its top
        return out
    d = _rotate(pixel_rays(cam), c2w[:3, :3])  # world directions; the ray parameter is the depth
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[2] / d[..., 2]
    hit = o + np.where(np.isfinite(t), t, 0.0)[..., None] * d
    near = np.zeros(t.shape, bool)
    for b in bases:
        near |= np.hypot(hit[..., 0] - b[0], hit[..., 1] - b[1]) <= radius
    ok = np.isfinite(t) & (t > 0) & near
    out[ok] = t[ok]
    return out


# -- the scale fit ---------------------------------------------------------------------------------
@dataclass
class Fit:
    ok: bool
    message: str
    s: float = math.nan  # d = s / z + t
    t: float = math.nan
    inlier_fraction: float = 0.0  # of the table pixels used
    # median |z_fit - z_table| over the inliers, along the view axis. IN-SAMPLE: the same table
    # pixels the fit used. It cannot see a wrong camera pose: on a plane 1/z is affine in the
    # pixel position, so a wrongly placed camera often still fits the table perfectly (s and t
    # absorb the error) while points above the table come out wrong.
    median_mm: float = math.nan
    plane_pixels: int = 0  # table pixels the fit used
    depth_ratio: float = math.nan  # far / near table depth over the pixels used
    inliers: np.ndarray | None = field(default=None, repr=False)

    def numbers(self) -> dict[str, float | int | None]:
        def num(x: float, nd: int) -> float | None:
            return None if not math.isfinite(x) else round(x, nd)

        return {"s": num(self.s, 6), "t": num(self.t, 6),
                "inlier_fraction": round(self.inlier_fraction, 4),
                "median_mm": num(self.median_mm, 2), "plane_pixels": self.plane_pixels,
                "depth_ratio": num(self.depth_ratio, 3)}  # fmt: skip


def _refit(d: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Least squares of x = a d + b, weighted by 1/x so each pixel's error counts relative to its
    depth, the same measure the inlier test uses."""
    m = np.stack([d / x, 1.0 / x], axis=1)
    (a, b), *_ = np.linalg.lstsq(m, np.ones_like(x), rcond=None)
    return float(a), float(b)


def fit_scale(disparity: np.ndarray, z_plane: np.ndarray, seed: int = 0) -> Fit:
    """Recover s and t of d = s / z + t from the pixels where the table's depth is known."""
    disparity = np.asarray(disparity, dtype=np.float64)
    hits = np.isfinite(z_plane) & np.isfinite(disparity)
    too_near = hits & (z_plane < MIN_TABLE_DEPTH_M)
    cand = hits & ~too_near
    n = int(cand.sum())
    need = max(MIN_PLANE_PIXELS, int(MIN_PLANE_FRACTION * disparity.size))
    if n < need:
        if int(too_near.sum()) > n:
            return Fit(False, f"The camera is too close to the table to set the scale: most of "
                       f"the table it sees is nearer than {MIN_TABLE_DEPTH_M * 100:.0f} cm.",
                       plane_pixels=n)  # fmt: skip
        return Fit(False, f"The camera sees too little of the table near the arm to set the "
                   f"scale ({n} pixels, needs {need}). Check where it sits in the 3D view, or use "
                   f"a camera that sees the desk.", plane_pixels=n)  # fmt: skip
    d = disparity[cand]
    x = 1.0 / z_plane[cand]
    lo, hi = np.percentile(x, [5, 95])
    ratio = float(hi / lo)
    if ratio < MIN_DEPTH_RATIO:
        return Fit(False, "The table sits at nearly one distance from this camera, so the depth "
                   "model's scale and offset cannot be told apart. A camera that looks at the "
                   "desk at a slant works; one square on to it does not.",
                   plane_pixels=n, depth_ratio=ratio)  # fmt: skip

    rng = np.random.default_rng(seed)
    score = rng.choice(n, size=min(n, RANSAC_SCORE_SAMPLE), replace=False)
    i = rng.integers(0, n, RANSAC_ITERATIONS)
    j = rng.integers(0, n, RANSAC_ITERATIONS)
    dd = d[i] - d[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (x[i] - x[j]) / dd
    b = x[i] - a * d[i]
    good = np.isfinite(a) & (np.abs(dd) > 1e-12) & (a > 0)  # a > 0: larger d must be nearer
    best = 0
    a0 = b0 = math.nan
    if good.any():
        ag, bg = a[good], b[good]
        res = np.abs(ag[:, None] * d[score][None, :] + bg[:, None] - x[score][None, :])
        counts = (res <= INLIER_TOLERANCE * x[score][None, :]).sum(axis=1)
        k = int(np.argmax(counts))
        best, a0, b0 = int(counts[k]), float(ag[k]), float(bg[k])
    if best == 0:
        return Fit(False, "No single scale makes the table flat: the depth picture does not grow "
                   f"nearer where the table does. {CHECK_PLACEMENT}",
                   plane_pixels=n, depth_ratio=ratio)  # fmt: skip
    inl = np.zeros(n, bool)
    for _ in range(3):  # refit on the inliers, then re-take the inliers under the refit
        inl = np.abs(a0 * d + b0 - x) <= INLIER_TOLERANCE * x
        if inl.sum() < 2:
            break
        a0, b0 = _refit(d[inl], x[inl])
    inl = np.abs(a0 * d + b0 - x) <= INLIER_TOLERANCE * x
    frac = float(inl.mean())
    s, t = (1.0 / a0, -b0 / a0) if a0 > 0 else (math.nan, math.nan)
    zfit = 1.0 / np.where(a0 * d + b0 > 0, a0 * d + b0, np.nan)
    med = float(np.nanmedian(np.abs(zfit[inl] - 1.0 / x[inl]))) * 1000 if inl.any() else math.nan
    mask = np.zeros(disparity.shape, bool)
    mask[cand] = inl
    fit = Fit(True, "", s, t, frac, med, n, ratio, mask)
    if not a0 > 0:
        fit.ok, fit.message = False, ("The fit came out with brighter meaning farther, which is "
                                      f"backwards. {CHECK_PLACEMENT}")
    elif frac < MIN_INLIER_FRACTION:
        fit.ok = False
        fit.message = (f"Only {frac:.0%} of the table pixels agree on one scale (needs "
                       f"{MIN_INLIER_FRACTION:.0%}). {CHECK_PLACEMENT}")  # fmt: skip
    return fit


def metric_depth(disparity: np.ndarray, fit: Fit, far: float = FAR_CAP_M) -> np.ndarray:
    """z = s / (d - t) per pixel, metres along the optical axis; NaN where d - t <= 0 or z > far."""
    if not fit.ok:
        raise ScaleRefused(fit.message, fit)
    shifted = np.asarray(disparity, dtype=np.float64) - fit.t
    with np.errstate(divide="ignore", invalid="ignore"):
        z = fit.s / shifted
    return np.where((shifted > 0) & (z <= far), z, np.nan)


# -- the arm, for hiding its points ----------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """An oriented box: centre, axes (columns of a rotation), half sizes. World frame, metres."""

    centre: np.ndarray
    axes: np.ndarray
    half: np.ndarray


@lru_cache(maxsize=32)
def _stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Min and max corner of a binary STL's vertices, in the file's units."""
    raw = path.read_bytes()
    (count,) = struct.unpack_from("<I", raw, 80)
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path.name} is not a binary STL")
    rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                             ("a", "<u2")]), count=count, offset=84)  # fmt: skip
    v = rec["v"].reshape(-1, 3).astype(np.float64)
    return v.min(axis=0), v.max(axis=0)


def arm_boxes(model: robot_model.RobotModel, q: Mapping[str, float],
              base: Sequence[float]) -> list[Box]:  # fmt: skip
    """One box per drawn part at joint angles q (MJCF radians), the arm's base at `base`. Each is
    the part's CAD bounding box in its own mesh frame, carried by forward kinematics: an
    approximation that is a little larger than the part."""
    world = robot_model.forward_kinematics(model, q)
    out = []
    for g in model.geoms:
        if g.role != "visual" or g.mesh is None:
            continue
        lo, hi = _stl_bounds(model.mesh_path(g.mesh))
        scale = np.array(model.meshes[g.mesh].scale)
        pose = robot_model.geom_pose(model, world, g)
        out.append(Box(pose[:3, :3] @ ((lo + hi) / 2 * scale) + pose[:3, 3] + np.asarray(base),
                       pose[:3, :3], np.abs(hi - lo) / 2 * scale))  # fmt: skip
    return out


def on_arm(points: np.ndarray, boxes: Sequence[Box], margin: float = ARM_MARGIN_M) -> np.ndarray:
    """Whether each world point lies within `margin` of any box."""
    hit = np.zeros(len(points), bool)
    for b in boxes:
        local = _rotate(points - b.centre, b.axes.T)
        hit |= np.all(np.abs(local) <= b.half + margin, axis=1)
    return hit


def wrist_camera_pose(model: robot_model.RobotModel, q: Mapping[str, float],
                      base: Sequence[float]) -> np.ndarray:  # fmt: skip
    """Camera-to-world of the wrist camera at joint angles q (MJCF radians), as the 3D view
    places it: robot_model.WRIST_CAMERA in the body wrist_camera, looking down its -z."""
    wc = robot_model.WRIST_CAMERA
    world = robot_model.forward_kinematics(model, q)
    pos = wc["pos"]
    quat = wc["quat"]
    assert isinstance(pos, tuple) and isinstance(quat, tuple)
    out = world[str(wc["body"])] @ robot_model.transform(pos, robot_model.quat_matrix(quat))
    out[:3, 3] += np.asarray(base, dtype=np.float64)
    return out


# -- the whole picture -----------------------------------------------------------------------------
def working_size(width: int, height: int, cap: int = MAX_POINTS) -> tuple[int, int]:
    """The picture shrunk by the smallest whole factor that brings it to at most `cap` pixels.
    Each side is floored, so when a side does not divide, the two sides shrink by slightly
    different factors; the caller passes that difference to Pinhole.from_fovy as x_stretch."""
    n = 1
    while (width // n) * (height // n) > cap:
        n += 1
    return width // n, height // n


@dataclass
class Cloud:
    positions: np.ndarray  # (n, 3) world, metres; points off the arm first
    colors: np.ndarray  # (n, 3) uint8 sRGB from the picture
    n_off_arm: int
    fit: Fit


def reconstruct(rgb: np.ndarray, disparity: np.ndarray, cam: Pinhole, c2w: np.ndarray,
                bases: Iterable[Sequence[float]], arms: Sequence[Sequence[Box]],
                max_points: int = MAX_POINTS, margin: float = ARM_MARGIN_M,
                seed: int = 0) -> Cloud:  # fmt: skip
    """A metric cloud from one picture and its model output, both at the camera's resolution.
    Raises ScaleRefused with the reason when the scale from the table is not trustworthy."""
    if rgb.shape[:2] != (cam.height, cam.width) or disparity.shape != (cam.height, cam.width):
        raise ValueError("picture, depth and camera sizes differ")
    if c2w[2, 3] <= 0:
        raise ScaleRefused("The camera is placed at or below the table, so it cannot see the "
                           "table top. Check its height in the 3D view.",
                           Fit(False, "camera below the table"))  # fmt: skip
    fit = fit_scale(disparity, table_depth(cam, c2w, list(bases)), seed=seed)
    if not fit.ok:
        raise ScaleRefused(fit.message, fit)
    z = metric_depth(disparity, fit)
    uu, vv = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
    valid = np.isfinite(z)
    pts = unproject(cam, c2w, uu[valid], vv[valid], z[valid])
    col = rgb[valid]
    if len(pts) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(len(pts), max_points, replace=False))
        pts, col = pts[keep], col[keep]
    flat = [b for arm in arms for b in arm]
    flags = on_arm(pts, flat, margin) if flat else np.zeros(len(pts), bool)
    order = np.argsort(flags, kind="stable")
    return Cloud(pts[order], np.ascontiguousarray(col[order], dtype=np.uint8),
                 int((~flags).sum()), fit)  # fmt: skip


def pack(cloud: Cloud) -> bytes:
    """b"PCL1", uint32 n, uint32 n_off_arm (little-endian), float32 xyz * n, uint8 rgb * n.
    WHY the positions first: they start at byte 12, so the page can view them as a Float32Array
    without a copy (a Float32Array must start on a multiple of 4)."""
    n = len(cloud.positions)
    return (MAGIC + struct.pack("<II", n, cloud.n_off_arm)
            + np.ascontiguousarray(cloud.positions, dtype="<f4").tobytes()
            + np.ascontiguousarray(cloud.colors, dtype=np.uint8).tobytes())  # fmt: skip


def unpack(blob: bytes) -> tuple[int, int, np.ndarray, np.ndarray]:
    if blob[:4] != MAGIC:
        raise ValueError("not a point cloud")
    n, n_off = struct.unpack_from("<II", blob, 4)
    pos = np.frombuffer(blob, dtype="<f4", count=3 * n, offset=12).reshape(n, 3)
    col = np.frombuffer(blob, dtype=np.uint8, count=3 * n, offset=12 + 12 * n).reshape(n, 3)
    return n, n_off, pos, col
