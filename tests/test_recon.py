"""Environment reconstruction: the scale fit and back-projection against synthetic ground truth.

The scene is built by ray casting (table plane z = 0, a box on it, a wall behind the arm), so the
true optical-axis depth of every pixel is known. The fake model output is d = s/z + t + noise, the
affine-invariant inverse depth Depth Anything V2 predicts (arXiv 2406.09414). The fit must recover
the metric cloud from the table alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from phi_studio import recon, robot_model

W, H, FOVY = 320, 240, 45.0
BOX_MIN = np.array([0.15, -0.05, 0.0])
BOX_MAX = np.array([0.25, 0.05, 0.06])
WALL_X = -0.25  # a wall behind the arm base, facing +x


def cast(cam: recon.Pinhole, c2w: np.ndarray, box: bool = True, wall: bool = True) -> np.ndarray:
    """True optical-axis depth per pixel (h, w); inf where the ray hits nothing."""
    rays = recon.pixel_rays(cam).reshape(-1, 3)  # camera frame, z component -1
    o = c2w[:3, 3]
    d = np.einsum("ij,nj->ni", c2w[:3, :3], rays)  # world direction; the ray parameter is depth
    best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[2] / d[:, 2]
        best = np.where((t > 0) & np.isfinite(t), np.minimum(best, t), best)
        if wall:
            t = (WALL_X - o[0]) / d[:, 0]
            p = o + t[:, None] * d
            ok = (t > 0) & (p[:, 2] >= 0) & (p[:, 2] <= 1.0) & (np.abs(p[:, 1]) <= 1.0)
            best = np.where(ok, np.minimum(best, t), best)
        if box:
            t0 = (BOX_MIN - o) / d
            t1 = (BOX_MAX - o) / d
            near = np.nanmax(np.minimum(t0, t1), axis=1)
            far = np.nanmin(np.maximum(t0, t1), axis=1)
            ok = (far >= near) & (near > 0)
            best = np.where(ok, np.minimum(best, near), best)
    return best.reshape(cam.height, cam.width)


def fake_model(z: np.ndarray, s: float, t: float, noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inv = np.where(np.isfinite(z), 1.0 / z, 0.0)  # nothing hit: as far as it gets
    d = s * inv + t
    return d + rng.normal(0.0, noise, d.shape)


def front_camera() -> tuple[recon.Pinhole, np.ndarray]:
    cam = recon.Pinhole.from_fovy(W, H, FOVY)
    return cam, recon.look_at((0.75, 0.12, 0.38), (0.12, 0.0, 0.04), (0, 0, 1))


def test_project_and_back_project_round_trip() -> None:
    cam, c2w = front_camera()
    rng = np.random.default_rng(1)
    pts = rng.uniform([-0.2, -0.3, 0.0], [0.4, 0.3, 0.3], size=(500, 3))
    u, v, z = recon.project(cam, c2w, pts)
    assert np.all(z > 0)
    back = recon.unproject(cam, c2w, u, v, z)
    assert np.max(np.abs(back - pts)) < 1e-9


def test_camera_convention_matches_three_lookat() -> None:
    """three.js: a camera looks down its own -z with +y up, and Matrix4.lookAt builds z = eye -
    target, x = up x z, y = z x x. From +x looking at the origin with z up, +y world is on the
    image's right and +z is above the centre (image v grows downward)."""
    cam = recon.Pinhole.from_fovy(W, H, FOVY)
    c2w = recon.look_at((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 0, 1))
    assert np.allclose(c2w[:3, 2], [1, 0, 0]) and np.allclose(c2w[:3, 0], [0, 1, 0])
    u, v, z = recon.project(cam, c2w, np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0, 0, 0]]))
    assert u[0] > cam.cx and abs(v[0] - cam.cy) < 1e-9
    assert v[1] < cam.cy and abs(u[1] - cam.cx) < 1e-9
    assert abs(z[2] - 1.0) < 1e-12
    # Vertical field of view: a point on the top edge of the picture sits fovy/2 above the axis.
    top = np.tan(np.radians(FOVY / 2))
    _, v, _ = recon.project(cam, c2w, np.array([[0.0, 0.0, top]]))
    assert abs(v[0]) < 1e-9


def test_fit_recovers_metric_scene() -> None:
    cam, c2w = front_camera()
    z_true = cast(cam, c2w)
    disp = fake_model(z_true, s=1.7, t=0.35, noise=0.002)
    fit = recon.fit_scale(disp, recon.table_depth(cam, c2w, bases=[(0.0, 0.0)]))
    assert fit.ok, fit.message
    # s and t are recovered: they are only defined up to the noise.
    assert fit.s == pytest.approx(1.7, rel=0.02) and fit.t == pytest.approx(0.35, abs=0.02)
    assert fit.median_mm < 2.0 and fit.inlier_fraction > 0.8

    z = recon.metric_depth(disp, fit)
    pts = recon.unproject(cam, c2w, *np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5), z)
    on_table = np.isfinite(z_true) & np.isclose(cast(cam, c2w, box=False, wall=False), z_true)
    table_err = np.abs(pts[on_table][:, 2]) * 1000
    assert np.median(table_err) < 2.0, np.median(table_err)

    true_pts = recon.unproject(cam, c2w, *np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5),
                               z_true)  # fmt: skip
    top = np.isfinite(z_true) & (np.abs(true_pts[..., 2] - BOX_MAX[2]) < 1e-6)
    assert top.sum() > 100  # the camera sees the box top
    box_err = np.abs(pts[top][:, 2] - BOX_MAX[2]) * 1000
    assert np.median(box_err) < 5.0, np.median(box_err)

    wall = np.isfinite(z_true) & (np.abs(true_pts[..., 0] - WALL_X) < 1e-6)
    assert wall.sum() > 100
    assert np.median(np.abs(pts[wall][:, 0] - WALL_X)) * 1000 < 10.0


def test_fit_survives_forty_percent_non_table_pixels() -> None:
    cam, c2w = front_camera()
    z_true = cast(cam, c2w, box=False, wall=False)
    plane = recon.table_depth(cam, c2w, bases=[(0.0, 0.0)])
    cand = np.isfinite(plane)
    rng = np.random.default_rng(3)
    clutter = cand & (rng.random(z_true.shape) < 0.4)  # objects 2 to 30 cm nearer than the table
    z_seen = np.where(clutter, z_true * rng.uniform(0.7, 0.98, z_true.shape), z_true)
    disp = fake_model(z_seen, s=0.9, t=-0.1, noise=0.001, seed=4)
    fit = recon.fit_scale(disp, plane)
    assert fit.ok, fit.message
    assert 0.5 < fit.inlier_fraction < 0.7  # the table, and only the table, agrees
    z = recon.metric_depth(disp, fit)
    pts = recon.unproject(cam, c2w, *np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5), z)
    table = cand & ~clutter
    assert np.median(np.abs(pts[table][:, 2])) * 1000 < 2.0


def test_ceiling_camera_is_refused() -> None:
    cam = recon.Pinhole.from_fovy(W, H, FOVY)
    c2w = recon.look_at((0.3, 0.0, 0.5), (0.3, 0.0, 1.5), (1, 0, 0))
    disp = fake_model(np.full((H, W), 1.0), 1.0, 0.0, 0.001)
    fit = recon.fit_scale(disp, recon.table_depth(cam, c2w, bases=[(0.0, 0.0)]))
    assert not fit.ok and "table" in fit.message
    with pytest.raises(recon.ScaleRefused):
        recon.metric_depth(disp, fit)


def test_wrist_camera_touching_the_table_is_refused() -> None:
    cam = recon.Pinhole.from_fovy(W, H, 85.0)
    c2w = recon.look_at((0.2, 0.0, 0.004), (0.2, 0.0, 0.0), (1, 0, 0))
    z_true = cast(cam, c2w, box=False, wall=False)
    fit = recon.fit_scale(fake_model(z_true, 1.0, 0.1, 0.001),
                          recon.table_depth(cam, c2w, bases=[(0.0, 0.0)]))  # fmt: skip
    assert not fit.ok and "close" in fit.message


def test_camera_square_on_to_the_table_is_refused() -> None:
    """Looking straight down, every table pixel is at one depth: s and t cannot be told apart."""
    cam = recon.Pinhole.from_fovy(W, H, 30.0)
    c2w = recon.look_at((0.2, 0.0, 0.7), (0.2, 0.0, 0.0), (1, 0, 0))
    z_true = cast(cam, c2w, box=False, wall=False)
    fit = recon.fit_scale(fake_model(z_true, 1.0, 0.1, 0.001),
                          recon.table_depth(cam, c2w, bases=[(0.0, 0.0)]))  # fmt: skip
    assert not fit.ok and "one distance" in fit.message


def test_inverted_depth_is_refused() -> None:
    """Brighter must be nearer (s > 0). A map that grows with distance fits only with s < 0."""
    cam, c2w = front_camera()
    z_true = cast(cam, c2w, box=False, wall=False)
    fit = recon.fit_scale(fake_model(z_true, -1.0, 3.0, 0.001),
                          recon.table_depth(cam, c2w, bases=[(0.0, 0.0)]))  # fmt: skip
    assert not fit.ok


def test_camera_under_the_table_is_refused() -> None:
    cam = recon.Pinhole.from_fovy(W, H, FOVY)
    c2w = recon.look_at((0.3, 0.0, -0.2), (0.3, 0.0, 0.5), (1, 0, 0))
    fit = recon.fit_scale(np.ones((H, W)), recon.table_depth(cam, c2w, bases=[(0.0, 0.0)]))
    assert not fit.ok


def test_wrist_camera_pose_follows_the_arm() -> None:
    m = robot_model.load()
    a = recon.wrist_camera_pose(m, {}, base=(0.0, 0.0, 0.0))
    b = recon.wrist_camera_pose(m, {"shoulder_pan": 0.5}, base=(0.0, 0.2, 0.0))
    assert a.shape == (4, 4) and np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3))
    assert not np.allclose(a, b)
    world = robot_model.forward_kinematics(m, {})
    lens = world["wrist_camera"] @ robot_model.transform(robot_model.WRIST_CAMERA["pos"], np.eye(3))
    assert np.allclose(a[:3, 3], lens[:3, 3])


def test_points_on_the_arm_are_flagged() -> None:
    m = robot_model.load()
    boxes = recon.arm_boxes(m, {}, base=(0.0, 0.0, 0.0))
    assert len(boxes) > 10
    centre = boxes[0].centre
    pts = np.array([centre, centre + [0.0, 0.0, 2.0], [0.6, 0.6, 0.0]])
    assert recon.on_arm(pts, boxes, margin=0.01).tolist() == [True, False, False]


def test_reconstruct_packs_and_caps() -> None:
    cam, c2w = front_camera()
    z_true = cast(cam, c2w)
    disp = fake_model(z_true, 1.7, 0.35, 0.002)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = 200
    m = robot_model.load()
    cloud = recon.reconstruct(rgb, disp, cam, c2w, bases=[(0.0, 0.0)],
                              arms=[recon.arm_boxes(m, {}, base=(0.0, 0.0, 0.0))],
                              max_points=5000)  # fmt: skip
    assert cloud.fit.ok and 0 < len(cloud.positions) <= 5000
    assert 0 < cloud.n_off_arm < len(cloud.positions)  # the arm stands in this view
    blob = recon.pack(cloud)
    n, n_off, pos, col = recon.unpack(blob)
    assert n == len(cloud.positions) and n_off == cloud.n_off_arm
    assert np.allclose(pos, cloud.positions.astype(np.float32)) and (col[:, 0] == 200).all()
    # Off-arm points come first, so the page hides the arm with a draw range alone.
    flagged = recon.on_arm(pos.astype(np.float64), recon.arm_boxes(m, {}, base=(0, 0, 0)),
                           margin=recon.ARM_MARGIN_M)  # fmt: skip
    assert not flagged[:n_off].any() and flagged[n_off:].all()


def test_reconstruct_refusal_carries_the_reason() -> None:
    cam = recon.Pinhole.from_fovy(W, H, FOVY)
    c2w = recon.look_at((0.3, 0.0, 0.5), (0.3, 0.0, 1.5), (1, 0, 0))
    with pytest.raises(recon.ScaleRefused) as e:
        recon.reconstruct(np.zeros((H, W, 3), np.uint8), np.ones((H, W)), cam, c2w,
                          bases=[(0.0, 0.0)], arms=[])  # fmt: skip
    assert e.value.fit is not None and not e.value.fit.ok


def test_working_size_keeps_aspect_under_the_cap() -> None:
    assert recon.working_size(640, 480, 80_000) == (320, 240)
    w, h = recon.working_size(1920, 1080, 80_000)
    assert w * h <= 80_000 and abs(w / h - 1920 / 1080) < 0.02
    assert recon.working_size(200, 100, 80_000) == (200, 100)


def test_uneven_shrink_keeps_projection() -> None:
    """848x480 floors to 282x160: across shrinks by 3.007, down by 3.0. With x_stretch the working
    camera projects every point where the full camera does, scaled."""
    w0, h0 = 848, 480
    w, h = recon.working_size(w0, h0, 50_000)
    assert (w0 / w) != (h0 / h)
    full = recon.Pinhole.from_fovy(w0, h0, 50.0)
    work = recon.Pinhole.from_fovy(w, h, 50.0, x_stretch=(w / w0) / (h / h0))
    c2w = recon.look_at((0.6, -0.3, 0.4), (0.1, 0.0, 0.0), (0, 0, 1))
    pts = np.random.default_rng(5).uniform([-0.1, -0.2, 0], [0.3, 0.2, 0.2], (200, 3))
    uf, vf, _ = recon.project(full, c2w, pts)
    uw, vw, _ = recon.project(work, c2w, pts)
    assert np.allclose(uw, uf * w / w0, atol=1e-9) and np.allclose(vw, vf * h / h0, atol=1e-9)


def test_table_fit_cannot_see_a_wrong_camera_height() -> None:
    """A documented limit, pinned so the UI keeps saying it: on a plane 1/z is affine in the pixel
    position, so a camera placed 15 cm too low still fits the table perfectly, with the wrong s
    and t, and points above the table come back wrong. The fit numbers cannot catch this."""
    cam = recon.Pinhole.from_fovy(W, H, 50.0)
    true = recon.look_at((0.7, 0.0, 0.60), (0.1, 0.0, 0.0), (0, 0, 1))
    wrong = recon.look_at((0.7, 0.0, 0.45), (0.1, 0.0, 0.0), (0, 0, 1))
    disp = fake_model(cast(cam, true, box=False, wall=False), 0.7, 0.3, 0.0)
    fit = recon.fit_scale(disp, recon.table_depth(cam, wrong, bases=[(0.0, 0.0)]))
    assert fit.ok and fit.inlier_fraction > 0.99 and fit.median_mm < 0.01
    assert abs(fit.s - 0.7) > 0.1  # the scale is wrong all the same


def small_cloud_blob() -> bytes:
    """Three points, two off the arm, with colours that pin the byte order and the colour LUT."""
    pos = np.array([[0.25, -0.5, 0.0], [1.0, 2.0, 3.0], [-0.125, 0.0625, 0.5]], dtype=np.float64)
    col = np.array([[0, 128, 255], [255, 0, 1], [10, 20, 30]], dtype=np.uint8)
    return recon.pack(recon.Cloud(pos, col, 2, recon.Fit(True, "")))


def test_pack_matches_the_web_fixture() -> None:
    """web/tests/recon.test.ts parses this same file with the page's parser: the two sides of the
    binary format are checked against one set of bytes."""
    from pathlib import Path

    fixture = Path(__file__).parents[1] / "web" / "tests" / "fixtures" / "cloud3.pcl"
    assert fixture.read_bytes() == small_cloud_blob()
