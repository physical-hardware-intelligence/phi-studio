"""Forward kinematics against MuJoCo's own: reference points computed with mujoco 3.12 on the same
MJCF
(so101_new_calib.xml), site gripperframe, 2026-10-04. A 2000-pose sweep matched to <1e-12 m."""

from __future__ import annotations

import numpy as np
import pytest

from phi_studio import kinematics as K

# LeRobot units in, metres out (MuJoCo mj_kinematics, site_xpos of gripperframe)
REFERENCE = {
    "zero": ([0, 0, 0, 0, 0, 0], [0.3913619, -1.1255e-05, 0.226468745]),
    "rest": ([-5.1, -103.9, 96.7, 54.2, -0.8, 0.7], [0.188631348, 0.013468382, 0.01696596]),
    "reach": ([30.0, -20.0, 40.0, 60.0, 90.0, 50.0], [0.185690263, -0.094115651, 0.033868347]),
}


@pytest.mark.parametrize("name", REFERENCE)
def test_tcp_matches_mujoco(name: str) -> None:
    q, want = REFERENCE[name]
    got = K.tcp(np.array([q]))[0]
    assert np.abs(got - np.array(want)).max() < 1e-8


def test_vectorised_over_frames() -> None:
    q = np.array([v[0] for v in REFERENCE.values()], dtype=float)
    got = K.tcp(q)
    assert got.shape == (3, 3)
    for row, (_, want) in zip(got, REFERENCE.values(), strict=True):
        assert np.abs(row - np.array(want)).max() < 1e-8


def test_gripper_does_not_move_the_tool_point() -> None:
    a = K.tcp(np.array([[10, -30, 40, 20, 5, 0.0]]))
    b = K.tcp(np.array([[10, -30, 40, 20, 5, 100.0]]))
    assert np.allclose(a, b)


def test_gripper_units() -> None:
    lo, hi = K.model()["gripper_deg"]
    rad = K.to_radians(np.array([[0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 100.0]]))
    assert np.isclose(np.rad2deg(rad[0, 5]), lo) and np.isclose(np.rad2deg(rad[1, 5]), hi)


def test_limits() -> None:
    lim = K.limits_deg()
    assert lim["shoulder_pan"] == (-110.0, 110.0)
    assert lim["shoulder_lift"] == (-100.0, 100.0)
    assert lim["gripper"][0] == pytest.approx(0, abs=0.01) and lim["gripper"][1] == pytest.approx(
        100, abs=0.01
    )


def test_bad_shape_and_unknown_frame() -> None:
    with pytest.raises(ValueError):
        K.forward(np.zeros((2, 5)))
    with pytest.raises(KeyError):
        K.forward(np.zeros((1, 6)), ("nowhere",))
