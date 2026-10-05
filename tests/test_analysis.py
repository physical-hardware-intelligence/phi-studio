"""Episode analysis: every rule on a signal whose answer is known, and the whole-dataset pass on
disk."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from support.lerobot_data import REST, episode_signals, write_v3

from phi_studio import analysis as A
from phi_studio.datasets import Dataset, split_arms

SINGLE = split_arms([f"{j}.pos" for j in A.JOINTS])
FPS = 30.0


def frames(action: np.ndarray, state: np.ndarray) -> dict[str, np.ndarray]:
    n = len(action)
    return {
        "timestamp": np.arange(n) / FPS,
        "frame_index": np.arange(n),
        "action": action,
        "state": state,
    }


def test_savgol_derivative_is_exact_on_a_cubic() -> None:
    t = np.arange(200) / FPS
    x = (0.5 * t**3 - 2 * t**2 + t)[:, None]
    v = A.smooth_deriv(x, FPS, 1)[:, 0]
    want = 1.5 * t**2 - 4 * t + 1
    inner = slice(10, -10)  # edges are padded, so only the interior is exact
    assert np.abs(v[inner] - want[inner]).max() < 1e-9


def test_runs() -> None:
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert A.runs(m) == [(1, 3), (4, 7), (8, 9)]
    assert A.runs(m, min_len=3) == [(4, 7)]
    assert A.runs(np.zeros(4, bool)) == []


def test_lag_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(1)
    x = (
        np.convolve(rng.standard_normal(600), np.ones(15) / 15, mode="same") * 400
    )  # smooth, deg/s scale
    y = np.concatenate([np.full(4, x[0]), x[:-4]])
    lag, corr = A.estimate_lag(x, y, 15)
    assert lag == pytest.approx(4, abs=0.25) and corr > 0.95


def test_lag_is_nan_without_motion() -> None:
    lag, corr = A.estimate_lag(np.zeros(100), np.zeros(100), 10)
    assert math.isnan(lag) and corr == 0.0


def test_gripper_events() -> None:
    g = np.concatenate(
        [np.full(30, 1.0), np.full(30, 40.0), np.full(30, 3.0), np.full(30, 40.0), np.full(30, 2.0)]
    )
    ev, closed = A.gripper_events(g, FPS)
    assert [e["kind"] for e in ev] == ["release", "grasp", "release", "grasp"]
    assert closed == [
        (60, 90),
        (120, 150),
    ]  # the closed-at-rest stretch at the start is not a grasp


def test_gripper_that_never_moves_has_no_events() -> None:
    assert A.gripper_events(np.full(100, 30.0), FPS) == ([], [])


def test_sparc_prefers_smooth() -> None:
    t = np.linspace(0, 1, 120)
    smooth = np.sin(np.pi * t)
    shaky = smooth * (
        1 + 0.3 * np.sin(2 * np.pi * 4 * t)
    )  # a 4 Hz wobble: inside SPARC's 10 Hz band
    assert A.sparc(smooth, 120) > A.sparc(shaky, 120)
    assert math.isnan(A.sparc(np.zeros(50), 30))


def test_clean_episode_has_one_grasp_and_its_lag() -> None:
    act, st = episode_signals(240, FPS, seed=0, dims=6, idle_s=0.3)
    a = A.analyse_episode(frames(act, st), FPS, SINGLE, ["pick up the cube"])
    m = a["metrics"]
    assert m["grasps"] == 1
    assert m["lag_ms"] == pytest.approx(100, abs=10)  # the fixture's follower trails by 3 frames
    assert not [f for f in a["flags"] if f["severity"] in ("warn", "error")]
    assert a["series"]["tcp"][""] and len(a["series"]["vel"]) == 240


def test_idle_start_is_flagged() -> None:
    act, st = episode_signals(300, FPS, seed=0, dims=6, idle_s=3.0)
    a = A.analyse_episode(frames(act, st), FPS, SINGLE)
    kinds = {f["kind"]: f for f in a["flags"]}
    # Motion starts at 3.0 s with a cosine ease-in, so the fastest joint passes IDLE_SPEED (3
    # deg/s) at
    # 3.0 + 6.5 * asin(3 / 14.5) / pi = 3.43 s: that, not 3.0, is when the arm is first seen
    # to move.
    assert "idle_start" in kinds and kinds["idle_start"]["value"] == pytest.approx(3.43, abs=0.12)


def test_a_one_frame_jump_is_an_error() -> None:
    act, st = episode_signals(200, FPS, seed=0, dims=6, idle_s=0.2)
    st[100, 2] += 40  # 40 degrees in 1/30 s: 1200 deg/s, more than any STS3215 can turn
    a = A.analyse_episode(frames(act, st), FPS, SINGLE)
    jump = [f for f in a["flags"] if f["kind"] == "jump"]
    assert jump and jump[0]["joint"] == "elbow_flex" and jump[0]["severity"] == "error"


def test_blocked_follower_reads_as_held() -> None:
    n = 200
    act = np.repeat(REST[None, :], n, axis=0)
    act[:, 1] = REST[1] + np.clip(
        (np.arange(n) - 40) * 0.5, 0, 40
    )  # leader lifts 40 degrees, slowly
    st = act.copy()
    st[60:, 1] = st[60, 1]  # the follower stops at 10 degrees: something holds it
    a = A.analyse_episode(frames(act, st), FPS, SINGLE)
    t = [f for f in a["flags"] if f["kind"] == "tracking"]
    assert t and t[0]["text"].startswith("Follower held") and t[0]["joint"] == "shoulder_lift"


def test_leader_faster_than_follower_reads_as_outran() -> None:
    n = 120
    act = np.repeat(REST[None, :], n, axis=0)
    act[30:, 4] = REST[4] - 80  # wrist roll snapped 80 degrees in one frame
    st = act.copy()
    for i in range(1, n):  # the follower turns at most 4 degrees a frame (120 deg/s)
        st[i, 4] = st[i - 1, 4] + np.clip(act[i, 4] - st[i - 1, 4], -4, 4)
    a = A.analyse_episode(frames(act, st), FPS, SINGLE)
    t = [f for f in a["flags"] if f["kind"] == "tracking"]
    assert t and t[0]["text"].startswith("Leader outran") and t[0]["joint"] == "wrist_roll"


def test_squeeze_on_an_object() -> None:
    act, st = episode_signals(300, FPS, seed=0, dims=6, idle_s=0.3)
    closed = act[:, 5] < 5
    st[closed, 5] = 18.0  # the object stops the jaw at 18 while the leader asks for 2
    a = A.analyse_episode(frames(act, st), FPS, SINGLE)
    assert any(f["kind"] == "squeeze" for f in a["flags"])


def test_a_static_episode_is_json_safe() -> None:
    n = 60
    still = np.repeat(REST[None, :], n, axis=0)
    a = A.analyse_episode(frames(still, still), FPS, SINGLE, ["pick it up"])
    kinds = {f["kind"] for f in a["flags"]}
    assert {"no_motion", "no_grasp", "short"} <= kinds
    assert a["metrics"]["sparc"] is None and a["arms"][""]["sparc"] is None
    json.dumps(a, allow_nan=False)


def test_video_shorter_than_data() -> None:
    act, st = episode_signals(90, FPS, seed=0, dims=6)
    a = A.analyse_episode(
        frames(act, st), FPS, SINGLE, videos={"observation.images.top": {"from": 0.0, "to": 2.5}}
    )
    assert any(f["kind"] == "video" for f in a["flags"])


def test_dataset_pass(tmp_path) -> None:
    lengths = [120, 125, 130, 128, 122, 126, 124, 500]  # the last is far longer than the rest
    ds = Dataset(write_v3(tmp_path / "u" / "d", n_eps=len(lengths), lengths=lengths))
    r = A.analyse_dataset(ds)
    assert [e["index"] for e in r["episodes"]] == list(range(len(lengths)))
    assert any(f["kind"] == "outlier" for f in r["episodes"][-1]["flags"])
    assert not any(f["kind"] == "outlier" for e in r["episodes"][:-1] for f in e["flags"])
    assert set(r["distributions"]) == set(A.JOINTS)
    assert r["workspace"][""]["points"] and len(r["rest"][""]) == 6
    assert r["episodes"][0]["grasps_t"]
    # the fixture writes no mp4 files: the health check says so
    assert any(h["kind"] == "video_missing" for h in r["health"])
    json.dumps(r, allow_nan=False)


def test_bimanual_dataset_pass(tmp_path) -> None:
    ds = Dataset(write_v3(tmp_path / "u" / "bi", n_eps=3, bimanual=True))
    r = A.analyse_dataset(ds)
    assert set(r["workspace"]) == {"left", "right"}
    assert r["episodes"][0]["metrics"]["grasps"] == 2  # one per arm


def test_empty_dataset(tmp_path) -> None:
    ds = Dataset(write_v3(tmp_path / "u" / "e", n_eps=0, lengths=[]))
    r = A.analyse_dataset(ds)
    assert r["episodes"] == [] and r["health"][0]["kind"] == "empty"
