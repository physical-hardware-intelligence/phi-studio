"""Camera align on synthetic pictures and a fake dataset tree: no camera, no real dataset."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from phi_studio import align

# WHY a mark, not a module-level importorskip: matching needs only numpy and Pillow and runs in CI;
# camera_realign imports LeRobot and OpenCV when it loads (through camera_backend).
needs_realign = pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None or importlib.util.find_spec("cv2") is None,
    reason="camera_realign needs LeRobot and OpenCV",
)


def _scene(seed: int, w: int = 640, h: int = 480) -> np.ndarray:
    """A made-up rig photo: shapes with edges on a lit background, uint8 RGB."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(60, 180, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3))
    img = Image.fromarray(ramp.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(14):
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        x1, y1 = x0 + int(rng.integers(30, 200)), y0 + int(rng.integers(30, 160))
        fill = tuple(int(c) for c in rng.integers(0, 256, 3))
        (draw.rectangle if rng.random() < 0.5 else draw.ellipse)((x0, y0, x1, y1), fill=fill)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(1.5)))


def _relit(img: np.ndarray, seed: int, gain: float = 0.7, bias: float = 40.0) -> np.ndarray:
    """Another day's light: contrast and brightness change plus sensor noise."""
    noise = np.random.default_rng(seed).normal(0, 8, img.shape)
    return np.clip(gain * img.astype(np.float32) + bias + noise, 0, 255).astype(np.uint8)


# -- measure: offset, sign and hint ---------------------------------------------------------------


@needs_realign
def test_the_hint_names_the_shift_that_undoes_the_offset() -> None:
    ref = _scene(1)
    live = np.roll(ref, (2, 4), axis=(0, 1))  # the live scene sits 4 px right and 2 px down
    m = align.measure(live, ref)
    assert m["dx"] == pytest.approx(4, abs=0.5) and m["dy"] == pytest.approx(2, abs=0.5)
    assert m["hint"] == "Move the camera so the picture shifts 4 px left and 2 px up"
    assert not m["aligned"] and not m["low_match"]


@needs_realign
@pytest.mark.parametrize("dx, dy", [(4, 2), (-6, 3), (0, -5), (9, -7), (-3, 0)])
def test_offset_sign_and_size_for_a_known_shift(dx: int, dy: int) -> None:
    ref = _scene(2)
    live = np.roll(ref, (dy, dx), axis=(0, 1))
    m = align.measure(live, ref)
    assert m["dx"] == pytest.approx(dx, abs=0.5) and m["dy"] == pytest.approx(dy, abs=0.5)
    words = {"left": dx > 0, "right": dx < 0, "up": dy > 0, "down": dy < 0}
    assert all((f"px {w}" in m["hint"]) == want for w, want in words.items()), m["hint"]
    # The proof the sign is right: shifting the picture the way the hint says lands it on the
    # reference.
    back = np.roll(live, (-round(m["dy"]), -round(m["dx"])), axis=(0, 1))
    assert align.measure(back, ref)["aligned"]


@needs_realign
def test_a_camera_that_panned_reads_as_the_scene_moving_the_other_way() -> None:
    # A wider view; the camera's window onto it moves 8 px left and 5 px up (no wrap-around).
    world = _scene(3, w=720, h=560)
    ref = world[40:520, 40:680]
    live = world[35:515, 32:672]
    m = align.measure(live, ref)
    assert m["dx"] == pytest.approx(8, abs=0.5) and m["dy"] == pytest.approx(5, abs=0.5)
    assert m["hint"] == "Move the camera so the picture shifts 8 px left and 5 px up"


@needs_realign
def test_sub_pixel_shift_by_warp() -> None:
    from phi_studio.camera_backend import cv2

    ref = _scene(4)
    live = cv2.warpAffine(ref, np.float32([[1, 0, -2.5], [0, 1, 6.5]]), (640, 480),
                          borderMode=cv2.BORDER_REFLECT)  # fmt: skip
    m = align.measure(live, ref)
    # The warp moved the scene 2.5 px left and 6.5 px down, so the fix shifts it right and up.
    assert m["dx"] == pytest.approx(-2.5, abs=0.3) and m["dy"] == pytest.approx(6.5, abs=0.3)
    assert "px right" in m["hint"] and "px up" in m["hint"]


@needs_realign
def test_offsets_are_in_live_pixels_when_sizes_differ() -> None:
    from phi_studio.camera_backend import cv2

    ref = _scene(5)
    live = cv2.resize(np.roll(ref, (0, 5), axis=(0, 1)), (1280, 960))
    m = align.measure(live, ref)
    assert m["dx"] == pytest.approx(10, abs=1.0) and m["size"] == [1280, 960]
    assert "resized from 640x480 to 1280x960" in m["note"] and "aspect" not in m["note"]
    wide = align.measure(cv2.resize(ref, (1280, 720)), ref)
    assert "aspect ratios differ" in wide["note"]


@needs_realign
def test_aligned_is_within_aligned_px_on_both_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    from phi_studio import camera_realign as cr

    ref = _scene(6)
    for (dx, dy), want in [((2.0, -2.0), True), ((2.01, 0.0), False), ((0.0, -2.2), False)]:
        monkeypatch.setattr(cr, "offset", lambda *_a, dx=dx, dy=dy: (dx, dy, 0.5))
        m = align.measure(ref, ref)
        assert m["aligned"] is want, (dx, dy)
        assert m["hint"].startswith("Aligned") is want
    monkeypatch.undo()
    assert align.measure(np.roll(ref, (1, -1), axis=(0, 1)), ref)["aligned"]
    assert not align.measure(np.roll(ref, (0, 3), axis=(0, 1)), ref)["aligned"]


@needs_realign
def test_unrelated_pictures_are_a_low_match_not_an_offset() -> None:
    m = align.measure(_scene(7), _scene(8))
    assert m["low_match"] and not m["aligned"] and m["response"] < align.LOW_RESPONSE
    assert m["hint"].startswith("Low match: wrong camera, the scene changed")
    same = align.measure(_scene(7), _scene(7))
    assert same["response"] == 1.0 and same["aligned"]  # OpenCV's >1 is clipped


# -- match_cameras --------------------------------------------------------------------------------


def test_match_recovers_a_shuffled_assignment_under_new_light_and_noise() -> None:
    keys = ["observation.images.wrist", "observation.images.front", "observation.images.top",
            "observation.images.side"]  # fmt: skip
    refs = {k: _scene(10 + i) for i, k in enumerate(keys)}
    # Device indices as macOS hands them out today: a different order from the dataset's.
    today = {0: keys[2], 1: keys[0], 2: keys[3], 3: keys[1]}
    live = {idx: _relit(np.roll(refs[k], (6, -9), axis=(0, 1)), idx) for idx, k in today.items()}
    r = align.match_cameras(live, refs)
    assert r["assignment"] == today
    assert not r["unsure"] and r["why"] is None and r["margin"] > align.UNSURE_MARGIN
    assert set(r["scores"]) == set(today) and all(set(v) == set(keys) for v in r["scores"].values())
    assert not any(p["weak"] for p in r["pairs"])
    assert r["unmatched_live"] == [] and r["unmatched_refs"] == []


def test_near_duplicate_pictures_are_unsure() -> None:
    a = _scene(20)
    refs = {"observation.images.front": a, "observation.images.top": _relit(a, 1, 1.0, 0.0)}
    live = {0: _relit(a, 2), 1: _relit(a, 3)}
    r = align.match_cameras(live, refs)
    assert r["unsure"] and r["margin"] < align.UNSURE_MARGIN
    assert r["why"].startswith("Unsure about 0, 1:")


def test_a_camera_that_matches_nothing_is_placed_by_elimination() -> None:
    # The real wrist camera: at rest it sees whatever was left by the gripper, so it rarely looks
    # like its reference. The two others still pin it down.
    refs = {"wrist": _scene(30), "front": _scene(31), "top": _scene(32)}
    live = {"cam0": _relit(refs["top"], 0), "cam1": _scene(99), "cam2": _relit(refs["front"], 2)}
    r = align.match_cameras(live, refs)
    assert r["assignment"] == {"cam0": "top", "cam1": "wrist", "cam2": "front"}
    weak = {p["live"]: p["weak"] for p in r["pairs"]}
    assert weak == {"cam0": False, "cam1": True, "cam2": False}
    assert not r["unsure"]


def test_nothing_matching_is_unsure_and_extra_cameras_are_unmatched() -> None:
    r = align.match_cameras({5: _scene(40)}, {"front": _scene(41)})
    assert r["margin"] is None and r["unsure"] and "No camera looks like" in r["why"]

    refs = {"front": _scene(50), "top": _scene(51)}
    live = {0: _relit(refs["top"], 0), 1: _scene(52), 2: _relit(refs["front"], 2)}
    r = align.match_cameras(live, refs)
    assert r["assignment"] == {0: "top", 2: "front"} and r["unmatched_live"] == [1]
    r = align.match_cameras({0: _relit(refs["top"], 0)}, refs)
    assert r["assignment"] == {0: "top"} and r["unmatched_refs"] == ["front"]


def test_match_refuses_empty_input() -> None:
    with pytest.raises(align.AlignError):
        align.match_cameras({}, {"front": _scene(1)})


# -- datasets and references ----------------------------------------------------------------------


def _dataset(root: Path, cameras: list[str], episodes: int = 3) -> Path:
    features = {k: {"dtype": "video", "shape": [480, 640, 3]} for k in cameras}
    features["observation.state"] = {"dtype": "float32", "shape": [6]}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(
        {"total_episodes": episodes, "total_frames": 100 * episodes, "features": features}))
    return root


THREE = ["observation.images.wrist", "observation.images.front", "observation.images.top"]


@needs_realign
def test_list_datasets_reads_every_layout_and_flags_transposed_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    _dataset(tmp_path / "Parv-09" / "cubes_20260801", THREE[:2])  # recorded locally
    _dataset(tmp_path / "hub" / "datasets--BrutalCaesar--phi_so101_8bin_v1" / "snapshots" / "abc",
             THREE)  # pulled from the Hub  # fmt: skip
    _dataset(tmp_path / "stray", ["observation.images.wrist_cam"])  # no org directory
    _dataset(tmp_path / "Parv-09" / "aborted", THREE, episodes=0)  # empty recording: left out

    got = {d["repo_id"]: d for d in align.list_datasets()}
    assert set(got) == {"Parv-09/cubes_20260801", "BrutalCaesar/phi_so101_8bin_v1",
                        "unknown/stray"}  # fmt: skip
    cubes = got["Parv-09/cubes_20260801"]
    assert cubes["episodes"] == 3 and cubes["frames"] == 300 and cubes["cameras"] == THREE[:2]
    assert cubes["note"] is None and cubes["root"] == str(tmp_path / "Parv-09" / "cubes_20260801")
    assert cubes["physical"] == {THREE[0]: "wrist", THREE[1]: "front"}
    eightbin = got["BrutalCaesar/phi_so101_8bin_v1"]
    assert eightbin["physical"]["observation.images.top"] == "wrist"
    assert "transposed" in eightbin["note"]
    assert "observation.images.top shows the wrist camera" in eightbin["note"]
    assert got["unknown/stray"]["physical"] == {} and got["unknown/stray"]["note"] is None


@needs_realign
def test_references_decode_once_then_come_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phi_studio import camera_realign as cr

    home, cache = tmp_path / "home", tmp_path / "cache"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    keys = ["observation.images.wrist_cam", "observation.images.front"]
    root = _dataset(home / "Parv-09" / "pens", keys, episodes=5)
    calls: list[tuple] = []

    def decode(ds, names, episode, frame):  # stands in for LeRobotDataset + the video decode
        calls.append((names, episode, frame, ds.key_map[0]))
        return {n: _scene(60 + i) for i, n in enumerate(names)}

    monkeypatch.setattr(cr, "resting_frame", lambda ds, episode, search: (12, 0.003))
    monkeypatch.setattr(cr, "reference_frames", decode)

    first = align.references(str(root), episode=2, cache_dir=cache)
    again = align.references(str(root), episode=2, cache_dir=cache)
    assert len(calls) == 1 and not first["cached"] and again["cached"]
    # Decoded by dataset key, each key standing for itself (wrist_cam is no physical name).
    assert calls[0] == (keys, 2, 12, {k: k for k in keys})
    assert again["frame"] == 12 and again["motion"] == 0.003 and again["episode"] == 2
    assert again["physical"] == {"observation.images.front": "front"}
    for k in keys:
        assert again["images"][k].dtype == np.uint8 and again["images"][k].shape == (480, 640, 3)
        assert np.array_equal(again["images"][k], first["images"][k])  # PNG is lossless
        assert Path(again["paths"][k]).is_file() and again["paths"][k] == first["paths"][k]
    meta = json.loads((Path(first["paths"][keys[0]]).parent / "reference.json").read_text())
    assert meta["root"] == str(root) and meta["frame"] == 12

    align.references(str(root), episode=3, cache_dir=cache)  # another episode: its own entry
    assert len(calls) == 2
    assert align.references(str(root), episode=2)["cached"] is False  # no cache_dir, no cache
    assert len(calls) == 3


@needs_realign
def test_reference_problems_are_readable_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phi_studio import camera_realign as cr

    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    root = _dataset(tmp_path / "Parv-09" / "pens", THREE, episodes=2)
    with pytest.raises(align.AlignError, match="No LeRobot dataset"):
        align.references(str(tmp_path / "nowhere"))
    with pytest.raises(align.AlignError, match="episodes 0-1"):
        align.references(str(root), episode=2)

    def no_parquet(ds, episode, search):
        raise SystemExit(f"no parquet under {ds.root}/data")

    monkeypatch.setattr(cr, "resting_frame", no_parquet)
    with pytest.raises(align.AlignError, match="no parquet under"):
        align.references(str(root))
