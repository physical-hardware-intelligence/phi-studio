"""Which camera is which after a restart (camcheck): fingerprints, matching, and the check that
rewrites only the numbers macOS moved. 2026-10-09 on the rig: the same camera scored 0.81-1.00
across probes, two different cameras at most 0.09."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from phi_studio import camcheck

CONFIG = """\
studio:
  name: "Phi Left Pair"
robot:
  type: so101_follower
  id: "phi_follower"
  port: "/dev/tty.usbmodemF1"
  cameras:
    wrist:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
    front:
      type: opencv
      index_or_path: 1
      width: 640
      height: 480
      fps: 30
    top:
      type: opencv
      index_or_path: 2
      width: 640
      height: 480
      fps: 30
teleop:
  type: so101_leader
  id: "phi_leader"
  port: "/dev/tty.usbmodemL1"
"""


def scene(seed: int, noise: float = 0.0) -> bytes:
    """A camera's view: shapes placed by the seed (a desk, an arm, a wall); noise stands in for a
    later probe of the same camera."""
    from PIL import ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (160, 120), tuple(int(v) for v in rng.integers(40, 200, 3)))
    draw = ImageDraw.Draw(img)
    for _ in range(9):
        x, y = int(rng.integers(0, 140)), int(rng.integers(0, 100))
        w, h = int(rng.integers(10, 60)), int(rng.integers(10, 50))
        box = (x, y, x + w, y + h)
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)(box, fill=colour)
    arr = np.asarray(img, dtype=np.float32)
    if noise:
        arr = arr + np.random.default_rng(seed + 1000).normal(0, noise, arr.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def refs(tmp: Path) -> list:
    # wrist 0, front 1, top 2, and the Mac's own camera 3 with no role
    camcheck.save_refs(tmp, {0: scene(1), 1: scene(2), 2: scene(3), 3: scene(4)},
                       {"wrist": 0, "front": 1, "top": 2})  # fmt: skip
    loaded = camcheck.load_refs(tmp)
    assert loaded is not None
    return loaded


def live(views: dict[int, bytes]) -> dict:
    return {s: camcheck.fingerprint(j) for s, j in views.items()}


def test_the_same_view_matches_itself_and_not_another() -> None:
    a, b = camcheck.fingerprint(scene(1)), camcheck.fingerprint(scene(1, noise=12))
    assert float(np.dot(a, b)) > 0.8 and abs(float(np.dot(a, camcheck.fingerprint(scene(2))))) < 0.3


def test_renumbered_cameras_are_found_by_what_they_see(tmp_path: Path) -> None:
    res = camcheck.resolve(refs(tmp_path), live({0: scene(3, 10), 1: scene(1, 10), 2: scene(4, 10),
                                                 3: scene(2, 10)}))  # fmt: skip
    assert res["assign"] == {"top": 0, "wrist": 1, "front": 3} and not res["missing"]
    assert all(d["how"] == "picture" for d in res["detail"].values())


def test_one_changed_view_is_placed_by_elimination_only_if_unambiguous(tmp_path: Path) -> None:
    # The wrist moved with the arm: its view is new. The Mac's own camera is recognised, so one
    # camera is left for the one role left.
    r = refs(tmp_path)
    res = camcheck.resolve(r, live({0: scene(99), 1: scene(2, 8), 2: scene(3, 8), 3: scene(4, 8)}))
    assert res["assign"]["wrist"] == 0 and res["detail"]["wrist"]["how"] == "elimination"
    # The Mac's camera changed too: two cameras left for one role, so no guess.
    res = camcheck.resolve(r, live({0: scene(99), 1: scene(2, 8), 2: scene(3, 8), 3: scene(77)}))
    assert res["missing"] == ["wrist"] and "wrist" not in res["assign"]


def fake_probe(views: dict[int, bytes]):
    def probe(indices):
        return [{"source": i, "ok": i in views, "thumbnail": views.get(i),
                 "error": None if i in views else "Failed to open OpenCVCamera"} for i in indices]

    return probe


def test_check_rewrites_only_the_moved_numbers_and_backs_up(tmp_path: Path) -> None:
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(CONFIG)
    refs(tmp_path)
    moved = {0: scene(2, 10), 1: scene(1, 10), 2: scene(3, 10), 3: scene(4, 10)}  # wrist<->front
    res = camcheck.check(cfg, tmp_path, probe=fake_probe(moved))
    assert res["status"] == "renumbered" and res["moves"] == {"wrist": [0, 1], "front": [1, 0]}
    text = cfg.read_text()
    assert "wrist:\n      type: opencv\n      index_or_path: 1" in text
    assert "front:\n      type: opencv\n      index_or_path: 0" in text
    assert "top:\n      type: opencv\n      index_or_path: 2" in text  # untouched
    assert Path(res["backup"]).read_text() == CONFIG
    assert camcheck.check(cfg, tmp_path, probe=fake_probe(moved))["status"] == "same"


def test_check_changes_nothing_when_unsure_or_never_shown(tmp_path: Path) -> None:
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(CONFIG)
    assert camcheck.check(cfg, tmp_path, probe=fake_probe({}))["status"] == "unverified"
    refs(tmp_path)
    blind = {0: scene(90), 1: scene(91), 2: scene(3, 8), 3: scene(4, 8)}  # wrist and front covered
    res = camcheck.check(cfg, tmp_path, probe=fake_probe(blind))
    assert res["status"] == "unsure" and "wrist" in res["message"] and "front" in res["message"]
    assert cfg.read_text() == CONFIG


def test_a_camera_that_fails_its_first_read_is_asked_again(tmp_path: Path, monkeypatch) -> None:
    """2026-10-09: the two cameras swapped a minute before failed their first read, and the check
    could not see them."""
    monkeypatch.setattr(camcheck, "RETRY_S", 0.0)
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(CONFIG)
    refs(tmp_path)
    views = {0: scene(3, 10), 1: scene(2, 10), 2: scene(1, 10), 3: scene(4, 10)}  # wrist, top
    calls: list[list[int]] = []

    def probe(indices):
        idx = list(indices)
        calls.append(idx)
        shy = len(calls) == 1 and {0, 2} or set()

        def row(i: int) -> dict:
            err = "read failed" if i in shy else None if i in views else "Failed to open camera"
            return {"source": i, "ok": err is None, "thumbnail": views.get(i), "error": err}

        return [row(i) for i in idx]

    res = camcheck.check(cfg, tmp_path, probe=probe)
    assert calls[1] == [0, 2]  # only the cameras that were there and failed
    assert res["status"] == "renumbered" and res["moves"] == {"wrist": [0, 2], "top": [2, 0]}


def test_a_sure_check_refreshes_the_references(tmp_path: Path) -> None:
    cfg = tmp_path / "robot-config.yaml"
    cfg.write_text(CONFIG)
    refs(tmp_path)
    later = {0: scene(1, 25), 1: scene(2, 25), 2: scene(3, 25), 3: scene(4, 25)}
    assert camcheck.check(cfg, tmp_path, probe=fake_probe(later))["status"] == "same"
    fresh = {r["source"]: r for r in camcheck.load_refs(tmp_path) or []}
    assert fresh[0]["role"] == "wrist" and fresh[3]["role"] is None
    assert float(np.dot(fresh[0]["fp"], camcheck.fingerprint(later[0]))) > 0.99
