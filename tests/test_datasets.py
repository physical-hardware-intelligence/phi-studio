"""Reading LeRobot datasets from disk with pyarrow: discovery, metadata, frames, both formats."""

from __future__ import annotations

import json

import numpy as np
import pytest
from support.lerobot_data import write_v3, write_v21

from phi_studio.datasets import Dataset, Library, dataset_id, find_datasets, split_arms

pytest.importorskip("pyarrow")


def test_discovery_finds_nested_and_skips_noise(tmp_path) -> None:
    write_v3(tmp_path / "alice" / "cubes", n_eps=2)
    write_v3(tmp_path / "bob" / "shelf" / "deep", n_eps=1)
    (tmp_path / "calibration" / "robots").mkdir(
        parents=True
    )  # LeRobot keeps calibration here: not a dataset
    write_v3(tmp_path / "calibration" / "fake", n_eps=1)
    (tmp_path / "._junk").mkdir()
    found = sorted(
        p.relative_to(tmp_path).as_posix() for p in find_datasets([tmp_path, tmp_path / "missing"])
    )
    assert found == ["alice/cubes", "bob/shelf/deep"]


def test_v3_metadata_and_frames(tmp_path) -> None:
    ds = Dataset(
        write_v3(tmp_path / "u" / "d", n_eps=3, cameras=("top", "wrist"), tasks=("a", "b"))
    )
    assert (
        ds.v3
        and ds.fps == 30
        and ds.cameras == ["observation.images.top", "observation.images.wrist"]
    )
    eps = ds.episodes()
    assert [e["index"] for e in eps] == [0, 1, 2] and [e["length"] for e in eps] == [90, 105, 120]
    assert eps[1]["videos"]["observation.images.top"]["from"] == pytest.approx(
        3.0
    )  # after episode 0's 3 s
    assert ds.tasks() == ["a", "b"]
    f = ds.frames(1)
    assert f["action"].shape == (105, 6) and f["state"].shape == (105, 6)
    assert np.all(np.diff(f["frame_index"]) == 1) and f["timestamp"][-1] == pytest.approx(
        104 / 30, abs=1e-6
    )
    allf = ds.all_frames()
    assert allf["action"].shape == (315, 6) and set(np.unique(allf["episode_index"])) == {0, 1, 2}


def test_summary(tmp_path) -> None:
    ds = Dataset(write_v3(tmp_path / "u" / "bi", n_eps=2, bimanual=True))
    s = ds.summary()
    assert s["bimanual"] and s["episodes"] == 2 and s["dims"]["action"] == 12
    assert [a["name"] for a in s["arms"]] == ["left", "right"]
    assert s["id"] == dataset_id(ds.root) and len(s["id"]) == 12


def test_v21(tmp_path) -> None:
    ds = Dataset(write_v21(tmp_path / "old"))
    assert not ds.v3 and ds.tasks() == ["wave"]
    assert ds.episodes()[1]["length"] == 60
    assert ds.frames(1)["state"].shape == (60, 6)


def test_unknown_episode(tmp_path) -> None:
    ds = Dataset(write_v3(tmp_path / "u" / "d", n_eps=1))
    with pytest.raises(KeyError):
        ds.frames(5)


def test_split_arms() -> None:
    single = split_arms(
        [
            f"{j}.pos"
            for j in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        ]
    )
    assert single == [
        {"name": "", "joints": single[0]["joints"], "index": [0, 1, 2, 3, 4, 5], "so101": True}
    ]
    bi = split_arms(
        [
            f"{s}_{j}.pos"
            for s in ("right", "left")
            for j in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        ]
    )
    assert [a["name"] for a in bi] == ["right", "left"] and bi[1]["index"][0] == 6
    odd = split_arms(["x", "y"])
    assert odd[0]["so101"] is False


def test_library_rescan_and_get(tmp_path) -> None:
    write_v3(tmp_path / "u" / "one", n_eps=1)
    lib = Library([tmp_path])
    assert len(lib.scan()) == 1
    write_v3(tmp_path / "u" / "two", n_eps=1)
    assert len(lib.scan(force=True)) == 2
    two = next(d for d in lib.scan() if d.root.name == "two")
    assert lib.get(two.id) is two
    assert lib.size(two) > 0
    with pytest.raises(KeyError):
        lib.get("nope")


def test_broken_info_is_skipped(tmp_path) -> None:
    write_v3(tmp_path / "u" / "good", n_eps=1)
    bad = tmp_path / "u" / "bad" / "meta"
    bad.mkdir(parents=True)
    (bad / "info.json").write_text("{not json")
    assert [d.root.name for d in Library([tmp_path]).scan()] == ["good"]


def test_names_from_v2_dict_form(tmp_path) -> None:
    root = write_v21(tmp_path / "old")
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["action"]["names"] = {"motors": info["features"]["action"]["names"]}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    assert Dataset(root).names("action")[0] == "shoulder_pan.pos"
