"""Write small LeRobot datasets to disk for tests: v3.0 (what LeRobot 0.6.0 writes) and v2.1, with
the same
columns and file layout, so the reader is tested against the format and not against itself."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REST = np.array([-5.1, -103.9, 96.7, 54.2, -0.8, 0.7])


def names(bimanual: bool) -> list[str]:
    if bimanual:
        return [f"{s}_{j}.pos" for s in ("left", "right") for j in JOINTS]
    return [f"{j}.pos" for j in JOINTS]


def episode_signals(
    n: int, fps: float, seed: int, dims: int, idle_s: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """A plausible teleop episode: still at rest for idle_s, then a smooth reach and one grasp. The
    follower
    (state) trails the leader (action) by 3 frames, as a real follower lags."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    base = np.tile(REST, dims // 6)
    act = np.repeat(base[None, :], n, axis=0).astype(np.float64)
    move = np.clip((t - idle_s) / max(1e-6, (n / fps) - idle_s - 0.5), 0, 1)
    s = 0.5 - 0.5 * np.cos(np.pi * move)  # 0 -> 1, smooth
    for a in range(dims // 6):
        o = a * 6
        act[:, o + 0] += 25 * np.sin(2 * np.pi * 0.5 * s) * (1 + 0.1 * rng.standard_normal())
        act[:, o + 1] += 60 * s
        act[:, o + 2] -= 50 * s
        act[:, o + 3] -= 30 * s
        g = np.where((s > 0.4) & (s < 0.8), 2.0, 30.0)  # open, closed on the object, open again
        g[s < 0.05] = 0.7  # closed at rest
        # a real jaw takes ~0.3 s to open or close (recorded p99 ~75 %/s), never one frame
        k = max(1, int(0.3 * fps))
        act[:, o + 5] = np.convolve(np.pad(g, (k, k), mode="edge"), np.ones(k) / k, mode="same")[
            k:-k
        ]
    lag = 3
    state = np.vstack([np.repeat(act[:1], lag, axis=0), act[:-lag]])
    state = np.round(state / (360 / 4095)) * (360 / 4095)  # encoder steps
    return act, state


def write_v3(
    root: Path,
    n_eps: int = 4,
    fps: int = 30,
    bimanual: bool = False,
    cameras: tuple[str, ...] = ("top",),
    lengths: list[int] | None = None,
    tasks: tuple[str, ...] = ("pick up the cube",),
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dims = 12 if bimanual else 6
    lengths = lengths or [90 + 15 * i for i in range(n_eps)]
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    feats = {
        "action": {"dtype": "float32", "shape": [dims], "names": names(bimanual)},
        "observation.state": {"dtype": "float32", "shape": [dims], "names": names(bimanual)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for c in cameras:
        feats[f"observation.images.{c}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.codec": "av1"},
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "bi_so_follower" if bimanual else "so_follower",
        "fps": fps,
        "total_episodes": n_eps,
        "total_frames": int(sum(lengths)),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": feats,
        "splits": {"train": f"0:{n_eps}"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    pq.write_table(
        pa.table({"task_index": list(range(len(tasks)))}, metadata=None).append_column(
            "task", pa.array(list(tasks))
        ),
        root / "meta" / "tasks.parquet",
    )

    rows = {
        k: []
        for k in (
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        )
    }
    eps = {
        k: []
        for k in (
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        )
    }
    for c in cameras:
        for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            eps[f"videos/observation.images.{c}/{k}"] = []
    g, vt = 0, 0.0
    for e, n in enumerate(lengths):
        act, st = episode_signals(n, fps, seed=e, dims=dims)
        rows["action"] += act.astype(np.float32).tolist()
        rows["observation.state"] += st.astype(np.float32).tolist()
        rows["timestamp"] += (np.arange(n) / fps).astype(np.float32).tolist()
        rows["frame_index"] += list(range(n))
        rows["episode_index"] += [e] * n
        rows["index"] += list(range(g, g + n))
        rows["task_index"] += [e % len(tasks)] * n
        eps["episode_index"].append(e)
        eps["tasks"].append([tasks[e % len(tasks)]])
        eps["length"].append(n)
        eps["data/chunk_index"].append(0)
        eps["data/file_index"].append(0)
        eps["dataset_from_index"].append(g)
        eps["dataset_to_index"].append(g + n)
        eps["meta/episodes/chunk_index"].append(0)
        eps["meta/episodes/file_index"].append(0)
        for c in cameras:
            p = f"videos/observation.images.{c}/"
            eps[p + "chunk_index"].append(0)
            eps[p + "file_index"].append(0)
            eps[p + "from_timestamp"].append(vt)
            eps[p + "to_timestamp"].append(vt + n / fps)
        g += n
        vt += n / fps
    t = pa.table(
        {
            "action": pa.array(rows["action"], type=pa.list_(pa.float32(), dims)),
            "observation.state": pa.array(
                rows["observation.state"], type=pa.list_(pa.float32(), dims)
            ),
            "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
            **{
                k: pa.array(rows[k], type=pa.int64())
                for k in ("frame_index", "episode_index", "index", "task_index")
            },
        }
    )
    pq.write_table(t, root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table(eps), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def write_v21(root: Path, n_eps: int = 2, fps: int = 30) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so100",
        "fps": fps,
        "total_episodes": n_eps,
        "total_frames": 60 * n_eps,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": names(False)},
            "observation.state": {"dtype": "float32", "shape": [6], "names": names(False)},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "wave"}) + "\n")
    lines = []
    for e in range(n_eps):
        act, st = episode_signals(60, fps, seed=e, dims=6)
        pq.write_table(
            pa.table(
                {
                    "action": pa.array(
                        act.astype(np.float32).tolist(), type=pa.list_(pa.float32(), 6)
                    ),
                    "observation.state": pa.array(
                        st.astype(np.float32).tolist(), type=pa.list_(pa.float32(), 6)
                    ),
                    "timestamp": pa.array((np.arange(60) / fps).astype(np.float32)),
                    "frame_index": pa.array(np.arange(60)),
                    "episode_index": pa.array([e] * 60),
                }
            ),
            root / "data" / "chunk-000" / f"episode_{e:06d}.parquet",
        )
        lines.append(json.dumps({"episode_index": e, "tasks": ["wave"], "length": 60}))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    return root
