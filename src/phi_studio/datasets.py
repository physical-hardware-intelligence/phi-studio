"""LeRobot datasets on this Mac, read directly with pyarrow: no torch, no LeRobot import.

Formats (LeRobot 0.6.0 writes v3.0; older Hub datasets are v2.1):
  v3.0  meta/info.json, meta/episodes/chunk-*/file-*.parquet (one row per episode: length, tasks,
        the data file that holds it, and per camera the video file and its from/to timestamps),
        data/chunk-*/file-*.parquet (many episodes per file), videos/<key>/chunk-*/file-*.mp4 (many
        episodes per file). Frame i of an episode is at file time from_timestamp + frame_index / fps
        (lerobot datasets/dataset_writer.py:208, dataset_reader.py:277-281).
  v2.1  meta/episodes.jsonl, meta/tasks.jsonl, one data parquet and one video per episode,
  from_timestamp 0.

A dataset's id is a short hash of its folder, so a URL never carries a path.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SKIP_DIRS = {
    "calibration",
    ".cache",
    "hub",
    "outputs",
    "inference_logs",
    "ports",
    "robots",
    "node_modules",
}
MAX_DEPTH = 4


def lerobot_home() -> Path:
    """Where LeRobot keeps datasets (lerobot/utils/constants.py: HF_LEROBOT_HOME, else
    HF_HOME/lerobot)."""
    if os.environ.get("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return hf / "lerobot"


def default_roots(rig_dir: Path | None = None) -> list[Path]:
    roots = [lerobot_home()]
    for p in os.environ.get("PHI_STUDIO_DATA_ROOTS", "").split(os.pathsep):
        if p.strip():
            roots.append(Path(p).expanduser())
    if rig_dir is not None:
        roots.append(rig_dir / "data")
    return roots


def dataset_id(root: Path) -> str:
    return hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]


def find_datasets(roots: list[Path]) -> list[Path]:
    """Every folder under the roots that holds meta/info.json, at most MAX_DEPTH levels down."""
    out: list[Path] = []
    seen: set[Path] = set()

    def walk(d: Path, depth: int) -> None:
        if (d / "meta" / "info.json").is_file():
            r = d.resolve()
            if r not in seen:
                seen.add(r)
                out.append(d)
            return
        if depth >= MAX_DEPTH:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for c in children:
            if c.name in SKIP_DIRS or c.name.startswith((".", "._")):
                continue
            walk(c, depth + 1)

    for root in roots:
        if root.is_dir():
            walk(root, 0)
    return out


def _folder_size(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for f in filenames:
            if not f.startswith("._"):
                try:
                    total += os.stat(os.path.join(dirpath, f)).st_size
                except OSError:
                    pass
    return total


def split_arms(names: list[str]) -> list[dict[str, Any]]:
    """Group a motor list into arms: ["left_shoulder_pan.pos", ...] -> left and right, each with its
    six indices in the SO-101 joint order. A single arm has the name ""."""
    base = [n.removesuffix(".pos") for n in names]
    arms: dict[str, dict[str, int]] = {}
    for i, n in enumerate(base):
        side = ""
        for prefix in ("left_", "right_"):
            if n.startswith(prefix):
                side, n = prefix[:-1], n[len(prefix) :]
                break
        arms.setdefault(side, {})[n] = i
    order = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    out = []
    for side, joints in arms.items():
        idx = [joints.get(j) for j in order]
        out.append(
            {
                "name": side,
                "joints": list(order),
                "index": idx,
                "so101": all(i is not None for i in idx),
            }
        )
    return out


@dataclass
class Dataset:
    root: Path
    info: dict[str, Any] = field(default_factory=dict)
    _episodes: list[dict[str, Any]] | None = None
    _tasks: list[str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())

    # -- shape -----------------------------------------------------------------------------------
    @property
    def id(self) -> str:
        return dataset_id(self.root)

    @property
    def version(self) -> str:
        return str(self.info.get("codebase_version", ""))

    @property
    def v3(self) -> bool:
        return self.version.startswith("v3")

    @property
    def fps(self) -> float:
        return float(self.info.get("fps", 30))

    @property
    def features(self) -> dict[str, Any]:
        return self.info.get("features", {})

    def names(self, key: str) -> list[str]:
        f = self.features.get(key) or {}
        n = f.get("names")
        if isinstance(n, dict):  # some v2 datasets: {"motors": [...]}
            n = next(iter(n.values()), [])
        if not n:
            dim = (f.get("shape") or [0])[0]
            n = [f"{key}_{i}" for i in range(dim)]
        return [str(x) for x in n]

    @property
    def cameras(self) -> list[str]:
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]

    @property
    def repo_id(self) -> str:
        home = lerobot_home().resolve()
        r = self.root.resolve()
        try:
            return str(r.relative_to(home))
        except ValueError:
            return r.name

    def arms(self) -> list[dict[str, Any]]:
        return split_arms(
            self.names("action") if "action" in self.features else self.names("observation.state")
        )

    # -- metadata --------------------------------------------------------------------------------
    def tasks(self) -> list[str]:
        if self._tasks is None:
            if self.v3:
                import pyarrow.parquet as pq

                # WHY pyarrow alone: Studio's own install has no pandas (LeRobot brings it, a
                # clean install does not, and CI failed on it). LeRobot writes the tasks with
                # pandas, the task string as the index, which pyarrow reads back as a column:
                # "task", or whatever the file's pandas metadata names the index.
                t = pq.read_table(self.root / "meta" / "tasks.parquet")
                index = (t.schema.pandas_metadata or {}).get("index_columns", [])
                key = "task" if "task" in t.column_names else next(
                    (c for c in index if isinstance(c, str) and c in t.column_names), None
                )
                if key is None:
                    raise ValueError(f"{self.root}: tasks.parquet has no task column")
                pairs = zip(t.column("task_index").to_pylist(), t.column(key).to_pylist(),
                            strict=True)  # fmt: skip
                ordered = sorted(pairs)
                self._tasks = [str(task) for _, task in ordered]
            else:
                rows = [
                    json.loads(x)
                    for x in (self.root / "meta" / "tasks.jsonl").read_text().splitlines()
                    if x.strip()
                ]
                self._tasks = [r["task"] for r in sorted(rows, key=lambda r: r["task_index"])]
        return self._tasks

    def episodes(self) -> list[dict[str, Any]]:
        """One dict per episode: index, length, duration, tasks, where its data and video segments
        are."""
        with self._lock:
            if self._episodes is None:
                self._episodes = self._read_episodes_v3() if self.v3 else self._read_episodes_v2()
            return self._episodes

    def _read_episodes_v3(self) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        files = [f for f in files if not f.name.startswith("._")]
        if not files:
            return []
        keep = None
        out: list[dict[str, Any]] = []
        for f in files:
            schema = pq.read_schema(f)
            keep = [n for n in schema.names if not n.startswith("stats/")] + [
                n
                for n in schema.names
                if n.startswith(("stats/action/", "stats/observation.state/"))
                and n.rsplit("/", 1)[1] in ("min", "max", "mean")
            ]
            for row in pq.read_table(f, columns=keep).to_pylist():
                vids = {}
                for cam in self.cameras:
                    p = f"videos/{cam}/"
                    if p + "chunk_index" in row:
                        vids[cam] = {
                            "chunk": int(row[p + "chunk_index"]),
                            "file": int(row[p + "file_index"]),
                            "from": float(row[p + "from_timestamp"]),
                            "to": float(row[p + "to_timestamp"]),
                        }
                n = int(row["length"])
                out.append(
                    {
                        "index": int(row["episode_index"]),
                        "length": n,
                        "duration": n / self.fps,
                        "tasks": list(row.get("tasks") or []),
                        "data": {
                            "chunk": int(row["data/chunk_index"]),
                            "file": int(row["data/file_index"]),
                            "from": int(row["dataset_from_index"]),
                            "to": int(row["dataset_to_index"]),
                        },
                        "videos": vids,
                        "range": {
                            k.split("/")[1] + "." + k.rsplit("/", 1)[1]: row[k]
                            for k in row
                            if k.startswith("stats/")
                        },
                    }
                )
        out.sort(key=lambda e: int(e["index"]))
        return out

    def _read_episodes_v2(self) -> list[dict[str, Any]]:
        path = self.root / "meta" / "episodes.jsonl"
        if not path.is_file():
            return []
        chunks = int(self.info.get("chunks_size", 1000))
        out: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i, n = int(r["episode_index"]), int(r["length"])
            vids = {
                cam: {"chunk": i // chunks, "file": i, "from": 0.0, "to": n / self.fps}
                for cam in self.cameras
            }
            out.append(
                {
                    "index": i,
                    "length": n,
                    "duration": n / self.fps,
                    "tasks": list(r.get("tasks") or []),
                    "data": {"chunk": i // chunks, "file": i, "from": None, "to": None},
                    "videos": vids,
                    "range": {},
                }
            )
        out.sort(key=lambda e: int(e["index"]))
        return out

    def episode_meta(self, index: int) -> dict[str, Any]:
        for e in self.episodes():
            if e["index"] == index:
                return e
        raise KeyError(f"episode {index} is not in this dataset")

    # -- frames ----------------------------------------------------------------------------------
    def data_path(self, chunk: int, file: int) -> Path:
        tpl = self.info.get("data_path") or (
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            if self.v3
            else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        return self.root / tpl.format(
            chunk_index=chunk, file_index=file, episode_chunk=chunk, episode_index=file
        )

    def video_path(self, key: str, chunk: int, file: int) -> Path:
        tpl = self.info.get("video_path") or (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            if self.v3
            else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        return self.root / tpl.format(
            video_key=key,
            chunk_index=chunk,
            file_index=file,
            episode_chunk=chunk,
            episode_index=file,
        )

    def frames(self, index: int) -> dict[str, np.ndarray]:
        """One episode's columns: timestamp [N], frame_index [N], action [N, D], state [N, D]."""
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        e = self.episode_meta(index)
        path = self.data_path(e["data"]["chunk"], e["data"]["file"])
        cols = [
            c
            for c in ("timestamp", "frame_index", "episode_index", "action", "observation.state")
            if c in pq.read_schema(path).names
        ]
        t = pq.read_table(
            path, columns=cols, filters=[("episode_index", "==", index)] if self.v3 else None
        )
        if "frame_index" in t.column_names:
            t = t.take(pc.sort_indices(t["frame_index"]))
        out: dict[str, np.ndarray] = {}
        n = t.num_rows
        out["timestamp"] = (
            t["timestamp"].to_numpy().astype(np.float64)
            if "timestamp" in t.column_names
            else np.arange(n) / self.fps
        )
        out["frame_index"] = (
            t["frame_index"].to_numpy() if "frame_index" in t.column_names else np.arange(n)
        )
        for col, key in (("action", "action"), ("observation.state", "state")):
            if col in t.column_names:
                arr = t[col].combine_chunks()
                out[key] = arr.flatten().to_numpy().astype(np.float64).reshape(n, -1)
        return out

    def all_frames(self) -> dict[str, np.ndarray]:
        """Every frame of the dataset (episode_index, frame_index, action, state), for dataset-wide
        stats. v3 reads each data file once."""
        import pyarrow.parquet as pq

        paths: list[Path] = []
        if self.v3:
            seen = set()
            for e in self.episodes():
                key = (e["data"]["chunk"], e["data"]["file"])
                if key not in seen:
                    seen.add(key)
                    paths.append(self.data_path(*key))
        else:
            paths = [self.data_path(e["data"]["chunk"], e["data"]["file"]) for e in self.episodes()]
        parts: dict[str, list[np.ndarray]] = {
            "episode_index": [],
            "frame_index": [],
            "action": [],
            "state": [],
        }
        for p in paths:
            names = pq.read_schema(p).names
            cols = [
                c
                for c in ("episode_index", "frame_index", "action", "observation.state")
                if c in names
            ]
            t = pq.read_table(p, columns=cols)
            n = t.num_rows
            parts["episode_index"].append(t["episode_index"].to_numpy())
            parts["frame_index"].append(t["frame_index"].to_numpy())
            for col, part in (("action", "action"), ("observation.state", "state")):
                if col in names:
                    parts[part].append(
                        t[col]
                        .combine_chunks()
                        .flatten()
                        .to_numpy()
                        .astype(np.float64)
                        .reshape(n, -1)
                    )
        return {k: np.concatenate(v) for k, v in parts.items() if v}

    # -- summary ---------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        info = self.info
        eps = int(info.get("total_episodes", 0))
        frames = int(info.get("total_frames", 0))
        arms = self.arms()
        try:
            mtime = max(
                p.stat().st_mtime
                for p in (self.root / "meta").iterdir()
                if not p.name.startswith("._")
            )
        except (OSError, ValueError):
            mtime = 0.0
        cams = []
        for c in self.cameras:
            f = self.features[c]
            shape = f.get("shape") or [0, 0, 0]
            vinfo = f.get("info") or {}
            cams.append(
                {
                    "key": c,
                    "name": c.removeprefix("observation.images."),
                    "h": shape[0],
                    "w": shape[1],
                    "codec": vinfo.get("video.codec"),
                }
            )
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "name": self.root.name,
            "path": str(self.root),
            "version": self.version,
            "robot_type": info.get("robot_type"),
            "fps": self.fps,
            "episodes": eps,
            "frames": frames,
            "duration_s": frames / self.fps if self.fps else 0.0,
            "tasks": int(info.get("total_tasks", 0)),
            "cameras": cams,
            "arms": [{"name": a["name"], "so101": a["so101"]} for a in arms],
            "bimanual": len(arms) > 1,
            "dims": {
                k: (self.features.get(k, {}).get("shape") or [0])[0]
                for k in ("action", "observation.state")
            },
            "modified": mtime,
        }


class Library:
    """Datasets under the roots, found once and refreshed on request. Thread-safe."""

    def __init__(self, roots: list[Path]) -> None:
        self.roots = roots
        self._by_id: dict[str, Dataset] = {}
        self._sizes: dict[str, tuple[float, int]] = {}
        self._scanned = 0.0
        self._lock = threading.Lock()

    def scan(self, force: bool = False) -> list[Dataset]:
        with self._lock:
            if force or not self._by_id or time.monotonic() - self._scanned > 30:
                found: dict[str, Dataset] = {}
                for root in find_datasets(self.roots):
                    did = dataset_id(root)
                    old = self._by_id.get(did)
                    try:
                        found[did] = old if old and not force else Dataset(root)
                    except (OSError, ValueError, KeyError):
                        continue  # an unreadable info.json: not a dataset Studio can show
                self._by_id = found
                self._scanned = time.monotonic()
            return list(self._by_id.values())

    def get(self, did: str) -> Dataset:
        if did not in self._by_id:
            self.scan(force=True)
        if did not in self._by_id:
            raise KeyError(f"no dataset with id {did}")
        return self._by_id[did]

    def size(self, ds: Dataset) -> int:
        """Bytes on disk, cached by the meta folder's mtime."""
        stamp = (ds.root / "meta" / "info.json").stat().st_mtime
        cached = self._sizes.get(ds.id)
        if cached and cached[0] == stamp:
            return cached[1]
        n = _folder_size(ds.root)
        self._sizes[ds.id] = (stamp, n)
        return n
