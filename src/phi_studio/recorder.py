"""Recording LeRobot datasets from Studio's own control loop: the files lerobot-record writes
(LeRobot 0.6.0, dataset v3.0), so lerobot-train and every viewer read them as their own.

WHY a writer of its own: the bus loop must never wait on a disk or a video encoder. It hands each
frame over and moves on; one writer owns the dataset: create or resume, add_frame, save_episode
or clear_episode_buffer, finalize. The worker's runs as a process (WriterProcess). Frames that
do not fit are dropped and counted, and the take says so. finalize runs in a finally, always: a
dataset never finalized has no meta/episodes and cannot be read (LeRobot then looks for it on
the Hub), and episodes added after finalize on the same object make it unreadable.

Names and order are LeRobot's (so_follower.py, bi_so_follower.py _motors_ft): six "<joint>.pos"
per arm, a bimanual rig's left arm first with a left_ prefix. The action is the leaders'
positions, as lerobot-record records the teleop action. Cameras are observation.images.<key>,
RGB uint8 H x W x 3.

Episodes, driven by the worker's clock (worker.py _rec_tick):
  warmup  a few seconds to settle the hands on the leaders; nothing recorded
  record  up to episode_s. Next keeps the take, Redo throws it away and records it again
  reset   reset_s, teleop live, nothing recorded. The take is saved when the reset ends, so Redo
          during the reset can still throw it away (as lerobot-record). A take that runs to its
          time limit is kept: LeLab drops it, which loses good takes
  paused  a camera gave no fresh picture for CAMERA_GRACE_S: the take is dropped, teleop goes
          on, and Resume records it again (Parv's RealCamera reopens a lost camera by itself)
  done    after the last episode, or Stop: a take in progress is dropped, one waiting in the
          reset is kept, and the dataset is finalized

WHY a pause, not an end: a camera that hiccups (a USB reset, a hub browning out) is the most
common way a LeRobot recording dies midway (OpenCVCamera's read times out and the record loop
raises). Here a short gap reuses the camera's last picture and counts it late, and a long one
costs one take, never the session.

A take ends on its frame count, round(episode_s * fps): the dataset's clock is frame_index / fps,
so a take of N frames is N / fps seconds long, exactly.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phi_studio.rig import JOINTS

MAX_PENDING = 90  # frames waiting for a thread writer: 3 s at 30 Hz, then they are dropped
SLOTS = 48  # frames in flight per camera to the writer process: 1.6 s at 30 Hz
WARMUP_S = 3.0
LATE_S = 0.1  # a camera frame older than this when recorded is late (LeRobot accepts 0.5 s)
CAMERA_GRACE_S = 1.0  # a camera with no fresh picture this long pauses the recording
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


@dataclass
class RecSpec:
    repo_id: str  # owner/name
    root: Path  # the dataset's folder
    fps: int
    task: str
    episodes: int
    episode_s: float
    reset_s: float
    resume: bool = False  # add episodes to an existing dataset
    warmup_s: float = WARMUP_S


def dataset_home() -> Path:
    """Where LeRobot keeps datasets (HF_LEROBOT_HOME), resolved its own way."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME)


def check_repo_id(repo_id: Any) -> str:
    """owner/name, each part a plain file name: it becomes a folder under the dataset home."""
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise ValueError("A dataset name is owner/name, for example phi/cube_pick.")
    owner, name = repo_id.split("/")
    if not NAME_RE.match(owner) or not NAME_RE.match(name):
        raise ValueError("Use letters, digits, dot, dash and underscore in the dataset name.")
    return repo_id


def stamped(repo_id: str, now: float | None = None) -> str:
    """A new dataset's name with lerobot-record's date suffix: create refuses an existing folder."""
    return f"{repo_id}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}"


Cols = list[tuple[str, str]]  # per feature name: (arm, joint)


def columns(pairs: list[tuple[Any, Any]]) -> tuple[list[str], Cols, Cols]:
    """Feature names, and per name the (follower, joint) it reads for the state and the
    (leader, joint) for the action. `pairs`: (leader, follower) arms, as the worker pairs them."""
    sided = len(pairs) > 1
    rank = {"left": 0, "right": 1}
    order = sorted(pairs, key=lambda p: rank.get(getattr(p[1], "side", None) or "", 2))
    names: list[str] = []
    state: list[tuple[str, str]] = []
    action: list[tuple[str, str]] = []
    for lead, fol in order:
        side = getattr(fol, "side", None)
        for j in JOINTS:
            names.append(f"{side}_{j}.pos" if sided and side else f"{j}.pos")
            state.append((fol.name, j))
            action.append((lead.name, j))
    return names, state, action


def features(names: list[str], cameras: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """LeRobot's features for these joints and cameras (feature_utils.hw_to_dataset_features)."""
    from lerobot.utils.feature_utils import hw_to_dataset_features

    hw: dict[str, Any] = {n: float for n in names}
    cams: dict[str, Any] = dict(cameras)
    return {**hw_to_dataset_features(hw, "action", use_video=True),
            **hw_to_dataset_features({**hw, **cams}, "observation", use_video=True)}  # fmt: skip


def open_lerobot_dataset(spec: RecSpec, feats: dict[str, Any], robot_type: str) -> Any:
    """A new LeRobot dataset, or an existing one to add episodes to, recording video as it goes."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    # WHY vcodec auto: a hardware encoder where there is one (h264_videotoolbox on a Mac), so
    # encoding leaves the CPU to the control loop; libsvtav1 otherwise (video.py).
    enc = RGBEncoderConfig(vcodec="auto")
    if spec.resume:
        return LeRobotDataset.resume(spec.repo_id, root=spec.root, streaming_encoding=True,
                                     rgb_encoder=enc)  # fmt: skip
    return LeRobotDataset.create(spec.repo_id, fps=spec.fps, features=feats, root=spec.root,
                                 robot_type=robot_type, use_videos=True,
                                 streaming_encoding=True, rgb_encoder=enc)  # fmt: skip


def serve(
    ds: Any,
    get: Callable[[], tuple[Any, ...]],
    send: Callable[[dict[str, Any]], None],
    spec: RecSpec,
    frame_of: Callable[[tuple[Any, ...]], dict[str, Any]],
) -> int:
    """The writer's loop, in a thread or a process: ops until finish. Returns episodes saved."""
    saved = 0
    send({"type": "rec_ready", "repo_id": spec.repo_id, "root": str(spec.root),
          "episodes": int(ds.num_episodes)})  # fmt: skip
    while True:
        op = get()
        if op[0] == "frame":
            ds.add_frame(frame_of(op))
        elif op[0] == "save":
            ds.save_episode()
            saved += 1
            send({"type": "rec_saved", "root": str(spec.root),
                  "episode": int(ds.num_episodes) - 1})  # fmt: skip
        elif op[0] == "discard":
            ds.clear_episode_buffer()
        elif op[0] == "finish":
            return saved


class Writer:
    """The writer as a thread in the caller's process (tests, with a stand-in dataset). The worker
    uses WriterProcess. frame/save/discard/finish never block."""

    def __init__(self, spec: RecSpec, feats: dict[str, Any], robot_type: str,
                 send: Callable[[dict[str, Any]], None],
                 open_dataset: Callable[..., Any] | None = None) -> None:  # fmt: skip
        self.spec, self.feats, self.robot_type, self.send = spec, feats, robot_type, send
        self.open_dataset = open_dataset or (lambda: open_lerobot_dataset(spec, feats, robot_type))
        self.q: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.pending = 0
        self.dropped = 0
        self.saved = 0
        self.error: str | None = None
        self.ready = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name="phi-recorder", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def frame(self, f: dict[str, Any]) -> bool:
        with self.lock:
            if self.pending >= MAX_PENDING or self.error:
                self.dropped += 1
                return False
            self.pending += 1
        self.q.put(("frame", f))
        return True

    def save(self) -> None:
        self.q.put(("save",))

    def discard(self) -> None:
        self.q.put(("discard",))

    def finish(self) -> None:
        self.q.put(("finish",))

    def _frame_of(self, op: tuple[Any, ...]) -> dict[str, Any]:
        with self.lock:
            self.pending -= 1
        return op[1]

    def _send(self, m: dict[str, Any]) -> None:
        if m["type"] == "rec_ready":
            self.ready.set()
        elif m["type"] == "rec_saved":
            self.saved += 1
        self.send(m)

    def _run(self) -> None:
        ds = None
        try:
            ds = self.open_dataset()
            serve(ds, self.q.get, self._send, self.spec, self._frame_of)
        except Exception as e:  # a full disk, an encoder that will not open: say so, keep the rest
            self.error = f"{type(e).__name__}: {e}"
            self.send({"type": "rec_error", "message": f"Recording stopped: {self.error}"})
        finally:
            if ds is not None:
                try:
                    ds.finalize()
                except Exception as e:
                    self.send({"type": "rec_error",
                               "message": f"Could not finish the dataset: {e}"})  # fmt: skip
            self.done.set()
            self.send({"type": "rec_finished", "root": str(self.spec.root), "saved": self.saved})


def _writer_main(
    spec: RecSpec,
    feats: dict[str, Any],
    robot_type: str,
    rings: dict[str, str],
    shapes: dict[str, tuple[int, ...]],
    ops: Any,
    out: Any,
    taken: Any,
) -> None:
    """The writer process: frames come through shared memory, everything else through `ops`."""
    from multiprocessing import shared_memory

    shms = {k: shared_memory.SharedMemory(name=n) for k, n in rings.items()}
    views = {k: np.ndarray((SLOTS, *shapes[k]), np.uint8, buffer=shms[k].buf) for k in rings}
    send = out.send

    def frame_of(op: tuple[Any, ...]) -> dict[str, Any]:
        _, slot, small = op
        f = dict(small)
        for k, v in views.items():  # copied out before the slot is handed back for reuse
            f[f"observation.images.{k}"] = v[slot].copy()
        taken.value += 1
        return f

    ds, saved = None, 0
    try:
        ds = open_lerobot_dataset(spec, feats, robot_type)
        saved = serve(ds, ops.get, send, spec, frame_of)
    except Exception as e:
        send({"type": "rec_error", "message": f"Recording stopped: {type(e).__name__}: {e}"})
    finally:
        if ds is not None:
            try:
                ds.finalize()
            except Exception as e:
                send({"type": "rec_error", "message": f"Could not finish the dataset: {e}"})
        send({"type": "rec_finished", "root": str(spec.root), "saved": saved})
        views.clear()  # WHY: a live numpy view on the buffer makes close() raise BufferError
        for shm in shms.values():
            shm.close()


class WriterProcess:
    """The writer in a process of its own. WHY: LeRobot's save_episode holds the GIL for tens of
    milliseconds (pandas, pyarrow), which as a thread stalled the 30 Hz loop to 65 ms (measured,
    bimanual, four cameras); and a crash in the encoder's native code ends this process only, not
    the worker holding the arms. Frames cross in shared memory, SLOTS per camera; the rest of a
    frame, and every op, in a queue. A frame with no free slot is dropped and counted."""

    def __init__(self, spec: RecSpec, feats: dict[str, Any], robot_type: str,
                 send: Callable[[dict[str, Any]], None],
                 shapes: dict[str, tuple[int, ...]]) -> None:  # fmt: skip
        import multiprocessing as mp
        from multiprocessing import shared_memory

        self.spec, self.send = spec, send
        ctx = mp.get_context("spawn")
        self.shms = {k: shared_memory.SharedMemory(create=True, size=SLOTS * int(np.prod(sh)))
                     for k, sh in shapes.items()}  # fmt: skip
        self.views = {k: np.ndarray((SLOTS, *sh), np.uint8, buffer=self.shms[k].buf)
                      for k, sh in shapes.items()}  # fmt: skip
        self.ops = ctx.Queue()
        self.taken = ctx.Value("q", 0, lock=False)  # frames the writer has copied out
        self.written = 0
        self.dropped = 0
        self.saved = 0
        self.error: str | None = None
        self.ready = threading.Event()
        self.done = threading.Event()
        r, w = ctx.Pipe(duplex=False)
        self._events = r
        self.proc = ctx.Process(
            target=_writer_main, name="phi-recorder", daemon=True,
            args=(spec, feats, robot_type, {k: m.name for k, m in self.shms.items()},
                  shapes, self.ops, w, self.taken),
        )  # fmt: skip
        self._w = w
        self.listener = threading.Thread(target=self._listen, name="phi-recorder-events",
                                         daemon=True)  # fmt: skip

    def start(self) -> None:
        self.proc.start()
        self._w.close()  # WHY: otherwise recv() never sees EOF if the writer dies
        self.listener.start()

    def frame(self, f: dict[str, Any]) -> bool:
        if self.error or self.done.is_set() or self.written - self.taken.value >= SLOTS:
            self.dropped += 1
            return False
        slot = self.written % SLOTS
        small = {}
        for key, v in f.items():
            cam = key.removeprefix("observation.images.")
            if cam in self.views:
                self.views[cam][slot] = v
            else:
                small[key] = v
        self.ops.put(("frame", slot, small))
        self.written += 1
        return True

    def save(self) -> None:
        self.ops.put(("save",))

    def discard(self) -> None:
        self.ops.put(("discard",))

    def finish(self) -> None:
        self.ops.put(("finish",))

    def _listen(self) -> None:
        try:
            while True:
                m = self._events.recv()
                if m["type"] == "rec_ready":
                    self.ready.set()
                elif m["type"] == "rec_saved":
                    self.saved += 1
                elif m["type"] == "rec_error":
                    self.error = m["message"]
                self.send(m)
                if m["type"] == "rec_finished":
                    break
        except (EOFError, OSError):  # the writer died without saying so
            self.error = self.error or ("The recording process stopped unexpectedly; the "
                                        "episodes saved before may need a repair.")  # fmt: skip
            self.send({"type": "rec_error", "message": self.error})
            self.send({"type": "rec_finished", "root": str(self.spec.root), "saved": self.saved})
        finally:
            self.proc.join(5)
            self.views.clear()
            self.done.set()
            for shm in self.shms.values():
                try:
                    shm.close()
                    shm.unlink()
                except (FileNotFoundError, BufferError):
                    pass


@dataclass
class Take:
    """The episode being recorded, kept in memory too, for its health check during the reset."""

    index: int
    t0: float
    state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    late: int = 0  # camera frames older than LATE_S when recorded
    dropped0: int = 0  # the writer's dropped count when the take began
    ticks: list[float] = field(default_factory=list)
    discarded: bool = False  # thrown away (Redo): its health check, if late, is not reported

    def health(self, fps: float, names: list[str], dropped: int) -> dict[str, Any]:
        """analysis.analyse_episode on the take, plus what only the recording knows: frames the
        writer dropped, late camera frames, and the loop rate."""
        from phi_studio.analysis import analyse_episode
        from phi_studio.datasets import split_arms

        n = len(self.state)
        out: dict[str, Any] = {"episode": self.index, "frames": n, "seconds": round(n / fps, 2),
                               "dropped": dropped, "late": self.late, "flags": [],
                               "health": "ok"}  # fmt: skip
        if len(self.ticks) > 1:
            out["hz"] = round((len(self.ticks) - 1) / (self.ticks[-1] - self.ticks[0]), 1)
        if n >= 3:
            a = analyse_episode({"state": np.stack(self.state), "action": np.stack(self.action)},
                                fps, split_arms(names), series=False)  # fmt: skip
            out["flags"], out["health"] = a["flags"], a["health"]
        rank = ["ok", "info", "warn", "error"]
        extra = []
        if dropped:
            extra.append({"kind": "dropped", "severity": "error",
                          "text": f"{dropped} frames dropped: the writer fell behind"})  # fmt: skip
        if self.late:
            extra.append(
                {
                    "kind": "late",
                    "severity": "warn",
                    "text": f"{self.late} camera frames older than {LATE_S * 1000:.0f} ms",
                }
            )
        hz = out.get("hz")
        if hz is not None and hz < 0.9 * fps:
            extra.append(
                {
                    "kind": "slow",
                    "severity": "warn",
                    "text": f"loop ran at {hz} Hz, under the dataset's {fps:g}",
                }
            )
        for f in extra:
            out["flags"].append(f)
            if rank.index(f["severity"]) > rank.index(out["health"]):
                out["health"] = f["severity"]
        return out


@dataclass
class Recording:
    """One recording session in the worker: the spec, the writer, and where the episodes are."""

    spec: RecSpec
    writer: Writer | WriterProcess
    names: list[str]
    state_cols: list[tuple[str, str]]
    action_cols: list[tuple[str, str]]
    cameras: list[Any]
    phase: str = "warmup"  # warmup | record | reset | paused | done
    why: str | None = None  # why it paused
    seen: dict[str, tuple[Any, float]] = field(default_factory=dict)  # camera -> last frame, time
    episode: int = 0  # the take in progress, or the one waiting in the reset
    t0: float = 0.0  # when this phase began
    take: Take | None = None
    waiting: bool = False  # a take in the reset, to save when it ends
    takes: dict[int, dict[str, Any]] = field(default_factory=dict)  # each finished take's health
    first: int = 0  # the dataset's episodes before this session (resume)

    @property
    def target(self) -> int:
        """Frames in a full take."""
        return max(1, round(self.spec.episode_s * self.spec.fps))

    def view(self, now: float) -> dict[str, Any]:
        limit = {"warmup": self.spec.warmup_s, "record": self.spec.episode_s,
                 "reset": self.spec.reset_s}.get(self.phase)  # fmt: skip
        return {"repo_id": self.spec.repo_id, "root": str(self.spec.root), "task": self.spec.task,
                "phase": self.phase, "episode": self.episode, "of": self.spec.episodes,
                "t": round(max(0.0, now - self.t0), 2), "limit": limit,
                "frames": len(self.take.state) if self.take and self.phase == "record" else 0,
                "target": self.target, "why": self.why,
                "late": self.take.late if self.take and self.phase == "record" else 0,
                "dropped": self.writer.dropped, "saved": self.writer.saved,
                "error": self.writer.error, "finished": self.writer.done.is_set(),
                "ready": self.writer.ready.is_set(),
                "takes": [self.takes[k] for k in sorted(self.takes)][-50:]}  # fmt: skip
