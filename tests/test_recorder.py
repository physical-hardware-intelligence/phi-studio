"""Recording LeRobot datasets from the worker's control loop: names and order as LeRobot's, the
writer thread that owns the dataset, and whole sessions on the mock rig written by LeRobot itself
and read back by Studio's own reader."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phi_studio import recorder as R
from phi_studio.mock import FakeClock, mock_rig
from phi_studio.worker import RigWorker

HZ = 30
DT = 1 / HZ


def test_columns_follow_lerobots_names_and_order() -> None:
    one = [(SimpleNamespace(name="leader", side=None), SimpleNamespace(name="follower", side=None))]
    names, state, action = R.columns(one)
    assert names[0] == "shoulder_pan.pos" and names[-1] == "gripper.pos" and len(names) == 6
    assert state[0] == ("follower", "shoulder_pan") and action[0] == ("leader", "shoulder_pan")

    def pair(side: str) -> tuple[SimpleNamespace, SimpleNamespace]:
        return (SimpleNamespace(name=f"{side}_leader", side=side),
                SimpleNamespace(name=f"{side}_follower", side=side))  # fmt: skip

    two = [pair("right"), pair("left")]
    names, state, _ = R.columns(two)
    assert names[:2] == ["left_shoulder_pan.pos", "left_shoulder_lift.pos"]  # the left arm first
    assert names[6] == "right_shoulder_pan.pos" and state[6] == ("right_follower", "shoulder_pan")


def test_repo_ids_are_checked_and_new_ones_stamped() -> None:
    assert R.check_repo_id("phi/cube_pick") == "phi/cube_pick"
    for bad in ("cube", "a/b/c", "../x", "phi/", "phi/a b"):
        with pytest.raises(ValueError):
            R.check_repo_id(bad)
    assert R.stamped("phi/cube", 0).startswith("phi/cube_19")


class FakeDS:
    def __init__(self, fail_on: int | None = None, slow: threading.Event | None = None) -> None:
        self.frames: list = []
        self.buffer = 0
        self.saved: list[int] = []
        self.finalized = False
        self.num_episodes = 0
        self.fail_on, self.slow = fail_on, slow

    def add_frame(self, f) -> None:
        if self.slow is not None:
            self.slow.wait(5)
        if self.fail_on is not None and len(self.frames) == self.fail_on:
            raise OSError("No space left on device")
        self.frames.append(f)
        self.buffer += 1

    def save_episode(self) -> None:
        self.saved.append(self.buffer)
        self.buffer = 0
        self.num_episodes += 1

    def clear_episode_buffer(self) -> None:
        self.buffer = 0

    def finalize(self) -> None:
        self.finalized = True


def writer(ds: FakeDS) -> tuple[R.Writer, list[dict]]:
    sent: list[dict] = []
    spec = R.RecSpec("phi/x", Path("/tmp/x"), HZ, "t", 3, 1.0, 1.0)
    return R.Writer(spec, {}, "so_follower", sent.append, open_dataset=lambda: ds), sent


def test_the_writer_keeps_saves_discards_and_always_finalizes() -> None:
    ds = FakeDS()
    w, sent = writer(ds)
    w.start()
    for _ in range(4):
        w.frame({"i": 1})
    w.discard()
    for _ in range(3):
        w.frame({"i": 2})
    w.save()
    w.finish()
    assert w.done.wait(5)
    assert ds.saved == [3] and ds.finalized
    assert [m["type"] for m in sent] == ["rec_ready", "rec_saved", "rec_finished"]


def test_a_failing_disk_stops_the_recording_but_still_finalizes() -> None:
    ds = FakeDS(fail_on=2)
    w, sent = writer(ds)
    w.start()
    for _ in range(5):
        w.frame({"i": 1})
    w.save()
    assert w.done.wait(5)
    assert ds.finalized and w.error and "No space" in w.error
    assert any(m["type"] == "rec_error" for m in sent)
    assert not w.frame({"i": 3})  # refused once it has failed


def test_a_writer_that_falls_behind_drops_frames_and_counts_them(monkeypatch) -> None:
    monkeypatch.setattr(R, "MAX_PENDING", 5)
    gate = threading.Event()
    ds = FakeDS(slow=gate)
    w, _ = writer(ds)
    w.start()
    took = [w.frame({"i": i}) for i in range(12)]
    assert took.count(False) == w.dropped >= 6  # the bus loop never waited
    gate.set()
    w.finish()
    assert w.done.wait(5) and ds.finalized


# -- whole sessions through the worker, written by LeRobot itself ----------------------------------
WORKERS: list[RigWorker] = []


@pytest.fixture(autouse=True)
def finish_every_recording():
    """No writer outlives its test: one still encoding while the next starts, or while the
    interpreter exits, is how native code crashes. The worker's own shutdown does it."""
    yield
    while WORKERS:
        WORKERS.pop().shutdown(finish_s=30)


def session(tmp_path, monkeypatch, pairs: int = 1, cameras=("front",)):
    pytest.importorskip("lerobot")
    monkeypatch.setattr(R, "dataset_home", lambda: tmp_path / "lerobot")
    clock = FakeClock()
    rig = mock_rig(pairs=pairs, cameras=cameras, clock=clock)
    out: list[dict] = []
    w = RigWorker(rig, send=out.append, clock=clock, loop_hz=HZ)
    WORKERS.append(w)
    for c in ("connect", "confirm", "arm"):
        w.handle({"cmd": c})
    w.handle({"cmd": "start", "activity": "teleop"})
    assert w.session.state.name == "MOVING", out
    return w, clock, out


def tick(w: RigWorker, clock: FakeClock, seconds: float) -> None:
    for _ in range(int(round(seconds * HZ))):
        w.handle({"cmd": "heartbeat"})
        clock.advance(DT)
        w.tick()
        keep_pace(w)


def keep_pace(w: RigWorker) -> None:
    """A fake clock runs the loop far faster than real time; let the writer process keep up,
    as it does at the real rate (test_real_time_four_cameras_drop_nothing)."""
    r = w.rec
    wr = getattr(r, "writer", None)
    if wr is None or not hasattr(wr, "taken"):
        return
    give_up = time.monotonic() + 30
    while wr.written - wr.taken.value > 8 and not wr.done.is_set() and time.monotonic() < give_up:
        time.sleep(0.002)


def finished(w: RigWorker, out: list[dict]) -> Path:
    assert w.rec is not None and w.rec.writer.done.wait(60), w.rec and w.rec.writer.error
    assert w.rec.writer.error is None, w.rec.writer.error
    return w.rec.spec.root


def wait_for(out: list[dict], kind: str, n: int = 1, timeout: float = 10) -> list[dict]:
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        got = [m for m in out if m.get("type") == kind]
        if len(got) >= n:
            return got
        time.sleep(0.02)
    raise AssertionError(f"no {n} x {kind}")


def start(w: RigWorker, **over) -> None:
    msg = {"cmd": "rec_start", "repo_id": "phi/test", "task": "Pick up the cube",
           "episodes": 2, "episode_s": 1.0, "reset_s": 0.5} | over  # fmt: skip
    w.handle(msg)
    if w.rec is not None:  # the writer process takes a moment; the warm-up waits for it
        assert w.rec.writer.ready.wait(60), w.rec.writer.error


def test_a_session_writes_a_dataset_studio_and_lerobot_both_read(tmp_path, monkeypatch) -> None:
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch)
    start(w)
    assert w.rec is not None and w.rec.phase == "warmup", [m for m in out if m["type"] == "error"]
    tick(w, clock, R.WARMUP_S + 1.0 + 0.5 + 1.0 + 0.2)
    root = finished(w, out)
    takes = wait_for(out, "rec_take", 2)
    assert {t["episode"] for t in takes} == {0, 1} and all(t["frames"] == 30 for t in takes)
    ds = Dataset(root)
    assert [e["length"] for e in ds.episodes()] == [30, 30]
    assert ds.names("observation.state")[0] == "shoulder_pan.pos"
    assert ds.tasks() == ["Pick up the cube"]
    # LeRobot reads it as its own, frames decoded, in a process of its own. WHY a subprocess: the
    # decoder (torchcodec) brings its own FFmpeg, and two FFmpeg builds in one process next to a
    # live encoder can crash it. The worker never decodes, so it only ever loads PyAV's.
    import subprocess
    import sys

    code = (
        "import sys; from lerobot.datasets import LeRobotDataset as D\n"
        "d = D(sys.argv[1], root=sys.argv[2]); x = d[0]\n"
        "s = tuple(x['observation.images.front'].shape)\n"
        "print(d.num_episodes, d.fps, d.meta.robot_type, s)"
    )
    got = subprocess.run([sys.executable, "-c", code, "phi/" + root.name, str(root)],
                         capture_output=True, text=True, timeout=180)  # fmt: skip
    assert got.returncode == 0, got.stderr[-2000:]
    assert got.stdout.split()[:3] == ["2", "30", "so_follower"] and "(3," in got.stdout


def test_redo_throws_the_take_away_in_record_and_in_reset(tmp_path, monkeypatch) -> None:
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch)
    start(w)
    tick(w, clock, R.WARMUP_S + 0.5)
    w.handle({"cmd": "rec_redo"})  # mid-take: again from the start
    tick(w, clock, 1.0 + 0.2)  # the take runs out: in the reset now
    assert w.rec.phase == "reset"
    w.handle({"cmd": "rec_redo"})  # thrown away during the reset: recorded again
    assert w.rec.phase == "record" and w.rec.episode == 0
    tick(w, clock, 1.0 + 0.5 + 1.0 + 0.2)
    root = finished(w, out)
    assert [e["length"] for e in Dataset(root).episodes()] == [30, 30]


def test_next_hurries_each_phase(tmp_path, monkeypatch) -> None:
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch)
    start(w, episode_s=10.0, reset_s=10.0)
    w.handle({"cmd": "rec_next"})  # skip the warm-up
    tick(w, clock, 0.5)
    w.handle({"cmd": "rec_next"})  # keep a short take
    assert w.rec.phase == "reset"
    w.handle({"cmd": "rec_next"})  # skip the reset
    tick(w, clock, 0.3)
    w.handle({"cmd": "rec_next"})
    root = finished(w, out)
    assert [e["length"] for e in Dataset(root).episodes()] == [15, 9]


def test_stop_drops_the_take_in_progress_and_keeps_what_was_saved(tmp_path, monkeypatch) -> None:
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch)
    start(w, episodes=3)
    tick(w, clock, R.WARMUP_S + 1.0 + 0.5 + 0.4)  # take 1 saved, take 2 under way
    w.handle({"cmd": "stop"})  # the rig's Stop: the arms freeze, the recording ends
    assert w.session.state.name == "STOPPED" and w.rec.phase == "done"
    root = finished(w, out)
    assert [e["length"] for e in Dataset(root).episodes()] == [30]


def test_bimanual_with_both_wrist_cameras(tmp_path, monkeypatch) -> None:
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch, pairs=2, cameras=None)
    start(w, episodes=1)
    tick(w, clock, R.WARMUP_S + 1.0 + 0.2)
    root = finished(w, out)
    ds = Dataset(root)
    assert len(ds.names("observation.state")) == 12
    assert {"observation.images.left_wrist", "observation.images.right_wrist"} <= set(ds.cameras)


def test_recording_needs_teleop(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lerobot")
    monkeypatch.setattr(R, "dataset_home", lambda: tmp_path)
    clock = FakeClock()
    out: list[dict] = []
    w = RigWorker(mock_rig(cameras=("front",), clock=clock), send=out.append, clock=clock)
    w.handle({"cmd": "connect"})
    start(w)
    assert w.rec is None and "teleop" in out[-1]["message"]


def test_a_camera_hiccup_costs_nothing_and_a_dark_one_costs_one_take(tmp_path, monkeypatch) -> None:
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch, cameras=("front",))
    start(w, episodes=2, episode_s=3.0, reset_s=0.5)
    cam = w.rig.cameras[0]
    tick(w, clock, R.WARMUP_S + 0.3)
    cam.inject("unplug")  # gone for half a second: its last picture stands in, counted late
    tick(w, clock, 0.5)
    cam.inject("replug")
    tick(w, clock, 2.5)
    assert w.rec.phase == "reset", w.rec.view(clock())
    first = wait_for(out, "rec_take", 1)[0]
    assert first["frames"] == 90 and first["late"] >= 10
    assert any(f["kind"] == "late" for f in first["flags"])
    tick(w, clock, 0.4)  # the reset ends; take 2 under way
    cam.inject("unplug")  # dark for good: the take is dropped and the recording waits
    tick(w, clock, R.CAMERA_GRACE_S + 0.2)
    assert w.rec.phase == "paused" and "front" in (w.rec.why or "")
    assert w.session.state.name == "MOVING"  # teleop goes on
    cam.inject("replug")
    w.handle({"cmd": "rec_next"})  # Resume: take 2 again, from the start
    assert w.rec.phase == "record" and w.rec.episode == 1
    tick(w, clock, 3.2)
    root = finished(w, out)
    assert [e["length"] for e in Dataset(root).episodes()] == [90, 90]


def test_a_take_reports_its_health_and_what_only_recording_knows() -> None:
    names = [f"{j}.pos" for j in R.JOINTS]
    t = R.Take(0, 0.0)
    for i in range(60):
        x = np.full(6, 10.0 + i * 0.5, np.float32)
        t.state.append(x)
        t.action.append(x)
        t.ticks.append(i / 15)  # the loop ran at 15 Hz
    t.late = 4
    h = t.health(30, names, dropped=3)
    kinds = {f["kind"] for f in h["flags"]}
    assert {"dropped", "late", "slow"} <= kinds and h["health"] == "error"


def test_a_worker_closing_mid_take_leaves_a_dataset_that_reads(tmp_path, monkeypatch) -> None:
    """The server gone, or Studio closed, mid-recording: the saved takes survive, finalized."""
    from phi_studio.datasets import Dataset

    w, clock, out = session(tmp_path, monkeypatch)
    start(w, episodes=3)
    tick(w, clock, R.WARMUP_S + 1.0 + 0.5 + 0.4)
    w.shutdown()
    assert w.rec.writer.done.is_set() and w.rec.writer.error is None
    assert [e["length"] for e in Dataset(w.rec.spec.root).episodes()] == [30]
    assert not any(a.torque for a in w.rig.arms)


def test_real_time_four_cameras_drop_nothing(tmp_path, monkeypatch) -> None:
    """At the real rate, with the real queue and LeRobot's own writer: a bimanual rig with four
    cameras keeps every frame, and the loop never waits on the disk."""
    from phi_studio.datasets import Dataset

    pytest.importorskip("lerobot")
    monkeypatch.setattr(R, "dataset_home", lambda: tmp_path / "lerobot")
    rig = mock_rig(pairs=2, cameras=("front", "top", "left_wrist", "right_wrist"))
    out: list[dict] = []
    w = RigWorker(rig, send=out.append, loop_hz=HZ)
    WORKERS.append(w)
    for c in ("connect", "confirm", "arm"):
        w.handle({"cmd": c})
    w.handle({"cmd": "start", "activity": "teleop"})
    start(w, episodes=2, episode_s=1.5, reset_s=0.3)
    w.handle({"cmd": "rec_next"})  # no warm-up
    worst, nxt = 0.0, time.monotonic()
    while w.rec.phase != "done" and time.monotonic() < nxt + 20:
        a = time.perf_counter()
        w.handle({"cmd": "heartbeat"})
        w.tick()
        worst = max(worst, time.perf_counter() - a)
        nxt += DT
        time.sleep(max(0.0, nxt - time.monotonic()))
    assert w.rec.writer.done.wait(60) and w.rec.writer.error is None
    assert w.rec.writer.dropped == 0
    assert [e["length"] for e in Dataset(w.rec.spec.root).episodes()] == [45, 45]
    assert worst < DT, f"a tick took {worst * 1000:.1f} ms"


def test_a_writer_that_crashes_takes_only_itself_down(tmp_path, monkeypatch) -> None:
    """Native code in the encoder can crash; it runs in a process of its own, so the worker,
    the arms and teleop carry on, and the page says what happened."""
    import os
    import signal

    w, clock, out = session(tmp_path, monkeypatch)
    start(w, episodes=3)
    tick(w, clock, R.WARMUP_S + 0.5)
    os.kill(w.rec.writer.proc.pid, signal.SIGKILL)
    assert w.rec.writer.done.wait(30)
    assert "stopped unexpectedly" in (w.rec.writer.error or "")
    tick(w, clock, 0.5)  # frames now refused, counted, and the loop goes on
    assert w.session.state.name == "MOVING" and w.rec.writer.dropped > 0
    assert any(m.get("type") == "rec_error" for m in out)


def test_recording_through_the_real_process_chain(tmp_path, monkeypatch) -> None:
    """As deployed: the browser's socket -> Studio -> the worker process -> the writer process.
    The in-process tests missed that a daemonic worker may not start the writer at all."""
    import asyncio
    import json

    import aiohttp
    from aiohttp.test_utils import TestServer
    from test_server import free_port, until, ws

    from phi_studio.datasets import Dataset
    from phi_studio.server import Studio

    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot"))  # the worker inherits it

    async def go() -> None:
        port = free_port()
        studio = Studio({"kind": "mock", "pairs": 1, "cameras": ["front"]}, port, token="t0k",
                        data_dir=tmp_path / "data")  # fmt: skip
        server = TestServer(studio.app(), host="127.0.0.1", port=port)
        await server.start_server()
        session = aiohttp.ClientSession()
        try:
            a = await ws(session, port)
            await until(a, lambda d: d["type"] == "hello")
            for cmd in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": cmd}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d["type"] == "state" and d["state"] == "MOVING", timeout=15)
            beat = asyncio.ensure_future(heartbeat(a))
            await a.send_str(json.dumps({"cmd": "rec_start", "repo_id": "phi/chain",
                                         "task": "Pick up the cube", "episodes": 1,
                                         "episode_s": 1.0, "reset_s": 0.0}))  # fmt: skip
            await until(a, lambda d: d["type"] == "rec_ready", timeout=60)
            done = await until(a, lambda d: d["type"] == "rec_finished", timeout=60)
            beat.cancel()
            assert done["saved"] == 1
            assert not [m for m in studio.log if m["kind"] == "error"], list(studio.log)
            assert [e["length"] for e in Dataset(Path(done["root"])).episodes()] == [30]
        finally:
            await session.close()
            await server.close()

    async def heartbeat(sock) -> None:
        while True:
            await sock.send_str('{"cmd": "heartbeat"}')
            await asyncio.sleep(0.25)

    asyncio.run(go())
