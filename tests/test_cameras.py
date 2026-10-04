"""Real camera feeds: probe, the capture thread, reopen after a failure, and pause for LeRobot.

No test opens a real camera: macOS camera permission prompts hang a non-interactive process. The
LeRobot path runs on video files written with cv2.VideoWriter; everything else uses fakes.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phi_studio import cameras, rigspec
from phi_studio.cameras import RealCamera, cameras_from_spec, probe

JPEG = b"\xff\xd8"

# WHY a mark, not a module-level importorskip: the fake-backed tests must still run in CI, which
# installs neither OpenCV nor LeRobot.
needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None or importlib.util.find_spec("lerobot") is None,
    reason="OpenCV and LeRobot are not installed",
)
# WHY: at the end of a file LeRobot's read loop raises inside its own thread
# (camera_opencv.py:464-469). That is the event under test, not a failure of it.
ends_in_thread = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")


def wait_for(pred: Callable[[], Any], timeout: float = 3.0) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


def frame_of(cam: RealCamera) -> tuple[np.ndarray, float, int] | None:
    try:
        return cam.read_latest()
    except ConnectionError:
        return None


def error_of(cam: RealCamera) -> ConnectionError | None:
    try:
        cam.read_latest()
    except ConnectionError as e:
        return e
    return None


# -- fakes with LeRobot's camera surface: async_read, disconnect, fps ------------------------------
class FakeSource:
    """A device the test controls: up (frames flow), stall (open but silent) or gone."""

    def __init__(self, fps: float = 100.0, shape: tuple[int, ...] = (48, 64, 3)) -> None:
        self.fps, self.shape = fps, shape
        self.state = "up"
        self.first_s = 0.0  # a new device's first frame comes this long after open
        self.release_s = 0.0  # disconnect blocks this long, like a read stuck in the driver
        self.opens: list[dict[str, Any]] = []
        self.live: list[FakeDevice] = []  # handed out and not yet released

    def open(self, fields: dict[str, Any]) -> FakeDevice:
        self.opens.append(dict(fields))
        if self.state == "gone":
            raise ConnectionError("Failed to open FakeCamera")
        dev = FakeDevice(self)
        self.live.append(dev)
        return dev


class FakeDevice:
    def __init__(self, src: FakeSource) -> None:
        self.src, self.fps, self.n = src, src.fps, 0
        self.first_at = time.monotonic() + src.first_s

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if self.src.state == "gone":
            raise RuntimeError("read thread is not running")
        if self.src.state == "stall" or time.monotonic() < self.first_at:
            time.sleep(timeout_ms / 1000)
            raise TimeoutError("timed out waiting for frame")
        time.sleep(1 / self.fps)
        self.n += 1
        return np.full(self.src.shape, self.n % 256, np.uint8)

    def disconnect(self) -> None:
        time.sleep(self.src.release_s)
        self.src.live.remove(self)


FIELDS = {"type": "opencv", "index_or_path": 0}


# -- RealCamera ------------------------------------------------------------------------------------
def test_read_latest_gives_the_newest_frame_with_rising_seq() -> None:
    src = FakeSource()
    cam = RealCamera("front", FIELDS, opener=src.open)
    try:
        frame, t, seq = wait_for(lambda: frame_of(cam))
        assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
        _, t2, seq2 = wait_for(lambda: (f := frame_of(cam)) and f[2] > seq and f)
        assert t2 >= t
    finally:
        cam.close()
    assert src.live == []  # close released the device


def test_building_a_camera_opens_nothing_until_it_is_read() -> None:
    src = FakeSource()
    cam = RealCamera("front", FIELDS, opener=src.open)
    time.sleep(0.05)
    assert src.opens == []
    cam.close()
    assert src.opens == []


def test_read_latest_does_not_wait_on_a_hung_open() -> None:
    gate = threading.Event()

    def hung(fields: dict[str, Any]) -> Any:
        gate.wait(5)
        raise ConnectionError("gave up")

    cam = RealCamera("top", FIELDS, opener=hung)
    t0 = time.monotonic()
    with pytest.raises(ConnectionError, match="camera top: opening 0"):
        cam.read_latest()
    assert time.monotonic() - t0 < 0.05
    gate.set()
    cam.close()


def test_a_stall_becomes_a_connection_error_and_the_camera_comes_back() -> None:
    src = FakeSource()  # 100 fps, so 5 frames is 50 ms
    cam = RealCamera("front", FIELDS, opener=src.open, retry_s=0.1, stall_frames=5)
    try:
        _, _, seq = wait_for(lambda: frame_of(cam))
        src.state = "stall"
        err = wait_for(lambda: error_of(cam))
        assert "camera front: no frame from 0 for" in str(err)
        src.state = "gone"
        wait_for(lambda: src.live == [])  # the stalled device was dropped, not held
        assert "could not open 0: Failed to open FakeCamera" in str(wait_for(
            lambda: (e := error_of(cam)) and "could not open" in str(e) and e))  # fmt: skip
        src.state = "up"
        _, _, seq2 = wait_for(lambda: frame_of(cam))
        assert seq2 > seq  # seq never goes back across a reopen
    finally:
        cam.close()
    assert src.live == []


def test_a_stall_is_reported_before_a_slow_release_finishes() -> None:
    src = FakeSource()
    src.release_s = 0.5  # a read stuck in the driver; LeRobot's stop can wait up to 2 s for it
    cam = RealCamera("front", FIELDS, opener=src.open, retry_s=0.1, stall_frames=5)
    try:
        wait_for(lambda: frame_of(cam))
        src.state = "stall"
        t0 = time.monotonic()
        wait_for(lambda: error_of(cam))
        assert time.monotonic() - t0 < 0.4  # the 50 ms stall limit, not that plus the release
    finally:
        src.release_s = 0.0
        cam.close()


def test_a_slow_first_frame_is_not_a_stall() -> None:
    src = FakeSource()
    src.first_s = 0.3  # past the 50 ms stall limit, inside LeRobot's 1 s warmup allowance
    cam = RealCamera("front", FIELDS, opener=src.open, retry_s=0.1, stall_frames=5)
    try:
        wait_for(lambda: frame_of(cam))
        assert len(src.opens) == 1  # never dropped and reopened while it was warming up
    finally:
        cam.close()


def test_a_source_that_dies_is_reported_and_reopened() -> None:
    src = FakeSource()
    cam = RealCamera("wrist", FIELDS, opener=src.open, retry_s=0.1)
    try:
        wait_for(lambda: frame_of(cam))
        src.state = "gone"
        err = wait_for(lambda: error_of(cam))
        assert "camera wrist: 0 stopped delivering: read thread is not running" in str(err)
        src.state = "up"
        wait_for(lambda: frame_of(cam))
    finally:
        cam.close()


def test_reopen_waits_retry_s_between_attempts() -> None:
    src = FakeSource()
    src.state = "gone"
    cam = RealCamera("front", FIELDS, opener=src.open, retry_s=0.2)
    try:
        assert error_of(cam) is not None
        time.sleep(0.7)
        assert 2 <= len(src.opens) <= 5  # about one attempt per 0.2 s, not a busy loop
    finally:
        cam.close()


def test_an_unknown_camera_type_says_so() -> None:
    cam = RealCamera("front", {"type": "gopro", "index_or_path": 0})
    try:
        err = wait_for(lambda: (e := error_of(cam)) and "could not open" in str(e) and e)
        assert "camera type 'gopro' is not one Studio can open" in str(err)
    finally:
        cam.close()


def test_pause_releases_the_device_and_resume_takes_it_back() -> None:
    src = FakeSource()
    cam = RealCamera("front", FIELDS, opener=src.open)
    try:
        _, _, seq = wait_for(lambda: frame_of(cam))
        assert cam.pause(timeout_s=2.0)
        assert src.live == []  # free for lerobot-record the moment pause returns
        with pytest.raises(ConnectionError, match="released for a LeRobot command"):
            cam.read_latest()
        n = len(src.opens)
        time.sleep(0.2)
        assert len(src.opens) == n  # stays released: no retries while paused
        cam.resume()
        _, _, seq2 = wait_for(lambda: frame_of(cam))
        assert seq2 > seq
        assert len(src.live) == 1
    finally:
        cam.close()
    assert src.live == []


def test_pause_before_any_read_opens_nothing() -> None:
    src = FakeSource()
    cam = RealCamera("front", FIELDS, opener=src.open)
    assert cam.pause()
    with pytest.raises(ConnectionError):
        cam.read_latest()
    assert src.opens == []
    cam.resume()
    wait_for(lambda: frame_of(cam))
    cam.close()


def test_close_is_final_and_idempotent() -> None:
    src = FakeSource()
    cam = RealCamera("front", FIELDS, opener=src.open)
    wait_for(lambda: frame_of(cam))
    cam.close()
    cam.close()
    assert src.live == []
    with pytest.raises(ConnectionError, match="closed"):
        cam.read_latest()
    cam.resume()  # no effect after close
    assert src.live == []


# -- probe -----------------------------------------------------------------------------------------
def test_probe_reports_good_and_bad_devices_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = FakeSource(shape=(480, 640, 3))
    src.fps = 30.0

    def opener(fields: dict[str, Any]) -> Any:
        if fields["index_or_path"] != 0:
            raise ConnectionError(f"Failed to open OpenCVCamera({fields['index_or_path']})")
        return src.open(fields)

    monkeypatch.setattr(cameras, "_open", opener)
    good, bad = probe([0, "7"])
    assert good["ok"] and (good["width"], good["height"], good["fps"]) == (640, 480, 30.0)
    assert good["error"] is None and good["thumbnail"].startswith(JPEG)
    assert bad["source"] == 7 and not bad["ok"] and "Failed to open" in bad["error"]
    assert src.live == []  # every probe releases


def test_probe_falls_back_to_the_native_size(monkeypatch: pytest.MonkeyPatch) -> None:
    src = FakeSource(shape=(240, 320, 3))

    def opener(fields: dict[str, Any]) -> Any:
        if "width" in fields:
            raise RuntimeError("failed to set capture_width=640")
        return src.open(fields)

    monkeypatch.setattr(cameras, "_open", opener)
    (r,) = probe([2])
    assert r["ok"] and (r["width"], r["height"]) == (320, 240)


def test_probe_times_out_on_a_hung_device_and_still_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = FakeSource()
    gate = threading.Event()

    def hung(fields: dict[str, Any]) -> Any:
        gate.wait(5)
        return src.open(fields)

    monkeypatch.setattr(cameras, "_open", hung)
    monkeypatch.setattr(cameras, "RELEASE_GRACE_S", 0.1)
    t0 = time.monotonic()
    (r,) = probe([3], timeout_s=0.2)
    assert time.monotonic() - t0 < 1.0
    assert not r["ok"] and "no answer from 3 in 0.2 s" in r["error"]
    gate.set()  # the driver finally answers, after probe gave up
    wait_for(lambda: len(src.opens) == 1 and src.live == [])


def test_probe_reports_lerobots_timeout_for_a_slow_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    src = FakeSource()
    src.first_s = 10.0  # opens, then never delivers a frame

    def slow(fields: dict[str, Any]) -> Any:
        time.sleep(0.15)
        return src.open(fields)

    monkeypatch.setattr(cameras, "_open", slow)
    t0 = time.monotonic()
    (r,) = probe([0], timeout_s=0.3)
    assert time.monotonic() - t0 < 0.6  # the frame wait got only what the open left
    assert not r["ok"] and "timed out waiting for frame" in r["error"]  # not "no answer"
    assert src.live == []  # released before the next candidate could start


# -- cameras_from_spec -----------------------------------------------------------------------------
SINGLE = """\
robot:
  type: so101_follower
  id: f
  port: /dev/tty.usbmodemF
  cameras:
    front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, hardware: Brio 101}
    wrist: {type: opencv, index_or_path: TBD}
    top: {type: opencv}
teleop: {type: so101_leader, id: l, port: /dev/tty.usbmodemL}
"""

BIMANUAL = """\
robot:
  type: bi_so_follower
  id: f
  left_arm_config:
    port: /dev/tty.usbmodemFL
    cameras:
      wrist: {type: opencv, index_or_path: 1}
  right_arm_config:
    port: /dev/tty.usbmodemFR
    cameras:
      wrist: {type: opencv, index_or_path: 2}
  cameras:
    top: {type: opencv, index_or_path: 0}
teleop:
  type: bi_so_leader
  id: l
  left_arm_config: {port: /dev/tty.usbmodemLL}
  right_arm_config: {port: /dev/tty.usbmodemLR}
"""


def _no_open(fields: dict[str, Any]) -> Any:
    raise AssertionError("cameras_from_spec must not open a device")


def test_cameras_from_spec_skips_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cameras, "_open", _no_open)
    cams = cameras_from_spec(rigspec.parse(SINGLE))
    assert [c.key for c in cams] == ["front"]
    assert cams[0].fields == {"type": "opencv", "index_or_path": 1, "width": 640,
                              "height": 480, "fps": 30}  # the note stays behind  # fmt: skip
    for c in cams:
        c.close()


def test_cameras_from_spec_names_per_arm_cameras_like_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cameras, "_open", _no_open)
    spec = rigspec.parse(BIMANUAL)
    cams = cameras_from_spec(spec)
    assert sorted(c.key for c in cams) == ["left_wrist", "right_wrist", "top"]
    feats = {c.feature.removeprefix("observation.images.") for c in spec.cameras}
    assert {c.key for c in cams} == feats
    assert {c.key: c.source for c in cams} == {"left_wrist": 1, "right_wrist": 2, "top": 0}
    for c in cams:
        c.close()


# -- the LeRobot path, on video files -------------------------------------------------------------
@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """64x48 red frames with a moving bar. 12000 of them, because LeRobot reads a file as fast
    as it decodes, with no pacing: 60 frames are gone before connect() returns."""
    import cv2

    path = tmp_path_factory.mktemp("video") / "red.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(12000):
        f = np.zeros((48, 64, 3), np.uint8)
        f[:, :, 2] = 200  # red, in OpenCV's BGR
        f[:, i % 60 : i % 60 + 4] = 255
        out.write(f)
    out.release()
    return path


@needs_cv2
def test_probe_on_a_video_file_and_a_missing_path(video: Path, tmp_path: Path) -> None:
    good, bad = probe([str(video), str(tmp_path / "missing.avi")])
    # A file refuses CAP_PROP_FRAME_WIDTH, so this also takes the native-size fallback.
    assert good["ok"], good["error"]
    assert (good["width"], good["height"], good["fps"]) == (64, 48, 30.0)
    assert good["thumbnail"].startswith(JPEG)
    assert not bad["ok"] and "Failed to open" in bad["error"] and bad["thumbnail"] is None


@needs_cv2
@ends_in_thread
def test_real_camera_plays_a_file_reports_its_end_and_reopens(video: Path) -> None:
    cam = RealCamera("front", {"type": "opencv", "index_or_path": str(video)}, retry_s=0.3)
    try:
        frame, _, seq = wait_for(lambda: frame_of(cam))
        assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
        r, g, b = (int(x) for x in np.median(frame.reshape(-1, 3), axis=0))
        assert r > 150 and b < 60  # RGB, not OpenCV's BGR
        wait_for(lambda: (f := frame_of(cam)) and f[2] > seq)
        err = wait_for(lambda: error_of(cam), timeout=10.0)
        assert str(err).startswith("camera front: ")
        _, _, seq2 = wait_for(lambda: frame_of(cam))  # the file "comes back": reopened
        assert seq2 > seq
    finally:
        cam.close()


@needs_cv2
@pytest.mark.skipif(importlib.util.find_spec("pyrealsense2") is not None,
                    reason="pyrealsense2 is installed; this test is for its absence")  # fmt: skip
def test_realsense_without_its_package_is_a_readable_error() -> None:
    cam = RealCamera("wrist", {"type": "intelrealsense", "serial_number_or_name": "123"})
    try:
        err = wait_for(lambda: (e := error_of(cam)) and "could not open" in str(e) and e)
        assert "camera wrist: could not open 123" in str(err) and "pyrealsense2" in str(err)
    finally:
        cam.close()
