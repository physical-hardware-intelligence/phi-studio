"""Real camera feeds for Studio, independent of whether the arms are mock or real.

Everything opens through LeRobot's own camera classes, built from the same fields robot-config.yaml
gives lerobot-record, so the preview fails, frames and rotates the way a recording would. This
module never touches cv2 (tests/test_no_raw_opencv.py).

WHY not phi.utils.camera_backend.open_camera: it forces BGR (camera_backend.py:110) and passes only
width, height, fps and fourcc, so a spec's rotation and backend would not apply; and its
`_Cam.read` turns every failure into (False, None) (camera_backend.py:64-68), which hides the error
a probe must report. Same rule as that module, though: if LeRobot does it, call LeRobot.

Two processes on one device: LeRobot's OpenCVCamera takes no lock and checks for no other user.
connect() builds a VideoCapture on index_or_path and raises ConnectionError when it does not open
(camera_opencv.py:156-165). So whether lerobot-record can open a camera Studio holds is the OS
backend's call (AVFoundation, V4L2, MSMF/DirectShow), and that is unverified here: no real device
is opened in tests. RealCamera.pause() releases the device so a LeRobot command can have it,
which is safe on every platform whatever the answer.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from phi.studio import rigspec

RETRY_S = 2.0  # wait between attempts to reopen a camera that failed
STALL_FRAMES = 30  # frame periods with no frame before a camera counts as gone (1 s at 30 fps)
DEFAULT_FPS = 30.0  # when the device reports none
THUMB = (160, 120)
# WHY: after a probe's deadline, releasing a camera whose read is blocked can take LeRobot up to
# 2 s (camera_opencv.py:487). The next candidate waits for that, so probes stay one at a time.
RELEASE_GRACE_S = 2.5


def _source(v: Any) -> Any:
    """An index typed as text ("2") is a device index, as `index_or_path: 2` is in YAML."""
    # WHY isdecimal, not isdigit: "²".isdigit() is True and int("²") raises.
    return int(v) if isinstance(v, str) and v.strip().isdecimal() else v


def _release(cam: Any) -> None:
    # WHY swallow: LeRobot raises DeviceNotConnectedError when there is nothing left to release
    # (camera_opencv.py:586-587); releasing must never be what fails.
    try:
        cam.disconnect()
    except Exception:
        pass


def _open(fields: dict[str, Any]) -> Any:
    """A connected LeRobot camera for one camera's config fields, delivering RGB. Raises what
    LeRobot raises: ConnectionError when the device does not open, RuntimeError when it will not
    take the size or fps asked (camera_opencv.py:240-244, 276-289)."""
    kind = str(fields.get("type", "opencv"))
    kw = {k: v for k, v in fields.items() if k != "type"}
    # WHY force RGB: the rig Camera protocol is RGB and the worker JPEG-encodes it as RGB.
    kw["color_mode"] = "rgb"
    cam: Any
    if kind == "opencv":
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        kw["index_or_path"] = _source(kw.get("index_or_path"))
        cam = OpenCVCamera(OpenCVCameraConfig(**kw))
    elif kind == "intelrealsense":
        # WHY delegated but unverified: RealSenseCamera has the same connect / async_read /
        # disconnect / fps surface (camera_realsense.py:162, 565, 672), but no device and no
        # pyrealsense2 here. Without the package its constructor raises ImportError
        # (camera_realsense.py:119, import_utils.py:86-96), which the tile shows as the reason.
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        kw["serial_number_or_name"] = str(kw.get("serial_number_or_name"))
        cam = RealSenseCamera(RealSenseCameraConfig(**kw))
    else:
        raise ValueError(f"camera type {kind!r} is not one Studio can open "
                         "(opencv, intelrealsense)")  # fmt: skip
    try:
        cam.connect(warmup=False)
    except BaseException:
        # WHY: a size or fps the device refuses raises after VideoCapture opened, and connect()
        # does not release it (camera_opencv.py:158-168); left open, it holds the device.
        _release(cam)
        raise
    return cam


def _thumbnail(frame: np.ndarray) -> bytes:
    from PIL import Image

    img = Image.fromarray(frame)
    img.thumbnail(THUMB)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _probe_one(src: Any, deadline: float) -> dict[str, Any]:
    out: dict[str, Any] = {"source": src, "ok": False, "width": None, "height": None,
                           "fps": None, "error": None, "thumbnail": None}  # fmt: skip
    cam = None
    try:
        try:
            cam = _open({"type": "opencv", "index_or_path": src, "width": 640, "height": 480})
        except RuntimeError:
            # WHY: it opened but refused 640x480 (camera_opencv.py:276-289); at its own size it is
            # still a camera worth listing.
            cam = _open({"type": "opencv", "index_or_path": src})
        # WHY only what is left: the opens spent part of the budget, and a frame that comes after
        # the deadline would be reported as a hang instead of as LeRobot's own timeout.
        frame = cam.async_read(timeout_ms=max(0.0, deadline - time.monotonic()) * 1000)
        fps = getattr(cam, "fps", None)
        out.update(ok=True, width=int(frame.shape[1]), height=int(frame.shape[0]),
                   fps=float(fps) if fps else None, thumbnail=_thumbnail(frame))  # fmt: skip
    except Exception as e:
        out["error"] = str(e) or type(e).__name__
    finally:
        if cam is not None:
            _release(cam)
    return out


def probe(indices: Iterable[int | str] = range(6), timeout_s: float = 3.0) -> list[dict[str, Any]]:
    """Try each camera once: open at 640x480 (else its own size), grab one frame, release.
    Returns {source, ok, width, height, fps, error, thumbnail (small JPEG bytes)} per candidate and
    never raises. A device Studio is already streaming may report not ok on an OS that refuses a
    second open; pause the cameras first.

    WHY one at a time: several uncompressed streams starting together can overrun a USB 2.0 bus
    (camera_backend.py:100), which would report a good camera as broken. Worst case is
    len(indices) * (timeout_s + RELEASE_GRACE_S), only when devices hang."""
    return [_probe_timed(_source(raw), timeout_s) for raw in indices]


def _probe_timed(src: Any, timeout_s: float) -> dict[str, Any]:
    box: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    # WHY a thread: an open that hangs in the driver must not freeze Studio. If it ever returns,
    # _probe_one's finally still releases the device.
    t = threading.Thread(target=lambda: box.append(_probe_one(src, deadline)),
                         name=f"probe {src}", daemon=True)  # fmt: skip
    t.start()
    t.join(timeout_s + RELEASE_GRACE_S)
    if box:
        return box[0]
    return {"source": src, "ok": False, "width": None, "height": None, "fps": None,
            "error": f"no answer from {src} in {timeout_s:g} s: hung, or held by another program",
            "thumbnail": None}  # fmt: skip


class RealCamera:
    """One camera from robot-config.yaml, read through LeRobot. Satisfies rig.Camera.

    Nothing opens until the first read_latest() (or resume()), so building one never touches a
    device. A capture thread then owns the device: it opens it, takes each new frame, and on a
    failure or a stall longer than `stall_frames` frame periods drops it and retries every
    `retry_s` seconds. read_latest() only reads what that thread stored, so it never waits on the
    device.
    """

    def __init__(
        self,
        key: str,
        fields: dict[str, Any],
        *,
        opener: Callable[[dict[str, Any]], Any] | None = None,
        retry_s: float = RETRY_S,
        stall_frames: int = STALL_FRAMES,
    ) -> None:
        self.key = key
        self.fields = dict(fields)
        src = self.fields.get(rigspec.CAMERA_SOURCE.get(str(self.fields.get("type")), ""))
        self.source = "?" if src is None else src
        self.retry_s = retry_s
        self.stall_frames = stall_frames
        self._opener = opener or _open
        self._lock = threading.Lock()
        self._wake = threading.Event()  # pause, resume and close interrupt any wait
        self._released = threading.Event()  # set by the thread once a pause has freed the device
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0  # frames delivered since construction; keeps counting across reopens
        self._error: str | None = None
        self._paused = False
        self._closed = False

    # -- rig.Camera ------------------------------------------------------------------------------
    def read_latest(self) -> tuple[np.ndarray, float, int]:
        """(RGB uint8 HxWx3, arrival time on time.monotonic, seq). Raises ConnectionError while
        the camera is opening, failed, paused or closed."""
        with self._lock:
            if self._closed:
                raise ConnectionError(f"camera {self.key}: closed")
            if self._paused:
                raise ConnectionError(f"camera {self.key}: released for a LeRobot command")
            self._start()
            frame, t, seq, err = self._frame, self._t, self._seq, self._error
        if err:
            raise ConnectionError(err)
        if frame is None:
            raise ConnectionError(f"camera {self.key}: opening {self.source}")
        return frame, t, seq

    def close(self) -> None:
        """Stop the capture thread and release the device. Final."""
        with self._lock:
            self._closed = True
            t = self._thread
        self._wake.set()
        if t is not None:
            t.join(timeout=3.0)  # LeRobot's own stop waits up to 2 s (camera_opencv.py:487)

    # -- sharing the device with LeRobot ---------------------------------------------------------
    def pause(self, timeout_s: float = 5.0) -> bool:
        """Release the device so lerobot-record or lerobot-teleoperate can open it. True once it
        is released; False if the capture thread is still stuck in the driver after timeout_s."""
        with self._lock:
            self._paused = True
            self._released.clear()
            running = self._thread is not None and self._thread.is_alive()
        if not running:
            return True
        self._wake.set()
        return self._released.wait(timeout_s)

    def resume(self) -> None:
        """Take the device back: reopen now, not after the retry interval."""
        with self._lock:
            if self._closed:
                return
            self._paused = False
            self._start()
        self._wake.set()

    # -- capture thread --------------------------------------------------------------------------
    def _start(self) -> None:
        """Caller holds the lock."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"camera {self.key}",
                                            daemon=True)  # fmt: skip
            self._thread.start()

    def _wait(self, timeout: float | None) -> None:
        # WHY clearing after the wait is safe: every flag is written before _wake.set(), and the
        # loop re-reads the flags after each wait.
        self._wake.wait(timeout)
        self._wake.clear()

    def _fail(self, why: str, cam: Any = None) -> None:
        # WHY the error goes up before the release: releasing a camera whose read is blocked can
        # take LeRobot up to 2 s (camera_opencv.py:487), and until the error is set read_latest
        # would keep serving the last frame as if it were live.
        with self._lock:
            self._error, self._frame = f"camera {self.key}: {why}", None
        if cam is not None:
            _release(cam)
        self._wait(self.retry_s)

    def _run(self) -> None:
        cam: Any = None
        period = last = 0.0
        got = False  # a frame since this open
        # WHY a first-frame allowance: LeRobot itself gives a new camera warmup_s (default 1 s)
        # to deliver its first frame (camera_opencv.py:170-177, configuration_opencv.py:64).
        first_s = float(self.fields.get("warmup_s") or 1)
        while True:
            with self._lock:
                closed, paused = self._closed, self._paused
            if closed or paused:
                if cam is not None:
                    _release(cam)
                    cam = None
                if closed:
                    self._released.set()  # a pause() racing close() must not wait it out
                    return
                with self._lock:
                    self._frame, self._error = None, None  # never show a pre-pause frame
                self._released.set()
                self._wait(None)
                continue
            if cam is None:
                try:
                    cam = self._opener(self.fields)
                except Exception as e:
                    self._fail(f"could not open {self.source}: {e}")
                    continue
                period = 1.0 / float(getattr(cam, "fps", None) or DEFAULT_FPS)
                last, got = time.monotonic(), False
            try:
                frame = cam.async_read(timeout_ms=1000 * min(2 * period, 0.2))
            except TimeoutError:  # no new frame yet (camera_opencv.py:526-530)
                quiet = time.monotonic() - last
                limit = self.stall_frames * period
                if quiet > (limit if got else max(limit, first_s)):
                    self._fail(f"no frame from {self.source} for {quiet:.1f} s", cam)
                    cam = None
                continue
            except Exception as e:
                # WHY: LeRobot's read loop tolerates 11 failed reads in a row and dies on the
                # next; async_read then raises RuntimeError (camera_opencv.py:464-469, 523-524).
                self._fail(f"{self.source} stopped delivering: {e}", cam)
                cam = None
                continue
            # WHY stamp on arrival: LeRobot stamps frames with time.perf_counter
            # (camera_opencv.py:454), not the time.monotonic the rig protocol uses.
            last, got = time.monotonic(), True
            with self._lock:
                self._frame, self._t, self._error = frame, last, None
                self._seq += 1


def cameras_from_spec(spec: rigspec.RigSpec) -> list[RealCamera]:
    """One RealCamera per usable camera in robot-config.yaml. Each is keyed like its dataset
    feature: a per-arm camera of a bimanual rig carries its side (left_wrist, bi_so_follower.py:
    91-98). Placeholders such as TBD are left out, by the same rule the LeRobot commands use."""
    return [RealCamera(c.feature.removeprefix("observation.images."), c.lerobot_fields())
            for c in spec.cameras if rigspec._usable(c)]  # fmt: skip
