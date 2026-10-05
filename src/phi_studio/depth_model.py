"""The depth model behind environment reconstruction, and its one-time download.

Depth Anything V2 Small (relative), the Hugging Face transformers port, pinned to one revision.
Apache-2.0. It predicts affine-invariant inverse depth (arXiv 2406.09414, Sections 3 and 7.2):
never depth, never metric. phi_studio.recon turns it metric with the table.

Studio never downloads it silently. Loading uses local_files_only=True, so a computer without it
gets a clear "not downloaded" answer, and the download happens only when the person with control
presses the button, through hf_hub_download at the pinned revision with token=False: the model is
public, and Studio never reads, shows or logs a Hugging Face token.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, cast

import numpy as np

from phi_studio.errors import Refusal

REPO = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
LICENSE = "Apache-2.0"
# The files from_pretrained reads, with the weights' size [RUN: the cached blob, 99,173,660 bytes].
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
WEIGHTS_BYTES = 99_173_660


class DepthUnavailable(Refusal):
    """The model cannot run here. The message says why in plain words."""


def cached(cache_dir: Path | None = None) -> bool:
    """Whether every file the model needs is in the Hugging Face cache at the pinned revision.
    Reads the cache only; never touches the network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for f in FILES:
        p = try_to_load_from_cache(REPO, f, revision=REVISION, cache_dir=cache_dir)
        if not isinstance(p, str) or not Path(p).is_file():
            return False
    return True


def pick_device() -> str:
    """MPS on an Apple GPU, else CPU. [RUN] 640x480: 75 ms a frame on MPS, 352 ms on CPU."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class DepthModel:
    """Loaded on first use, in whatever thread calls it; Studio calls it only from its one depth
    thread, so the model is never used from two threads at once."""

    def __init__(self, cache_dir: Path | None = None, device: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not cached(self.cache_dir):
            raise DepthUnavailable("The depth model is not on this computer yet. Download it "
                                   "first (99 MB, Apache-2.0).")  # fmt: skip
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise DepthUnavailable(f"Depth needs torch and transformers in Studio's Python "
                                   f"environment: {e}") from e  # fmt: skip
        kw: dict[str, Any] = {"revision": REVISION, "local_files_only": True}
        if self.cache_dir is not None:
            kw["cache_dir"] = str(self.cache_dir)
        self.device = self.device or pick_device()
        self._processor = AutoImageProcessor.from_pretrained(REPO, **kw)
        self._model = AutoModelForDepthEstimation.from_pretrained(REPO, **kw).to(self.device)
        self._model.eval()

    def __call__(self, rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """The model's output for one RGB uint8 picture, resized to size = (width, height):
        (height, width) float32, larger = nearer, in the model's own arbitrary units."""
        self.load()
        import torch
        from PIL import Image

        inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs).predicted_depth  # (1, h', w'), the processor's size
            out = torch.nn.functional.interpolate(out[:, None], size=(size[1], size[0]),
                                                  mode="bilinear", align_corners=False)  # fmt: skip
        return cast(np.ndarray, out[0, 0].float().cpu().numpy())


Downloader = Callable[..., Any]
DOWNLOAD_THREAD = "phi-depth-download"


class Download:
    """One download of the model's files, on its own one-thread executor, with progress and cancel.

    The same pattern as the Models page (hub_api.HubApi.fetcher, hub.download): its own executor,
    a threading.Event to cancel, and hub._silent_bar as the progress bar.
    WHY its own executor, not Studio's depth thread or asyncio's default one: a 99 MB fetch must
    not hold up depth jobs, and the default executor is shared with the rest of Studio.
    WHY cancel works over plain HTTP: hf_hub_download calls the bar's update() for every chunk it
    writes; on_bytes raises once cancel is pressed, which ends the transfer, and a later download
    resumes from the .incomplete file. On the Xet path the exception may come back wrapped, or only
    when the file is done, so cancel is checked first on any error and again between files.
    NOT VERIFIED against the network: Xet cancel has only been exercised with a fake downloader."""

    def __init__(self, on_change: Callable[[dict[str, Any]], None],
                 cache_dir: Path | None = None,
                 downloader: Downloader | None = None) -> None:  # fmt: skip
        self.on_change = on_change
        self.cache_dir = cache_dir
        self.downloader = downloader
        self.state = "idle"  # idle | running | done | cancelled | failed
        self.done = 0
        self.total = WEIGHTS_BYTES
        self.error: str | None = None
        self._cancel = threading.Event()
        self._pool = ThreadPoolExecutor(1, thread_name_prefix=DOWNLOAD_THREAD)
        self._future: Future[None] | None = None
        self._last_push = 0.0

    def view(self) -> dict[str, Any]:
        return {"state": self.state, "done": self.done, "total": self.total, "error": self.error}

    @property
    def running(self) -> bool:
        return self.state == "running"

    def start(self) -> bool:
        if self.running:
            return False
        self._cancel.clear()
        self.state, self.done, self.error = "running", 0, None
        self._future = self._pool.submit(self._run)
        self.on_change(self.view())
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self._cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def join(self, timeout: float | None = None) -> None:
        if self._future is not None:
            wait([self._future], timeout)

    def _run(self) -> None:
        from phi_studio import hub

        finished = 0  # bytes of the files already complete

        def on_bytes(n: int) -> None:
            self.done += n
            now = time.monotonic()
            # WHY throttle: chunks come many times a second; 4 pushes a second is plenty for a bar
            if now - self._last_push > 0.25:
                self._last_push = now
                self.on_change(self.view())
            if self._cancel.is_set():
                raise hub.Cancelled("Download cancelled.")

        try:
            if self.downloader is None:
                from huggingface_hub import hf_hub_download

                fetch: Downloader = hf_hub_download
            else:
                fetch = self.downloader
            for f in FILES:
                if self._cancel.is_set():
                    raise hub.Cancelled("Download cancelled.")
                try:
                    path = fetch(REPO, f, revision=REVISION, cache_dir=self.cache_dir, token=False,
                                 tqdm_class=cast(Any, hub._silent_bar(on_bytes)))  # fmt: skip
                except Exception:
                    if self._cancel.is_set():
                        raise hub.Cancelled("Download cancelled.") from None
                    raise
                finished += Path(path).stat().st_size if Path(path).is_file() else 0
                self.done = finished  # WHY: retries and Range resets can skew the summed chunks
            self.state = "done" if cached(self.cache_dir) else "failed"
            if self.state == "failed":
                self.error = "The files arrived but the cache does not show them. Try again."
        except hub.Cancelled:
            self.state, self.error = "cancelled", None
        except Exception as e:  # WHY all: a network or disk error must reach the page, not a log
            self.state = "failed"
            # WHY the pattern only, not hub._scrub: that reads the stored token to blank it, and
            # this download never reads one (token=False sends none).
            msg = f"The download failed: {type(e).__name__}: {e}"
            self.error = hub._TOKEN_RE.sub("<token>", msg)
        self.on_change(self.view())
