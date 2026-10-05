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
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
LICENSE = "Apache-2.0"
# The files from_pretrained reads, with the weights' size [RUN: the cached blob, 99,173,660 bytes].
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
WEIGHTS_BYTES = 99_173_660


class DepthUnavailable(ValueError):
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


class Cancelled(Exception):
    pass


Downloader = Callable[..., Any]


class Download:
    """One download of the model's files, in its own thread, with progress and cancel.

    WHY its own thread, not Studio's depth thread or asyncio's default executor: a 99 MB fetch
    must not hold up depth jobs, and the default executor is shared with the terminal.
    WHY cancel works over plain HTTP: hf_hub_download calls the progress bar's update() for every
    chunk it writes (huggingface_hub file_download.py http_get); update() raises once cancel is
    pressed, which ends the transfer, and a later download resumes from the .incomplete file.
    NOT VERIFIED for a file served through Xet (hf_xet is installed): that path reports progress
    from its own callback, and whether it passes the exception up is untested. Cancel is also
    checked between files."""

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
        self._thread: threading.Thread | None = None
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
        self._thread = threading.Thread(target=self._run, name="phi-depth-download", daemon=True)
        self._thread.start()
        self.on_change(self.view())
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _progress(self, n: int) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        self.done += n
        now = time.monotonic()
        # WHY throttle: chunks come many times a second; 4 pushes a second is plenty for a bar
        if now - self._last_push > 0.25:
            self._last_push = now
            self.on_change(self.view())

    def _run(self) -> None:
        job = self
        finished: list[int] = []  # bytes of the files already complete, so progress adds up

        class Bar:
            """The progress bar hf_hub_download builds per file. Not a tqdm: it prints nothing."""

            def __init__(self, **kw: Any) -> None:
                self.n = int(kw.get("initial") or 0)
                job.done = sum(finished) + self.n

            def __enter__(self) -> Bar:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def update(self, n: int = 1) -> None:
                self.n += n
                job._progress(n)

            def close(self) -> None:
                return None

            def __getattr__(self, name: str) -> Any:
                # WHY: huggingface_hub's Xet path (XetDownloadProgressReporter) also calls
                # set_postfix_str and other tqdm methods; a missing one failed every download.
                return lambda *a, **k: None

        try:
            if self.downloader is None:
                from huggingface_hub import hf_hub_download

                fetch: Downloader = hf_hub_download
            else:
                fetch = self.downloader
            for f in FILES:
                if self._cancel.is_set():
                    raise Cancelled()
                path = fetch(REPO, f, revision=REVISION, cache_dir=self.cache_dir, token=False,
                             tqdm_class=cast(Any, Bar))  # fmt: skip
                finished.append(Path(path).stat().st_size if Path(path).is_file() else 0)
            self.state = "done" if cached(self.cache_dir) else "failed"
            if self.state == "failed":
                self.error = "The files arrived but the cache does not show them. Try again."
        except Cancelled:
            self.state, self.error = "cancelled", None
        except Exception as e:  # WHY all: a network or disk error must reach the page, not a log
            self.state = "failed"
            self.error = f"The download failed: {type(e).__name__}: {e}"
        self.on_change(self.view())
