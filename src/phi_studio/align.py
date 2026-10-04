"""Camera align: put each camera back where it was when a dataset was recorded.

The rig is rebuilt every day, and a camera that moved since recording silently hurts every policy
trained on that dataset. Studio picks the dataset's resting frame (the arm at rest, the one pose
you can put back by hand), works out which live camera shows which dataset camera by picture (so
macOS renumbering camera indices stops mattering), and says per camera how far it is off and which
way to move it. The dataset side reuses phi_studio.camera_realign, so the CLI and Studio agree on
the reference frame and on which way is left.

WHY the fix is moving the camera, not warping the image in software. Shifting or warping the live
picture to line it up drops the pixels that slide off one edge and fills the other edge with black
(or smeared edge pixels) the policy never saw in training. A camera that only turned about its own
centre maps back by a homography, but that warp crops and pads the same way. A camera that moved
to a new position sees near things shift more than far things (parallax), so no single 2D warp
lines up the whole scene: it can match one plane, say the table top, and the arm and objects above
it stay off. So Studio measures the offset and the user moves the camera until it reads aligned.
[RUN] Checked with a pinhole model (f = 500 px): after a 3 degree turn a homography maps the old
view to the new one to 1e-4 px; after a 2 cm sideways move, points 0.3-1.5 m away shift 7-33 px
and the best homography still misses by 19 px, while points on one plane fit to 1e-4 px.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# [JUDGEMENT] Below this phaseCorrelate response the offset is not trusted. On real resting frames
# (phi_so101_8bin_v1, cubes_cylinder_v1) the same camera in one session read 0.11-0.21, a wrong
# camera 0.00-0.12, so the line sits between them and some readings near it go either way.
LOW_RESPONSE = 0.1
# Opening frames searched for the resting pose; camera_realign's --search default (3 s at 30 fps).
SEARCH_FRAMES = 90
# Matching compares gradient pictures computed at GRAD_SIZE and shrunk to MATCH_SIZE (w, h).
GRAD_SIZE = (128, 96)
MATCH_SIZE = (16, 12)
# [JUDGEMENT] Scores are correlations in -1..1. [RUN] Six live/reference pairings of those real
# frames, within one dataset and across datasets recorded on different days: a front or top camera
# scored 0.47-0.97 against its own reference, any camera at most 0.14 against another one's, and
# the wrist camera -0.10-0.30 against its own (what it sees at rest changes per episode), so the
# wrist is usually placed by elimination. Every assignment was right, ahead of the next-best by
# 0.45-1.18. Shifting the live pictures 48 px across and 24 px down kept all six right (lead >=
# 0.16); at 64 px and 32 px two of six went wrong. Match first, then line up by measure().
MIN_MATCH = 0.2
UNSURE_MARGIN = 0.1
MAX_CAMERAS = 8  # brute force over permutations: 8! = 40320 sums


class AlignError(ValueError):
    """A problem the user can act on, with a message Studio can show as is."""


@contextmanager
def _readable() -> Iterator[None]:
    """camera_realign is a CLI and exits on bad input; Studio needs an exception it can show.

    WHY catch SystemExit: a SystemExit escaping an aiohttp handler stops the server, not the
    request. A missing LeRobot, OpenCV or pandas becomes the same kind of readable error.
    """
    try:
        yield
    except SystemExit as e:
        raise AlignError(str(e.code)) from None
    except ImportError as e:
        raise AlignError(f"Camera align needs LeRobot's environment (LeRobot, OpenCV, pandas): "
                         f"{e}") from e  # fmt: skip


def _physical(mapping: dict[str, str], cameras: list[str]) -> dict[str, str]:
    """Dataset key -> the physical camera (wrist, front, top) it holds, where known."""
    return {key: name for name, key in mapping.items() if key in cameras}


def list_datasets() -> list[dict[str, Any]]:
    """Every local LeRobot dataset camera_realign can find, with its camera keys."""
    with _readable():
        from phi_studio import camera_realign as cr

        found = cr.discover()
    out = []
    for d in found:
        mapping, override = d.key_map
        physical = _physical(mapping, d.cameras)
        note = None
        if override:
            shows = ", ".join(f"{k} shows the {n} camera" for k, n in sorted(physical.items()))
            note = f"Camera keys are transposed in this dataset (matched '{override}'): {shows}."
        out.append(
            {"name": d.name, "repo_id": d.repo_id, "root": d.root, "episodes": d.episodes,
             "frames": d.frames, "task": d.task, "cameras": list(d.cameras),
             "physical": physical, "note": note}
        )  # fmt: skip
    return out


def _cache_entry(cache_dir: Path, root: str, episode: int, frame: int) -> Path:
    digest = hashlib.sha1(f"{root}\0{episode}\0{frame}".encode()).hexdigest()[:12]
    return cache_dir / f"ep{episode}_f{frame}_{digest}"


def _load_cached(entry: Path, root: str) -> dict[str, Any] | None:
    try:
        meta = json.loads((entry / "reference.json").read_text())
        if meta["root"] != root:
            return None
        images = {
            key: np.asarray(Image.open(entry / f).convert("RGB"), dtype=np.uint8)
            for key, f in meta["files"].items()
        }
    except (OSError, ValueError, KeyError):
        return None
    paths = {key: str(entry / f) for key, f in meta["files"].items()}
    return {**{k: meta[k] for k in ("root", "episode", "frame", "motion", "physical")},
            "images": images, "paths": paths, "cached": True}  # fmt: skip


def _save_cached(entry: Path, ref: dict[str, Any]) -> dict[str, str]:
    entry.mkdir(parents=True, exist_ok=True)
    files = {key: re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".png" for key in ref["images"]}
    for key, f in files.items():
        Image.fromarray(ref["images"][key]).save(entry / f)  # PNG: lossless, so offsets are exact
    meta = {k: ref[k] for k in ("root", "episode", "frame", "motion", "physical")}
    # WHY the JSON last, via rename: a crash mid-write leaves no JSON, so the entry is redone.
    tmp = entry / "reference.json.tmp"
    tmp.write_text(json.dumps({**meta, "files": files}, indent=1))
    os.replace(tmp, entry / "reference.json")
    return {key: str(entry / f) for key, f in files.items()}


def references(root: str, episode: int = 0, cache_dir: Path | None = None) -> dict[str, Any]:
    """The resting frame of one episode, decoded once per camera key as uint8 RGB.

    Returns root, episode, frame, motion, images {key: HxWx3}, physical {key: wrist|front|top}
    where known, paths {key: cached PNG} when cache_dir is given, and cached (True on a hit). For a
    Hub dataset whose videos are not all local, LeRobot downloads the episode's video files first.
    """
    with _readable():
        from phi_studio import camera_realign as cr

        want = os.path.realpath(root)
        ds = next((d for d in cr.discover() if os.path.realpath(d.root) == want), None)
        if ds is None:
            raise AlignError(f"No LeRobot dataset at {root} (looked under {cr.lerobot_home()}).")
        if not 0 <= episode < ds.episodes:
            raise AlignError(
                f"Episode {episode} is out of range: this dataset has episodes 0-{ds.episodes - 1}."
            )
        frame, motion = cr.resting_frame(ds, episode, SEARCH_FRAMES)

        entry = None
        if cache_dir is not None:
            entry = _cache_entry(Path(cache_dir), ds.root, episode, frame)
            hit = _load_cached(entry, ds.root)
            if hit is not None:
                return hit

        # WHY decode by dataset key, not physical name: matching is by picture onto the keys the
        # policy reads, and datasets use keys outside wrist/front/top (wrist_cam, up, side).
        class ByKey(cr.Dataset):
            @property
            def key_map(self) -> tuple[dict[str, str], str | None]:
                return {k: k for k in self.cameras}, None

        images = cr.reference_frames(ByKey(**dataclasses.asdict(ds)), ds.cameras, episode, frame)

    ref: dict[str, Any] = {
        "root": ds.root, "episode": episode, "frame": frame, "motion": motion,
        "physical": _physical(ds.key_map[0], ds.cameras), "images": images, "paths": {},
        "cached": False,
    }  # fmt: skip
    if entry is not None:
        ref["paths"] = _save_cached(entry, ref)
    return ref


def _hint(dx: float, dy: float, low: bool, aligned: bool, limit: float) -> str:
    if low:
        return ("Low match: wrong camera, the scene changed, or the camera is far off. "
                "Line it up roughly by eye against the reference picture first")  # fmt: skip
    if aligned:
        return f"Aligned: within {limit:g} px on both axes"
    # The picture must move by (-dx, -dy) to land on the reference: the same correction
    # camera_realign.annotate prints as "move image x -dx y -dy".
    parts = []
    if round(abs(dx)) >= 1:
        parts.append(f"{abs(dx):.0f} px {'left' if dx > 0 else 'right'}")
    if round(abs(dy)) >= 1:
        parts.append(f"{abs(dy):.0f} px {'up' if dy > 0 else 'down'}")
    return "Move the camera so the picture shifts " + " and ".join(parts)


def measure(live_rgb: np.ndarray, ref_rgb: np.ndarray) -> dict[str, Any]:
    """How far the live picture sits from the reference, in live-image pixels.

    dx > 0: the live scene sits dx px right of where the reference has it; dy > 0: dy px lower.
    Phase correlation sees translation only, so a camera that tilted or zoomed reads as a low
    response rather than as an offset.
    """
    with _readable():
        from phi_studio import camera_realign as cr
        from phi_studio.camera_backend import cv2

    live = cv2.cvtColor(live_rgb, cv2.COLOR_RGB2GRAY) if live_rgb.ndim == 3 else live_rgb
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY) if ref_rgb.ndim == 3 else ref_rgb
    h, w = live.shape
    note = None
    if ref.shape != live.shape:
        rh, rw = ref.shape
        note = f"Reference resized from {rw}x{rh} to {w}x{h} to match the live camera"
        if abs(rw / rh - w / h) > 0.01:
            note += "; the aspect ratios differ, so the offset is approximate"
        ref = cv2.resize(ref, (w, h))
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    dx, dy, resp = cr.offset(live, ref, window)
    # WHY clip: OpenCV's response runs a little past 1 for identical pictures and below 0 for
    # unrelated ones ([RUN] 1.07 and -0.02 on synthetic images).
    response = float(min(max(resp, 0.0), 1.0))
    low = response < LOW_RESPONSE
    # WHY aligned needs a trusted match: a wrong camera can put the correlation peak near 0, 0.
    aligned = not low and abs(dx) <= cr.ALIGNED_PX and abs(dy) <= cr.ALIGNED_PX
    return {
        "dx": round(float(dx), 2), "dy": round(float(dy), 2), "response": round(response, 3),
        "aligned": aligned, "low_match": low, "hint": _hint(dx, dy, low, aligned, cr.ALIGNED_PX),
        "size": [w, h], "note": note,
    }  # fmt: skip


def _signature(rgb: np.ndarray) -> np.ndarray:
    """A small zero-mean, unit-length gradient picture; the dot of two is their correlation.

    WHY gradient magnitude under zero-mean normalized correlation: correlation alone ignores a
    global brightness and contrast change (auto-exposure, a lamp switched on), and gradients drop
    the slow light falloff that would otherwise dominate the plain grey picture, so edges (table
    edge, base, box) decide. WHY gradients at 128x96, then shrunk to 16x12: the shrink blurs them,
    so a camera that moved by several percent of the frame still matches its own reference, and the
    rig's three views differ in gross layout, which 16x12 keeps. [RUN] On the real frames above,
    the worst lead of the right assignment over the next-best was 0.45 with this, 0.29 at 32x24,
    and 0.05 to 0.11 for plain grey pictures at 32x24 or 16x12.
    """
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    small = np.asarray(Image.fromarray(img).resize(GRAD_SIZE, Image.Resampling.BOX))
    gy, gx = np.gradient(small)
    mag = Image.fromarray(np.hypot(gx, gy).astype(np.float32))
    v = np.asarray(mag.resize(MATCH_SIZE, Image.Resampling.BOX), dtype=np.float64).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)  # a blank picture matches nothing


def match_cameras(live: dict[Any, np.ndarray], refs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Which dataset camera key each live camera shows, by picture.

    Returns assignment {live: key}; pairs [{live, ref, score, weak}], where weak means the picture
    looks like no reference and the pair stands only because the others fit better; margin, the
    best total score minus the next-best assignment's (None when there is no other); unsure and
    why; unmatched_live and unmatched_refs; and scores {live: {key: correlation}}.
    """
    lives, keys = list(live), list(refs)
    if not lives or not keys:
        raise AlignError("Need at least one live camera and one reference picture to match.")
    if max(len(lives), len(keys)) > MAX_CAMERAS:
        raise AlignError(f"Matching handles up to {MAX_CAMERAS} cameras a side.")
    sig_live = np.stack([_signature(live[k]) for k in lives])
    sig_ref = np.stack([_signature(refs[k]) for k in keys])
    # WHY einsum, not @: [RUN] in a process that had run OpenCV, `@` here (Apple Accelerate on
    # this Mac) printed divide-by-zero and overflow warnings on finite inputs, intermittently. Its
    # values were finite and equal to einsum's, and the same run with einsum printed none.
    s = np.einsum("ik,jk->ij", sig_live, sig_ref)  # s[i, j]: live i against reference j

    # Every one-to-one assignment of the smaller side into the larger, best total first.
    if len(lives) <= len(keys):
        options = [dict(zip(range(len(lives)), p, strict=True))
                   for p in itertools.permutations(range(len(keys)), len(lives))]  # fmt: skip
    else:
        options = [dict(zip(p, range(len(keys)), strict=True))
                   for p in itertools.permutations(range(len(lives)), len(keys))]  # fmt: skip
    totals = [float(sum(s[i, j] for i, j in o.items())) for o in options]
    order = sorted(range(len(options)), key=lambda n: -totals[n])
    best = options[order[0]]
    margin = totals[order[0]] - totals[order[1]] if len(order) > 1 else None

    pairs = [{"live": lives[i], "ref": keys[j], "score": round(float(s[i, j]), 3),
              "weak": bool(s[i, j] < MIN_MATCH)} for i, j in sorted(best.items())]  # fmt: skip
    why = None
    if margin is not None and margin < UNSURE_MARGIN:
        second = options[order[1]]
        torn = [str(lives[i]) for i in sorted(set(best) | set(second))
                if best.get(i) != second.get(i)]  # fmt: skip
        why = (f"Unsure about {', '.join(torn)}: placing them differently scores within "
               f"{margin:.2f}. Two cameras look alike, or one does not show its usual view.")
    elif all(p["weak"] for p in pairs):
        why = ("No camera looks like its reference picture. Put the arm at rest, and check the "
               "cameras are the rig's and are not covered.")  # fmt: skip
    return {
        "assignment": {p["live"]: p["ref"] for p in pairs},
        "pairs": pairs,
        "margin": None if margin is None else round(margin, 3),
        "unsure": why is not None,
        "why": why,
        "unmatched_live": [k for i, k in enumerate(lives) if i not in best],
        "unmatched_refs": [k for j, k in enumerate(keys) if j not in best.values()],
        "scores": {lk: {rk: round(float(s[i, j]), 3) for j, rk in enumerate(keys)}
                   for i, lk in enumerate(lives)},  # fmt: skip
    }
