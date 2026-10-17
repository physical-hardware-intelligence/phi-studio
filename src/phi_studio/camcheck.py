"""Which camera is which, each time Studio starts the real arms.

macOS hands OpenCV its camera numbers in an order that changes with USB ports, plug order, hub
power-up timing and reboots (phi/src/phi/utils/camera_realign.py: "DO NOT TRUST A WRITTEN-DOWN
INDEX ORDER"), and OpenCV exposes no name or serial that would pin them (2026-10-09: the wrist and
top cameras both report 1920x1080 at 30 fps). So Studio recognises its cameras by what they see:

  * Onboarding's Save cameras stores a fingerprint of every camera it probed (save_refs): the edge
    map of a 64x48 grey thumbnail, with the role it was given or none (the Mac's own camera).
  * Before the arm worker opens the cameras (Studio start, every rig reload), check() probes every
    index, matches the fingerprints and, when macOS renumbered them, rewrites only those numbers in
    robot-config.yaml (configedit, with a backup).

Measured on the rig, 2026-10-09: the same camera in two probes a minute apart scores 0.81-1.00;
twenty minutes later, with people and the arm moved, 0.53-0.83; two different cameras never more
than 0.15. A role is matched at MATCH or more with a LEAD over every rival; one role left over takes
the one camera left over (the wrist sees the most change: it moves with the arm). After a sure
check the references are refreshed, so they follow the room. When it is not sure, it says so and
changes nothing: recording and policies wait (server.py).
"""

from __future__ import annotations

import base64
import io
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

REF_FILE = "camera-refs.json"
SIZE = (64, 48)
MATCH = 0.35  # same camera, scene changed: 0.53+; different cameras: 0.15 at most
LEAD = 0.25
PROBE_RANGE = range(8)
RETRY_S = 1.5  # a camera just plugged in can fail its first read


def fingerprint(jpeg: bytes) -> np.ndarray:
    """Edge map of a 64x48 grey thumbnail, zero mean and unit length: the dot product of two is
    their normalised cross-correlation. WHY edges: plain grey cross-matched up to 0.39 on the rig,
    edges at most 0.09, and a change of light shifts grey levels more than edges."""
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32)
    y = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx, gy = np.zeros_like(y), np.zeros_like(y)
    gx[:, 1:-1], gy[1:-1, :] = y[:, 2:] - y[:, :-2], y[2:, :] - y[:-2, :]
    e = np.hypot(gx, gy).ravel()
    e -= e.mean()
    n = float(np.linalg.norm(e))
    return (e / n if n > 1e-6 else e).astype(np.float32)


def _encode(fp: np.ndarray) -> str:
    return base64.b64encode(fp.astype(np.float16).tobytes()).decode()


def _decode(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


def save_refs(data_dir: Path, thumbs: dict[Any, bytes], roles: dict[str, Any]) -> Path:
    """Store a fingerprint of every probed camera: `thumbs` source -> JPEG, `roles` role -> source.
    Cameras with no role are kept too, so the Mac's own camera is recognised and never taken for a
    missing one."""
    by_source = {str(src): role for role, src in roles.items()}
    cams = [{"source": src, "role": by_source.get(str(src)), "fp": _encode(fingerprint(j))}
            for src, j in thumbs.items()]  # fmt: skip
    path = data_dir / REF_FILE
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"saved": time.time(), "cameras": cams}, indent=1))
    tmp.replace(path)
    return path


def load_refs(data_dir: Path) -> list[dict[str, Any]] | None:
    try:
        data = json.loads((data_dir / REF_FILE).read_text())
        return [{**c, "fp": _decode(c["fp"])} for c in data["cameras"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def resolve(refs: list[dict[str, Any]], live: dict[Any, np.ndarray]) -> dict[str, Any]:
    """Match each role's reference to one live camera. Returns {"assign": {role: source}, "detail":
    {role: {"source", "score", "how"}}, "missing": [roles], "scores": {role: {source: score}}}."""
    roles = [r for r in refs if r.get("role")]
    others = [r for r in refs if not r.get("role")]
    srcs = list(live)
    score = {r["role"]: {s: float(np.dot(r["fp"], live[s])) for s in srcs} for r in roles}
    assign: dict[str, Any] = {}
    detail: dict[str, dict[str, Any]] = {}
    for role, row in score.items():
        if not row:
            continue
        best = max(row, key=row.__getitem__)
        rival_src = max((v for s, v in row.items() if s != best), default=-1.0)
        rival_role = max((score[o][best] for o in score if o != role), default=-1.0)
        if row[best] >= MATCH and row[best] - max(rival_src, rival_role) >= LEAD:
            assign[role] = best
            detail[role] = {"source": best, "score": round(row[best], 2), "how": "picture"}
    missing = [r["role"] for r in roles if r["role"] not in assign]
    if len(missing) == 1:
        # One role left, one camera left: by elimination, after the cameras with no role (the
        # Mac's own) have been recognised as themselves.
        known = {s for s in srcs for o in others if float(np.dot(o["fp"], live[s])) >= MATCH}
        left = [s for s in srcs if s not in assign.values() and s not in known]
        if len(left) == 1:
            role = missing.pop()
            assign[role] = left[0]
            detail[role] = {"source": left[0], "score": round(score[role][left[0]], 2),
                            "how": "elimination"}  # fmt: skip
    table = {role: {str(src): round(v, 2) for src, v in row.items()} for role, row in score.items()}
    return {"assign": assign, "detail": detail, "missing": missing, "scores": table}


Probe = Callable[[Iterable[int]], list[dict[str, Any]]]


def check(config: Path, data_dir: Path, probe: Probe | None = None) -> dict[str, Any]:
    """Probe every camera, recognise the config's ones and fix their numbers if macOS moved them.
    Returns {"status": same | renumbered | unsure | unverified | none, "message", "fix", ...}."""
    from phi_studio import configedit, rigspec

    spec = rigspec.parse(config.read_text())
    cams = [c for c in spec.cameras if c.fields.get("type") == "opencv"]
    if not cams:
        return {"status": "none", "message": "", "fix": ""}
    refs = load_refs(data_dir)
    if not refs:
        return {"status": "unverified",
                "message": "Studio has not seen these cameras yet, and macOS can renumber them at "
                           "any restart.",
                "fix": "Open Rig setup, Cameras, and save once, so Studio can recognise "
                       "them."}  # fmt: skip
    if probe is None:
        from phi_studio import cameras

        probe = cameras.probe
    rows = probe(PROBE_RANGE)
    # WHY again: 2026-10-09 the two cameras swapped a minute before failed their first read and
    # the check could not see them. "Failed to open" is an index with no camera: not retried.
    shy = [r["source"] for r in rows
           if not r.get("ok") and "Failed to open" not in str(r.get("error"))]  # fmt: skip
    if shy:
        time.sleep(RETRY_S)
        again = {r["source"]: r for r in probe(shy)}
        rows = [again.get(r["source"], r) for r in rows]
    thumbs = {r["source"]: r["thumbnail"] for r in rows if r.get("ok") and r.get("thumbnail")}
    live = {s: fingerprint(j) for s, j in thumbs.items()}
    res = resolve(refs, live)
    unknown = [c.key for c in cams if not any(r.get("role") == c.key for r in refs)]
    missing = sorted(set(res["missing"]) | set(unknown))
    # WHY the whole table: an unsure check must show why (every role against every camera)
    out: dict[str, Any] = {"detail": res["detail"], "seen": sorted(live, key=str),
                           "scores": res["scores"]}  # fmt: skip
    if missing:
        return {**out, "status": "unsure",
                "message": f"Studio could not tell which camera is the {', '.join(missing)} one.",
                "fix": "Open Rig setup, Cameras: check the pictures and save. Recording and "
                       "policies wait until then."}  # fmt: skip
    moves = {c.key: (c.fields.get("index_or_path"), res["assign"][c.key]) for c in cams
             if c.fields.get("index_or_path") != res["assign"][c.key]}  # fmt: skip
    if all(d["how"] == "picture" for d in res["detail"].values()):
        # WHY: the room changes (light, people, the arm); a sure match refreshes what it is
        # matched against. Not after an elimination: that view did not match its reference.
        save_refs(data_dir, thumbs, res["assign"])
    if not moves:
        return {**out, "status": "same", "message": "", "fix": ""}
    changes = {configedit.camera_source_path(c, spec): res["assign"][c.key]
               for c in cams if c.key in moves}  # fmt: skip
    backup = configedit.write_values(config, changes, data_dir / "config-backups")
    said = ", ".join(f"{k} #{a} → #{b}" for k, (a, b) in moves.items())
    return {**out, "status": "renumbered", "moves": {k: list(v) for k, v in moves.items()},
            "backup": str(backup),
            "message": f"macOS renumbered the cameras; Studio updated robot-config.yaml: {said}.",
            "fix": "Nothing to do. Before a policy runs, Align checks they still point where "
                   "they did."}  # fmt: skip
