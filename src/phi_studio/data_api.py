"""Datasets, episodes, analysis, video and notes for the web app.

Bulk data goes over HTTP (an episode's arrays, a video file with Range requests, the 3D model),
because a <video> element and fetch() stream it far better than the control WebSocket. Notes go over
the WebSocket, so every window hears a change at once.

Every HTTP route here checks the same three things as the WebSocket (server.py:311-319): an allowed
Host (DNS rebinding), an allowed Origin when the browser sends one, and the per-launch token. A
<video> cannot send headers, so the token may come as ?token=. Nothing here can move an arm.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import logging
import math
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from aiohttp import web

from phi_studio import analysis, kinematics
from phi_studio.datasets import Dataset, Library, default_roots
from phi_studio.errors import Refusal
from phi_studio.notes import NoteStore

if TYPE_CHECKING:
    from phi_studio.server import Client, Studio

log = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent / "assets" / "so101"
MODEL_FILES = {"model.json": "application/json", "meshes.bin": "application/octet-stream"}
MEMO = 24  # whole-dataset analyses kept in memory (each a few hundred kB)


class DataAPI:
    def __init__(self, studio: Studio) -> None:
        self.studio = studio
        rig_dir = getattr(studio, "rig_dir", None)
        self.library = Library(default_roots(Path(rig_dir) if rig_dir else None))
        data_dir = studio.data_dir
        self.cache_dir = data_dir / "cache" / "analysis" if data_dir else None
        self.notes = NoteStore(data_dir / "studio.db") if data_dir else None
        self.author = getpass.getuser()
        self._memo: dict[tuple[str, float], dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # -- auth ------------------------------------------------------------------------------------
    def refuse(self, request: web.Request) -> str | None:
        s = self.studio
        if request.host not in s.allowed_hosts:
            return f"host {request.host!r} not allowed"
        origin = request.headers.get("Origin")
        if origin is not None and origin not in s.allowed_origins:
            return "origin not allowed"
        tok = request.headers.get("X-Studio-Token") or request.query.get("token", "")
        if not secrets.compare_digest(tok.encode(), s.token.encode()):
            return "bad or missing token"
        return None

    def guarded(self, fn: Any) -> Any:
        async def handler(request: web.Request) -> web.StreamResponse:
            why = self.refuse(request)
            if why:
                return web.Response(status=403, text=why)
            try:
                return await fn(request)
            except KeyError as e:
                return web.json_response({"error": str(e).strip("'")}, status=404)
            except Refusal as e:
                return web.json_response({"error": str(e), "fix": e.fix}, status=400)

        return handler

    @staticmethod
    def packed(data: Any) -> web.Response:
        resp = web.json_response(
            data,
            dumps=lambda o: json.dumps(
                _finite(o), separators=(",", ":"), allow_nan=False, default=_jsonable
            ),
        )
        resp.enable_compression()
        return resp

    # -- analysis cache --------------------------------------------------------------------------
    def stamp(self, ds: Dataset) -> float:
        return (ds.root / "meta" / "info.json").stat().st_mtime

    def remember(self, key: tuple[str, float], value: dict[str, Any]) -> None:
        """Keep at most MEMO analyses in memory, least recently used out first; the disk cache keeps
        all."""
        self._memo.pop(key, None)
        self._memo[key] = value
        while len(self._memo) > MEMO:
            self._memo.pop(next(iter(self._memo)))

    def cached(self, ds: Dataset) -> dict[str, Any] | None:
        key = (ds.id, self.stamp(ds))
        if key in self._memo:
            self.remember(key, self._memo[key])
            return self._memo[key]
        if self.cache_dir is not None:
            p = self.cache_dir / f"{ds.id}-{int(key[1])}-v{analysis.VERSION}.json"
            if p.is_file():
                try:
                    self.remember(key, json.loads(p.read_text()))
                    return self._memo[key]
                except (OSError, ValueError):
                    pass  # a torn or old cache file: compute again
        return None

    async def analysed(self, ds: Dataset) -> dict[str, Any]:
        hit = self.cached(ds)
        if hit is not None:
            return hit
        lock = self._locks.setdefault(ds.id, asyncio.Lock())
        async with lock:  # one computation per dataset, however many windows ask
            hit = self.cached(ds)
            if hit is not None:
                return hit
            started = time.monotonic()
            result = await asyncio.to_thread(analysis.analyse_dataset, ds)
            result["computed_s"] = round(time.monotonic() - started, 2)
            key = (ds.id, self.stamp(ds))
            self.remember(key, result)
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                p = self.cache_dir / f"{ds.id}-{int(key[1])}-v{analysis.VERSION}.json"
                tmp = p.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(_finite(result), separators=(",", ":"), default=_jsonable)
                )
                tmp.replace(p)
            return result

    def dismissed_marks(self, ds_id: str, flags: list[dict[str, Any]], episode: int) -> None:
        if self.notes is None:
            return
        gone = self.notes.dismissed(ds_id)
        for f in flags:
            f["key"] = NoteStore.flag_key(f)
            f["dismissed"] = (episode, f["kind"], f["key"]) in gone

    # -- routes ----------------------------------------------------------------------------------
    async def datasets(self, request: web.Request) -> web.Response:
        found = await asyncio.to_thread(self.library.scan, request.query.get("refresh") == "1")
        out = []
        for ds in found:
            s = ds.summary()
            s["size"] = await asyncio.to_thread(self.library.size, ds)
            a = self.cached(ds)
            s["analysis"] = (
                None
                if a is None
                else {
                    "health": a["health"],
                    "flag_counts": a["flag_counts"],
                    "episodes_by_health": _count([e["health"] for e in a["episodes"]]),
                }
            )
            s["notes"] = len(self.notes.list(ds.id, status="open")) if self.notes else 0
            out.append(s)
        out.sort(key=lambda s: -s["modified"])
        return self.packed({"datasets": out, "roots": [str(r) for r in self.library.roots]})

    async def dataset(self, request: web.Request) -> web.Response:
        ds = self.library.get(request.match_info["id"])
        eps = await asyncio.to_thread(ds.episodes)
        tasks = await asyncio.to_thread(ds.tasks)
        summary = ds.summary()
        summary["size"] = await asyncio.to_thread(self.library.size, ds)
        return self.packed(
            {
                "summary": summary,
                "tasks": tasks,
                "arms": ds.arms(),
                "names": ds.names("action")
                if "action" in ds.features
                else ds.names("observation.state"),
                "episodes": [
                    {k: e[k] for k in ("index", "length", "duration", "tasks")} for e in eps
                ],
                "excluded": self.notes.excluded(ds.id) if self.notes else [],
            }
        )

    async def dataset_analysis(self, request: web.Request) -> web.Response:
        ds = self.library.get(request.match_info["id"])
        result = await self.analysed(ds)
        for e in result["episodes"]:
            self.dismissed_marks(ds.id, e["flags"], e["index"])
        return self.packed(result)

    async def episode(self, request: web.Request) -> web.Response:
        ds = self.library.get(request.match_info["id"])
        index = int(request.match_info["ep"])
        meta = ds.episode_meta(index)
        frames = await asyncio.to_thread(ds.frames, index)
        result = await asyncio.to_thread(
            analysis.analyse_episode, frames, ds.fps, ds.arms(), meta["tasks"], meta["videos"], True
        )
        whole = self.cached(ds)
        if whole is not None:  # dataset-level flags (outliers) only exist in the whole-dataset pass
            for e in whole["episodes"]:
                if e["index"] == index:
                    result["flags"] += [f for f in e["flags"] if f["kind"] == "outlier"]
        self.dismissed_marks(ds.id, result["flags"], index)
        state = frames.get("state", frames.get("action"))
        action = frames.get("action", state)
        return self.packed(
            {
                "dataset": {
                    "id": ds.id,
                    "name": ds.root.name,
                    "repo_id": ds.repo_id,
                    "fps": ds.fps,
                    "episodes": len(ds.episodes()),
                },
                "names": ds.names("action")
                if "action" in ds.features
                else ds.names("observation.state"),
                "arms": ds.arms(),
                "episode": {k: meta[k] for k in ("index", "length", "duration", "tasks", "videos")},
                "t": np.round(frames["timestamp"], 4).tolist(),
                "state": np.round(np.asarray(state), 3).tolist(),
                "action": np.round(np.asarray(action), 3).tolist(),
                "analysis": result,
                "limits": kinematics.limits_deg(),
            }
        )

    async def video(self, request: web.Request) -> web.StreamResponse:
        ds = self.library.get(request.match_info["id"])
        key = request.match_info["key"]
        if key not in ds.cameras:
            raise KeyError(f"no camera {key} in this dataset")
        path = ds.video_path(key, int(request.match_info["chunk"]), int(request.match_info["file"]))
        if not path.is_file():
            raise KeyError(f"video file missing: {path.name}")
        # WHY FileResponse: it answers Range requests, which a <video> needs to seek inside a file
        # that holds many episodes.
        return web.FileResponse(
            path, headers={"Cache-Control": "private, max-age=3600", "Content-Type": "video/mp4"}
        )

    async def model(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in MODEL_FILES:
            raise KeyError(f"no model file {name}")
        return web.FileResponse(
            MODEL_DIR / name,
            headers={"Cache-Control": "private, max-age=86400", "Content-Type": MODEL_FILES[name]},
        )

    # -- notes over the WebSocket ------------------------------------------------------------------
    def _need_notes(self) -> NoteStore:
        if self.notes is None:
            raise Refusal("Notes need a data folder.", "Start Studio with --data-dir.")
        return self.notes

    async def notes_list(self, client: Client, msg: dict[str, Any]) -> None:
        store = self._need_notes()
        items = await asyncio.to_thread(
            store.list, msg.get("dataset"), msg.get("episode"), msg.get("status")
        )
        client.push(
            {
                "type": "notes",
                "dataset": msg.get("dataset"),
                "episode": msg.get("episode"),
                "items": items,
            }
        )

    async def note_save(self, client: Client, msg: dict[str, Any]) -> None:
        store = self._need_notes()
        given = msg.get("note")
        if not isinstance(given, dict):
            raise Refusal("A note must be an object.")
        note = await asyncio.to_thread(store.save, given, self.author)
        client.push({"type": "note_saved", "note": note, "ref": msg.get("ref")})
        self.studio._fanout({"type": "notes_changed", "dataset": note["dataset"]})

    async def note_delete(self, client: Client, msg: dict[str, Any]) -> None:
        store = self._need_notes()
        nid = msg.get("id")
        if not isinstance(nid, str):
            raise Refusal("Which note? The request had no id.")
        await asyncio.to_thread(store.delete, nid)
        self.studio._fanout({"type": "notes_changed", "dataset": msg.get("dataset")})

    async def flag_dismiss(self, client: Client, msg: dict[str, Any]) -> None:
        store = self._need_notes()
        ds, ep, kind, key = msg.get("dataset"), msg.get("episode"), msg.get("kind"), msg.get("key")
        if not (
            isinstance(ds, str)
            and isinstance(ep, int)
            and isinstance(kind, str)
            and isinstance(key, str)
        ):
            raise Refusal("A dismissal needs a dataset, episode, kind and key.")
        await asyncio.to_thread(
            store.dismiss, ds, ep, kind, key, self.author, bool(msg.get("undo"))
        )
        self.studio._fanout({"type": "notes_changed", "dataset": ds})


def _finite(o: Any) -> Any:
    """NaN and infinity become null: JSON has no spelling for them, and a metric that cannot be
    computed (the smoothness of an arm that never moves) is unknown, not a number."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def _count(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def register(studio: Studio) -> None:
    api = DataAPI(studio)
    studio.data_api = api  # type: ignore[attr-defined]
    g = api.guarded
    studio.routes += [
        web.get("/api/data/datasets", g(api.datasets)),
        web.get("/api/data/{id}", g(api.dataset)),
        web.get("/api/data/{id}/analysis", g(api.dataset_analysis)),
        web.get("/api/data/{id}/episode/{ep:\\d+}", g(api.episode)),
        web.get("/api/data/{id}/video/{key}/{chunk:\\d+}/{file:\\d+}.mp4", g(api.video)),
        web.get("/api/model/so101/{name}", g(api.model)),
    ]
    # WHY control=False for writes: a note changes no rig state and no job, and two people reviewing
    # the same dataset from two windows should both be able to write. Every write is broadcast to
    # all windows.
    studio.handle("notes_list", api.notes_list, control=False)
    studio.handle("note_save", api.note_save, control=False)
    studio.handle("note_delete", api.note_delete, control=False)
    studio.handle("flag_dismiss", api.flag_dismiss, control=False)

    async def close() -> None:
        if api.notes is not None:
            api.notes.close()

    studio.on_close.append(close)
