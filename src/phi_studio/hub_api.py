"""The Models page's commands: Hugging Face login status, search, a model's fit with this rig, one
verified download at a time, the models on this Mac, and the lerobot-rollout command for one of
them. Loaded by server.register_features.

Any window may read (whoami, search, inspect, local, rollout command). Only the window with control
may download or cancel: a download writes gigabytes to this Mac's disk.

The token: hub.py never returns it, and nothing here reads it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from phi_studio import hub, rigspec
from phi_studio.rigspec import RigSpec

# [JUDGEMENT] The Hub limits whoami calls, and who is logged in changes only when the person runs
# `hf auth login`; the page has a Refresh for that.
WHOAMI_TTL_S = 300.0
# WHY a deadline here: list_models takes no timeout and huggingface_hub's client has none
# (utils/_http.py:304-312, utils/_pagination.py:27-36), so a stalled link would hang the reply.
CALL_TIMEOUT_S = 30.0
INSPECT_TIMEOUT_S = 60.0  # model_info plus config.json, each with hub.TIMEOUT_S
PROGRESS_EVERY_S = 0.2  # about 5 progress messages a second


class HubApi:
    def __init__(self, studio: Any, cache_dir: Path | None = None) -> None:
        self.studio = studio
        self.cache_dir = cache_dir  # None: the HF cache, which LeRobot reads too
        self.who: dict[str, Any] | None = None
        self.who_at = 0.0  # monotonic
        self.who_wall = 0.0  # unix seconds, shown as "checked at"
        self.who_lock: asyncio.Lock | None = None
        self.job: dict[str, Any] | None = None  # the newest download, running or finished
        self.cancel: threading.Event | None = None  # set while a download runs

    def register(self) -> None:
        s = self.studio
        s.handle("hub_whoami", self.whoami, control=False)
        s.handle("hub_search", self.search, control=False)
        s.handle("hub_inspect", self.inspect, control=False)
        s.handle("hub_local", self.local, control=False)
        s.handle("hub_rollout", self.rollout, control=False)
        s.handle("hub_download", self.download, control=True)
        s.handle("hub_cancel", self.cancel_download, control=True)

    # -- rig ------------------------------------------------------------------------------------
    def rig(self) -> tuple[RigSpec | None, dict[str, Any]]:
        """The rig as robot-config.yaml says, read from the file the Files page shows."""
        view = self.studio.files.lerobot()
        if view is None:
            return None, {"error": "Studio found no robot-config.yaml, so it cannot check the fit.",
                          "cameras": [], "joints": 0}  # fmt: skip
        root = self.studio.files.roots.get(view["file"]["root"])
        if "error" in view or root is None:
            return None, {"error": f"robot-config.yaml could not be read: {view.get('error')}",
                          "cameras": [], "joints": 0}  # fmt: skip
        try:
            spec = rigspec.parse((root.path / view["file"]["path"]).read_text())
        except (yaml.YAMLError, OSError) as e:
            return None, {"error": f"robot-config.yaml could not be read: {e}", "cameras": [],
                          "joints": 0}  # fmt: skip
        cams = []
        for c in spec.cameras:
            w, h = c.fields.get("width"), c.fields.get("height")
            cams.append({"key": c.feature, "name": c.feature.removeprefix(hub.IMAGES),
                         "size": f"{w}x{h}" if isinstance(w, int) and isinstance(h, int) else None,
                         "usable": c.source is not None})  # fmt: skip
        return spec, {"error": None, "cameras": cams, "joints": len(spec.action_features()),
                      "bimanual": spec.bimanual}  # fmt: skip

    @staticmethod
    def fit(info: dict[str, Any], spec: RigSpec | None, rename: dict[str, str] | None = None
            ) -> dict[str, Any]:  # fmt: skip
        """compatibility() split into what stops a run and what does not."""
        if spec is None:
            return {"problems": [], "warnings": [], "checked": False}
        lines = hub.compatibility(hub.as_seen_by_rig(info, rename or {}), spec)
        return {"problems": [x for x in lines if not x.startswith(hub.WARNING)],
                "warnings": [x.removeprefix(hub.WARNING) for x in lines
                             if x.startswith(hub.WARNING)], "checked": True}  # fmt: skip

    @staticmethod
    def proposal(info: dict[str, Any], spec: RigSpec | None) -> dict[str, str]:
        if spec is None:
            return {}
        rig = [c.feature for c in spec.cameras if c.source is not None]
        return hub.propose_rename_map(list(info.get("cameras") or {}), rig)

    def guessed_fit(self, info: dict[str, Any], spec: RigSpec | None) -> dict[str, Any]:
        """The fit after Studio's proposed camera mapping, with the guess stated first.
        WHY: checked before mapping, a model whose names differ shows only "map a camera" and
        hides what is still wrong once the names line up, such as a frame size."""
        rename = self.proposal(info, spec)
        out = self.fit(info, spec, rename)
        if rename:
            pairs = [f"`{r.removeprefix(hub.IMAGES)}` feeds `{m.removeprefix(hub.IMAGES)}`"
                     for r, m in rename.items()]  # fmt: skip
            out["warnings"].insert(0, f"Studio guesses that {hub.join_and(pairs)}, from the "
                                   "camera names and their order. Check it, and change it when "
                                   "you run the model.")  # fmt: skip
        return {**out, "rename_map": rename}

    # -- read-only commands ---------------------------------------------------------------------
    async def whoami(self, client: Any, msg: dict[str, Any]) -> None:
        if self.who_lock is None:
            self.who_lock = asyncio.Lock()
        async with self.who_lock:  # WHY: two windows opening the page make one Hub call, not two
            stale = time.monotonic() - self.who_at > WHOAMI_TTL_S
            if self.who is None or stale or msg.get("refresh") is True:
                try:
                    who = await asyncio.wait_for(asyncio.to_thread(hub.whoami), CALL_TIMEOUT_S)
                except TimeoutError:
                    late = (f"The Hugging Face Hub did not answer within {CALL_TIMEOUT_S:.0f} s. "
                            "Check this Mac's connection.")  # fmt: skip
                    who = {"user": None, "orgs": [], "error": late}
                self.who, self.who_at, self.who_wall = who, time.monotonic(), time.time()
        client.push({"type": "hub_whoami", **self.who, "checked_at": self.who_wall,
                     "login_cmd": hub.LOGIN_CMD})  # fmt: skip

    async def search(self, client: Any, msg: dict[str, Any]) -> None:
        q, s = msg.get("query"), msg.get("sort")
        query = q if isinstance(q, str) else ""
        sort = s if isinstance(s, str) else "downloads"
        reply: dict[str, Any] = {"type": "hub_search", "query": query, "results": [], "error": None}
        try:
            reply["results"] = await asyncio.wait_for(
                asyncio.to_thread(hub.search_models, query, 30, sort), CALL_TIMEOUT_S
            )
        except hub.HubError as e:
            reply["error"] = str(e)
        except TimeoutError:
            reply["error"] = (f"The Hugging Face Hub did not answer within {CALL_TIMEOUT_S:.0f} s. "
                              "Check this Mac's connection, then search again.")  # fmt: skip
        client.push(reply)

    async def inspect(self, client: Any, msg: dict[str, Any]) -> None:
        asked = str(msg.get("repo_id") or "")
        r = msg.get("revision")
        rev = r if isinstance(r, str) and r else None
        try:
            info = await asyncio.wait_for(
                asyncio.to_thread(hub.inspect_model, asked, rev), INSPECT_TIMEOUT_S
            )
        except hub.HubError as e:
            client.push({"type": "hub_model", "asked": asked, "error": str(e)})
            return
        except TimeoutError:
            client.push({"type": "hub_model", "asked": asked,
                         "error": "The Hugging Face Hub did not answer in time. Try again."})
            return
        spec, rig = await asyncio.to_thread(self.rig)
        here = await asyncio.to_thread(self._local_paths)
        client.push({"type": "hub_model", "asked": asked, "error": None, "info": info,
                     "rig": rig, **self.guessed_fit(info, spec),
                     "local": here.get((info["repo_id"], info["revision"]))})  # fmt: skip

    def _local_paths(self) -> dict[tuple[str, str], str]:
        return {(m["repo_id"], m["revision"]): m["path"]
                for m in hub.local_models(self.cache_dir) if m["has_weights"]}  # fmt: skip

    def _local(self) -> list[dict[str, Any]]:
        spec, _ = self.rig()
        out = []
        for m in hub.local_models(self.cache_dir):
            row = {**m, "size": hub.snapshot_size(m["path"]), "cameras": {}, "error": None,
                   "problems": [], "warnings": [], "rename_map": {}}  # fmt: skip
            try:
                info = hub.inspect_local(m["path"])
            except hub.HubError as e:
                row["error"] = str(e)
            else:
                row |= {"cameras": info["cameras"], "state_shape": info["state_shape"],
                        "action_shape": info["action_shape"],
                        **self.guessed_fit(info, spec)}  # fmt: skip
            out.append(row)
        return out

    async def local(self, client: Any, msg: dict[str, Any]) -> None:
        models = await asyncio.to_thread(self._local)
        _, rig = await asyncio.to_thread(self.rig)
        client.push({"type": "hub_local", "models": models, "rig": rig, "download": self.job})

    async def rollout(self, client: Any, msg: dict[str, Any]) -> None:
        """The lerobot-rollout command for a model already on this Mac, with the fit after the
        camera mapping. It only builds text; the page types it into the terminal."""
        reply: dict[str, Any] = {"type": "hub_rollout", "seq": msg.get("seq"), "cmd": None,
                                 "error": None, "problems": [], "warnings": []}  # fmt: skip
        spec, _ = await asyncio.to_thread(self.rig)
        rid, rev = msg.get("repo_id"), msg.get("revision")
        path = (await asyncio.to_thread(self._local_paths)).get((str(rid), str(rev)))
        rename = msg.get("rename_map") if isinstance(msg.get("rename_map"), dict) else {}
        if spec is None:
            reply["error"] = "Studio cannot read robot-config.yaml, so it cannot build the command."
        elif path is None:
            reply["error"] = f"{rid} at this revision is not on this Mac with its weights."
        else:
            try:
                info = await asyncio.to_thread(hub.inspect_local, path)
                reply |= self.fit(info, spec, rename)
                reply["cmd"] = rigspec.rollout_command(
                    spec, path, str(msg.get("task") or ""), _number(msg.get("duration_s"), 60),
                    str(msg.get("strategy") or "base"), msg.get("dataset_repo_id") or None, rename,
                    int(_number(msg.get("episodes"), 10)), msg.get("upload") is True,
                )  # fmt: skip
            except (ValueError, hub.HubError) as e:
                reply["error"] = str(e)
        client.push(reply)

    # -- download -------------------------------------------------------------------------------
    def _publish(self, **patch: Any) -> None:
        assert self.job is not None
        self.job = {**self.job, **patch}
        self.studio._fanout({"type": "hub_progress", **self.job})

    async def download(self, client: Any, msg: dict[str, Any]) -> None:
        if self.cancel is not None and self.job is not None:
            # WHY one at a time: two downloads would split the link and the disk check.
            client.push({"type": "error", "cmd": "hub_download",
                         "message": f"Another download is running: {self.job['repo_id']}.",
                         "fix": "Wait for it to finish, or cancel it first."})  # fmt: skip
            return
        rid = str(msg.get("repo_id") or "")
        rev = msg.get("revision") if isinstance(msg.get("revision"), str) else None
        cancel = threading.Event()
        self.cancel = cancel  # set before any await, so a second request sees it
        self.job = {"repo_id": rid, "revision": rev, "state": "running", "done": 0, "total": None,
                    "message": None, "path": None}  # fmt: skip
        self._publish()
        loop = asyncio.get_running_loop()
        last = 0.0

        def progress(done: int, total: int) -> None:  # runs on the download thread
            nonlocal last
            now = time.monotonic()
            if now - last < PROGRESS_EVERY_S and done < total:
                return
            last = now
            loop.call_soon_threadsafe(lambda: self._publish(done=done, total=total))

        try:
            snap = await asyncio.to_thread(hub.download, rid, rev, self.cache_dir, progress, cancel)
        except hub.Cancelled:
            self._publish(state="cancelled", message=f"Download of {rid} cancelled.")
        except hub.HubError as e:
            self._publish(state="error", message=str(e))
        except Exception as e:  # WHY all: a stuck "running" bar would block every later download
            self._publish(state="error", message=f"The download failed: {type(e).__name__}: {e}")
        else:
            self._publish(state="done", path=str(snap), message=f"{rid} is on this Mac.")
        finally:
            self.cancel = None
        models = await asyncio.to_thread(self._local)
        _, rig = await asyncio.to_thread(self.rig)
        self.studio._fanout({"type": "hub_local", "models": models, "rig": rig,
                             "download": self.job})  # fmt: skip

    async def cancel_download(self, client: Any, msg: dict[str, Any]) -> None:
        if self.cancel is None:
            client.push({"type": "error", "cmd": "hub_cancel", "message": "No download is running.",
                         "fix": ""})  # fmt: skip
            return
        self.cancel.set()
        self._publish(state="cancelling")


def _number(v: Any, default: float) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def register(studio: Any) -> None:
    studio.hub = HubApi(studio)
    studio.hub.register()
