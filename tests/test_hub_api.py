"""The Models page's server commands, with hub.py's network calls replaced. No Hub, no LeRobot."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from phi_studio import hub, hub_api
from phi_studio.files import Files, Root

CONFIG = """\
robot:
  type: so101_follower
  id: f
  port: /dev/tty.F
teleop:
  type: so101_leader
  id: l
  port: /dev/tty.L
cameras:
  front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
  top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  wrist: {type: opencv, index_or_path: TBD, width: 640, height: 480, fps: 30}
"""
SHA = "c" * 40
IMG = "observation.images."


def _feat(kind: str, *shape: int) -> dict[str, Any]:
    return {"type": kind, "shape": list(shape)}


def _info(cams: tuple[str, ...], joints: int = 6) -> dict[str, Any]:
    inputs = {IMG + c: _feat("VISUAL", 3, 480, 640) for c in cams}
    inputs["observation.state"] = _feat("STATE", joints)
    return {"repo_id": "me/act", "revision": SHA, "policy_type": "act", "input_features": inputs,
            "output_features": {"action": _feat("ACTION", joints)},
            "cameras": {k: f["shape"] for k, f in inputs.items() if f["type"] == "VISUAL"}}


class FakeStudio:
    def __init__(self, root: Path) -> None:
        self.files = Files([Root("repo", "Phi checkout", root)])
        self.handlers: dict[str, tuple[Any, bool]] = {}
        self.fanned: list[dict[str, Any]] = []

    def handle(self, cmd: str, fn: Any, *, control: bool) -> None:
        self.handlers[cmd] = (fn, control)

    def _fanout(self, msg: dict[str, Any]) -> None:
        self.fanned.append(msg)


class Client:
    def __init__(self) -> None:
        self.pushed: list[dict[str, Any]] = []

    def push(self, msg: dict[str, Any]) -> None:
        self.pushed.append(msg)


@pytest.fixture
def api(tmp_path):
    rig = tmp_path / "rig"
    rig.mkdir()
    (rig / "robot-config.yaml").write_text(CONFIG)
    cache = tmp_path / "cache"
    cache.mkdir()
    studio = FakeStudio(rig)
    a = hub_api.HubApi(studio, cache_dir=cache)
    a.register()
    return a


def _snapshot(cache: Path, rid: str, cfg: dict[str, Any], weights: bool = True) -> Path:
    snap = cache / f"models--{rid.replace('/', '--')}" / "snapshots" / SHA
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(cfg))
    if weights:
        (snap / "model.safetensors").write_bytes(b"w" * 64)
    return snap


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_only_download_and_cancel_need_control(api):
    need = {cmd: control for cmd, (_, control) in api.studio.handlers.items()}
    assert need == {"hub_whoami": False, "hub_search": False, "hub_inspect": False,
                    "hub_local": False, "hub_rollout": False, "hub_download": True,
                    "hub_cancel": True}  # fmt: skip


def test_the_real_server_loads_the_hub_commands():
    pytest.importorskip("aiohttp")
    from phi_studio.server import Studio

    studio = Studio({"kind": "mock"}, 0, token="t")
    assert studio.handlers["hub_download"][1] is True and studio.handlers["hub_search"][1] is False


def test_whoami_is_cached_until_refresh_or_five_minutes(api, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "whoami", lambda: calls.append(1) or {"user": "parv", "orgs": [],
                                                                   "error": None})  # fmt: skip
    c = Client()
    run(api.whoami(c, {}))
    run(api.whoami(c, {}))
    assert len(calls) == 1 and c.pushed[-1]["user"] == "parv"
    assert c.pushed[-1]["login_cmd"] == "hf auth login"
    run(api.whoami(c, {"refresh": True}))
    assert len(calls) == 2
    api.who_at -= hub_api.WHOAMI_TTL_S + 1
    run(api.whoami(c, {}))
    assert len(calls) == 3


def test_search_errors_and_a_stalled_hub_reach_the_window(api, monkeypatch):
    def offline(*a: Any) -> Any:
        raise hub.HubError("Cannot search the Hugging Face Hub (no connection).")

    monkeypatch.setattr(hub, "search_models", offline)
    c = Client()
    run(api.search(c, {"query": "act"}))
    assert c.pushed[-1]["results"] == [] and c.pushed[-1]["query"] == "act"
    assert c.pushed[-1]["error"] == "Cannot search the Hugging Face Hub (no connection)."
    monkeypatch.setattr(hub, "search_models", lambda *a: time.sleep(0.5) or [])
    monkeypatch.setattr(hub_api, "CALL_TIMEOUT_S", 0.05)
    run(api.search(c, {"query": "act"}))
    assert "did not answer" in c.pushed[-1]["error"]


def test_inspect_splits_problems_from_warnings_and_proposes_a_mapping(api, monkeypatch):
    monkeypatch.setattr(hub, "inspect_model", lambda rid, rev: _info(("camera1", "camera2")))
    c = Client()
    run(api.inspect(c, {"repo_id": "https://huggingface.co/me/act"}))
    m = c.pushed[-1]
    assert m["type"] == "hub_model" and m["error"] is None and m["local"] is None
    assert len(m["problems"]) == 2 and all("your rig has" in p for p in m["problems"])
    assert all(not w.startswith(hub.WARNING) for w in m["warnings"]) and len(m["warnings"]) == 3
    # wrist has no device, so only front and top can be mapped, in the rig's order
    assert m["rename_map"] == {IMG + "front": IMG + "camera1", IMG + "top": IMG + "camera2"}
    assert [c["name"] for c in m["rig"]["cameras"]] == ["front", "top", "wrist"]
    assert m["rig"]["cameras"][2]["usable"] is False and m["rig"]["joints"] == 6


def test_inspect_error_is_one_message(api, monkeypatch):
    def gated(rid: str, rev: Any) -> Any:
        raise hub.HubError("me/act is gated.")

    monkeypatch.setattr(hub, "inspect_model", gated)
    c = Client()
    run(api.inspect(c, {"repo_id": "me/act"}))
    assert c.pushed == [{"type": "hub_model", "asked": "me/act", "error": "me/act is gated."}]


def test_local_lists_models_with_size_and_fit(api):
    cache = api.cache_dir
    _snapshot(cache, "me/act", {"type": "act", **{k: _info(("front", "top"))[k] for k in
                                                  ("input_features", "output_features")}})
    _snapshot(cache, "me/half", {"type": "act"}, weights=False)
    c = Client()
    run(api.local(c, {}))
    m = c.pushed[-1]
    assert m["type"] == "hub_local" and m["download"] is None
    rows = {r["repo_id"]: r for r in m["models"]}
    assert rows["me/act"]["problems"] == [] and rows["me/act"]["size"] > 64
    assert rows["me/act"]["has_weights"] and not rows["me/half"]["has_weights"]
    json.dumps(m)


def test_rollout_builds_the_command_for_a_local_model_only(api):
    snap = _snapshot(api.cache_dir, "me/smol", {"type": "smolvla", **{
        k: _info(("camera1",))[k] for k in ("input_features", "output_features")}})
    c = Client()
    base = {"repo_id": "me/smol", "revision": SHA, "task": "Pick up the cube", "duration_s": 20,
            "strategy": "base", "seq": 7}  # fmt: skip
    run(api.rollout(c, base))
    m = c.pushed[-1]
    assert m["seq"] == 7 and m["error"] is None and f"--policy.path={snap}" in m["cmd"]
    assert any("camera1" in p for p in m["problems"])  # not mapped yet
    run(api.rollout(c, {**base, "rename_map": {IMG + "front": IMG + "camera1"}}))
    m = c.pushed[-1]
    assert m["problems"] == [] and "--rename_map=" in m["cmd"]
    run(api.rollout(c, {**base, "strategy": "sentry"}))
    assert "owner/rollout_name" in c.pushed[-1]["error"] and c.pushed[-1]["cmd"] is None
    run(api.rollout(c, {**base, "repo_id": "me/other"}))
    assert "not on this Mac" in c.pushed[-1]["error"]


def test_one_download_at_a_time_progress_is_throttled_and_cancel_works(api, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def fake_download(rid, rev, cache_dir, progress, cancel):
        for i in range(2000):  # far more callbacks than the page should see
            progress(i, 2000)
        started.set()
        while not cancel.is_set() and not release.is_set():
            time.sleep(0.01)
        if cancel.is_set():
            raise hub.Cancelled(f"Download of {rid} cancelled.")
        progress(2000, 2000)
        return api.cache_dir / "snap"

    monkeypatch.setattr(hub, "download", fake_download)
    a, b = Client(), Client()

    async def go() -> None:
        first = asyncio.create_task(api.download(a, {"repo_id": "me/act", "revision": SHA}))
        await asyncio.to_thread(started.wait, 5)
        await api.download(b, {"repo_id": "me/other"})
        assert "Another download is running: me/act" in b.pushed[-1]["message"]
        await api.cancel_download(a, {})
        await first

    run(go())
    states = [m["state"] for m in api.studio.fanned if m["type"] == "hub_progress"]
    assert states[0] == "running" and states[-2:] == ["cancelling", "cancelled"]
    assert len(states) < 20  # 2000 callbacks inside 0.2 s collapse to a handful
    assert api.studio.fanned[-1]["type"] == "hub_local"
    # the slot is free again, and a clean run ends in done
    release.set()
    started.clear()
    run(api.download(a, {"repo_id": "me/act", "revision": SHA}))
    done = [m for m in api.studio.fanned if m["type"] == "hub_progress"][-1]
    assert done["state"] == "done" and done["done"] == done["total"] == 2000


def test_cancel_without_a_download_says_so(api):
    c = Client()
    run(api.cancel_download(c, {}))
    assert c.pushed[-1]["message"] == "No download is running."


def test_a_failed_download_frees_the_slot(api, monkeypatch):
    def boom(*a: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(hub, "download", boom)
    c = Client()
    run(api.download(c, {"repo_id": "me/act"}))
    last = [m for m in api.studio.fanned if m["type"] == "hub_progress"][-1]
    assert last["state"] == "error" and "disk full" in last["message"]
    assert api.cancel is None
