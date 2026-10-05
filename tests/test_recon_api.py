"""Environment reconstruction in Studio: the cloud route's auth, the download behind control, the
depth thread (never asyncio's default executor), latest-wins, keep updating, and plain refusals.
A real worker (mock rig) streams the frames; the depth model is a fake that returns the inverse
depth of the synthetic scene in test_recon, so no model loads and nothing downloads."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("PIL")
from aiohttp.test_utils import TestServer  # noqa: E402
from test_recon import cast, fake_model  # noqa: E402

from phi_studio import depth_model, recon, recon_api, robot_model  # noqa: E402
from phi_studio.server import Studio  # noqa: E402

POSE = {"pos": [0.75, 0.12, 0.38], "target": [0.12, 0.0, 0.04], "up": [0, 0, 1], "fovy_deg": 45.0,
        "placed": True}  # fmt: skip
DATASET = Path.home() / (".cache/huggingface/lerobot/hub/datasets--BrutalCaesar--phi_so101_8bin_v1"
                         "/snapshots/34c7a026e27d4272911843fbc45af9a3d946c115")  # fmt: skip


class FakeModel:
    """Stands in for DepthModel: the synthetic scene's d = s/z + t at the requested size."""

    device = "fake"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.threads: list[str] = []
        self.calls = 0

    def __call__(self, rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        self.calls += 1
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        cam = recon.Pinhole.from_fovy(size[0], size[1], POSE["fovy_deg"])
        c2w = recon.look_at(POSE["pos"], POSE["target"], POSE["up"])
        return fake_model(cast(cam, c2w), 1.7, 0.35, 0.002)


class Recording(concurrent.futures.ThreadPoolExecutor):
    """asyncio's default executor, recording what is sent to it."""

    def __init__(self) -> None:
        super().__init__(max_workers=4)
        self.seen: list[str] = []

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        f = getattr(fn, "func", fn)
        self.seen.append(f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', f)}")
        return super().submit(fn, *args, **kwargs)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def started(cameras: list[str]) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": cameras}, port, token="t0k")
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


async def ws(session: aiohttp.ClientSession, port: int) -> Any:
    return await session.ws_connect(f"http://127.0.0.1:{port}/ws?token=t0k",
                                    headers={"Origin": f"http://127.0.0.1:{port}"})  # fmt: skip


async def until(sock: Any, pred: Any, timeout: float = 8.0) -> dict[str, Any]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = await sock.receive(timeout=0.2)
        except TimeoutError:
            continue
        if m.type == aiohttp.WSMsgType.TEXT:
            d = json.loads(m.data)
            if pred(d):
                return d
    raise AssertionError("condition not met in time")


async def frames(studio: Studio, key: str = "front", timeout: float = 8.0) -> None:
    end = time.monotonic() + timeout
    while key not in studio.latest_frame:
        assert time.monotonic() < end, "no frame from the mock camera"
        await asyncio.sleep(0.05)


def plain(text: str) -> bool:
    return "Traceback" not in text and "\u2014" not in text and "\u2013" not in text


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_cloud_route_needs_host_origin_and_token() -> None:
    async def go() -> None:
        studio, server, session = await started([])
        port = server.port
        recon_api.state_of(studio).clouds["abc"] = ("front", b"PCL1blob")
        url = f"http://127.0.0.1:{port}/api/recon/cloud/abc"
        try:
            cases = [({}, 403, "token"), ({"X-Phi-Token": "wrong"}, 403, "token"),
                     ({"X-Phi-Token": "t0k", "Host": f"attacker.example:{port}"}, 403, "host"),
                     ({"X-Phi-Token": "t0k", "Origin": "https://evil.example"}, 403, "origin"),
                     ({"X-Phi-Token": "té"}, 403, "token")]  # fmt: skip
            for headers, status, word in cases:
                async with session.get(url, headers=headers) as r:
                    assert r.status == status and word in await r.text(), headers
            async with session.get(url, headers={"X-Phi-Token": "t0k"}) as r:
                assert r.status == 200 and await r.read() == b"PCL1blob"
                assert r.headers["Cache-Control"] == "no-store"
            async with session.get(url, params={"token": "t0k"}) as r:
                assert r.status == 403  # the query is not where the token goes
            async with session.get(url.replace("abc", "nope"), headers={"X-Phi-Token": "t0k"}) as r:
                assert r.status == 404
        finally:
            await session.close()
            await server.close()

    run(go())


def fake_hub(cache: Path, calls: list[dict[str, Any]], slow: bool = False) -> Any:
    def download(repo: str, filename: str, **kw: Any) -> str:
        calls.append({"repo": repo, "filename": filename, **kw})
        bar = kw["tqdm_class"](total=1000, initial=0, desc=filename)
        with bar:
            for _ in range(1000 if slow else 3):
                bar.update(1)
                if slow:
                    time.sleep(0.005)
        snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / kw["revision"]
        snap.mkdir(parents=True, exist_ok=True)
        (snap / filename).write_bytes(b"x")
        return str(snap / filename)

    return download


def test_download_needs_control_and_is_pinned(tmp_path: Path) -> None:
    async def go() -> None:
        studio, server, session = await started([])
        st = recon_api.state_of(studio)
        st.cache_dir = tmp_path
        calls: list[dict[str, Any]] = []
        st.download.downloader = fake_hub(tmp_path, calls)
        try:
            first = await ws(session, server.port)
            second = await ws(session, server.port)
            await second.send_str(json.dumps({"cmd": "recon_download"}))
            err = await until(second, lambda m: m.get("type") == "error")
            assert err["message"] == "Another window has control." and not calls
            await first.send_str(json.dumps({"cmd": "recon_status"}))
            s = await until(first, lambda m: m.get("kind") == "status")
            assert s["model"]["cached"] is False and s["model"]["bytes"] == 99_173_660
            await first.send_str(json.dumps({"cmd": "recon_download"}))
            s = await until(first, lambda m: m.get("kind") == "status"
                            and m["download"]["state"] == "done")  # fmt: skip
            assert s["model"]["cached"] is True
            assert [c["filename"] for c in calls] == list(depth_model.FILES)
            for c in calls:
                assert c["repo"] == depth_model.REPO and c["revision"] == depth_model.REVISION
                assert c["token"] is False  # Studio never reads the person's token
        finally:
            await session.close()
            await server.close()

    run(go())


def test_download_cancel(tmp_path: Path) -> None:
    async def go() -> None:
        studio, server, session = await started([])
        st = recon_api.state_of(studio)
        st.cache_dir = tmp_path
        st.download.downloader = fake_hub(tmp_path, [], slow=True)
        try:
            first = await ws(session, server.port)
            await first.send_str(json.dumps({"cmd": "recon_download"}))
            await until(first, lambda m: m.get("kind") == "status"
                        and m["download"]["state"] == "running" and m["download"]["done"] > 0)
            await first.send_str(json.dumps({"cmd": "recon_download_cancel"}))
            s = await until(first, lambda m: m.get("kind") == "status"
                            and m["download"]["state"] == "cancelled")  # fmt: skip
            assert s["model"]["cached"] is False and 0 < s["download"]["done"] < 1000
        finally:
            await session.close()
            await server.close()

    run(go())


def test_live_capture_runs_on_the_depth_thread_only() -> None:
    async def go() -> None:
        loop = asyncio.get_running_loop()
        default = Recording()
        loop.set_default_executor(default)
        studio, server, session = await started(["front"])
        st = recon_api.state_of(studio)
        fake = FakeModel()
        st.model = fake
        try:
            sock = await ws(session, server.port)
            await frames(studio)
            await sock.send_str(json.dumps({"cmd": "recon_capture", "source": "live",
                                            "key": "front", "camera": "front",
                                            "poses": {"front": POSE}}))  # fmt: skip
            msg = await until(sock, lambda m: m.get("kind") in ("cloud", "refused"))
            assert msg["kind"] == "cloud", msg
            assert msg["fit"]["inlier_fraction"] > 0.8 and msg["fit"]["median_mm"] < 2
            assert msg["estimate"]["placement"] == "placed by you"
            assert "distortion" in msg["estimate"]["lens"]
            async with session.get(f"http://127.0.0.1:{server.port}{msg['url']}",
                                   headers={"X-Phi-Token": "t0k"}) as r:  # fmt: skip
                n, n_off, pos, col = recon.unpack(await r.read())
            assert n == msg["n"] and n_off == msg["n_off_arm"] and n <= recon.MAX_POINTS
            assert fake.threads and all(t.startswith(recon_api.DEPTH_THREAD) for t in fake.threads)
            ours = [s for s in default.seen if "phi_studio" in s]
            assert not ours, ours
        finally:
            await session.close()
            await server.close()
            default.shutdown()

    run(go())


def test_live_capture_without_frames_says_how_to_start() -> None:
    async def go() -> None:
        studio, server, session = await started([])
        recon_api.state_of(studio).model = FakeModel()
        try:
            sock = await ws(session, server.port)
            await sock.send_str(json.dumps({"cmd": "recon_capture", "source": "live",
                                            "key": "front", "camera": "front",
                                            "poses": {"front": POSE}}))  # fmt: skip
            msg = await until(sock, lambda m: m.get("kind") == "refused")
            assert "Start teleop or camera align" in msg["message"] and plain(msg["message"])
        finally:
            await session.close()
            await server.close()

    run(go())


def test_capture_without_the_model_says_download(tmp_path: Path) -> None:
    async def go() -> None:
        studio, server, session = await started(["front"])
        st = recon_api.state_of(studio)
        st.model = depth_model.DepthModel(cache_dir=tmp_path)  # an empty cache: not downloaded
        try:
            sock = await ws(session, server.port)
            await frames(studio)
            await sock.send_str(json.dumps({"cmd": "recon_capture", "source": "live",
                                            "key": "front", "camera": "front",
                                            "poses": {"front": POSE}}))  # fmt: skip
            msg = await until(sock, lambda m: m.get("kind") == "refused")
            assert msg["model_missing"] and "Download" in msg["message"] and plain(msg["message"])
        finally:
            await session.close()
            await server.close()

    run(go())


@pytest.mark.parametrize("req,words", [
    ({"source": "dataset", "root": "/nowhere/at/all", "episode": 0, "frame": 0,
      "key": "observation.images.front", "camera": "front"}, "No LeRobot dataset"),
    ({"source": "live", "key": "front", "camera": "front", "poses": {}}, "Place the front camera"),
    ({"source": "live", "key": "front", "camera": "front",
      "poses": {"front": {**POSE, "fovy_deg": 400}}}, "field of view"),
    ({"source": "elsewhere", "camera": "front"}, "Pick where the picture comes from"),
])  # fmt: skip
def test_refusals_are_plain(req: dict[str, Any], words: str) -> None:
    async def go() -> None:
        studio, server, session = await started(["front"])
        recon_api.state_of(studio).model = FakeModel()
        try:
            sock = await ws(session, server.port)
            await frames(studio)
            await sock.send_str(json.dumps({"cmd": "recon_capture", **req}))
            msg = await until(sock, lambda m: m.get("kind") == "refused", timeout=30)
            assert words in msg["message"] and plain(msg["message"]), msg
        finally:
            await session.close()
            await server.close()

    run(go())


def test_latest_wins_per_camera() -> None:
    async def go() -> None:
        studio, server, session = await started(["front"])
        st = recon_api.state_of(studio)
        fake = FakeModel(delay=0.3)
        st.model = fake
        try:
            await frames(studio)
            req = {"source": "live", "key": "front", "camera": "front", "poses": {"front": POSE}}
            first = asyncio.ensure_future(st.submit(recon_api.Job("front", req)))
            await asyncio.sleep(0.05)  # the first is running now
            second = asyncio.ensure_future(st.submit(recon_api.Job("front", req)))
            third = asyncio.ensure_future(st.submit(recon_api.Job("front", req)))
            a, b, c = await asyncio.gather(first, second, third)
            assert a["kind"] == "cloud" and b is None and c["kind"] == "cloud"
            assert fake.calls == 2
        finally:
            await session.close()
            await server.close()

    run(go())


def test_keep_updating_is_live_only_and_at_most_two_a_second() -> None:
    async def go() -> None:
        studio, server, session = await started(["front"])
        recon_api.state_of(studio).model = FakeModel()
        try:
            sock = await ws(session, server.port)
            await frames(studio)
            req = {"cmd": "recon_keep", "on": True, "key": "front", "camera": "front",
                   "poses": {"front": POSE}}  # fmt: skip
            await sock.send_str(json.dumps({**req, "source": "dataset"}))
            msg = await until(sock, lambda m: m.get("kind") == "refused")
            assert "live cameras only" in msg["message"]
            await sock.send_str(json.dumps({**req, "source": "live"}))
            stamps = []
            for _ in range(3):
                await until(sock, lambda m: m.get("kind") == "cloud")
                stamps.append(time.monotonic())
            gaps = np.diff(stamps)
            assert (gaps >= recon_api.KEEP_PERIOD_S * 0.8).all(), gaps
            await sock.send_str(json.dumps({"cmd": "recon_keep", "on": False}))
            await until(sock, lambda m: m.get("kind") == "status" and m["keep"] is None)
        finally:
            await session.close()
            await server.close()

    run(go())


@pytest.mark.skipif(not (DATASET.is_dir() and depth_model.cached()),
                    reason="needs phi_so101_8bin_v1 and the depth model in the local cache")
def test_real_frame_end_to_end_smoke() -> None:
    """Frame 60 of the front camera through the real model. NOT an accuracy claim: the true
    camera pose for this dataset is unknown. One placement is phi's starting guess for its own
    rig, whose table grid visibly lands on this picture's wall; the other was fitted by eye."""

    class Bare:
        frame_clock: dict[str, Any] = {}
        latest_frame: dict[str, Any] = {}
        telemetry = None
        loop = None

        def _fanout(self, m: dict[str, Any]) -> None:
            pass

    st = recon_api.Recon(Bare())  # type: ignore[arg-type]
    req = {"source": "dataset", "root": str(DATASET), "episode": 0, "frame": 60,
           "key": "observation.images.front"}  # fmt: skip
    guess = {**robot_model.CAMERA_DEFAULTS["front"], "placed": False}
    by_eye = {"pos": [0.75, 0.0, 0.15], "target": [0.0, 0.0, 0.1], "up": [0, 0, 1],
              "fovy_deg": 36.0, "placed": True}  # fmt: skip
    try:
        out = st.compute(recon_api.Job("front", {**req, "poses": {"front": guess}}))
        print("starting guess:", out.get("kind"), out.get("fit"), out.get("message"))
        assert out["kind"] == "refused" and plain(out["message"])
        out = st.compute(recon_api.Job("front", {**req, "poses": {"front": by_eye}}))
        print("by eye:", out.get("kind"), out.get("fit"), out.get("ms"))
        assert out["kind"] == "cloud" and out["n"] > 0
    finally:
        st.close()


def test_download_survives_tqdm_extras(tmp_path: Path) -> None:
    """huggingface_hub's Xet path calls set_postfix_str and friends, not only update()."""
    calls: list[dict[str, Any]] = []
    plain_fetch = fake_hub(tmp_path, calls)

    def xet_like(repo: str, filename: str, **kw: Any) -> str:
        bar = kw["tqdm_class"](total=10, initial=0, desc=filename)
        bar.update(1)
        bar.set_postfix_str("xet")
        bar.refresh()
        return plain_fetch(repo, filename, **kw)

    seen: list[dict[str, Any]] = []
    d = depth_model.Download(seen.append, cache_dir=tmp_path, downloader=xet_like)
    assert d.start()
    d.join(5)
    assert d.state == "done", d.error


def test_only_the_exact_wrist_key_rides_on_the_arm() -> None:
    assert recon_api._is_wrist("wrist")
    assert not recon_api._is_wrist("left_wrist") and not recon_api._is_wrist("front")
