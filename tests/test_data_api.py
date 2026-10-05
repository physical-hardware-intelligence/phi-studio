"""The data API end to end on a real Studio: auth on every route, datasets, analysis cache, episode
payload,
video Range requests, the 3D model, and notes over the WebSocket."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("pyarrow")
pytest.importorskip("PIL")
from aiohttp.test_utils import TestServer  # noqa: E402
from support.lerobot_data import write_v3  # noqa: E402

from phi_studio.server import Studio  # noqa: E402

TOKEN = "t0k"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Rig:
    def __init__(self, tmp: Path) -> None:
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.studio = Studio(
            {"kind": "mock", "pairs": 1, "cameras": []},
            self.port,
            token=TOKEN,
            data_dir=tmp / "data",
        )
        self.server = TestServer(self.studio.app(), host="127.0.0.1", port=self.port)

    async def __aenter__(self) -> Rig:
        await self.server.start_server()
        self.http = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.http.close()
        await self.server.close()

    async def get(
        self, path: str, token: str | None = TOKEN, **headers: str
    ) -> aiohttp.ClientResponse:
        if token is not None:
            headers.setdefault("X-Studio-Token", token)
        return await self.http.get(self.base + path, headers=headers)


@pytest.fixture
def lerobot_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "lerobot"
    write_v3(home / "lab" / "cubes", n_eps=3)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    monkeypatch.delenv("PHI_STUDIO_DATA_ROOTS", raising=False)
    return home


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_every_route_needs_the_token_host_and_origin(tmp_path, lerobot_home) -> None:
    async def main() -> None:
        async with Rig(tmp_path) as r:
            assert (await r.get("/api/data/datasets", token=None)).status == 403
            assert (await r.get("/api/data/datasets", token="wrong")).status == 403
            assert (await r.get("/api/data/datasets")).status == 200
            assert (
                await r.http.get(f"{r.base}/api/data/datasets?token={TOKEN}")
            ).status == 200  # <video> style
            assert (await r.get("/api/data/datasets", Host="evil.example:80")).status == 403
            assert (await r.get("/api/data/datasets", Origin="http://evil.example")).status == 403
            assert (await r.get("/api/data/datasets", Origin=r.base)).status == 200
            assert (await r.get("/api/model/so101/model.json", token=None)).status == 403

    run(main())


def test_datasets_detail_analysis_and_episode(tmp_path, lerobot_home) -> None:
    async def main() -> None:
        async with Rig(tmp_path) as r:
            d = await (await r.get("/api/data/datasets")).json()
            assert [s["repo_id"] for s in d["datasets"]] == ["lab/cubes"]
            s = d["datasets"][0]
            assert s["episodes"] == 3 and s["size"] > 0 and s["analysis"] is None
            did = s["id"]

            det = await (await r.get(f"/api/data/{did}")).json()
            assert [e["length"] for e in det["episodes"]] == [90, 105, 120] and det[
                "excluded"
            ] == []

            t0 = time.monotonic()
            a1 = await (await r.get(f"/api/data/{did}/analysis")).json()
            first = time.monotonic() - t0
            assert len(a1["episodes"]) == 3 and a1["version"] >= 1
            cache = list((tmp_path / "data" / "cache" / "analysis").glob(f"{did}-*.json"))
            assert len(cache) == 1
            t0 = time.monotonic()
            a2 = await (await r.get(f"/api/data/{did}/analysis")).json()
            assert a2["episodes"] == a1["episodes"] and time.monotonic() - t0 <= first + 0.5

            d2 = await (await r.get("/api/data/datasets")).json()
            assert d2["datasets"][0]["analysis"][
                "episodes_by_health"
            ]  # the list now carries health

            ep = await (await r.get(f"/api/data/{did}/episode/1")).json()
            assert len(ep["t"]) == len(ep["state"]) == len(ep["action"]) == 105
            assert ep["episode"]["videos"]["observation.images.top"]["from"] == pytest.approx(3.0)
            assert set(ep["analysis"]["series"]) >= {"vel", "track", "err", "tcp"}
            assert ep["limits"]["shoulder_pan"] == [-110.0, 110.0]
            assert all("key" in f and f["dismissed"] is False for f in ep["analysis"]["flags"])

            assert (await r.get(f"/api/data/{did}/episode/99")).status == 404
            assert (await r.get("/api/data/nope")).status == 404

    run(main())


def test_video_and_model(tmp_path, lerobot_home) -> None:
    async def main() -> None:
        async with Rig(tmp_path) as r:
            did = (await (await r.get("/api/data/datasets")).json())["datasets"][0]["id"]
            path = f"/api/data/{did}/video/observation.images.top/0/0.mp4"
            assert (await r.get(path)).status == 404  # the fixture writes no video
            assert (
                await r.get(f"/api/data/{did}/video/observation.images.nope/0/0.mp4")
            ).status == 404
            mp4 = (
                lerobot_home
                / "lab"
                / "cubes"
                / "videos"
                / "observation.images.top"
                / "chunk-000"
                / "file-000.mp4"
            )
            mp4.parent.mkdir(parents=True)
            mp4.write_bytes(bytes(range(256)) * 64)
            resp = await r.http.get(
                f"{r.base}{path}?token={TOKEN}", headers={"Range": "bytes=100-199"}
            )
            assert resp.status == 206 and len(await resp.read()) == 100

            model = await (await r.get("/api/model/so101/model.json")).json()
            assert model["joints"][0] == "shoulder_pan" and model["tcp_site"] == "gripperframe"
            blob = await (await r.get("/api/model/so101/meshes.bin")).read()
            assert (
                len(blob)
                == (Path(__file__).parents[1] / "src/phi_studio/assets/so101/meshes.bin")
                .stat()
                .st_size
            )
            assert (await r.get("/api/model/so101/..%2fserver.py")).status == 404

    run(main())


def test_notes_over_the_websocket(tmp_path, lerobot_home) -> None:
    async def main() -> None:
        async with Rig(tmp_path) as r:
            did = (await (await r.get("/api/data/datasets")).json())["datasets"][0]["id"]
            ws = await r.http.ws_connect(f"{r.base}/ws?token={TOKEN}", headers={"Origin": r.base})
            other = await r.http.ws_connect(
                f"{r.base}/ws?token={TOKEN}", headers={"Origin": r.base}
            )

            async def until(sock: Any, kind: str) -> dict[str, Any]:
                end = time.monotonic() + 5
                while time.monotonic() < end:
                    m = await sock.receive(timeout=5)
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d.get("type") == kind:
                            return d
                raise AssertionError(f"no {kind}")

            # a window without control may still write notes; every window hears the change
            await other.send_str(
                json.dumps(
                    {
                        "cmd": "note_save",
                        "ref": "x",
                        "note": {
                            "dataset": did,
                            "episode": 1,
                            "kind": "issue",
                            "text": "jaw slips",
                            "t0": 2.5,
                        },
                    }
                )
            )
            saved = await until(other, "note_saved")
            assert saved["note"]["text"] == "jaw slips" and saved["ref"] == "x"
            assert (await until(ws, "notes_changed"))["dataset"] == did

            await ws.send_str(json.dumps({"cmd": "notes_list", "dataset": did, "episode": 1}))
            items = (await until(ws, "notes"))["items"]
            assert [n["text"] for n in items] == ["jaw slips"]

            await ws.send_str(
                json.dumps(
                    {"cmd": "note_save", "note": {"dataset": did, "kind": "note", "text": ""}}
                )
            )
            err = await until(ws, "error")
            assert err["cmd"] == "note_save" and "Write what you noticed" in err["message"]

            await ws.send_str(
                json.dumps(
                    {"cmd": "note_save", "note": {"dataset": did, "episode": 2, "kind": "bad"}}
                )
            )
            await until(ws, "note_saved")
            assert (await (await r.get(f"/api/data/{did}")).json())["excluded"] == [2]

            ep = await (await r.get(f"/api/data/{did}/episode/0")).json()
            if ep["analysis"]["flags"]:
                f = ep["analysis"]["flags"][0]
                await ws.send_str(
                    json.dumps(
                        {
                            "cmd": "flag_dismiss",
                            "dataset": did,
                            "episode": 0,
                            "kind": f["kind"],
                            "key": f["key"],
                        }
                    )
                )
                await until(ws, "notes_changed")
                ep2 = await (await r.get(f"/api/data/{did}/episode/0")).json()
                assert ep2["analysis"]["flags"][0]["dismissed"] is True

            await ws.send_str(
                json.dumps({"cmd": "note_delete", "id": saved["note"]["id"], "dataset": did})
            )
            await until(ws, "notes_changed")
            await ws.send_str(json.dumps({"cmd": "notes_list", "dataset": did, "episode": 1}))
            assert (await until(ws, "notes"))["items"] == []
            await ws.close()
            await other.close()

    run(main())
