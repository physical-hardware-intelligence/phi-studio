"""The 3D view's routes: model JSON and meshes, behind the same Host check as the rest of Studio."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestServer  # noqa: E402

from phi_studio import robot_model  # noqa: E402
from phi_studio.server import Studio  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def get(path: str, host: str | None = None) -> tuple[int, dict[str, str], bytes]:
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": []}, port, token="t0k")
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as s:
            headers = {"Host": host.format(port=port)} if host else {}
            async with s.get(f"http://127.0.0.1:{port}{path}", headers=headers) as r:
                return r.status, dict(r.headers), await r.read()
    finally:
        await server.close()


def test_model_json_needs_no_token() -> None:
    status, headers, body = asyncio.run(get("/api/scene/model"))
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    j = json.loads(body)
    assert j["joint_order"] == list(robot_model.JOINTS) and len(j["geoms"]) == 19
    assert j["mapping"]["gripper"]["unit"] == "0..100"


def test_mesh_is_the_stl_and_caches() -> None:
    url = robot_model.model_json()["meshes"]["sts3215_03a_v1"]["url"]
    status, headers, body = asyncio.run(get(url))
    path = robot_model.mesh_file("sts3215_03a_v1")
    assert path is not None
    assert status == 200 and body == path.read_bytes()
    assert headers["Content-Type"] == "model/stl" and "immutable" in headers["Cache-Control"]


@pytest.mark.parametrize("name", ["nope", "..%2Fso101_new_calib_camera.xml", "..%2F..%2Fserver.py",
                                  "sts3215_03a_v1.stl", "LICENSE"])  # fmt: skip
def test_only_model_meshes_are_served(name: str) -> None:
    status, _, _ = asyncio.run(get(f"/api/scene/mesh/{name}"))
    assert status == 404


@pytest.mark.parametrize("path", ["/api/scene/model", "/api/scene/mesh/sts3215_03a_v1"])
def test_foreign_host_is_refused(path: str) -> None:
    status, _, body = asyncio.run(get(path, host="attacker.example:{port}"))
    assert status == 403 and b"host" in body
