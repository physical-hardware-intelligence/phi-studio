"""The 3D view's routes: model JSON and meshes, behind the same Host check as the rest of Studio."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestServer  # noqa: E402
from test_server import until, ws  # noqa: E402

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


def test_mesh_without_its_hash_is_not_cached_for_good() -> None:
    url = robot_model.model_json()["meshes"]["sts3215_03a_v1"]["url"]
    base = url.split("?")[0]
    for path in (base, f"{base}?v=0000000000000000"):
        status, headers, _ = asyncio.run(get(path))
        assert status == 200 and headers["Cache-Control"] == "no-cache", path


def test_web_test_copy_of_the_model_is_current() -> None:
    """web/tests/scene.test.ts checks the browser's kinematics on this copy. Regenerate it with
    json.dumps(robot_model.model_json(), indent=1) plus a newline, written to the path below."""
    copy = Path(__file__).parents[1] / "web" / "tests" / "fixtures" / "so101_model.json"
    assert json.loads(copy.read_text()) == json.loads(json.dumps(robot_model.model_json()))


# -- each arm's joint units ------------------------------------------------------------------------
CAL = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
       for i, j in enumerate(robot_model.JOINTS)}  # fmt: skip


def test_degrees_arm_needs_no_calibration() -> None:
    assert robot_model.arm_units(True, None) == {"unit": "degrees", "calibration": None,
                                                 "problem": None}  # fmt: skip


def test_m100_arm_carries_its_body_joint_ranges() -> None:
    u = robot_model.arm_units(False, CAL)
    assert u["unit"] == "m100" and u["problem"] is None
    assert set(u["calibration"]) == set(robot_model.ARM_JOINTS)
    assert u["calibration"]["elbow_flex"] == {"range_min": 1000, "range_max": 3000, "drive_mode": 0}


@pytest.mark.parametrize("cal", [None, {}, {"shoulder_pan": {"range_min": 5}}, "text",
                                 {j: {**c, "range_max": 10} for j, c in CAL.items()}])  # fmt: skip
def test_m100_arm_without_usable_ranges_says_why(cal: Any) -> None:
    u = robot_model.arm_units(False, cal)
    assert u["unit"] == "m100" and u["calibration"] is None
    assert "-100 to 100" in u["problem"] and "not drawn" in u["problem"]


def test_scene_arms_reads_units_from_the_rig_config(tmp_path: Path, monkeypatch: Any) -> None:
    rig = tmp_path / "rig"
    rig.mkdir()
    (rig / "robot-config.yaml").write_text(
        "robot: {type: so101_follower, id: f1, port: /dev/tty.usbmodemF, use_degrees: false}\n"
        "teleop: {type: so101_leader, id: l1, port: /dev/tty.usbmodemL}\n")  # fmt: skip
    home = tmp_path / "lerobot"
    cal = home / "calibration" / "robots" / "so_follower"
    cal.mkdir(parents=True)
    (cal / "f1.json").write_text(json.dumps(CAL))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))

    async def go() -> dict[str, Any]:
        port = free_port()
        studio = Studio({"kind": "mock", "pairs": 1, "cameras": []}, port, token="t0k",
                        data_dir=tmp_path / "data", rig_dir=rig)  # fmt: skip
        server = TestServer(studio.app(), host="127.0.0.1", port=port)
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as s:
                sock = await ws(s, port)
                await until(sock, lambda d: d["type"] == "hello")
                await sock.send_str(json.dumps({"cmd": "scene_arms"}))
                m: dict[str, Any] = await until(sock, lambda d: d["type"] == "scene_arms")
                return m
        finally:
            await server.close()

    arms = asyncio.run(go())["arms"]
    assert arms["follower"]["unit"] == "m100" and arms["follower"]["problem"] is None
    assert arms["follower"]["calibration"]["shoulder_pan"]["range_max"] == 3000
    assert arms["leader"] == {"unit": "degrees", "calibration": None, "problem": None}
