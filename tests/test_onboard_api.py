"""Onboarding: calibration ids on disk, ports.local.sh hints, the config it writes (read back by
rigspec), its refusals, and rewrites that back up the old file and keep its cameras."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from phi_studio import onboard_api as O
from phi_studio import rigspec
from phi_studio.errors import Refusal


def touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")


@pytest.fixture
def cal_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "calibration"
    for n in ("phi_follower", "phi_bi_follower_left", "phi_bi_follower_right", "lonely_left"):
        touch(root / "robots" / "so_follower" / f"{n}.json")
    for n in ("phi_leader", "phi_bi_leader_left", "phi_bi_leader_right"):
        touch(root / "teleoperators" / "so_leader" / f"{n}.json")
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(root))
    return root


def bi_answers(**over):
    a = {
        "name": "Bench A",
        "layout": "bimanual",
        "ids": {"follower": "phi_bi_follower", "leader": "phi_bi_leader"},
        "ports": {
            "left_leader": "/dev/tty.usbmodemA1",
            "left_follower": "/dev/tty.usbmodemA2",
            "right_leader": "/dev/tty.usbmodemB1",
            "right_follower": "/dev/tty.usbmodemB2",
        },
    }
    a.update(over)
    return a


def test_slug() -> None:
    assert (
        O.slug("Bench A") == "bench_a" and O.slug("  ") == "rig" and O.slug("Φ-lab #2") == "lab_2"
    )


def test_known_ids(cal_root) -> None:
    k = O.known_ids(cal_root)
    assert [r["id"] for r in k["bi_follower"]] == ["phi_bi_follower"]  # lonely_left has no _right
    assert [r["id"] for r in k["bi_leader"]] == ["phi_bi_leader"]
    assert {r["id"] for r in k["follower"]} >= {"phi_follower", "phi_bi_follower_left"}


def test_ports_hint(tmp_path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "ports.local.sh").write_text(
        'export LEFT_LEADER_PORT="/dev/tty.usbmodemL1"     # by hand\n'
        "export LEFT_FOLLOWER_PORT=/dev/tty.usbmodemL2\n"
        'export FOLLOWER_PORT="$LEFT_FOLLOWER_PORT"\n'
        'export BI_FOLLOWER_ID="phi_bi_follower"\n'
        'export LEADER_ID="bad id with spaces"\n'
        "echo not-an-assignment\n"
    )
    h = O.ports_hint(tmp_path)
    assert h["ports"] == {
        "left_leader": "/dev/tty.usbmodemL1",
        "left_follower": "/dev/tty.usbmodemL2",
        "follower": "/dev/tty.usbmodemL2",
    }
    assert h["ids"] == {"single": {}, "bimanual": {"follower": "phi_bi_follower"}}
    assert O.ports_hint(tmp_path / "nowhere") == {"ports": {}, "ids": {}}


def test_bimanual_config_reads_back() -> None:
    ans = O.check_answers(
        bi_answers(
            cameras={
                "top": {"source": 1, "width": 640, "height": 480, "fps": 30},
                "left_wrist": {"source": 2},
            }
        )
    )
    spec = rigspec.parse(O.build_yaml(ans))
    assert spec.problems == ()
    arms = {a.key: a for a in spec.arms}
    assert (
        arms["left_follower"].port == "/dev/tty.usbmodemA2"
        and arms["left_follower"].lerobot_id == "phi_bi_follower_left"
    )
    assert arms["right_leader"].lerobot_id == "phi_bi_leader_right"
    assert arms["left_follower"].max_relative_target == O.STEP_LIMIT
    cams = {c.key: c for c in spec.cameras}
    assert cams["top"].source == 1 and cams["left_wrist"].source == 2
    assert cams["top"].feature == "observation.images.top"


def test_single_config_without_ports_reads_back() -> None:
    ans = O.check_answers(
        {"name": "Desk", "layout": "single", "ids": {"follower": "f1", "leader": "l1"}, "ports": {}}
    )
    spec = rigspec.parse(O.build_yaml(ans))
    assert {a.key for a in spec.arms} == {"follower", "leader"} and all(
        a.port in (None, "") for a in spec.arms
    )
    assert not [p for p in spec.problems if "no id" in p]


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"name": " "}, "name"),
        ({"layout": "trimanual"}, "single arm or bimanual"),
        ({"ids": {"follower": "same", "leader": "same"}}, "different ids"),
        ({"ids": {"follower": "has space", "leader": "x"}}, "follower id"),
        ({"ports": {"left_leader": "usbmodem1"}}, "serial port"),
        (
            {"ports": {"left_leader": "/dev/tty.usbmodemX", "right_leader": "/dev/cu.usbmodemX"}},
            "share",
        ),
        ({"cameras": {"ceiling": {"source": 1}}}, "camera role"),
        ({"cameras": {"top": {"source": 99}}}, "camera number"),
        ({"cameras": {"top": {"source": 1}, "front": {"source": 1}}}, "both read camera 1"),
        ({"cameras": {"top": {"source": 1, "fps": 1000}}}, "fps"),
    ],
)
def test_refusals(over, msg) -> None:
    with pytest.raises(Refusal, match=msg):
        O.check_answers(bi_answers(**over))


def test_rewrite_backs_up_and_keeps_cameras(tmp_path) -> None:
    path, backups = tmp_path / "rig" / "robot-config.yaml", tmp_path / "backups"
    first = O.check_answers(
        bi_answers(cameras={"top": {"source": 1, "width": 640, "height": 480, "fps": 30}})
    )
    assert O.write(path, first, backups) is None
    second = O.check_answers(bi_answers(name="Bench B"))  # cameras not asked: keep them
    backup = O.write(path, second, backups)
    assert backup is not None and backup.is_file() and "Bench A" in backup.read_text()
    spec = rigspec.parse(path.read_text())
    assert {c.key for c in spec.cameras} == {"top"} and 'name: "Bench B"' in path.read_text()


@dataclass
class Root:
    path: Path


class Files:
    def __init__(self, roots):
        self.roots = roots


class FakeStudio:
    def __init__(self, rig: Path, code: Path):
        self.files = Files({"repo": Root(rig), "code": Root(code)})
        self.rig_dir = rig
        self.data_dir = None


def test_status(tmp_path, cal_root) -> None:
    rig, code = tmp_path / "phi", tmp_path / "studio"
    rig.mkdir()
    code.mkdir()
    s = FakeStudio(rig, code)
    st = O.status(s)
    assert not st["exists"] and st["path"] == str(rig / "robot-config.yaml")
    assert [r["id"] for r in st["known"]["bi_follower"]] == ["phi_bi_follower"]
    O.write(rig / "robot-config.yaml", O.check_answers(bi_answers()), tmp_path / "b")
    st = O.status(s)
    assert st["exists"] and st["name"] == "Bench A" and st["layout"] == "bimanual"
    by = {a["key"]: a for a in st["arms"]}
    assert by["left_follower"]["calibrated"] and by["right_leader"]["calibrated"]
    assert by["left_follower"]["port"] == "/dev/tty.usbmodemA2"


def test_config_lives_in_studio_data_without_a_rig_folder(tmp_path) -> None:
    """Studio stands alone: with no phi checkout, onboarding writes the config to Studio's data."""
    data, code = tmp_path / "data", tmp_path / "code"
    data.mkdir()
    code.mkdir()
    s = FakeStudio(tmp_path / "unused", code)
    s.files = Files({"code": Root(code), "studio": Root(data)})
    s.rig_dir = None
    assert O.config_target(s) == data / "robot-config.yaml"
    (code / "robot-config.yaml").write_text("robot: {}\n")  # a dev checkout's copy is the fallback
    assert O.config_target(s) == code / "robot-config.yaml"
    (data / "robot-config.yaml").write_text("robot: {}\n")  # Studio's own copy wins over it
    assert O.config_target(s) == data / "robot-config.yaml"


def test_hardware_reads_the_config_onboarding_wrote(tmp_path, monkeypatch) -> None:
    from phi_studio import cli, server

    seen: dict[str, object] = {}
    monkeypatch.setattr(server, "serve", lambda spec, **kw: seen.update(spec=spec, **kw))
    rig, data = tmp_path / "rig", tmp_path / "data"
    rig.mkdir()
    data.mkdir()
    argv = ["--hardware", "--no-browser", "--rig-dir", str(rig), "--data-dir", str(data)]
    assert cli.main(argv) == 1 and not seen  # nothing set up yet: refuse, never guess
    (data / "robot-config.yaml").write_text("robot: {}\n")
    assert cli.main(argv) == 0
    assert seen["spec"] == {"kind": "lerobot", "config": str(data / "robot-config.yaml")}
    (rig / "robot-config.yaml").write_text("robot: {}\n")  # a rig folder's copy comes first
    assert cli.main(argv) == 0
    assert seen["spec"] == {"kind": "lerobot", "config": str(rig.resolve() / "robot-config.yaml")}


# -- switching rigs without a restart ------------------------------------------------------------
def _studio(tmp_path, pairs: int = 1):
    import aiohttp
    from aiohttp.test_utils import TestServer
    from test_server import free_port

    from phi_studio.server import Studio

    port = free_port()
    studio = Studio({"kind": "mock", "pairs": pairs, "cameras": []}, port, token="t0k",
                    data_dir=tmp_path / "data")  # fmt: skip
    return studio, TestServer(studio.app(), host="127.0.0.1", port=port), aiohttp.ClientSession()


def test_switching_to_simulated_arms_restarts_the_worker_for_the_rigs_layout(tmp_path) -> None:
    import asyncio

    from test_server import until, ws

    async def go() -> None:
        studio, server, session = _studio(tmp_path)
        await server.start_server()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            first = await until(a, lambda d: d["type"] == "rig")
            assert len(first["arms"]) == 2
            O.write(tmp_path / "data" / "robot-config.yaml", O.check_answers(bi_answers()),
                    tmp_path / "b")  # fmt: skip
            await a.send_str('{"cmd": "rig_use", "hardware": false}')
            hello = await until(a, lambda d: d["type"] == "hello", timeout=15)
            assert hello["mock"] is True
            rig = await until(a, lambda d: d["type"] == "rig" and len(d["arms"]) == 4, timeout=15)
            assert {x["name"] for x in rig["arms"]} >= {"left_follower", "right_leader"}
            assert not [m for m in studio.log if "robot worker stopped" in m["text"]]
        finally:
            await session.close()
            await server.close()
            studio.stop_worker()

    asyncio.run(go())


def test_no_switch_while_an_arm_may_hold_torque_and_never_without_ports(tmp_path) -> None:
    import asyncio
    import json

    from test_server import until, ws

    async def go() -> None:
        studio, server, session = _studio(tmp_path)
        await server.start_server()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "rig_use", "hardware": True}))
            err = await until(a, lambda d: d["type"] == "error")
            assert "no robot-config.yaml" in err["message"]
            ans = bi_answers()
            ans["ports"].pop("left_leader")
            O.write(tmp_path / "data" / "robot-config.yaml", O.check_answers(ans), tmp_path / "b")
            await a.send_str(json.dumps({"cmd": "rig_use", "hardware": True}))
            err = await until(a, lambda d: d["type"] == "error")
            assert "needs a port" in err["message"] and "left_leader" in err["message"]
            for cmd in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": cmd}))
            await until(a, lambda d: d["type"] == "state" and d["state"] == "ARMED", timeout=10)
            await a.send_str(json.dumps({"cmd": "rig_use", "hardware": False}))
            err = await until(a, lambda d: d["type"] == "error")
            assert "holding torque" in err["message"]
        finally:
            await session.close()
            await server.close()
            studio.stop_worker()

    asyncio.run(go())


def test_switching_to_real_arms_starts_the_hardware_worker_even_with_nothing_plugged_in(
    tmp_path,
) -> None:
    """The whole --hardware start in the worker process: robot-config.yaml -> FeetechArms over
    LeRobot's bus. The ports do not exist here, so every arm is unreachable, and the rig must come
    up anyway, say it is real, and name each silent arm with a reason on Connect, not crash."""
    import asyncio
    import json

    import pytest
    from test_server import until, ws

    pytest.importorskip("lerobot")

    async def go() -> None:
        studio, server, session = _studio(tmp_path)
        await server.start_server()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            O.write(tmp_path / "data" / "robot-config.yaml", O.check_answers(bi_answers()),
                    tmp_path / "b")  # fmt: skip
            await a.send_str(json.dumps({"cmd": "rig_use", "hardware": True}))
            hello = await until(a, lambda d: d["type"] == "hello", timeout=20)
            assert hello["mock"] is False
            rig = await until(a, lambda d: d["type"] == "rig", timeout=20)
            assert rig["mock"] is False and len(rig["arms"]) == 4
            await a.send_str(json.dumps({"cmd": "connect"}))
            got = await until(a, lambda d: d["type"] == "identity", timeout=20)
            assert len(got["arms"]) == 4 and not any(x["ok"] for x in got["arms"])
            for x in got["arms"]:  # WHY each: one silent arm must not hide which of four it is
                text = x.get("error") or ""
                assert "usbmodem" in text or "not answering" in text or "port" in text.lower(), x
        finally:
            await session.close()
            await server.close()
            studio.stop_worker()

    asyncio.run(go())


def test_with_no_flag_studio_picks_real_arms_only_when_a_board_and_a_config_are_both_there(
    tmp_path, capsys
) -> None:
    from phi_studio.cli import rig_spec

    cfg = tmp_path / "robot-config.yaml"
    board = lambda: ["/dev/cu.usbmodem1"]  # noqa: E731
    none = lambda: []  # noqa: E731
    assert rig_spec(None, 2, [cfg], boards=board) == {"kind": "mock", "pairs": 2}
    assert "no robot-config.yaml yet" in capsys.readouterr().out
    cfg.write_text("robot: {}\n")
    assert rig_spec(None, 1, [cfg], boards=none) == {"kind": "mock", "pairs": 1}
    assert rig_spec(None, 1, [cfg], boards=board) == {"kind": "lerobot", "config": str(cfg)}
    assert "real arms (1 SO-101 driver board plugged in)" in capsys.readouterr().out
    assert rig_spec("mock", 1, [cfg], boards=board) == {"kind": "mock", "pairs": 1}


def test_an_arm_holding_torque_blocks_a_rig_switch_in_any_state() -> None:
    from types import SimpleNamespace

    from phi_studio.onboard_api import may_hold_torque

    s = SimpleNamespace(last={"state": {"state": "READY"}}, telemetry=None)
    assert not may_hold_torque(s)
    s.telemetry = {"arms": {"left_follower": {"torque": True}, "left_leader": {"torque": False}}}
    assert may_hold_torque(s)  # left on by an earlier session, while READY
    s.telemetry, s.last = None, {"state": {"state": "ARMED"}}
    assert may_hold_torque(s)
