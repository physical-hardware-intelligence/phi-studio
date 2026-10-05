"""The Set up page's server side: saving ports and cameras to robot-config.yaml with comments kept,
finding a moved arm by its USB serial number, and live camera align, all on fakes."""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from aiohttp.test_utils import TestServer  # noqa: E402
from PIL import Image  # noqa: E402
from test_server import free_port, until, ws  # noqa: E402

from phi_studio import rigspec, setup_api  # noqa: E402
from phi_studio.server import Studio  # noqa: E402
from phi_studio.setup_api import SetupError  # noqa: E402

CONFIG = """\
# Parv's rig. Comments stay.
robot:                                   # the follower
  type: so101_follower
  id: phi_follower
  port: /dev/tty.usbmodemOLDF            # found by unplugging

teleop:
  type: so101_leader
  id: phi_leader
  port: /dev/tty.usbmodemL

cameras:
  front:
    type: opencv
    index_or_path: TBD                   # confirm each session
    width: 640
    height: 480
    fps: 30
  top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}
"""


def row(name: str, serial: str | None) -> dict[str, Any]:
    dev = name.replace("/dev/tty.", "/dev/cu.")
    return {"device": dev, "tty": name, "description": "USB serial", "vid": 6790, "pid": 21971,
            "serial": serial, "manufacturer": None, "usb": True, "arm": None}  # fmt: skip


BT = {"device": "/dev/cu.Bluetooth-Incoming-Port", "tty": "/dev/tty.Bluetooth-Incoming-Port",
      "description": None, "vid": None, "pid": None, "serial": None, "manufacturer": None,
      "usb": False, "arm": None}  # fmt: skip


class FakeTerm:
    """Studio's terminal with `command` in the foreground. WHY not set term_running: with no
    terminal panel open Studio asks the terminal itself, since term_running may be stale."""

    def __init__(self, command: str | None) -> None:
        self.command = command

    def running(self) -> dict[str, Any] | None:
        return {"pgid": 1, "command": self.command} if self.command else None

    def close(self) -> None:
        pass


def jpeg(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (160, 120), color).save(buf, format="JPEG")
    return buf.getvalue()


# -- pure checks ---------------------------------------------------------------------------------
def test_only_usb_ports_count_and_carry_the_name_configs_use() -> None:
    rows = setup_api.usb_ports([row("/dev/tty.usbmodemA", "A"), BT])
    assert [r["name"] for r in rows] == ["/dev/tty.usbmodemA"]


def test_a_moved_arm_is_found_only_by_an_unambiguous_serial() -> None:
    spec = rigspec.parse(CONFIG)
    mem = {"follower:phi_follower": {"serial": "S1"}, "leader:phi_leader": {"serial": "S2"}}
    rows = [row("/dev/tty.usbmodemNEWF", "S1"), row("/dev/tty.usbmodemL", "S2")]
    assert setup_api.moved_ports(spec.arms, rows, mem) == [
        {"arm": "follower", "old": "/dev/tty.usbmodemOLDF", "new": "/dev/tty.usbmodemNEWF",
         "serial": "S1"}]  # fmt: skip
    twins = [row("/dev/tty.usbmodemX", "S1"), row("/dev/tty.usbmodemY", "S1")]
    assert setup_api.moved_ports(spec.arms, twins, mem) == []  # two boards, one serial: unsure
    assert setup_api.moved_ports(spec.arms, rows, {}) == []  # never saw this board's serial
    # The new name is already the leader's: no suggestion that would put two arms on one port.
    taken = [row("/dev/tty.usbmodemL", "S1")]
    assert setup_api.moved_ports(spec.arms, taken, mem) == []


def test_port_choices_refuse_unknown_arms_absent_ports_and_shared_ports() -> None:
    spec = rigspec.parse(CONFIG)
    rows = [row("/dev/tty.usbmodemA", "A"), row("/dev/tty.usbmodemL", "L")]
    ok = setup_api.check_port_choice(spec, {"follower": "/dev/tty.usbmodemA"}, rows)
    assert ok == {"follower": "/dev/tty.usbmodemA"}
    bad = [({"elbow": "/dev/tty.usbmodemA"}, "no arm called elbow"),
           ({"follower": "rm -rf /"}, "not a serial port name"),
           ({"follower": "/dev/tty.usbmodemGONE"}, "not plugged in"),
           # the leader keeps /dev/tty.usbmodemL, so the follower cannot take it too
           ({"follower": "/dev/tty.usbmodemL"}, "would share"),
           ({}, "Say which port")]  # fmt: skip
    for ask, why in bad:
        with pytest.raises(SetupError, match=why):
            setup_api.check_port_choice(spec, ask, rows)


def test_camera_choices_name_config_cameras_once_each() -> None:
    spec = rigspec.parse(CONFIG)
    assert setup_api.check_camera_choice(spec, {"front": 0, "observation.images.top": 1}) == {
        "observation.images.front": 0, "observation.images.top": 1}  # fmt: skip
    bad = [({"side": 0}, "no camera called side"), ({"front": "0; ls"}, "not a camera"),
           ({"front": True}, "not a camera"),
           ({"front": 1, "top": 1}, "both read camera 1")]  # fmt: skip
    for ask, why in bad:
        with pytest.raises(SetupError, match=why):
            setup_api.check_camera_choice(spec, ask)


# -- end to end through the server ---------------------------------------------------------------
async def started(tmp: Path, cameras: tuple[str, ...] = ("front",)) -> tuple[
        Studio, TestServer, Any]:  # fmt: skip
    rig = tmp / "rig"
    rig.mkdir()
    (rig / "robot-config.yaml").write_text(CONFIG)
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": list(cameras)}, port, token="t0k",
                    data_dir=tmp / "data", rig_dir=rig)  # fmt: skip
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


def test_saving_ports_keeps_comments_backs_up_and_remembers_serials(tmp_path: Path,
                                                                     monkeypatch: Any) -> None:
    rows = [row("/dev/tty.usbmodemNEWF", "S1"), row("/dev/tty.usbmodemL", "S2"), BT]
    monkeypatch.setattr(setup_api, "list_ports", lambda ident: {"ports": list(rows)})

    async def go() -> None:
        studio, server, session = await started(tmp_path)
        cfg = tmp_path / "rig" / "robot-config.yaml"
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            await until(b, lambda d: d["type"] == "hello")
            ask = {"cmd": "setup_ports_save", "ports": {"follower": "/dev/tty.usbmodemNEWF"}}
            await b.send_str(json.dumps(ask))
            err = await until(b, lambda d: d["type"] == "error")
            assert "control" in err["message"] and "NEWF" not in cfg.read_text()

            await a.send_str(json.dumps({"cmd": "setup_ports_save",
                                         "ports": {"follower": "/dev/tty.usbmodemNEWF",
                                                   "leader": "/dev/tty.usbmodemL"}}))  # fmt: skip
            saved = await until(a, lambda d: d["type"] == "setup_saved")
            assert saved["what"] == "ports" and Path(saved["backup"]).read_text() == CONFIG
            text = cfg.read_text()
            assert "port: /dev/tty.usbmodemNEWF            # found by unplugging" in text
            assert text.replace("NEWF", "OLDF") == CONFIG  # one value changed, nothing else
            files = await until(a, lambda d: d["type"] == "files")
            assert files["lerobot"]["arms"][0]["port"] == "/dev/tty.usbmodemNEWF"
            mem = json.loads((tmp_path / "data" / "ports.json").read_text())
            assert mem["follower:phi_follower"]["serial"] == "S1"
            assert any("Saved ports" in e["text"] for e in studio.log)

            # Next day the follower's board comes back under another name: Studio offers the fix.
            rows[0] = row("/dev/tty.usbmodemMOVED", "S1")
            await a.send_str(json.dumps({"cmd": "setup_ports"}))
            st = await until(a, lambda d: d["type"] == "setup_ports")
            assert st["moved"] == [{"arm": "follower", "old": "/dev/tty.usbmodemNEWF",
                                    "new": "/dev/tty.usbmodemMOVED", "serial": "S1"}]  # fmt: skip
            assert [p["name"] for p in st["ports"]] == ["/dev/tty.usbmodemMOVED",
                                                        "/dev/tty.usbmodemL"]  # fmt: skip

            # A LeRobot command in the terminal may hold the ports: no saving under it.
            studio.terminal = FakeTerm("/x/python /x/bin/lerobot-teleoperate")  # type: ignore[assignment]
            ask = {"cmd": "setup_ports_save", "ports": {"follower": "/dev/tty.usbmodemMOVED"}}
            await a.send_str(json.dumps(ask))
            err = await until(a, lambda d: d["type"] == "error")
            assert "Stop it before changing ports" in err["message"]
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def test_the_camera_finder_shows_pictures_and_saves_numbers(tmp_path: Path,
                                                            monkeypatch: Any) -> None:
    from phi_studio import cameras

    def probe(indices: Any, timeout_s: float = 3.0) -> list[dict[str, Any]]:
        return [{"source": 0, "ok": True, "width": 640, "height": 480, "fps": 30.0, "error": None,
                 "thumbnail": jpeg((200, 10, 10))},
                {"source": 1, "ok": False, "width": None, "height": None, "fps": None,
                 "error": "not there", "thumbnail": None}]  # fmt: skip

    monkeypatch.setattr(cameras, "probe", probe)

    async def go() -> None:
        studio, server, session = await started(tmp_path)
        cfg = tmp_path / "rig" / "robot-config.yaml"
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "setup_cameras_probe"}))
            got = await until(a, lambda d: d["type"] == "setup_cameras" and not d["probing"])
            first, second = got["cameras"]
            assert first["picture"].startswith("data:image/jpeg;base64,") and first["ok"]
            assert second["picture"] is None and second["error"] == "not there"
            await a.send_str(json.dumps({"cmd": "setup_cameras_save",
                                         "cameras": {"front": 0, "top": 3}}))  # fmt: skip
            await until(a, lambda d: d["type"] == "setup_saved" and d["what"] == "cameras")
            text = cfg.read_text()
            assert "    index_or_path: 0                     # confirm each session" in text
            assert "top: {type: opencv, index_or_path: 3," in text
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


class FakeCam:
    opened: list[Any] = []
    closed: list[Any] = []

    def __init__(self, key: str, fields: dict[str, Any]) -> None:
        self.key, self.fields = key, fields
        self.seq = 0
        FakeCam.opened.append(fields["index_or_path"])

    def read_latest(self) -> tuple[Any, float, int]:
        """A new frame each read, grey level 10 x the camera number, so a test can tell whose
        picture it holds; stamped on time.monotonic as RealCamera does."""
        self.seq += 1
        return np.full((480, 640, 3), 10 * self.fields["index_or_path"], np.uint8), \
            time.monotonic(), self.seq

    def close(self) -> None:
        FakeCam.closed.append(self.fields["index_or_path"])


def test_camera_align_matches_streams_offsets_and_releases(tmp_path: Path,
                                                           monkeypatch: Any) -> None:
    from phi_studio import align, cameras

    root = str(tmp_path / "ds")
    refs = {"observation.images.front": np.full((480, 640, 3), 90, np.uint8),
            "observation.images.top": np.full((480, 640, 3), 30, np.uint8)}  # fmt: skip
    monkeypatch.setattr(align, "list_datasets", lambda: [{"root": root, "name": "ds"}])
    monkeypatch.setattr(align, "references", lambda r, e, cache_dir=None: {
        "root": r, "episode": e, "frame": 3, "motion": 0.0, "images": refs, "physical": {}})
    monkeypatch.setattr(cameras, "probe", lambda idx, timeout_s=3.0: [
        {"source": 4, "ok": True, "thumbnail": jpeg((90, 90, 90))},
        {"source": 5, "ok": True, "thumbnail": jpeg((30, 30, 30))}])  # fmt: skip
    monkeypatch.setattr(align, "match_cameras", lambda live, r: {
        "assignment": {4: "observation.images.front", 5: "observation.images.top"},
        "scores": {4: {}, 5: {}}, "unsure": False, "why": None, "unmatched_refs": []})  # fmt: skip
    monkeypatch.setattr(align, "measure", lambda live, ref: {
        "dx": 4.0, "dy": -2.0, "response": 0.9, "aligned": False, "low_match": False,
        "hint": "Move the camera 4 px left", "note": None})  # fmt: skip
    monkeypatch.setattr(cameras, "RealCamera", FakeCam)
    FakeCam.opened, FakeCam.closed = [], []

    async def go() -> None:
        studio, server, session = await started(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "align_start", "root": "/etc", "episode": 0}))
            err = await until(a, lambda d: d["type"] == "error")
            assert "datasets on this Mac" in err["message"]  # only a listed dataset

            await a.send_str(json.dumps({"cmd": "align_start", "root": root, "episode": 0}))
            s = await until(a, lambda d: d["type"] == "align_session")
            assert s["assignment"] == [{"live": 4, "key": "observation.images.front"},
                                       {"live": 5, "key": "observation.images.top"}]  # fmt: skip
            assert s["config"] == {"observation.images.front": "front",
                                   "observation.images.top": "top"}  # fmt: skip
            assert set(s["references"]) == set(refs)
            tick = await until(a, lambda d: d["type"] == "align_tick")
            late = await ws(session, server.port)  # a reloaded page learns of the running session
            await until(late, lambda d: d["type"] == "hello")
            await late.send_str(json.dumps({"cmd": "align_datasets"}))
            await until(late, lambda d: d["type"] == "align_datasets")
            assert (await until(late, lambda d: d["type"] == "align_session"))["root"] == root
            assert tick["dx"] == 4.0 and tick["picture"].startswith("data:image/jpeg")
            assert sorted(FakeCam.opened) == [4, 5]

            swap = {"4": "observation.images.top", "5": "observation.images.front"}
            await a.send_str(json.dumps({"cmd": "align_assign", "assignment": swap}))
            s2 = await until(a, lambda d: d["type"] == "align_session"
                             and d["assignment"][0]["key"] == "observation.images.top")  # fmt: skip
            assert sorted(FakeCam.closed) == [4, 5]  # reopened under the new names

            # A LeRobot command starting in the terminal takes the cameras back.
            studio.terminal = FakeTerm("lerobot-record --robot.type=x")  # type: ignore[assignment]
            stop = await until(a, lambda d: d["type"] == "align_stopped", timeout=5)
            assert "lerobot-record" in stop["why"]
            assert sorted(FakeCam.closed) == [4, 4, 5, 5]
            assert s2  # silence the unused name
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def test_a_crash_while_measuring_frees_the_cameras_and_says_why(tmp_path: Path,
                                                                monkeypatch: Any) -> None:
    from phi_studio import align, cameras

    root = str(tmp_path / "ds")
    refs = {"observation.images.front": np.full((480, 640, 3), 90, np.uint8)}
    monkeypatch.setattr(align, "list_datasets", lambda: [{"root": root, "name": "ds"}])
    monkeypatch.setattr(align, "references", lambda r, e, cache_dir=None: {
        "root": r, "episode": e, "frame": 0, "motion": 0.0, "images": refs, "physical": {}})
    monkeypatch.setattr(cameras, "probe", lambda idx, timeout_s=3.0: [
        {"source": 4, "ok": True, "thumbnail": jpeg((90, 90, 90))}])  # fmt: skip
    monkeypatch.setattr(align, "match_cameras", lambda live, r: {
        "assignment": {4: "observation.images.front"}, "scores": {4: {}}, "unsure": False,
        "why": None, "unmatched_refs": []})  # fmt: skip

    def broken(live: Any, ref: Any) -> dict[str, Any]:
        raise RuntimeError("cv2 fell over")

    monkeypatch.setattr(align, "measure", broken)
    monkeypatch.setattr(cameras, "RealCamera", FakeCam)
    FakeCam.opened, FakeCam.closed = [], []

    async def go() -> None:
        studio, server, session = await started(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "align_start", "root": root, "episode": 0}))
            stop = await until(a, lambda d: d["type"] == "align_stopped", timeout=5)
            assert "cv2 fell over" in stop["why"]
            assert FakeCam.closed == [4] and studio.setup_api.align is None  # type: ignore[attr-defined]
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def test_one_device_under_its_two_mac_names_is_one_port() -> None:
    spec = rigspec.parse(CONFIG.replace("port: /dev/tty.usbmodemL", "port: /dev/cu.usbmodemL"))
    rows = [row("/dev/tty.usbmodemL", "L")]
    with pytest.raises(SetupError, match="would share"):
        setup_api.check_port_choice(spec, {"follower": "/dev/tty.usbmodemL"}, rows)
    # and a config written with /dev/cu. still reads as plugged in, so no false "moved"
    mem = {"leader:phi_leader": {"serial": "L"}}
    assert setup_api.moved_ports(spec.arms, rows, mem) == []


def test_two_boards_with_one_serial_get_no_moved_suggestion() -> None:
    spec = rigspec.parse(CONFIG.replace("port: /dev/tty.usbmodemL", "port: /dev/tty.usbmodemGONE"))
    mem = {"follower:phi_follower": {"serial": "X"}, "leader:phi_leader": {"serial": "X"}}
    assert setup_api.moved_ports(spec.arms, [row("/dev/tty.usbmodemNEW", "X")], mem) == []


def test_a_camera_number_already_in_the_config_cannot_be_given_twice() -> None:
    spec = rigspec.parse(CONFIG)  # top is camera 2 already
    with pytest.raises(SetupError, match="Top camera and Front camera would both read camera 2"):
        setup_api.check_camera_choice(spec, {"front": 2})
    assert setup_api.check_camera_choice(spec, {"front": 2, "top": 0}) == {
        "observation.images.front": 2, "observation.images.top": 0}  # fmt: skip


def align_fakes(monkeypatch: Any, root: str, probe_s: float = 0.0,
                sources: dict[int, str] | None = None) -> None:  # fmt: skip
    """Camera align on fakes: `sources` maps each live camera number to the dataset key it
    matches (default: camera 4 shows the front)."""
    from phi_studio import align, cameras

    sources = sources or {4: "observation.images.front"}
    refs = {k: np.full((480, 640, 3), 90, np.uint8) for k in sources.values()}

    def probe(idx: Any, timeout_s: float = 3.0) -> list[dict[str, Any]]:
        time.sleep(probe_s)
        return [{"source": s, "ok": True, "thumbnail": jpeg((90, 90, 90))} for s in sources]

    monkeypatch.setattr(align, "list_datasets", lambda: [{"root": root, "name": "ds"}])
    monkeypatch.setattr(align, "references", lambda r, e, cache_dir=None: {
        "root": r, "episode": e, "frame": 0, "motion": 0.0, "images": refs, "physical": {}})
    monkeypatch.setattr(cameras, "probe", probe)
    monkeypatch.setattr(align, "match_cameras", lambda live, r: {
        "assignment": dict(sources), "scores": {s: {} for s in sources}, "unsure": False,
        "why": None, "unmatched_refs": []})  # fmt: skip
    monkeypatch.setattr(align, "measure", lambda live, ref: {
        "dx": 0.0, "dy": 0.0, "response": 1.0, "aligned": True, "low_match": False,
        "hint": "Aligned", "size": [640, 480], "note": None})  # fmt: skip
    monkeypatch.setattr(cameras, "RealCamera", FakeCam)
    FakeCam.opened, FakeCam.closed = [], []


def test_overlapping_starts_and_a_stop_leave_no_camera_open(tmp_path: Path,
                                                            monkeypatch: Any) -> None:
    root = str(tmp_path / "ds")
    align_fakes(monkeypatch, root, probe_s=0.3)

    async def go() -> None:
        studio, server, session = await started(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            start = json.dumps({"cmd": "align_start", "root": root, "episode": 0})
            await a.send_str(start)
            await a.send_str(start)  # a reloaded page pressing Start again mid-start
            await a.send_str(json.dumps({"cmd": "align_stop"}))  # Stop while both are starting
            await until(a, lambda d: d["type"] == "align_stopped", timeout=5)
            await asyncio.sleep(0.5)
            api = studio.setup_api  # type: ignore[attr-defined]
            if api.align is not None:  # the stop ran between the starts; stop the survivor
                await a.send_str(json.dumps({"cmd": "align_stop"}))
                await asyncio.sleep(0.5)
            assert api.align is None
            assert sorted(FakeCam.opened) == sorted(FakeCam.closed)  # every open camera closed
            loops = [t for t in asyncio.all_tasks() if "AlignSession.run" in repr(t.get_coro())]
            assert loops == []
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def test_align_frees_the_cameras_when_no_window_shows_it(tmp_path: Path, monkeypatch: Any) -> None:
    root = str(tmp_path / "ds")
    align_fakes(monkeypatch, root)
    monkeypatch.setattr(setup_api, "ALIGN_IDLE_S", 0.6)

    async def go() -> None:
        studio, server, session = await started(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "align_start", "root": root, "episode": 0}))
            await until(a, lambda d: d["type"] == "align_session")
            for _ in range(3):  # a window showing it keeps it alive past the limit
                await asyncio.sleep(0.4)
                await a.send_str(json.dumps({"cmd": "align_alive"}))
            assert studio.setup_api.align is not None  # type: ignore[attr-defined]
            stop = await until(a, lambda d: d["type"] == "align_stopped", timeout=5)
            assert "no window showed camera align" in stop["why"] and FakeCam.closed == [4]
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def grey(msg: dict[str, Any]) -> float:
    """The mean grey level of a frame message's JPEG."""
    return float(setup_api.decode(msg["jpeg"]).mean())


async def frame_where(studio: Studio, pred: Any, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred(studio.latest_frame):
        assert time.monotonic() < end, f"frames now: {sorted(studio.latest_frame)}"
        await asyncio.sleep(0.05)


def test_align_offers_its_frames_under_the_rig_names_and_takes_them_back(
        tmp_path: Path, monkeypatch: Any) -> None:  # fmt: skip
    from phi_studio import recon_api

    root = str(tmp_path / "ds")
    align_fakes(monkeypatch, root, sources={4: "observation.images.front",
                                           5: "observation.images.top"})  # fmt: skip

    async def go() -> None:
        # WHY no worker cameras: every frame Studio holds must then be align's
        studio, server, session = await started(tmp_path, cameras=())
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "align_start", "root": root, "episode": 0}))
            await until(a, lambda d: d["type"] == "align_session")
            await frame_where(studio, lambda f: {"front", "top"} <= set(f))
            front, top = studio.latest_frame["front"], studio.latest_frame["top"]
            assert abs(grey(front) - 40) < 3 and abs(grey(top) - 50) < 3  # camera 4, camera 5
            assert front["from"] == "camera align" and (front["w"], front["h"]) == (640, 480)
            # The worker's frame clock and camera list, which the camera check reads, stay its own.
            assert studio.frame_clock == {} and studio.cameras == {}
            assert recon_api.state_of(studio).status()["live"] == ["front", "top"]

            swap = {"4": "observation.images.top", "5": "observation.images.front"}
            await a.send_str(json.dumps({"cmd": "align_assign", "assignment": swap}))
            await frame_where(studio, lambda f: "front" in f and abs(grey(f["front"]) - 50) < 3)

            # A worker frame that replaced align's is not align's to take back. WHY end the loop
            # first: its next tick would otherwise overwrite the planted frame before the stop.
            api = studio.setup_api  # type: ignore[attr-defined]
            api.align.task.cancel()
            await asyncio.sleep(0.05)
            worker = {"type": "frame", "key": "top", "seq": 99, "jpeg": b""}
            studio.latest_frame["top"] = worker
            await a.send_str(json.dumps({"cmd": "align_stop"}))
            await until(a, lambda d: d["type"] == "align_stopped")
            assert "front" not in studio.latest_frame and studio.latest_frame["top"] is worker
            assert recon_api.state_of(studio).status()["live"] == []
            # Only align opened cameras: once, then again for the swap; all closed now.
            assert sorted(FakeCam.opened) == [4, 4, 5, 5] == sorted(FakeCam.closed)
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def test_refusals_read_as_written_and_name_their_command(tmp_path: Path) -> None:
    async def go() -> None:
        studio, server, session = await started(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            await until(b, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "align_assign", "assignment": {}}))
            err = await until(a, lambda d: d["type"] == "error")
            assert err == {"type": "error", "cmd": "align_assign", "fix": "",
                           "message": "Camera align is not running."}  # fmt: skip
            await b.send_str(json.dumps({"cmd": "setup_ports_save", "ports": {}}))
            err = await until(b, lambda d: d["type"] == "error")
            assert err["cmd"] == "setup_ports_save"
            assert "Another window has control" in err["message"]
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())
