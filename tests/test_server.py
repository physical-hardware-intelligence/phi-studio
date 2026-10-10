"""Server: auth, control, frames, and the heartbeat path end to end with a real worker process
(mock rig). SPEC X-2, TEL-4, TEL-5."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("PIL")
from aiohttp.test_utils import TestServer  # noqa: E402

import phi_studio.server as server_mod  # noqa: E402
from phi_studio.server import Studio  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def started(pairs: int = 1) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": pairs, "cameras": ["front"]}, port, token="t0k")
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


async def ws(
    session: aiohttp.ClientSession, port: int, token: str = "t0k", origin: str | None = None
) -> Any:
    origin = origin or f"http://127.0.0.1:{port}"
    return await session.ws_connect(
        f"http://127.0.0.1:{port}/ws?token={token}", headers={"Origin": origin}
    )


async def until(sock: Any, pred, timeout: float = 5.0, beat: bool = False) -> dict[str, Any]:
    """Read JSON messages until pred(msg) holds. Optionally keep sending heartbeats."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if beat:
            await sock.send_str(json.dumps({"cmd": "heartbeat"}))
        try:
            m = await sock.receive(timeout=0.2)
        except TimeoutError:
            continue
        if m.type == aiohttp.WSMsgType.TEXT:
            d = json.loads(m.data)
            if pred(d):
                return d
    raise AssertionError("condition not met in time")


def run(coro):
    return asyncio.run(coro)


def test_refuses_bad_token_origin_and_host() -> None:
    async def go() -> None:
        studio, server, session = await started()
        port = server.port
        try:
            for kwargs in ({"token": "wrong"}, {"origin": "https://evil.example"}):
                with pytest.raises(aiohttp.WSServerHandshakeError) as e:
                    await ws(session, port, **kwargs)
                assert e.value.status == 403
            r = await session.get(
                f"http://127.0.0.1:{port}/ws?token=t0k",
                headers={"Host": f"attacker.example:{port}", "Origin": f"http://127.0.0.1:{port}"},
            )
            assert r.status == 403 and "host" in await r.text()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_connect_identify_and_state_reach_the_window() -> None:
    async def go() -> None:
        studio, server, session = await started(pairs=2)
        try:
            w = await ws(session, server.port)
            hello = await until(w, lambda d: d["type"] == "hello")
            assert hello["control"] is True and hello["mock"] is True
            await w.send_str(json.dumps({"cmd": "connect"}))
            ident = await until(w, lambda d: d["type"] == "identity")
            assert len(ident["arms"]) == 4 and all(a["exact"] for a in ident["arms"])
            await until(w, lambda d: d["type"] == "state" and d["state"] == "IDENTIFIED")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_camera_frames_arrive_as_binary_packets() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            w = await ws(session, server.port)
            end = time.monotonic() + 5
            while time.monotonic() < end:
                m = await w.receive(timeout=2)
                if m.type == aiohttp.WSMsgType.BINARY:
                    assert m.data[:1] == b"F"
                    (n,) = struct.unpack(">H", m.data[1:3])
                    head = json.loads(m.data[3 : 3 + n])
                    assert head["key"] == "front" and head["w"] == 640
                    assert m.data[3 + n : 3 + n + 2] == b"\xff\xd8"  # JPEG magic
                    return
            raise AssertionError("no frame")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_second_window_may_stop_but_not_arm() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            hb = await until(b, lambda d: d["type"] == "hello")
            assert hb["control"] is False
            await b.send_str(json.dumps({"cmd": "connect"}))
            err = await until(b, lambda d: d["type"] == "error")
            assert "control" in err["message"]
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await until(a, lambda d: d.get("state") == "ARMED", beat=True)
            await b.send_str(json.dumps({"cmd": "stop"}))
            await until(b, lambda d: d.get("state") == "STOPPED")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_closing_the_controlling_window_stops_motion() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            watcher = await ws(session, server.port)
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            t0 = time.monotonic()
            await a.close()
            s = await until(watcher, lambda d: d.get("state") == "STOPPED", timeout=3)
            assert s["stop_reason"] == "window closed" and time.monotonic() - t0 < 0.5
        finally:
            await session.close()
            await server.close()

    run(go())


def test_between_workers_a_window_keeps_its_link_and_a_closed_one_frees_control() -> None:
    """2026-10-09, a rig reload after onboarding saved: while switch_rig had no worker, a window's
    heartbeat raised in to_worker and closed its socket, and the same raise in that socket's
    cleanup left control pinned to the closed window, so no window could control the rig."""

    async def go() -> None:
        studio, server, session = await started()
        saved = studio.conn
        try:
            a = await ws(session, server.port)
            assert (await until(a, lambda d: d["type"] == "hello"))["control"]
            studio.conn = None  # the gap inside switch_rig
            for _ in range(3):
                await a.send_str(json.dumps({"cmd": "heartbeat"}))
            await asyncio.sleep(0.3)
            assert not a.closed and studio.controller is not None
            await a.close()
            await asyncio.sleep(0.2)
            b = await ws(session, server.port)
            assert (await until(b, lambda d: d["type"] == "hello"))["control"]  # control was freed
        finally:
            studio.conn = saved
            await session.close()
            await server.close()

    run(go())


def test_unrecognised_cameras_hold_recording_and_policies_but_not_teleop() -> None:
    """camcheck: a recording or a policy on the wrong camera is worse than none."""

    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            studio.camcheck = {"status": "unsure", "message": "Which is the top camera?",
                               "fix": "Open Rig setup, Cameras."}  # fmt: skip
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await until(a, lambda d: d.get("state") == "ARMED", beat=True)
            await a.send_str(json.dumps({"cmd": "start", "activity": "policy"}))
            err = await until(a, lambda d: d["type"] == "error", beat=True)
            assert err["message"] == "Which is the top camera?" and "Rig setup" in err["fix"]
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            await a.send_str(json.dumps({"cmd": "rec_start", "repo_id": "local/x", "task": "t"}))
            err = await until(a, lambda d: d["type"] == "error", beat=True)
            assert err["message"] == "Which is the top camera?"
        finally:
            await session.close()
            await server.close()

    run(go())


def test_worker_death_is_reported_to_every_window() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            w = await ws(session, server.port)
            await until(w, lambda d: d["type"] == "hello")
            studio.proc.kill()
            d = await until(w, lambda d: d["type"] == "worker_exit")
            assert "Restart" in d["message"]
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_reload_inside_the_heartbeat_window_still_stops() -> None:
    # Cmd+R mid-teleop: the new window takes control and beats within 1 s, which the heartbeat
    # watchdog alone would not catch.
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            await a.close()
            b = await ws(session, server.port)
            await until(b, lambda d: d.get("state") == "STOPPED", timeout=1.5, beat=True)
        finally:
            await session.close()
            await server.close()

    run(go())


def test_taking_control_mid_motion_stops_first() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            await b.send_str(json.dumps({"cmd": "take_control"}))
            s = await until(b, lambda d: d.get("state") == "STOPPED", timeout=1.0)
            assert s["stop_reason"] == "control moved"
        finally:
            await session.close()
            await server.close()

    run(go())


class _Fake:
    """A Client stand-in that records what the server pushes to it."""

    def __init__(self) -> None:
        self.got: list[dict[str, Any]] = []

    def push(self, msg: dict[str, Any]) -> None:
        self.got.append(msg)


def offline_studio() -> tuple[Studio, list[dict[str, Any]]]:
    studio = Studio({"kind": "mock"}, 1, token="t0k")
    forwarded: list[dict[str, Any]] = []
    studio.to_worker = forwarded.append  # type: ignore[method-assign]
    return studio, forwarded


def test_a_window_without_control_cannot_arm_and_its_heartbeats_are_ignored() -> None:
    studio, forwarded = offline_studio()
    a, b = _Fake(), _Fake()
    studio.controller = a  # type: ignore[assignment]
    for cmd in ("heartbeat", "arm", "start", "release", "disconnect"):
        studio._from_client(b, json.dumps({"cmd": cmd}))  # type: ignore[arg-type]
    assert forwarded == []
    studio._from_client(b, json.dumps({"cmd": "stop"}))  # type: ignore[arg-type]
    assert forwarded == [{"cmd": "stop"}]


@pytest.mark.parametrize("raw", ['{"cmd": ["stop"]}', '{"cmd": {"a": 1}}', "[1, 2]", '"stop"',
                                 '{"cmd": 7}', "not json"])  # fmt: skip
def test_malformed_messages_get_an_error_not_a_dropped_socket(raw: str) -> None:
    studio, forwarded = offline_studio()
    a = _Fake()
    studio.controller = a  # type: ignore[assignment]
    studio._from_client(a, raw)  # type: ignore[arg-type]
    assert forwarded == [] and a.got[-1]["type"] == "error"


def test_non_ascii_token_is_refused_not_a_server_error() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError) as e:
                await ws(session, server.port, token="%C3%A9")
            assert e.value.status == 403
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_busy_port_is_a_one_line_message_not_a_traceback(capsys: Any) -> None:
    from phi_studio.cli import main

    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        code = main(["--mock", "--no-browser", "--port", str(port)])
    err = capsys.readouterr().err
    assert code == 1
    assert f"Port {port} is in use" in err and "--port" in err
    assert "Traceback" not in err


async def started_with_data(tmp: Any) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": ["front"]}, port, token="t0k",
                    data_dir=tmp)  # fmt: skip
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


def test_the_rig_description_reaches_every_window(tmp_path: Any) -> None:
    async def go() -> None:
        studio, server, session = await started_with_data(tmp_path)
        try:
            a = await ws(session, server.port)
            rig = await until(a, lambda d: d["type"] == "rig")
            assert any(p["id"] == "mock-reach" and p["available"] for p in rig["policies"])
            b = await ws(session, server.port)  # a window opened later gets it replayed
            await until(b, lambda d: d["type"] == "rig")
        finally:
            await session.close()
            await server.close()

    run(go())


CARD = {"name": "Cube to box", "task": "cube in box", "success": "The cube ends in the box",
        "limit_s": 1, "rubric": [{"label": "Reached it", "points": 0.5},
                                 {"label": "In the box", "points": 1.0}],
        "failures": ["Missed the grasp"], "conditions": [{"label": "Spot 1", "axis": "train"}]}


def test_an_eval_runs_blind_on_the_rig_and_is_saved_for_every_window(tmp_path: Any) -> None:
    async def go() -> None:
        studio, server, session = await started_with_data(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "rig")  # the worker has said what can run
            send = lambda m: a.send_str(json.dumps(m))  # noqa: E731
            await send({"cmd": "eval_card_save", "card": CARD})
            cid = (await until(a, lambda d: d["type"] == "eval_card_saved"))["id"]
            await send({"cmd": "eval_begin", "card": cid, "policies": ["mock-reach"], "reps": 1,
                        "blind": True, "alpha": 0.05})  # fmt: skip
            ev = await until(a, lambda d: d["type"] == "eval" and d["current"] is not None)
            rid = ev["current"]["id"]
            assert ev["current"]["policies"] == [{"alias": "A"}]  # blind: a letter, no name
            assert ev["next"]["alias"] == "A" and ev["next"]["condition"]["label"] == "Spot 1"
            assert ev["current"]["stamps"]["camera_check"] in (None, "none", "unverified", "same")
            await send({"cmd": "eval_score", "stage": 2, "run_id": "made-up"})
            err = await until(a, lambda d: d["type"] == "error")
            assert "no policy run" in err["message"]  # the server scores only runs it saw

            for c in ("connect", "confirm", "arm"):
                await send({"cmd": c})
            await until(a, lambda d: d["type"] == "state" and d["state"] == "ARMED", beat=True)
            await send({"cmd": "start", "activity": "policy", "policy": "mock-reach",
                        "task": "cube in box", "limit_s": 1})  # fmt: skip
            err = await until(a, lambda d: d["type"] == "error", beat=True)
            assert "eval is running" in err["message"]  # trials start from the schedule only
            await send({"cmd": "eval_run"})
            t = await until(a, lambda d: d["type"] == "telemetry" and d["policy"]
                            and d["policy"]["running"], beat=True)  # fmt: skip
            assert t["policy"]["name"] == "Policy A" and t["policy"]["id"] == "policy-A"
            # The eval update leaves just before the telemetry that shows the run ended.
            ev = await until(a, lambda d: d["type"] == "eval" and d["awaiting"], beat=True)
            assert ev["next"]["alias"] == "A"  # windows learn the trial ended without a reload
            assert studio.run is not None and not studio.run["running"]
            run_id = studio.run["run_id"]
            await send({"cmd": "eval_run"})  # the trial waits for its score
            err = await until(a, lambda d: d["type"] == "error", beat=True)
            assert "not scored" in err["message"]
            await send({"cmd": "disconnect"})  # resetting the scene must not lose the trial
            await until(a, lambda d: d["type"] == "state" and d["state"] == "DISCONNECTED",
                        beat=True)  # fmt: skip

            await send({"cmd": "eval_score", "stage": 2, "failures": [], "note": "ok",
                        "run_id": run_id})  # fmt: skip
            ev = await until(a, lambda d: d["type"] == "eval" and d["current"] is None
                             or (d["type"] == "eval" and d["summary"]
                                 and d["summary"]["valid_trials"] == 1), beat=True)  # fmt: skip
            trial = json.loads((tmp_path / "evals" / f"{rid}.json").read_text())["trials"][0]
            assert trial["success"] and trial["duration_s"] == pytest.approx(1.0, abs=0.1)
            assert trial["metrics"]["ended"] == "time limit" and "clipped" in trial["metrics"]

            b = await ws(session, server.port)  # view-only window
            ev_b = await until(b, lambda d: d["type"] == "eval")
            assert ev_b["current"]["id"] == rid
            await b.send_str(json.dumps({"cmd": "eval_undo"}))
            err = await until(b, lambda d: d["type"] == "error")
            assert "control" in err["message"]

            await send({"cmd": "eval_end"})
            ev = await until(a, lambda d: d["type"] == "eval" and d["current"] is None, beat=True)
            assert ev["past"][0]["id"] == rid
            assert ev["past"][0]["results"][0]["policy"] == "Scripted reach and place"
            await b.send_str(json.dumps({"cmd": "eval_export", "id": rid, "format": "md"}))
            md = await until(b, lambda d: d["type"] == "eval_export")
            assert "Scripted reach and place" in md["text"] and md["name"].endswith(".md")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_bad_eval_file_does_not_break_new_windows(tmp_path: Any) -> None:
    # Before: a record without an id made every new window's handler raise after it was added to
    # the clients, leaking it (and possibly control) on each reconnect.
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "bad.json").write_text(json.dumps({"episodes": []}))

    async def go() -> None:
        studio, server, session = await started_with_data(tmp_path)
        try:
            a = await ws(session, server.port)
            ev = await until(a, lambda d: d["type"] == "eval")
            assert ev["current"] is None and ev["past"] == []
            await a.close()
            for _ in range(20):
                if not studio.clients:
                    break
                await asyncio.sleep(0.05)
            assert not studio.clients and studio.controller is None
        finally:
            await session.close()
            await server.close()

    run(go())


def test_evals_without_a_data_directory_say_so() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "eval_begin", "card": "x", "policies": ["p"],
                                         "reps": 1, "blind": False}))  # fmt: skip
            err = await until(a, lambda d: d["type"] == "error")
            assert "data folder" in err["message"]
        finally:
            await session.close()
            await server.close()

    run(go())


# -- assistant and file view ---------------------------------------------------------------------
FAKE_CLI = f"{sys.executable} {Path(__file__).parent / 'support' / 'fake_claude.py'}"


async def started_with_code(tmp: Path) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    code = tmp / "code"
    code.mkdir()
    (code / "robot-config.yaml").write_text("robot:\n  port: /dev/tty.usbmodem123\n")
    (code / ".env").write_text("HF_TOKEN=hf_x\n")
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": ["front"]}, port, token="t0k",
                    data_dir=tmp / "data", code_root=code)  # fmt: skip
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


def test_any_window_can_read_files_but_not_secrets(tmp_path: Path, monkeypatch) -> None:
    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            a = await ws(session, server.port)  # takes control
            b = await ws(session, server.port)  # view only
            await until(b, lambda d: d["type"] == "hello" and not d["control"])
            await b.send_str(json.dumps({"cmd": "files_index"}))
            ix = await until(b, lambda d: d["type"] == "files")
            assert [n["path"] for n in ix["notes"]] == ["robot-config.yaml"]
            await b.send_str(json.dumps({"cmd": "file_read", "path": "robot-config.yaml"}))
            f = await until(b, lambda d: d["type"] == "file")
            assert "usbmodem123" in f["text"] and f["root"] == "code"
            await b.send_str(json.dumps({"cmd": "file_read", "root": "code", "path": ".env"}))
            err = await until(b, lambda d: d["type"] == "file_error")
            assert "secrets" in err["message"]
            await b.send_str(json.dumps({"cmd": "files_search", "query": "usbmodem"}))
            res = await until(b, lambda d: d["type"] == "search")
            assert [h["path"] for h in res["hits"]] == ["robot-config.yaml"]
            await b.send_str(json.dumps({"cmd": "ports"}))
            await until(b, lambda d: d["type"] == "ports")
            await a.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_the_assistant_answers_only_the_window_that_asked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PHI_STUDIO_CLAUDE", FAKE_CLI)

    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            a = await ws(session, server.port)
            b = await ws(session, server.port)
            await until(b, lambda d: d["type"] == "hello")
            await b.send_str(json.dumps({"cmd": "assist_status"}))
            st = await until(b, lambda d: d["type"] == "assist_status")
            assert st["available"]
            await a.send_str(json.dumps({"cmd": "connect"}))  # gives the log a state change
            await until(a, lambda d: d["type"] == "state" and d["state"] == "IDENTIFIED")
            await b.send_str(json.dumps({"cmd": "assist_ask", "text": "what state?",
                                         "page": "overview",
                                         "focus": {"message": "boom"}}))  # fmt: skip
            done = await until(b, lambda d: d["type"] == "assist" and d["kind"] == "done")
            assert done["stopped"] is False
            await b.send_str(json.dumps({"cmd": "assist_context", "page": "overview",
                                         "focus": {"message": "boom"}}))  # fmt: skip
            ctx = (await until(b, lambda d: d["type"] == "assist_context"))["text"]
            assert '"IDENTIFIED"' in ctx and '"boom"' in ctx and "t0k" not in ctx
            assert '"robot-config.yaml"' not in ctx  # files are read on demand, not pushed
            with pytest.raises(AssertionError):  # the other window heard nothing
                await until(a, lambda d: d["type"] == "assist", timeout=0.5)
            proc = next(c for c in studio.clients if c.conversation).conversation.proc
            await b.close()
            await asyncio.wait_for(proc.wait(), 5)  # closing the window ends its Claude process
            await a.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_checks_run_from_any_window_and_reach_the_assistant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PHI_STUDIO_CLAUDE", f"{FAKE_CLI} --signed-out")

    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            a = await ws(session, server.port)
            b = await ws(session, server.port)  # view only
            await until(b, lambda d: d["type"] == "hello" and not d["control"])
            await b.send_str(json.dumps({"cmd": "checks_run"}))
            first = await until(b, lambda d: d["type"] == "checks")
            before = {r["id"]: r for r in first["results"]}
            assert len(before) == 21 and before["arms_answer"]["status"] == "skip"
            await a.send_str(json.dumps({"cmd": "connect"}))
            await until(b, lambda d: d["type"] == "telemetry" and d["arms"], beat=False)
            await b.send_str(json.dumps({"cmd": "checks_run"}))
            got = await until(b, lambda d: d["type"] == "checks", timeout=10)
            res = {r["id"]: r for r in got["results"]}
            assert res["arms_answer"]["status"] == "pass" and res["faults"]["status"] == "pass"
            assert res["assistant"]["status"] == "warn"
            assert "claude auth login" in res["assistant"]["fix"]
            assert res["rig_config"]["status"] == "warn"  # this config names a port but no id
            assert res["ports_identity"]["status"] == "skip"  # mock ports are not real
            await b.send_str(json.dumps({"cmd": "assist_context", "page": "checks"}))
            ctx = json.loads((await until(b, lambda d: d["type"] == "assist_context"))["text"])
            titles = [r["title"] for r in ctx["checks"]["not_passing"]]
            assert "Claude assistant" in titles and "Rig config" in titles
            assert ctx["checks"]["minutes_ago"] == 0 and "not live" in ctx["checks"]["note"]
            await a.close()
            await b.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_read_only_command_that_crashes_still_answers_its_window(
    tmp_path: Path, monkeypatch
) -> None:
    # WHY: an uncaught error here left the window's spinner up for good (review, 2026-10-04).
    def boom(_):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(server_mod, "run_checks", boom)

    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            b = await ws(session, server.port)
            await b.send_str(json.dumps({"cmd": "checks_run"}))
            got = await until(b, lambda d: d["type"] == "checks")
            assert "probe exploded" in got["error"] and "results" not in got
            await b.send_str(json.dumps({"cmd": "files_search", "query": ""}))
            err = await until(b, lambda d: d["type"] == "file_error")
            assert err["op"] == "search"
            await b.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_claudes_context_reports_the_largest_load_by_size_not_sign() -> None:
    # Load is signed by direction; the Checks page and the joint table compare its size.
    studio, _ = offline_studio()
    hp = {"temp": 30.0, "volt": 12.0, "faults": []}
    studio.telemetry = {"arms": {"follower": {"role": "follower", "health": {
        "gripper": {**hp, "load": -95.0}, "shoulder_pan": {**hp, "load": 5.0}}}}}
    arm = json.loads(studio.context())["arms"]["follower"]
    assert arm["max_load_pct"] == 95.0


def test_a_feature_command_needs_control_when_it_says_so() -> None:
    async def go() -> None:
        studio, server, session = await started()
        seen: list[str] = []

        async def change(client: Any, msg: dict) -> None:
            seen.append("change")
            client.push({"type": "changed"})

        async def read(client: Any, msg: dict) -> None:
            raise RuntimeError("boom")

        studio.handle("x_change", change, control=True)
        studio.handle("x_read", read, control=False)
        with pytest.raises(ValueError):
            studio.handle("stop", read, control=False)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            await until(b, lambda d: d["type"] == "hello")
            await b.send_str(json.dumps({"cmd": "x_change"}))
            err = await until(b, lambda d: d["type"] == "error")
            assert "control" in err["message"] and seen == []
            await a.send_str(json.dumps({"cmd": "x_change"}))
            await until(a, lambda d: d["type"] == "changed")
            await b.send_str(json.dumps({"cmd": "x_read"}))
            err = await until(b, lambda d: d["type"] == "error" and d.get("cmd") == "x_read")
            assert "RuntimeError: boom" in err["message"]
        finally:
            await session.close()
            await server.close()

    run(go())


def test_every_worker_command_reaches_the_worker() -> None:
    """A command the worker has but the server's list lacks is refused as unknown before it gets
    there (auto-calibration's were, the first time). The list must cover every _cmd_ handler."""
    from phi_studio.server import COMMANDS
    from phi_studio.worker import RigWorker

    handlers = {n.removeprefix("_cmd_") for n in dir(RigWorker) if n.startswith("_cmd_")}
    assert handlers <= COMMANDS, sorted(handlers - COMMANDS)
