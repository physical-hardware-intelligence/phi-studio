"""The terminal socket: same auth as /ws, only the window with control types, Run refused while
a command holds the terminal."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("PIL")

from test_studio_server import started, until, ws  # noqa: E402


async def term(session: Any, port: int, client: str, token: str = "t0k") -> Any:
    return await session.ws_connect(
        f"http://127.0.0.1:{port}/terminal?token={token}&client={client}",
        headers={"Origin": f"http://127.0.0.1:{port}"},
    )


async def read_until(sock: Any, want: bytes, timeout: float = 10.0) -> tuple[bytes, list[dict]]:
    """Collect output bytes until `want` appears; also return the JSON messages seen."""
    out, msgs = b"", []
    end = time.monotonic() + timeout
    while want not in out:
        if time.monotonic() > end:
            raise AssertionError(f"{want!r} never came; got {out[-300:]!r} and {msgs}")
        try:
            m = await sock.receive(timeout=0.2)
        except TimeoutError:
            continue
        if m.type == aiohttp.WSMsgType.BINARY:
            out += m.data
        elif m.type == aiohttp.WSMsgType.TEXT:
            msgs.append(json.loads(m.data))
    return out, msgs


async def json_until(sock: Any, pred, timeout: float = 10.0) -> dict:
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


def test_terminal_needs_the_token_and_only_the_controller_types(monkeypatch: Any) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")

    async def go() -> None:
        studio, server, session = await started()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await term(session, server.port, "x", token="wrong")
            a = await ws(session, server.port)
            hello = await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            hello_b = await until(b, lambda d: d["type"] == "hello")
            assert hello["control"] and not hello_b["control"]

            ta = await term(session, server.port, hello["client"])
            st = await json_until(ta, lambda d: d["t"] == "status")
            assert st["alive"] and st["typing"]
            await ta.send_str(json.dumps({"t": "run", "cmd": "echo from-$((20+22))"}))
            await read_until(ta, b"from-42")

            tb = await term(session, server.port, hello_b["client"])
            out, msgs = await read_until(tb, b"from-42")  # a late window gets the history
            if not any(m["t"] == "status" for m in msgs):
                msgs.append(await json_until(tb, lambda d: d["t"] == "status"))
            assert [m["typing"] for m in msgs if m["t"] == "status"] == [False]
            await tb.send_str(json.dumps({"t": "in", "d": "echo sneaky\n"}))
            err = await json_until(tb, lambda d: d["t"] == "error")
            assert "control" in err["message"]

            await b.send_str(json.dumps({"cmd": "take_control"}))  # control moves: so does typing
            await json_until(tb, lambda d: d["t"] == "status" and d["typing"])
            await json_until(ta, lambda d: d["t"] == "status" and not d["typing"])
        finally:
            await session.close()
            await server.close()

    asyncio.run(go())


def test_run_is_refused_while_a_command_holds_the_terminal(monkeypatch: Any) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")

    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            hello = await until(a, lambda d: d["type"] == "hello")
            t = await term(session, server.port, hello["client"])
            await t.send_str(json.dumps({"t": "run", "cmd": "sleep 30"}))
            st = await json_until(t, lambda d: d["t"] == "status" and "sleep 30" in (
                (d["running"] or {}).get("command", "")))
            assert st["running"]["pgid"] > 0
            await t.send_str(json.dumps({"t": "run", "cmd": "echo too-soon"}))
            err = await json_until(t, lambda d: d["t"] == "error")
            assert "still running sleep 30" in err["message"]
            await t.send_str(json.dumps({"t": "interrupt"}))
            await json_until(t, lambda d: d["t"] == "status" and d["running"] is None)
            await t.send_str(json.dumps({"t": "run", "cmd": "echo now-$((1+1))"}))
            await read_until(t, b"now-2")
            assert any("Ran in the terminal: sleep 30" in e["text"] for e in studio.log)
        finally:
            await session.close()
            await server.close()
        assert studio.terminal is not None and not studio.terminal.alive  # closed with Studio

    asyncio.run(go())
