"""The Studio server: static UI plus one WebSocket, bound to 127.0.0.1 (ARCHITECTURE ADR-S7, S12).

    browser  <--WebSocket-->  this process  <--Pipe-->  robot worker process

Security, because this socket can move motors:
  * binds 127.0.0.1 only
  * the WebSocket needs the per-launch token, an allowed Origin, and an allowed Host (DNS rebinding)
  * no CORS headers, so no other site's page can read responses

Evals live here, not in the worker: judging an episode moves nothing, and the record must survive a
worker crash. Every judgement is on disk before the windows hear about it.

The assistant (Claude) and the file view live here too, read-only. Neither has a path to the worker,
so neither can move an arm; any window may use them.

Control: one window holds control; any window may Stop. Control changing hands stops the rig:
closing or reloading the controlling window, or another window taking control, sends Stop at
once. The worker's heartbeat timeout is the backstop for a window that hangs without closing.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import multiprocessing as mp
import os
import secrets
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from phi_studio.assistant import ClaudeCLI, Conversation
from phi_studio.checks import Inputs, run_checks
from phi_studio.errors import Refusal
from phi_studio.evals import EvalError, EvalStore
from phi_studio.files import FileError, Files, default_roots, list_ports
from phi_studio.terminal import Terminal

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)
Handler = Callable[["Client", dict[str, Any]], Awaitable[None]]
# Modules that add commands to Studio, each with register(studio). A missing one is skipped, so a
# feature can land on its own.
FEATURES = ("setup_api", "hub_api", "train_api", "scene_api", "recon_api")

COMMANDS = {"heartbeat", "connect", "identify", "confirm", "arm", "start", "stop", "resume",
            "release", "clear", "disconnect", "inject",
            "cal_start", "cal_middle", "cal_finish", "cal_save", "cal_cancel"}  # fmt: skip
EVAL_COMMANDS = {"eval_begin", "eval_mark", "eval_undo", "eval_end"}  # answered by the server
ANYONE = {"stop", "take_control"}  # allowed from a window without control
# Answered by the server, from any window: they read, and none reaches the worker.
READ_ONLY = {"assist_status", "assist_context", "assist_ask", "assist_stop", "assist_reset",
             "files_index", "file_read", "files_search", "ports", "checks_run"}  # fmt: skip
CODE_ROOT = Path(__file__).resolve().parents[2]  # src/phi_studio/server.py -> the repo
LOG_SIZE = 200
REPLAYED = ("rig", "state", "identity", "worker_exit")  # newest of each, sent to a new window


def frame_packet(msg: dict[str, Any]) -> bytes:
    """b'F' + uint16 header length + JSON header + JPEG bytes."""
    head = json.dumps({k: msg[k] for k in ("key", "t", "seq", "w", "h")}).encode()
    return b"F" + struct.pack(">H", len(head)) + head + msg["jpeg"]


class Client:
    """One browser window. Holds the newest telemetry and the newest frame per camera, so a slow
    window drops stale data instead of queueing it."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        # WHY an id: the terminal is a second socket, and it must know whether its window has
        # control (only that window may type). The page sends this id when it opens the terminal.
        self.id = secrets.token_urlsafe(9)
        self.events: list[dict[str, Any]] = []  # state, identity, errors: every one delivered
        self.telemetry: dict[str, Any] | None = None
        self.frames: dict[str, dict[str, Any]] = {}
        self.wake = asyncio.Event()
        self.conversation: Conversation | None = None

    def push(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "telemetry":
            self.telemetry = msg
        elif kind == "frame":
            self.frames[msg["key"]] = msg
        else:
            self.events.append(msg)
        self.wake.set()

    async def run(self) -> None:
        while not self.ws.closed:
            await self.wake.wait()
            self.wake.clear()
            events, self.events = self.events, []
            for e in events:
                await self.ws.send_str(json.dumps(e))
            if self.telemetry is not None:
                t, self.telemetry = self.telemetry, None
                await self.ws.send_str(json.dumps(t))
            frames, self.frames = self.frames, {}
            for f in frames.values():
                await self.ws.send_bytes(frame_packet(f))


def register_features(studio: Studio) -> None:
    for name in FEATURES:
        try:
            mod = importlib.import_module(f"phi_studio.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"phi_studio.{name}":
                continue
            raise
        mod.register(studio)


class Studio:
    def __init__(
        self,
        spec: dict[str, Any],
        port: int,
        token: str | None = None,
        data_dir: Path | str | None = None,
        assistant_model: str | None = None,
        code_root: Path | None = None,
        rig_dir: Path | None = None,
    ) -> None:
        self.spec = {**spec, "data_dir": str(data_dir)} if data_dir else dict(spec)
        self.port = port
        self.data_dir = Path(data_dir) if data_dir else None
        self.evals = EvalStore(self.data_dir) if self.data_dir else None
        self.token = token or secrets.token_urlsafe(24)
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}
        self.clients: set[Client] = set()
        self.controller: Client | None = None
        self.last: dict[
            str, dict[str, Any]
        ] = {}  # newest state / identity, replayed to new windows
        self.run: dict[str, Any] | None = None  # telemetry's policy view, set by the reader thread
        self.telemetry: dict[str, Any] | None = None  # newest, for the assistant's context
        self.telemetry_at: float | None = None  # monotonic time it arrived
        self.log: deque[dict[str, Any]] = deque(maxlen=LOG_SIZE)  # state changes and errors
        # Per camera: online, the last error, and when the newest frame arrived with a smoothed
        # interval between frames, for the checks. Written by the reader thread as whole tuples.
        self.cameras: dict[str, dict[str, Any]] = {}
        self.frame_clock: dict[str, tuple[float, float | None]] = {}
        # The newest frame message per camera (JPEG and size), for features that look at what the
        # cameras already stream (recon_api's live capture). Written by the reader thread.
        self.latest_frame: dict[str, dict[str, Any]] = {}
        self.checks: dict[str, Any] | None = None  # the newest check run, for the assistant
        self.code_root = code_root or CODE_ROOT
        roots = default_roots(self.code_root, self.data_dir, rig_dir)
        self.files = Files(roots)
        self.terminal: Terminal | None = None  # started when a window first opens the panel
        self.term_socks: dict[web.WebSocketResponse, str] = {}  # each terminal socket's window id
        self.term_running: dict[str, Any] | None = None  # the command in the foreground, if any
        self.term_watch: asyncio.Task[None] | None = None
        self.cli = ClaudeCLI(cwd=roots[0].path, dirs=[r.path for r in roots[1:]],
                             model=assistant_model)  # fmt: skip
        self.loop: asyncio.AbstractEventLoop | None = None
        self.proc: Any = None
        self.conn: Any = None
        self._send_lock = threading.Lock()
        # Feature modules (hub_api, train_api, scene_api, setup_api) add commands and routes here.
        self.handlers: dict[str, tuple[Handler, bool]] = {}  # cmd -> (handler, needs control)
        self.routes: list[web.RouteDef] = []
        self.config_lock = asyncio.Lock()  # one robot-config.yaml write at a time
        self.on_close: list[Callable[[], Awaitable[Any]]] = []  # run when Studio shuts down
        register_features(self)

    # -- worker ---------------------------------------------------------------------------------
    def start_worker(self) -> None:
        from phi_studio.worker import run_worker

        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=run_worker, args=(child, self.spec), daemon=True)
        self.proc.start()
        child.close()  # WHY: otherwise recv() never sees EOF when the worker dies
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Worker -> windows. Runs in a thread because Pipe.recv blocks."""
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                code = self.proc.exitcode if self.proc else None
                msg = {"type": "worker_exit", "code": code,
                       "message": "The robot worker stopped. Restart Studio."}  # fmt: skip
                self._dispatch(msg)
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind in REPLAYED:
            self.last[kind] = msg
        if kind == "telemetry":
            self.telemetry, self.telemetry_at = msg, time.monotonic()
            self.run = msg.get("policy")  # the newest policy run: the only one an eval can judge
        elif kind == "state":
            why = msg.get("fault") or msg.get("stop_reason")
            self._note("state", f"{msg.get('label')}" + (f": {why}" if why else ""))
        elif kind == "error":
            self._note("error", msg.get("message", ""), msg.get("fix"))
        elif kind == "calibrated":
            self._note("calibrated", f"Saved calibration for {msg.get('arm')}")
        elif kind == "worker_exit":
            self._note("error", msg.get("message", ""), None)
        elif kind == "frame":
            self.latest_frame[msg["key"]] = msg
            self._frame(msg["key"])
        elif kind == "camera":
            self.cameras[msg["key"]] = {"online": msg.get("online"), "message": msg.get("message")}
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fanout, msg)

    def _frame(self, key: str) -> None:
        now = time.monotonic()
        last, gap = self.frame_clock.get(key, (None, None))
        if last is not None:
            dt = now - last
            gap = dt if gap is None else 0.9 * gap + 0.1 * dt  # smoothed over about ten frames
        self.frame_clock[key] = (now, gap)
        if not self.cameras.get(key, {}).get("online"):
            self.cameras[key] = {"online": True, "message": None}

    def _note(self, kind: str, text: str, fix: str | None = None) -> None:
        """One line of the log the assistant sees. Called from the reader thread and the loop;
        deque.append is atomic."""
        self.log.append({"t": round(time.time(), 1), "kind": kind, "text": text[:500],
                         "fix": (fix or None) and fix[:500]})  # fmt: skip

    def _tell(self, client: Client, message: str, fix: str = "", cmd: str | None = None) -> None:
        """An error for one window, logged so the assistant sees it too. `cmd`: the request it
        answers, so the page can clear what waits for that reply."""
        self._note("error", message, fix)
        client.push({"type": "error", "message": message, "fix": fix,
                     **({"cmd": cmd} if cmd else {})})  # fmt: skip

    def _fanout(self, msg: dict[str, Any]) -> None:
        for c in list(self.clients):
            c.push(msg)

    def to_worker(self, msg: dict[str, Any]) -> None:
        with self._send_lock:
            try:
                self.conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def stop_worker(self) -> None:
        if self.conn is not None:
            self.conn.close()
        if self.proc is not None:
            self.proc.join(3)
            if self.proc.is_alive():
                self.proc.terminate()

    # -- web --------------------------------------------------------------------------------------
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/terminal", self.terminal_handler)
        app.router.add_get("/api/health", self.health)
        app.router.add_get("/", self.index)
        if (STATIC / "assets").is_dir():
            app.router.add_static("/assets", STATIC / "assets")
        app.router.add_routes(self.routes)
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def _on_startup(self, app: web.Application) -> None:
        self.loop = asyncio.get_running_loop()
        if self.proc is None:
            self.start_worker()

    async def _on_shutdown(self, app: web.Application) -> None:
        for fn in self.on_close:  # features release cameras and stop their tasks first
            try:
                await fn()
            except Exception:
                log.exception("a feature failed to close")
        for c in list(self.clients):
            await c.ws.close()
        for ws in list(self.term_socks):
            await ws.close()
        if self.term_watch is not None:
            self.term_watch.cancel()
        if self.terminal is not None:
            self.terminal.close()  # the shell and anything it runs end with Studio
        self.stop_worker()

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "worker_alive": self.worker_alive()})

    async def index(self, request: web.Request) -> web.StreamResponse:
        page = STATIC / "index.html"
        if page.exists():
            return web.FileResponse(page, headers={"Cache-Control": "no-store"})
        return web.Response(text="Studio UI is not built. Run: npm --prefix web run build",
                            status=503)  # fmt: skip

    def _refuse(self, request: web.Request) -> str | None:
        if request.host not in self.allowed_hosts:
            return f"host {request.host!r} not allowed"
        if request.headers.get("Origin") not in self.allowed_origins:
            return "origin not allowed"
        # WHY bytes: compare_digest raises TypeError on non-ASCII str, which would be a 500.
        if not secrets.compare_digest(request.query.get("token", "").encode(), self.token.encode()):
            return "bad or missing token"
        return None

    async def ws_handler(self, request: web.Request) -> web.StreamResponse:
        why = self._refuse(request)
        if why:
            return web.Response(status=403, text=why)
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        client = Client(ws)
        self.clients.add(client)
        if self.controller is None:
            self.controller = client
        sender = asyncio.create_task(client.run())
        try:  # WHY everything after clients.add is inside: a failure must still run the cleanup
            client.push({"type": "hello", "control": self.controller is client,
                         "client": client.id,
                         "mock": self.spec.get("kind", "mock") == "mock"})  # fmt: skip
            for kind in REPLAYED:
                if kind in self.last:
                    client.push(self.last[kind])
            if self.evals is not None:
                client.push(self._eval_msg())
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    self._from_client(client, m.data)
        finally:
            self.clients.discard(client)
            if self.controller is client:
                # WHY stop now, not on the heartbeat timeout: a reload reconnects and beats again
                # within 1 s, so the timeout alone would let motion run on.
                self.to_worker({"cmd": "stop", "reason": "window closed"})
                self.controller = None
                for other in self.clients:
                    other.push({"type": "control", "available": True})
                self._term_status()
            sender.cancel()
            if client.conversation is not None:
                await client.conversation.aclose()
        return ws

    # -- terminal -------------------------------------------------------------------------------
    def _term_cwd(self) -> Path:
        r = self.files.roots.get("repo") or self.files.roots["code"]
        return r.path

    def _may_type(self, window: str) -> bool:
        return self.controller is not None and secrets.compare_digest(
            self.controller.id.encode(), window.encode()
        )

    def _term_status(self) -> None:
        t = self.terminal
        for ws, window in list(self.term_socks.items()):
            msg = {"t": "status", "alive": bool(t and t.alive), "running": self.term_running,
                   "typing": self._may_type(window)}  # fmt: skip
            asyncio.ensure_future(self._term_send(ws, msg))

    @staticmethod
    async def _term_send(ws: web.WebSocketResponse, msg: dict[str, Any]) -> None:
        if ws.closed:
            return
        try:
            await ws.send_str(json.dumps(msg))
        except (ConnectionError, RuntimeError):  # the window closed while this was queued
            pass

    async def _watch_terminal(self) -> None:
        """Polls which command holds the terminal, twice a second, while any panel is open."""
        while self.term_socks and self.terminal is not None:
            now = await asyncio.to_thread(self.terminal.running)
            if (now or {}).get("command") != (self.term_running or {}).get("command"):
                self.term_running = now
                self._term_status()
            await asyncio.sleep(0.5)
        self.term_watch = None

    async def terminal_handler(self, request: web.Request) -> web.StreamResponse:
        why = self._refuse(request)
        if why:
            return web.Response(status=403, text=why)
        window = request.query.get("client", "")
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        if self.terminal is None:
            self.terminal = Terminal(self._term_cwd())
            self.terminal.on_exit = self._term_status
        if not self.terminal.alive:
            self.terminal.start(loop)
        term = self.terminal
        # WHY one queue per socket, filled with the history before the listener is added (no
        # await between): output that arrives meanwhile is neither lost nor sent out of order.
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        if term.history():
            queue.put_nowait(term.history())
        term.listeners.add(queue.put_nowait)

        async def pump() -> None:
            while not ws.closed:
                data = await queue.get()
                while not queue.empty() and len(data) < 65536:  # coalesce a burst into one frame
                    data += queue.get_nowait()
                await ws.send_bytes(data)

        sender = loop.create_task(pump())
        self.term_socks[ws] = window
        self._term_status()
        if self.term_watch is None:
            self.term_watch = loop.create_task(self._watch_terminal())
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    await self._term_message(ws, window, m.data)
        finally:
            term.listeners.discard(queue.put_nowait)
            self.term_socks.pop(ws, None)
            sender.cancel()
        return ws

    async def _term_message(self, ws: web.WebSocketResponse, window: str, data: str) -> None:
        try:
            msg = json.loads(data)
            kind = msg["t"]
        except (ValueError, KeyError, TypeError):
            return
        term = self.terminal
        if kind == "resize":  # any window may size its own view; the shell gets the newest
            cols, rows = msg.get("cols"), msg.get("rows")
            if term is not None and isinstance(cols, int) and isinstance(rows, int):
                term.resize(cols, rows)
            return

        def refuse(message: str, fix: str = "") -> None:
            asyncio.ensure_future(self._term_send(ws, {"t": "error", "message": message,
                                                       "fix": fix}))  # fmt: skip

        if not self._may_type(window):
            refuse("Another window has control.", "Take control to use the terminal here.")
            return
        if kind == "restart":
            if term is not None:
                term.close()
                term.start(asyncio.get_running_loop())
                self._term_status()
            return
        if term is None or not term.alive:
            refuse("The shell is not running.", "Press Restart.")
            return
        if kind == "in" and isinstance(msg.get("d"), str):
            term.write(msg["d"].encode())
        elif kind == "interrupt":
            term.interrupt()
        elif kind == "run" and isinstance(msg.get("cmd"), str) and msg["cmd"].strip():
            cmd = msg["cmd"].strip()
            busy = await asyncio.to_thread(term.running)
            if busy:
                # WHY refuse: typed into a running command, the text becomes its input (an
                # "Enter" to lerobot-calibrate's prompt), not a new command.
                refuse(f"The terminal is still running {busy['command'] or 'a command'}.",
                       "Wait for it to finish, or stop it with Ctrl-C.")  # fmt: skip
                return
            self._note("state", f"Ran in the terminal: {cmd[:200]}")
            term.write(cmd.encode() + b"\r")

    def _from_client(self, client: Client, data: str) -> None:
        try:
            msg = json.loads(data)
            cmd = msg["cmd"]
        except (ValueError, KeyError, TypeError):
            cmd = None
        if not isinstance(cmd, str):
            self._tell(client, "malformed message")
            return
        if cmd == "take_control":
            if self.controller is not None and self.controller is not client:
                self.to_worker({"cmd": "stop", "reason": "control moved"})
                self.controller.push({"type": "control", "control": False})
            self.controller = client
            client.push({"type": "control", "control": True})
            self._term_status()
            return
        if cmd in self.handlers:
            fn, needs_control = self.handlers[cmd]
            if needs_control and client is not self.controller:
                self._tell(client, "Another window has control.",
                           "Take control to change the rig from here.", cmd)  # fmt: skip
                return
            asyncio.get_running_loop().create_task(self._feature(client, cmd, fn, msg))
            return
        if cmd in READ_ONLY:
            asyncio.get_running_loop().create_task(self._read_only(client, cmd, msg))
            return
        if cmd not in COMMANDS and cmd not in EVAL_COMMANDS:
            self._tell(client, f"unknown command {cmd!r}")
            return
        if client is not self.controller and cmd not in ANYONE:
            if cmd != "heartbeat":
                self._tell(client, "Another window has control.",
                           "Take control to operate the rig here.")  # fmt: skip
            return
        if cmd in EVAL_COMMANDS:
            self._eval(client, cmd, msg)
            return
        if cmd == "start" and msg.get("activity") == "policy" and self._unjudged():
            # A new run replaces the last one, and the unjudged episode could never be judged.
            self._tell(client, "The last eval episode is not judged yet.",
                       "Judge it on the Evaluate page, or end the eval.")  # fmt: skip
            return
        self.to_worker({k: v for k, v in msg.items() if isinstance(k, str)})

    def handle(self, cmd: str, fn: Handler, *, control: bool) -> None:
        """Answer `cmd` with `fn(client, msg)`. control=True: only the window with control may
        send it (it changes files, the rig or a job); False: any window (it only reads)."""
        if cmd in self.handlers or cmd in COMMANDS or cmd in READ_ONLY or cmd in EVAL_COMMANDS:
            raise ValueError(f"{cmd} is already a command")
        self.handlers[cmd] = (fn, control)

    async def _feature(self, client: Client, cmd: str, fn: Handler, msg: dict[str, Any]) -> None:
        try:
            await fn(client, msg)
        except Refusal as e:  # the person can act on it: shown as written, no traceback
            self._tell(client, str(e), e.fix, cmd)
        except Exception as e:  # WHY all: a dropped error leaves the window waiting forever
            log.exception("%s failed", cmd)
            fix = "Try again. If it repeats, ask Claude or restart Studio."
            client.push({"type": "error", "cmd": cmd, "fix": fix,
                         "message": f"Studio could not answer {cmd}: {type(e).__name__}: {e}"})

    # -- evals ----------------------------------------------------------------------------------
    def _unjudged(self) -> bool:
        return self.evals is not None and self.evals.awaiting(self.run)

    def _eval_msg(self) -> dict[str, Any]:
        assert self.evals is not None
        return {"type": "eval", "current": self.evals.current, "past": self.evals.list(20),
                "dir": str(self.evals.dir)}  # fmt: skip

    def _eval(self, client: Client, cmd: str, msg: dict[str, Any]) -> None:
        if self.evals is None:
            self._tell(client, "Evals need a data directory.", "Start Studio with --data-dir.")
            return
        e = self.evals
        try:
            if cmd == "eval_begin":
                e.begin(msg.get("policy"), msg.get("task", ""), msg.get("planned"),
                        msg.get("limit_s"))  # fmt: skip
            elif cmd == "eval_mark":
                e.mark(msg.get("outcome"), msg.get("note", ""), self.run, msg.get("run_id"))
            elif cmd == "eval_undo":
                e.undo()
            else:
                e.end()
        except EvalError as err:
            self._tell(client, str(err))
            return
        except OSError as err:  # the judgement is not saved, so do not show it as saved
            self._tell(client, f"Could not save the eval: {err}", f"Check {e.dir} is writable.")
            return
        self._fanout(self._eval_msg())

    # -- assistant and files ----------------------------------------------------------------------
    async def _read_only(self, client: Client, cmd: str, msg: dict[str, Any]) -> None:
        try:
            if cmd == "assist_status":
                client.push({"type": "assist_status", **(await self.cli.status())})
            elif cmd == "assist_context":  # what the next question would carry, shown on request
                client.push({"type": "assist_context",
                             "text": self.context(msg.get("page"), msg.get("focus"))})  # fmt: skip
            elif cmd == "assist_ask":
                if client.conversation is None:
                    client.conversation = Conversation(self.cli, client.push)
                ctx = self.context(msg.get("page"), msg.get("focus"))
                await client.conversation.ask(msg.get("text"), ctx)
            elif cmd == "assist_stop" and client.conversation is not None:
                client.conversation.stop()
            elif cmd == "assist_reset" and client.conversation is not None:
                client.conversation.reset()
            elif cmd == "checks_run":
                started = time.monotonic()
                results = await asyncio.to_thread(run_checks, await self._check_inputs())
                self.checks = {"type": "checks", "at": time.time(), "results": results,
                               "ms": round((time.monotonic() - started) * 1000)}  # fmt: skip
                client.push(self.checks)
            elif cmd == "files_index":
                client.push({"type": "files", **self.files.index()})
            elif cmd == "file_read":
                root, path = msg.get("root"), msg.get("path")
                if root is None:  # a path from an answer: find the folder that holds it
                    loc = self.files.locate(path)
                    if loc is None:
                        raise FileError(f"{path} is not in any folder Studio can show.")
                    root, path = loc
                got = await asyncio.to_thread(self.files.read, root, path)
                client.push({"type": "file", **got, "line": msg.get("line")})
            elif cmd == "files_search":
                got = await asyncio.to_thread(self.files.search, msg.get("query"))
                client.push({"type": "search", **got})
            elif cmd == "ports":
                ident = self.last.get("identity", {}).get("arms", [])
                client.push({"type": "ports", **(await asyncio.to_thread(list_ports, ident))})
        except Exception as e:  # WHY all: a dropped error leaves the window's spinner up forever
            self._read_only_failed(client, cmd, msg, e)

    def _read_only_failed(
        self, client: Client, cmd: str, msg: dict[str, Any], e: Exception
    ) -> None:
        if not isinstance(e, FileError | OSError):
            log.exception("%s failed", cmd)
        text = str(e) if isinstance(e, FileError) else f"{type(e).__name__}: {e}"
        if cmd == "checks_run":
            client.push({"type": "checks", "error": f"Checks could not run: {text}"})
        elif cmd in ("file_read", "files_search"):
            op = "search" if cmd == "files_search" else "read"
            if not isinstance(e, FileError):
                text = f"Could not read it: {text}"
            client.push({"type": "file_error", "op": op, "message": text, "path": msg.get("path")})
        else:
            fix = "Try again. If it repeats, ask Claude or restart Studio."
            client.push(
                {"type": "error", "message": f"Studio could not answer {cmd}: {text}", "fix": fix}
            )

    async def _check_inputs(self) -> Inputs:
        ident = self.last.get("identity", {}).get("arms", [])
        status, ports = await asyncio.gather(
            self.cli.status(), asyncio.to_thread(list_ports, ident)
        )
        now, cams = time.monotonic(), {}
        for key, c in list(self.cameras.items()):
            last, gap = self.frame_clock.get(key, (None, None))
            cams[key] = {**c, "fps": 1.0 / gap if gap else None,
                         "age_s": now - last if last is not None else None}  # fmt: skip
        cal_dir = self.last.get("rig", {}).get("cal_dir")
        return Inputs(
            rig_kind=self.spec.get("kind", "mock"),
            state=self.last.get("state"),
            identity=ident,
            telemetry=self.telemetry,
            cameras=cams,
            ports=ports,
            assistant=status,
            roots={k: r.path for k, r in self.files.roots.items()},
            rig_cal_dir=Path(cal_dir).expanduser().resolve() if cal_dir else None,
            data_dir=self.data_dir,
            code_root=self.code_root,
            static_dir=STATIC,
            worker_alive=self.worker_alive(),
            telemetry_age_s=now - self.telemetry_at if self.telemetry_at is not None else None,
        )

    def worker_alive(self) -> bool:
        return "worker_exit" not in self.last and bool(self.proc and self.proc.is_alive())

    def _checks_summary(self) -> dict[str, Any] | None:
        if self.checks is None:
            return None
        bad = [r for r in self.checks["results"] if r["status"] in ("warn", "fail")]
        age = round((time.time() - self.checks["at"]) / 60)
        return {
            "ran": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.checks["at"])),
            "minutes_ago": age,
            # WHY: the rig can change after a run; old failures must not read as current.
            "note": "A snapshot from that run, not live. If the rig or config changed since, "
            "ask the user to press Run again on the Checks page.",
            "not_passing": [{k: r[k] for k in ("title", "status", "detail", "fix")} for r in bad],
        }

    def context(self, page: Any = None, focus: Any = None) -> str:
        """What Studio sees right now, as text for the assistant. Never the token."""
        t = self.telemetry or {}
        arms = {}
        for name, a in (t.get("arms") or {}).items():
            h = a.get("health") or {}
            arms[name] = {"role": a.get("role"), "online": a.get("online"),
                          "torque": a.get("torque"),
                          "max_temp_c": max((x["temp"] for x in h.values()), default=None),
                          "max_load_pct": max((abs(x["load"]) for x in h.values()),
                                              default=None),  # load is signed by direction
                          "min_volt": min((x["volt"] for x in h.values()), default=None),
                          "faults": {j: x["faults"] for j, x in h.items() if x["faults"]},
                          "position_deg": a.get("pos")}  # fmt: skip
        cur = self.evals.current if self.evals else None
        ev = cur and {**{k: v for k, v in cur.items() if k != "episodes"},
                      "last_episodes": cur["episodes"][-5:]}  # fmt: skip
        ctx = {
            "now": time.strftime("%Y-%m-%d %H:%M:%S"),
            "page": page if isinstance(page, str) else None,
            "asked_about": focus if isinstance(focus, dict) else None,
            "rig_kind": self.spec.get("kind", "mock"),
            "folders": [r.public() for r in self.files.roots.values()],
            "session": {k: v for k, v in self.last.get("state", {}).items() if k != "type"},
            "rig": {k: v for k, v in self.last.get("rig", {}).items() if k != "type"},
            "identity": self.last.get("identity", {}).get("arms", []),
            "arms": arms,
            "loop": t.get("loop"),
            "calibration_in_progress": t.get("calibration"),
            "policy_run": t.get("policy"),
            "eval": ev,
            "checks": self._checks_summary(),
            "log": list(self.log)[-40:],
        }
        text = json.dumps(ctx, indent=1, default=str)
        return text.replace(self.token, "[token]")  # WHY: belt and braces; it should never be there


class PortInUse(RuntimeError):
    def __init__(self, port: int) -> None:
        super().__init__(f"Port {port} is in use, probably by another Studio.")
        self.port = port


def port_free(port: int) -> bool:
    """Whether Studio could bind 127.0.0.1:port. Same SO_REUSEADDR setting as aiohttp, so a port
    in TIME_WAIT after a recent Studio counts as free."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def serve(
    spec: dict[str, Any],
    port: int = 8765,
    open_browser: bool = True,
    data_dir: Path | None = None,
    assistant_model: str | None = None,
    rig_dir: Path | None = None,
) -> None:
    if not port_free(port):  # before printing a URL or opening a browser at the wrong server
        raise PortInUse(port)
    # WHY an env override: a fixed token lets a dev preview reload with the same URL.
    # The default is a fresh random token per launch.
    token = os.environ.get("PHI_STUDIO_TOKEN") or None
    studio = Studio(spec, port, token=token, data_dir=data_dir, assistant_model=assistant_model,
                    rig_dir=rig_dir)
    url = f"http://127.0.0.1:{port}/#token={studio.token}"
    print(f"Phi Studio on {url}\nCtrl+C stops Studio and releases torque.", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    web.run_app(studio.app(), host="127.0.0.1", port=port, print=None)
