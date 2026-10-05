"""The Set up page's server side: the port finder, the camera finder, live camera align, and saving
what they find to robot-config.yaml.

Writes go through configedit, which changes single values in place (comments stay) and saves a
backup first; config_lock makes them take turns. Every command that opens a camera or writes the
config needs control; the ones that only read do not.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phi_studio import configedit, rigspec
from phi_studio.errors import Refusal
from phi_studio.files import list_ports, root_order

if TYPE_CHECKING:
    import numpy as np

    from phi_studio.server import Client, Studio

PORT_RE = re.compile(r"^(/dev/[\w.\-]+|COM\d{1,3})$")
PROBE_RANGE = range(6)  # macOS numbers cameras from 0; six covers a rig of three and spares
ALIGN_PERIOD_S = 0.3  # one offset per camera about three times a second
ALIGN_IDLE_S = 120.0  # no window has shown the session for this long: free the cameras
ALIGN_ALIVE_S = 30  # how often a window showing it says so (lib/setup.ts sends align_alive)
IDLE_STOP = ("Stopped: no window showed camera align for two minutes, so the cameras are free "
             "again. Start it again to go on.")
PREVIEW_W = 480  # live and reference pictures sent to the page are this wide
MEMORY = "ports.json"  # each arm's USB serial number, so a moved port can be found again
BACKUPS = "config-backups"


class SetupError(Refusal):
    """A request Studio refuses; the message is shown as is."""


# -- files ---------------------------------------------------------------------------------------
def data_dir(studio: Studio) -> Path:
    return studio.data_dir or Path.home() / ".cache" / "phi" / "studio"


def config_file(studio: Studio) -> Path:
    for key in root_order("robot-config.yaml"):
        r = studio.files.roots.get(key)
        if r is not None and (r.path / "robot-config.yaml").is_file():
            return r.path / "robot-config.yaml"
    raise SetupError("There is no robot-config.yaml. Start Studio with --rig-dir set to the "
                     "folder that has it.")  # fmt: skip


def load_spec(path: Path) -> rigspec.RigSpec:
    return rigspec.parse(path.read_text())


def arm_id(a: rigspec.ArmSpec) -> str:
    """What the port memory files an arm under: its role and LeRobot id survive a reorder of the
    config, where a key like left_follower would not."""
    return f"{a.role}:{a.lerobot_id or a.key}"


def load_memory(path: Path) -> dict[str, dict[str, Any]]:
    try:
        got = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# -- ports ---------------------------------------------------------------------------------------
def usb_ports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """USB serial ports only, each with the name LeRobot configs use (the /dev/tty. form on macOS).
    WHY USB only: the arms' boards are USB; Bluetooth and debug ports are never an arm."""
    return [{**r, "name": r.get("tty") or r["device"]} for r in rows if r.get("usb")]


def moved_ports(arms: tuple[rigspec.ArmSpec, ...], rows: list[dict[str, Any]],
                memory: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:  # fmt: skip
    """Arms whose port in the config is gone while the board Studio remembers for them (by USB
    serial number) is plugged in under another name. Only an unambiguous match counts: one port
    with that serial, not already another arm's."""
    usb = usb_ports(rows)
    here = {port_key(r["name"]) for r in usb}
    taken = {port_key(a.port) for a in arms if a.port and port_key(a.port) in here}
    out = []
    for a in arms:
        if not a.port or port_key(a.port) in here:
            continue
        serial = (memory.get(arm_id(a)) or {}).get("serial")
        if not serial:
            continue
        hits = [r for r in usb if r.get("serial") == serial]
        if len(hits) == 1 and port_key(hits[0]["name"]) not in taken:
            out.append({"arm": a.key, "old": a.port, "new": hits[0]["name"], "serial": serial})
    # WHY: two boards can report one serial (a cloned or blank one); then neither guess is safe.
    news = [m["new"] for m in out]
    return [m for m in out if news.count(m["new"]) == 1]


def port_key(name: str) -> str:
    """One spelling per device: macOS lists each USB serial port twice, as /dev/cu.X and
    /dev/tty.X, and a config may use either."""
    return re.sub(r"^/dev/cu\.", "/dev/tty.", name)


def check_port_choice(spec: rigspec.RigSpec, ports: Any,
                      rows: list[dict[str, Any]]) -> dict[str, str]:  # fmt: skip
    """{arm key: port} as asked, refused unless every arm is in the config, every port is a USB
    serial port plugged in now, and no two arms share one."""
    if not isinstance(ports, dict) or not ports:
        raise SetupError("Say which port each arm is on.")
    by_key = {a.key: a for a in spec.arms}
    here = {port_key(r["name"]) for r in usb_ports(rows)}
    out: dict[str, str] = {}
    for key, port in ports.items():
        if key not in by_key:
            raise SetupError(f"robot-config.yaml has no arm called {key}.")
        if not isinstance(port, str) or not PORT_RE.match(port):
            raise SetupError(f"{port!r} is not a serial port name.")
        if port_key(port) not in here:
            raise SetupError(f"{port} is not plugged in now. Plug the arm in and try again.")
        out[key] = port
    # A port not being changed still counts: two arms on one port is the swap this guards against.
    final = {a.key: a.port for a in spec.arms if a.port} | out
    seen: dict[str, str] = {}
    for key, port in final.items():
        if port_key(port) in seen:
            raise SetupError(f"{label(seen[port_key(port)])} and {label(key)} would share {port}. "
                             "Each arm needs its own port.")  # fmt: skip
        seen[port_key(port)] = key
    return out


def label(key: str) -> str:
    return key.replace("_", " ")


# -- cameras -------------------------------------------------------------------------------------
def check_camera_choice(spec: rigspec.RigSpec, cams: Any) -> dict[str, int | str]:
    """{dataset key or camera key: index or path} as asked, refused unless each names a camera in
    the config and each device is used once."""
    if not isinstance(cams, dict) or not cams:
        raise SetupError("Say which camera is which.")
    by_name = {c.feature: c for c in spec.cameras} | {c.key: c for c in spec.cameras if not c.side}
    out: dict[str, int | str] = {}
    for name, src in cams.items():
        cam = by_name.get(name)
        if cam is None:
            raise SetupError(f"robot-config.yaml has no camera called {name}.")
        if cam.fields.get("type") != "opencv":
            raise SetupError(f"{cam.key} is a {cam.fields.get('type')} camera; Studio sets only "
                             "the number of ordinary (opencv) cameras.")  # fmt: skip
        if isinstance(src, bool) or not (
            (isinstance(src, int) and 0 <= src < 64)
            or (isinstance(src, str) and re.fullmatch(r"/dev/video\d{1,2}", src))
        ):
            raise SetupError(f"{src!r} is not a camera number.")
        out[cam.feature] = src
    # A camera not being changed still counts, as for ports.
    final = {c.feature: c.source for c in spec.cameras
             if c.fields.get("type") == "opencv" and c.source is not None} | out  # fmt: skip
    seen: dict[str, str] = {}
    for feature, src in final.items():
        if str(src) in seen:
            raise SetupError(f"{camera_label(seen[str(src)])} and {camera_label(feature)} would "
                             f"both read camera {src}. Give each its own.")  # fmt: skip
        seen[str(src)] = feature
    return out


def camera_label(feature: str) -> str:
    """observation.images.left_front reads Left Front camera, as on the page (lib/labels.ts)."""
    words = feature.removeprefix("observation.images.").split("_")
    return " ".join(w[:1].upper() + w[1:] for w in words if w) + " camera"


def picture(rgb: np.ndarray, width: int = PREVIEW_W) -> str:
    """A small JPEG as a data URL, for the page."""
    from PIL import Image

    img = Image.fromarray(rgb)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def full_jpeg(rgb: np.ndarray) -> bytes:
    """A frame at full size, the way the worker sends one (worker.encode_jpeg), for recon."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def rig_name(key: str) -> str:
    """observation.images.front -> front: how the worker keys a rig camera's frames
    (cameras.cameras_from_spec), so Studio.latest_frame readers find align's under the same name."""
    return key.removeprefix("observation.images.")


def decode(jpeg: bytes) -> np.ndarray:
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))


async def lerobot_busy(studio: Studio) -> str | None:
    """The LeRobot command running in Studio's terminal, if any. It may hold the arms' ports or
    the cameras, so Studio does not touch them while it runs. WHY ask the terminal when no panel
    is open: term_running is only kept fresh while one is, so it could name a finished command."""
    run = studio.term_running
    if not studio.term_socks:
        term = studio.terminal
        run = await asyncio.to_thread(term.running) if term is not None else None
    cmd = (run or {}).get("command") or ""
    return cmd if "lerobot-" in cmd else None


# -- the commands --------------------------------------------------------------------------------
class SetupApi:
    def __init__(self, studio: Studio) -> None:
        self.studio = studio
        self.align: AlignSession | None = None
        # WHY one lock for every step that opens or frees the cameras: two overlapping starts
        # used to leave a session nothing could stop, holding its camera.
        self.cam_lock = asyncio.Lock()

    @property
    def memory_file(self) -> Path:
        return data_dir(self.studio) / MEMORY

    def _ident(self) -> list[dict[str, Any]]:
        return list(self.studio.last.get("identity", {}).get("arms", []))

    async def ports(self, client: Client, msg: dict[str, Any]) -> None:
        rows = (await asyncio.to_thread(list_ports, self._ident())).get("ports", [])
        memory = load_memory(self.memory_file)
        moved: list[dict[str, Any]] = []
        try:
            spec = load_spec(config_file(self.studio))
            moved = moved_ports(spec.arms, rows, memory)
        except (SetupError, OSError, ValueError):
            pass  # no config, or one that does not parse: the page says so from the files index
        client.push({"type": "setup_ports", "ports": usb_ports(rows), "moved": moved,
                     "at": time.time(), "busy": await lerobot_busy(self.studio)})  # fmt: skip

    async def save_ports(self, client: Client, msg: dict[str, Any]) -> None:
        busy = await lerobot_busy(self.studio)
        if busy:
            raise SetupError(f"The terminal is running {busy}. Stop it before changing ports.")
        rows = (await asyncio.to_thread(list_ports, self._ident())).get("ports", [])
        path = config_file(self.studio)
        async with self.studio.config_lock:
            spec = load_spec(path)
            choice = check_port_choice(spec, msg.get("ports"), rows)
            arms = {a.key: a for a in spec.arms}
            changes = {configedit.port_path(arms[k]): v for k, v in choice.items()
                       if arms[k].port != v}  # fmt: skip
            backup = None
            if changes:
                backup = await asyncio.to_thread(configedit.write_values, path, changes,
                                                 data_dir(self.studio) / BACKUPS)  # fmt: skip
            # Remember each board by its USB serial number, so a moved port is found again.
            memory = load_memory(self.memory_file)
            usb = {r["name"]: r for r in usb_ports(rows)}
            for key, port in choice.items():
                r = usb[port]
                memory[arm_id(arms[key])] = {"port": port, "serial": r.get("serial"),
                                             "vid": r.get("vid"), "pid": r.get("pid"),
                                             "saved": time.time()}  # fmt: skip
            await asyncio.to_thread(save_json, self.memory_file, memory)
        done = ", ".join(f"{label(k)} on {v}" for k, v in choice.items())
        self.studio._note("state", f"Saved ports to robot-config.yaml: {done}")
        no_serial = [label(k) for k, v in choice.items() if not usb[v].get("serial")]
        self._saved(client, "ports", backup, [k for k in choice], no_serial=no_serial)

    async def probe(self, client: Client, msg: dict[str, Any]) -> None:
        from phi_studio import cameras

        async with self.cam_lock:
            busy = await lerobot_busy(self.studio)
            if busy:
                raise SetupError(f"The terminal is running {busy}, which may hold the cameras. "
                                 "Stop it first.")  # fmt: skip
            await self._stop_locked("Stopped to look for cameras.")
            self.studio._fanout({"type": "setup_cameras", "probing": True})
            try:
                rows = await asyncio.to_thread(cameras.probe, PROBE_RANGE)
            except BaseException:  # every window shows the spinner; every window must lose it
                self.studio._fanout({"type": "setup_cameras", "probing": False, "cameras": None,
                                     "at": time.time()})  # fmt: skip
                raise
        out = []
        for r in rows:
            thumb = r.pop("thumbnail", None)
            url = "data:image/jpeg;base64," + base64.b64encode(thumb).decode() if thumb else None
            out.append({**r, "picture": url})
        self.studio._fanout({"type": "setup_cameras", "probing": False, "cameras": out,
                             "at": time.time()})  # fmt: skip

    async def save_cameras(self, client: Client, msg: dict[str, Any]) -> None:
        path = config_file(self.studio)
        async with self.studio.config_lock:
            spec = load_spec(path)
            choice = check_camera_choice(spec, msg.get("cameras"))
            by_feature = {c.feature: c for c in spec.cameras}
            changes = {configedit.camera_source_path(by_feature[f], spec): v
                       for f, v in choice.items()}  # fmt: skip
            backup = await asyncio.to_thread(configedit.write_values, path, changes,
                                             data_dir(self.studio) / BACKUPS)  # fmt: skip
        done = ", ".join(f"{by_feature[f].key} = {v}" for f, v in choice.items())
        self.studio._note("state", f"Saved camera numbers to robot-config.yaml: {done}")
        self._saved(client, "cameras", backup, list(choice))

    def _saved(self, client: Client, what: str, backup: Path | None, keys: list[str],
               **extra: Any) -> None:  # fmt: skip
        client.push({"type": "setup_saved", "what": what, "keys": keys,
                     "backup": str(backup) if backup else None, **extra})  # fmt: skip
        try:
            self.studio._fanout({"type": "files", **self.studio.files.index()})
        except Exception:  # the save stands; the page re-reads on its own next time
            pass

    # -- camera align ----------------------------------------------------------------------------
    async def datasets(self, client: Client, msg: dict[str, Any]) -> None:
        from phi_studio import align

        found = await asyncio.to_thread(align.list_datasets)
        client.push({"type": "align_datasets", "datasets": found})
        # WHY: a reloaded page, or a second window, asks for datasets first; a session already
        # running keeps streaming ticks, so tell it what they belong to.
        if self.align is not None:
            client.push(self.align.summary())

    async def start_align(self, client: Client, msg: dict[str, Any]) -> None:
        try:
            async with self.cam_lock:
                await self._start_locked(msg)
        except BaseException:
            # every window: not starting
            self.studio._fanout({"type": "align_status", "text": None})
            raise

    async def _start_locked(self, msg: dict[str, Any]) -> None:
        from phi_studio import align, cameras

        busy = await lerobot_busy(self.studio)
        if busy:
            raise SetupError(f"The terminal is running {busy}, which may hold the cameras. "
                             "Stop it first.")  # fmt: skip
        root, episode = msg.get("root"), msg.get("episode", 0)
        known = {d["root"] for d in await asyncio.to_thread(align.list_datasets)}
        if not isinstance(root, str) or root not in known:
            raise SetupError("Pick one of the datasets on this Mac.")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise SetupError("The episode is a whole number from 0.")
        await self._stop_locked("Stopped to start again.")
        say = self._status
        say("Reading the dataset's resting frame")
        try:
            ref = await asyncio.to_thread(align.references, root, episode,
                                          data_dir(self.studio) / "align-cache")  # fmt: skip
        except align.AlignError as e:
            raise SetupError(str(e)) from None
        say("Looking for cameras")
        rows = [r for r in await asyncio.to_thread(cameras.probe, PROBE_RANGE) if r["ok"]]
        if not rows:
            raise SetupError("No camera answered. Plug them in, and check that the app you "
                             "started Studio from may use the camera (System Settings, "
                             "Privacy).")  # fmt: skip
        say("Matching each camera to the dataset by its picture")
        live = {r["source"]: decode(r["thumbnail"]) for r in rows}
        try:
            match = await asyncio.to_thread(align.match_cameras, live, ref["images"])
        except align.AlignError as e:
            raise SetupError(str(e)) from None
        spec = None
        try:
            spec = load_spec(config_file(self.studio))
        except (SetupError, OSError, ValueError):
            pass
        s = AlignSession(self, ref, match, spec)
        try:
            await asyncio.to_thread(s.open)
        except BaseException:
            await s.close()
            raise
        self.align = s
        self.studio._fanout(s.summary())
        s.task = asyncio.get_running_loop().create_task(s.run())

    async def stop_align(self, client: Client | None, msg: dict[str, Any]) -> None:
        async with self.cam_lock:
            await self._stop_locked(msg.get("why", "Stopped"))

    async def _stop_locked(self, why: str) -> None:
        s, self.align = self.align, None
        if s is not None:
            await s.stop()
            self.studio._fanout({"type": "align_stopped", "why": why})

    async def assign(self, client: Client, msg: dict[str, Any]) -> None:
        """The user changed which dataset camera a live camera shows."""
        async with self.cam_lock:
            if self.align is None:
                raise SetupError("Camera align is not running.")
            await self.align.reassign(msg.get("assignment"))
            self.studio._fanout(self.align.summary())

    async def alive(self, client: Client, msg: dict[str, Any]) -> None:
        """A window still shows the session, so it is not idle."""
        if self.align is not None:
            self.align.seen = time.monotonic()

    def _status(self, text: str) -> None:
        self.studio._fanout({"type": "align_status", "text": text})


class AlignSession:
    """Live cameras against one dataset's resting frame: which way to move each one."""

    def __init__(self, api: SetupApi, ref: dict[str, Any], match: dict[str, Any],
                 spec: rigspec.RigSpec | None) -> None:  # fmt: skip
        self.api = api
        self.ref = ref
        self.match = match
        self.spec = spec
        self.assignment: dict[Any, str] = dict(match["assignment"])  # live source -> dataset key
        self.cams: dict[Any, Any] = {}
        self.task: asyncio.Task[None] | None = None
        self.seen = time.monotonic()  # the last time a window said it shows this session
        self.refs_sent = {k: picture(img) for k, img in ref["images"].items()}
        # rig name -> the frame message this session put in Studio.latest_frame, so closing takes
        # back only its own (a worker frame that replaced one stays).
        self.published: dict[str, dict[str, Any]] = {}
        self.sent: dict[str, tuple[Any, int]] = {}  # rig name -> (camera, seq) last published

    def _fields(self, key: str, source: Any) -> dict[str, Any]:
        """Open a live camera the way the rig's config does for that dataset key, at its size."""
        base: dict[str, Any] = {"type": "opencv", "width": 640, "height": 480, "fps": 30}
        if self.spec is not None:
            cam = next((c for c in self.spec.cameras if c.feature == key), None)
            if cam is not None and cam.fields.get("type") == "opencv":
                base = {**base, **cam.lerobot_fields()}
        return {**base, "index_or_path": source}

    def open(self) -> None:
        from phi_studio.cameras import RealCamera

        for src, key in self.assignment.items():
            self.cams[src] = RealCamera(str(key), self._fields(key, src))

    async def reassign(self, assignment: Any) -> None:
        keys = set(self.ref["images"])
        if not isinstance(assignment, dict) or not all(
            isinstance(v, str) and v in keys for v in assignment.values()
        ):
            raise SetupError("Each camera must show one of the dataset's cameras.")
        if len(set(assignment.values())) != len(assignment):
            raise SetupError("Two cameras cannot show the same dataset camera.")
        live = {str(s): s for s in self.match["scores"]}
        new = {live[str(s)]: k for s, k in assignment.items() if str(s) in live}
        await self.close()
        self.assignment = new
        self.seen = time.monotonic()
        await asyncio.to_thread(self.open)

    def config_names(self) -> dict[str, str | None]:
        """Dataset key -> the config camera with that dataset key, or None when the config has
        none (the dataset used other names)."""
        feats = {c.feature: c.key for c in self.spec.cameras} if self.spec else {}
        return {k: feats.get(k) for k in self.ref["images"]}

    def summary(self) -> dict[str, Any]:
        m = self.match
        return {"type": "align_session", "root": self.ref["root"], "episode": self.ref["episode"],
                "frame": self.ref["frame"], "physical": self.ref.get("physical", {}),
                "references": self.refs_sent, "config": self.config_names(),
                "assignment": [{"live": s, "key": k} for s, k in self.assignment.items()],
                "scores": {str(s): v for s, v in m["scores"].items()},
                "unsure": m.get("unsure"), "why": m.get("why"),
                "unmatched_refs": m.get("unmatched_refs", [])}  # fmt: skip

    async def run(self) -> None:
        from phi_studio import align

        studio = self.api.studio
        try:
            while True:
                if time.monotonic() - self.seen > ALIGN_IDLE_S:
                    await self._end(IDLE_STOP)
                    return
                busy = await lerobot_busy(studio)
                if busy:
                    await self._end(f"Stopped: the terminal started {busy}, which needs the "
                                    "cameras.")  # fmt: skip
                    return
                for src, key in list(self.assignment.items()):
                    cam = self.cams.get(src)
                    if cam is None:
                        continue
                    try:
                        frame, t, seq = cam.read_latest()
                    except ConnectionError as e:
                        studio._fanout({"type": "align_tick", "live": src, "key": key,
                                        "error": str(e)})  # fmt: skip
                        continue
                    await self._publish(src, cam, frame, t, seq)
                    m = await asyncio.to_thread(align.measure, frame, self.ref["images"][key])
                    pic = await asyncio.to_thread(picture, frame)
                    studio._fanout({"type": "align_tick", "live": src, "key": key, **m,
                                    "picture": pic})  # fmt: skip
                await asyncio.sleep(ALIGN_PERIOD_S)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # WHY: a crash must still free the cameras and say why
            await self._end(f"Camera align stopped on an error: {type(e).__name__}: {e}")

    async def _publish(self, src: Any, cam: Any, frame: np.ndarray, t: float, seq: int) -> None:
        """Offer a new frame to Studio.latest_frame, as a worker frame would be, so recon can
        capture live while align holds the cameras. WHY "arrived" and not frame_clock: the
        frame clock and Studio.cameras are the worker's, and the camera check reads them; align's
        three frames a second there would read as a slow worker camera. WHY t: RealCamera stamps
        each frame on time.monotonic when it arrives, in this process."""
        key = self.assignment.get(src)
        if key is None:
            return
        name = rig_name(key)
        if self.sent.get(name) == (cam, seq):
            return  # no new frame since the last tick
        jpeg = await asyncio.to_thread(full_jpeg, frame)
        # WHY again after the encode: a reassign may have closed or renamed this camera meanwhile.
        if self.cams.get(src) is not cam or self.assignment.get(src) != key:
            return
        msg = {"type": "frame", "key": name, "t": t, "seq": seq, "arrived": t,
               "w": int(frame.shape[1]), "h": int(frame.shape[0]), "from": "camera align",
               "jpeg": jpeg}  # fmt: skip
        self.sent[name] = (cam, seq)
        self.published[name] = msg
        self.api.studio.latest_frame[name] = msg

    def _withdraw(self) -> None:
        latest = self.api.studio.latest_frame
        for name, msg in self.published.items():
            if latest.get(name) is msg:
                latest.pop(name, None)
        self.published, self.sent = {}, {}

    async def _end(self, why: str) -> None:
        """Stop this session from inside its own loop. WHY the identity check: a newer session may
        have replaced this one; then only this one's cameras close."""
        async with self.api.cam_lock:
            if self.api.align is self:
                await self.api._stop_locked(why)
                return
        await self.close()

    async def close(self) -> None:
        self._withdraw()  # WHY first: no capture may get a frame from a camera being closed
        cams, self.cams = self.cams, {}
        for c in cams.values():
            await asyncio.to_thread(c.close)

    async def stop(self) -> None:
        if self.task is not None and self.task is not asyncio.current_task():
            self.task.cancel()
        await self.close()


def register(studio: Studio) -> None:
    api = SetupApi(studio)
    studio.setup_api = api  # type: ignore[attr-defined]
    studio.handle("setup_ports", api.ports, control=False)
    studio.handle("setup_ports_save", api.save_ports, control=True)
    studio.handle("setup_cameras_probe", api.probe, control=True)
    studio.handle("setup_cameras_save", api.save_cameras, control=True)
    studio.handle("align_datasets", api.datasets, control=False)
    studio.handle("align_start", api.start_align, control=True)
    studio.handle("align_assign", api.assign, control=True)
    studio.handle("align_stop", api.stop_align, control=True)
    studio.handle("align_alive", api.alive, control=False)
    studio.on_close.append(lambda: api.stop_align(None, {}))
