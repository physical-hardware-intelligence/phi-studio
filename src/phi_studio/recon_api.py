"""Environment reconstruction in Studio: commands, the point-cloud route, and the depth thread.

    recon_status            any window    model on disk?, download progress, what runs, live cameras
    recon_datasets          any window    local LeRobot datasets (align.list_datasets)
    recon_capture           any window    one cloud for one camera, from a dataset frame or live
    recon_keep              any window    keep capturing one live camera, at most 2 per second
    recon_download          control       fetch the depth model (99 MB) at the pinned revision
    recon_download_cancel   control
    GET /api/recon/cloud/{id}             the cloud as binary (recon.pack); needs the token

Capture reads, it moves nothing, so any window may ask. The download writes 99 MB to disk, so
only the window with control may start or stop it.

Frames come only from what Studio already has: a dataset frame decoded from its video file, or
the newest frame the robot worker streams (teleop, policy). Nothing here opens a camera device.

Depth runs on one thread of its own (DEPTH_THREAD), one job at a time, latest wins per camera.
WHY not asyncio's default executor: it is shared with the terminal and the checks, and a 350 ms
CPU job queued there would stall them.

The route checks the Host like every Studio route, refuses any Origin but Studio's own, and wants
the launch token in the X-Phi-Token header. WHY a token when the 3D model routes need none: a
cloud is a picture of the person's room. WHY a header, not the query: a same-origin fetch sends no
Origin, so the token is what stops another local program or page; a custom header also forces a
CORS preflight from any other origin, which Studio never answers.
"""

from __future__ import annotations

import asyncio
import io
import math
import secrets
import time
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from aiohttp import web

from phi_studio import depth_model, recon, robot_model
from phi_studio.errors import Refusal

if TYPE_CHECKING:
    from phi_studio.server import Client, Studio

DEPTH_THREAD = "phi-depth"
KEEP_PERIOD_S = 0.5  # "keep updating": at most 2 captures a second, for one camera
# [JUDGEMENT] A camera counts as streaming when its newest frame is younger than this. WHY 2 s:
# the worker sends about 30 frames a second; 2 s without one means teleop or policy stopped.
LIVE_FRESH_S = 2.0
# WHY 2 kept: while keep-updating replaces a cloud, the page may still be fetching the last one.
CLOUDS_PER_CAMERA = 2
TOKEN_HEADER = "X-Phi-Token"
DEGREES = robot_model.arm_units(True, None)  # what an arm robot-config.yaml does not name reads
NO_LIVE = "No camera is streaming. Start teleop or camera align to capture live."

_states: weakref.WeakKeyDictionary[Studio, Recon] = weakref.WeakKeyDictionary()


def state_of(studio: Studio) -> Recon:
    return _states[studio]


@dataclass
class Job:
    camera: str  # the scene camera this cloud belongs to (front, top, wrist, ...)
    request: dict[str, Any]


Result = dict[str, Any] | None  # None: a newer capture of the same camera replaced this one


# -- frame sources ---------------------------------------------------------------------------------
@dataclass
class Frame:
    rgb: np.ndarray
    readings: dict[str, dict[str, float]]  # arm name -> joint -> reading (degrees, gripper 0..100)
    physical: str | None  # wrist | front | top when known
    source: dict[str, Any]


@lru_cache(maxsize=2)
def _lerobot_dataset(repo_id: str, root: str, episode: int) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # WHY pyav: the backend these datasets were recorded with (info.json video.video_backend),
    # and the decoder the brief for this feature names; torchcodec would also work.
    return LeRobotDataset(repo_id, root=root, episodes=[episode], video_backend="pyav")


def dataset_frame(root: str, episode: int, frame: int, key: str) -> Frame:
    """One recorded frame of one camera key, with that frame's joint readings. Finds the dataset
    the way camera align does (camera_realign.discover), so the two agree on which datasets exist
    and which camera a key shows."""
    from phi_studio import align

    try:
        from phi_studio import camera_realign as cr
    except ImportError as e:  # WHY: camera_realign loads LeRobot and OpenCV when imported
        raise Refusal(
            "Reading a dataset frame needs LeRobot and OpenCV, which Studio's Python does not "
            "have.",
            "Start Studio from the environment LeRobot is installed in.",
        ) from e

    with align._readable():
        want = str(Path(root).expanduser().resolve())
        ds = next((d for d in cr.discover() if str(Path(d.root).resolve()) == want), None)
        if ds is None:
            raise Refusal(f"No LeRobot dataset at {root}.")
        if not 0 <= episode < ds.episodes:
            raise Refusal(f"Episode {episode} is out of range: this dataset has episodes "
                          f"0 to {ds.episodes - 1}.")  # fmt: skip
        if key not in ds.cameras:
            raise Refusal(f"This dataset has no camera {key}. It has {', '.join(ds.cameras)}.")
        lds = _lerobot_dataset(ds.repo_id, ds.root, episode)
        if not 0 <= frame < len(lds):
            raise Refusal(f"Frame {frame} is out of range: episode {episode} has frames 0 to "
                          f"{len(lds) - 1}.")  # fmt: skip
        item = lds[frame]
        img = item[key]  # (3, H, W) float in 0..1
        rgb = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        names = lds.meta.features.get("observation.state", {}).get("names") or []
        state = item.get("observation.state")
        readings = {}
        # Read as degrees, the mode Studio's telemetry uses. ASSUMED: a dataset recorded with
        # use_degrees=False (-100..100) would place the wrist camera and arm boxes wrongly.
        if state is not None and len(names) == len(robot_model.JOINTS):
            readings["dataset"] = {str(n).removesuffix(".pos"): float(v)
                                   for n, v in zip(names, state.tolist(), strict=True)}  # fmt: skip
        physical = {k: n for n, k in ds.key_map[0].items()}.get(key)
    return Frame(rgb, readings, physical,
                 {"kind": "dataset", "name": ds.name, "root": ds.root, "episode": episode,
                  "frame": frame, "key": key})  # fmt: skip


def live_frame(studio: Studio, key: str) -> Frame:
    """The newest frame the worker streamed for `key`, with the newest joint readings."""
    from PIL import Image

    msg = studio.latest_frame.get(key)
    arrived = studio.frame_clock.get(key, (None, None))[0]
    if msg is None or arrived is None or time.monotonic() - arrived > LIVE_FRESH_S:
        raise Refusal(NO_LIVE)
    rgb = np.asarray(Image.open(io.BytesIO(msg["jpeg"])).convert("RGB"), dtype=np.uint8)
    arms = (studio.telemetry or {}).get("arms") or {}
    readings = {name: dict(a.get("pos") or {}) for name, a in arms.items()}
    return Frame(rgb, readings, None, {"kind": "live", "key": key, "seq": msg.get("seq")})


# -- request checking ------------------------------------------------------------------------------
def _vec(v: Any, what: str) -> tuple[float, float, float]:
    if not (isinstance(v, list | tuple) and len(v) == 3
            and all(isinstance(x, int | float) and math.isfinite(x) for x in v)):  # fmt: skip
        raise Refusal(f"{what} must be three numbers.")
    return float(v[0]), float(v[1]), float(v[2])


def _pose(p: Any, name: str) -> dict[str, Any]:
    if not isinstance(p, dict):
        raise Refusal(f"Place the {name} camera in the 3D view first.")
    fovy = p.get("fovy_deg")
    if not (isinstance(fovy, int | float) and 5 <= fovy <= 150):
        raise Refusal(f"The {name} camera's vertical field of view must be 5 to 150 degrees.")
    return {"pos": _vec(p.get("pos"), "Position"), "target": _vec(p.get("target"), "Looks at"),
            "up": _vec(p.get("up"), "Up"), "fovy_deg": float(fovy),
            "placed": bool(p.get("placed"))}  # fmt: skip


def _arms(v: Any) -> list[dict[str, Any]]:
    """[{name, base}] from the page: each follower drawn and where its base sits."""
    if not isinstance(v, list):
        return []
    out = []
    for a in v[:4]:
        if isinstance(a, dict) and isinstance(a.get("name"), str):
            out.append({"name": a["name"], "base": _vec(a.get("base", [0, 0, 0]), "Arm base")})
    return out


def _is_wrist(name: str) -> bool:
    """Only the exact key rides on the arm, as the 3D view draws it (engine.ts syncCameras); a
    left_wrist or right_wrist is a placed camera there too, until the view learns per-arm wrists."""
    return name == str(robot_model.WRIST_CAMERA["key"])


# -- the work, on the depth thread -----------------------------------------------------------------
class Recon:
    def __init__(self, studio: Studio) -> None:
        self.studio = studio
        self.cache_dir: Path | None = None  # None: the Hugging Face default (tests point it away)
        self.model: Any = depth_model.DepthModel()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=DEPTH_THREAD)
        self.download = depth_model.Download(self._download_changed)
        self.clouds: OrderedDict[str, tuple[str, bytes]] = OrderedDict()  # id -> (camera, blob)
        # camera -> the newest job not started yet, and the future its caller awaits
        self.pending: OrderedDict[str, tuple[Job, asyncio.Future[Result]]] = OrderedDict()
        self.running: str | None = None
        self.pump: asyncio.Task[None] | None = None
        self.keep: dict[str, Any] | None = None  # {"camera", "request", "client"}
        self.keep_task: asyncio.Task[None] | None = None
        self.closed = False

    # -- status ---------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        live = sorted(k for k, (t, _) in list(self.studio.frame_clock.items())
                      if t is not None and now - t <= LIVE_FRESH_S)  # fmt: skip
        self.download.cache_dir = self.cache_dir
        return {
            "type": "recon", "kind": "status",
            "model": {"cached": depth_model.cached(self.cache_dir), "repo": depth_model.REPO,
                      "revision": depth_model.REVISION, "license": depth_model.LICENSE,
                      "bytes": depth_model.WEIGHTS_BYTES,
                      "device": getattr(self.model, "device", None)},
            "download": self.download.view(),
            "running": self.running, "pending": list(self.pending),
            "keep": self.keep["camera"] if self.keep else None,
            "live": live,
            "limits": {"max_points": recon.MAX_POINTS, "keep_hz": 1 / KEEP_PERIOD_S,
                       "workspace_radius_m": recon.WORKSPACE_RADIUS_M,
                       "min_inlier_fraction": recon.MIN_INLIER_FRACTION},
        }  # fmt: skip

    def broadcast(self, msg: dict[str, Any]) -> None:
        self.studio._fanout(msg)

    def _download_changed(self, view: dict[str, Any]) -> None:
        """Called from the download thread."""
        loop = self.studio.loop
        if loop is not None and not self.closed:
            loop.call_soon_threadsafe(lambda: self.broadcast(self.status()))

    # -- jobs -----------------------------------------------------------------------------------
    async def submit(self, job: Job) -> Result:
        """Queue a capture. A newer job for the same camera replaces one not started yet; the
        replaced one resolves to None."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Result] = loop.create_future()
        old = self.pending.pop(job.camera, None)
        if old is not None and not old[1].done():
            old[1].set_result(None)
        self.pending[job.camera] = (job, fut)
        if self.pump is None or self.pump.done():
            self.pump = loop.create_task(self._pump())
        self.broadcast(self.status())
        return await fut

    async def _pump(self) -> None:
        loop = asyncio.get_running_loop()
        while self.pending and not self.closed:
            camera, (job, fut) = self.pending.popitem(last=False)
            self.running = camera
            self.broadcast(self.status())
            try:
                msg = await loop.run_in_executor(self.executor, self.compute, job)
            except Exception as e:  # WHY all: a crash must reach the page, not hang it
                msg = {"type": "recon", "kind": "refused", "camera": camera,
                       "message": f"Studio could not build the cloud: {type(e).__name__}: {e}"}
            self.running = None
            if msg.get("kind") == "cloud":
                self._store(camera, msg.pop("_blob"), msg["id"])
            self.broadcast(msg)
            if not fut.done():
                fut.set_result(msg)
        self.broadcast(self.status())

    def _store(self, camera: str, blob: bytes, cid: str) -> None:
        self.clouds[cid] = (camera, blob)
        mine = [k for k, (c, _) in self.clouds.items() if c == camera]
        for k in mine[:-CLOUDS_PER_CAMERA]:
            del self.clouds[k]

    def compute(self, job: Job) -> dict[str, Any]:
        """On the depth thread: frame, depth, scale from the table, cloud."""
        t0 = time.perf_counter()
        r = job.request
        base_msg: dict[str, Any] = {"type": "recon", "camera": job.camera}
        try:
            frame = self._frame(r)
            t_frame = time.perf_counter()
            out = self._cloud(job.camera, r, frame)
        except recon.ScaleRefused as e:  # a Refusal too: caught first for its fit numbers
            fit = e.fit.numbers() if e.fit is not None else None
            return {**base_msg, "kind": "refused", "message": str(e), "fit": fit}
        except Refusal as e:  # DepthUnavailable is one: the page then offers the download
            return {**base_msg, "kind": "refused", "message": str(e),
                    "model_missing": isinstance(e, depth_model.DepthUnavailable)}  # fmt: skip
        except ValueError as e:  # AlignError and other plain problems with the request
            return {**base_msg, "kind": "refused", "message": str(e)}
        out.setdefault("ms", {})["frame"] = round((t_frame - t0) * 1000)
        out["ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        return {**base_msg, **out}

    def _units(self) -> dict[str, Any]:
        """robot-config.yaml's units per arm; {} when this Studio has no file roots (tests)."""
        if getattr(self.studio, "files", None) is None:
            return {}
        from phi_studio import scene_api

        return scene_api.rig_units(self.studio)

    def _frame(self, r: dict[str, Any]) -> Frame:
        if r.get("source") == "dataset":
            root, episode, frame = r.get("root"), r.get("episode"), r.get("frame")
            key = r.get("key")
            if not isinstance(root, str) or not root:
                raise Refusal("Pick a dataset.")
            if not (isinstance(episode, int) and isinstance(frame, int)):
                raise Refusal("Episode and frame must be whole numbers.")
            if not isinstance(key, str):
                raise Refusal("Pick a camera of the dataset.")
            return dataset_frame(root, episode, frame, key)
        if r.get("source") == "live":
            key = r.get("key")
            if not isinstance(key, str):
                raise Refusal("Pick a camera.")
            return live_frame(self.studio, key)
        raise Refusal("Pick where the picture comes from: a dataset frame or live.")

    def _cloud(self, camera: str, r: dict[str, Any], frame: Frame) -> dict[str, Any]:
        model = robot_model.load()
        timings: dict[str, int] = {}
        arms = _arms(r.get("arms"))
        drawn_on: str | None = None  # the follower in the 3D view that shows a dataset frame's pose
        if frame.source["kind"] == "dataset":
            # A dataset holds one arm. WHY on the wrist arm's base: the page draws the frame's
            # pose on that follower, so the arm boxes and the wrist camera must sit where it does.
            want = r.get("wrist_arm")
            host = next((a for a in arms if a["name"] == want), arms[0] if arms else None)
            drawn_on = host["name"] if host else None
            arms = [{"name": "dataset", "base": host["base"] if host else (0.0, 0.0, 0.0)}]
        elif not arms:
            arms = [{"name": n, "base": (0.0, 0.0, 0.0)} for n in list(frame.readings)[:1]]
        # Each arm's units from robot-config.yaml, as the 3D view reads them (scene_api.rig_units,
        # web/src/lib/sceneCore.ts readArm). ASSUMED: a dataset frame was recorded in the units of
        # the follower that shows it; nothing in a LeRobot dataset says which mode it used.
        rig = self._units()
        units: dict[str, dict[str, Any]] = {}
        for a in arms:
            who = drawn_on if a["name"] == "dataset" else a["name"]
            units[a["name"]] = (rig.get(who) if who else None) or DEGREES
        angles: dict[str, dict[str, float]] = {}
        for a in arms:
            reads = frame.readings.get(a["name"])
            u = units[a["name"]]
            if u["problem"] is None and reads and all(j in reads for j in robot_model.JOINTS):
                angles[a["name"]] = robot_model.lerobot_to_mjcf(
                    model, reads, units=u["unit"], calibration=u["calibration"])

        estimate: dict[str, Any]
        if _is_wrist(camera):
            wrist_arm = r.get("wrist_arm") if frame.source["kind"] == "live" else "dataset"
            arm = next((a for a in arms if a["name"] == wrist_arm), arms[0] if arms else None)
            if arm is not None and units[arm["name"]]["problem"]:
                raise Refusal(f"The wrist camera cannot be placed: its arm "
                              f"{units[arm['name']]['problem']}.")  # fmt: skip
            if arm is None or arm["name"] not in angles:
                raise Refusal("The wrist camera rides on the arm, and there is no joint reading "
                              "for that arm to place it.")  # fmt: skip
            c2w = recon.wrist_camera_pose(model, angles[arm["name"]], arm["base"])
            fovy = float(robot_model.WRIST_CAMERA["fovy_deg"])  # type: ignore[arg-type]
            estimate = {"fovy_deg": fovy, "fov": "estimated, not measured",
                        "placement": "mount from the CAD, moved by the frame's joint angles"}
        else:
            p = _pose((r.get("poses") or {}).get(camera), camera)
            c2w = recon.look_at(p["pos"], p["target"], p["up"])
            fovy = p["fovy_deg"]
            estimate = {"fovy_deg": fovy, "fov": "set in the 3D view, an estimate",
                        "placement": "placed by you" if p["placed"] else "the starting guess"}
        estimate["lens"] = "pinhole, centred, no distortion model"

        h0, w0 = frame.rgb.shape[:2]
        w, h = recon.working_size(w0, h0, recon.MAX_POINTS)
        # WHY the x stretch: flooring each side can shrink width and height by slightly
        # different factors (848x480 -> 282x160), so fx must follow the width's own factor.
        cam = recon.Pinhole.from_fovy(w, h, fovy, x_stretch=(w / w0) / (h / h0))
        from PIL import Image

        small = np.asarray(Image.fromarray(frame.rgb).resize((w, h), Image.Resampling.BOX))
        t = time.perf_counter()
        disparity = np.asarray(self.model(frame.rgb, (w, h)), dtype=np.float64)
        timings["depth"] = round((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        boxes = [recon.arm_boxes(model, angles[a["name"]], a["base"])
                 for a in arms if a["name"] in angles]  # fmt: skip
        try:
            cloud = recon.reconstruct(small, disparity, cam, c2w,
                                      bases=[a["base"][:2] for a in arms] or [(0.0, 0.0)],
                                      arms=boxes)  # fmt: skip
        except recon.ScaleRefused as e:  # the numbers and labels still help the person fix it
            return {"kind": "refused", "message": str(e), "source": frame.source,
                    "estimate": estimate, "ms": timings,
                    "fit": e.fit.numbers() if e.fit is not None else None}
        timings["fit"] = round((time.perf_counter() - t) * 1000)
        blob = recon.pack(cloud)
        cid = secrets.token_urlsafe(12)
        pose = None
        if drawn_on is not None and "dataset" in frame.readings:
            # The raw readings, as telemetry sends them: the page converts both the same way.
            # The units too, so the page draws it, or refuses to, exactly as the boxes were made.
            pose = {"arm": drawn_on, "pos": frame.readings["dataset"], "units": units["dataset"],
                    "label": f"pose from episode {frame.source['episode']}, "
                             f"frame {frame.source['frame']}"}  # fmt: skip
        return {"kind": "cloud", "id": cid, "url": f"/api/recon/cloud/{cid}", "bytes": len(blob),
                "n": len(cloud.positions), "n_off_arm": cloud.n_off_arm,
                "arm_hidden_by": "boxes around each part of the drawn arm" if boxes else None,
                "fit": cloud.fit.numbers(), "source": frame.source, "estimate": estimate,
                "pose": pose, "size": [w, h], "frame_size": [w0, h0], "ms": timings,
                "device": getattr(self.model, "device", None), "at": time.time(),
                "_blob": blob}  # fmt: skip

    # -- keep updating --------------------------------------------------------------------------
    async def _keep_loop(self) -> None:
        last_seq: Any = None
        while self.keep is not None and not self.closed:
            k = self.keep
            client: Client = k["client"]
            if client.ws.closed:
                break
            t0 = time.monotonic()
            msg = self.studio.latest_frame.get(k["request"]["key"])
            seq = msg.get("seq") if msg else None
            if seq is not None and seq != last_seq:
                last_seq = seq
                out = await self.submit(Job(k["camera"], k["request"]))
                if out and out.get("kind") == "refused" and out.get("message") == NO_LIVE:
                    break
            elif msg is None or time.monotonic() - (self.studio.frame_clock.get(
                    k["request"]["key"], (0.0, None))[0] or 0.0) > LIVE_FRESH_S:  # fmt: skip
                self.broadcast({"type": "recon", "kind": "refused", "camera": k["camera"],
                                "message": NO_LIVE})  # fmt: skip
                break
            await asyncio.sleep(max(0.05, KEEP_PERIOD_S - (time.monotonic() - t0)))
        self.keep = None
        self.keep_task = None
        if not self.closed:
            self.broadcast(self.status())

    def close(self) -> None:
        self.closed = True
        self.download.close()
        for t in (self.pump, self.keep_task):
            if t is not None:
                t.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)


def _scene_camera(r: dict[str, Any]) -> str:
    """The scene camera a request draws into: the physical name for a dataset key when known."""
    cam = r.get("camera")
    if isinstance(cam, str) and cam:
        return cam
    key = r.get("key")
    return key.removeprefix("observation.images.") if isinstance(key, str) else ""


def register(studio: Studio) -> None:
    st = Recon(studio)
    _states[studio] = st
    async def close() -> None:
        st.close()

    studio.on_close.append(close)

    async def status(client: Client, msg: dict[str, Any]) -> None:
        client.push(st.status())

    async def datasets(client: Client, msg: dict[str, Any]) -> None:
        from phi_studio import align

        loop = asyncio.get_running_loop()
        try:
            found = await loop.run_in_executor(st.executor, align.list_datasets)
        except ValueError as e:
            client.push({"type": "recon", "kind": "datasets", "datasets": [], "error": str(e)})
            return
        client.push({"type": "recon", "kind": "datasets", "datasets": found, "error": None})

    async def capture(client: Client, msg: dict[str, Any]) -> None:
        r = dict(msg)
        camera = _scene_camera(r)
        if not camera:
            client.push({"type": "recon", "kind": "refused", "camera": "",
                         "message": "Pick a camera."})  # fmt: skip
            return
        await st.submit(Job(camera, r))

    async def keep(client: Client, msg: dict[str, Any]) -> None:
        if not msg.get("on"):
            st.keep = None
            client.push(st.status())
            return
        r = dict(msg)
        if r.get("source") != "live":
            client.push({"type": "recon", "kind": "refused", "camera": _scene_camera(r),
                         "message": "Keep updating works with live cameras only."})  # fmt: skip
            return
        camera = _scene_camera(r)
        st.keep = {"camera": camera, "request": r, "client": client}
        if st.keep_task is None or st.keep_task.done():
            st.keep_task = asyncio.get_running_loop().create_task(st._keep_loop())
        st.broadcast(st.status())

    async def download(client: Client, msg: dict[str, Any]) -> None:
        st.download.cache_dir = st.cache_dir
        if depth_model.cached(st.cache_dir):
            client.push(st.status())
            return
        st.download.start()

    async def cancel(client: Client, msg: dict[str, Any]) -> None:
        st.download.cancel()

    studio.handle("recon_status", status, control=False)
    studio.handle("recon_datasets", datasets, control=False)
    studio.handle("recon_capture", capture, control=False)
    studio.handle("recon_keep", keep, control=False)
    studio.handle("recon_download", download, control=True)
    studio.handle("recon_download_cancel", cancel, control=True)

    async def cloud(request: web.Request) -> web.StreamResponse:
        if request.host not in studio.allowed_hosts:
            return web.Response(status=403, text=f"host {request.host!r} not allowed")
        origin = request.headers.get("Origin")
        if origin is not None and origin not in studio.allowed_origins:
            return web.Response(status=403, text="origin not allowed")
        tok = request.headers.get(TOKEN_HEADER, "")
        if not secrets.compare_digest(tok.encode(), studio.token.encode()):
            return web.Response(status=403, text="bad or missing token")
        got = st.clouds.get(request.match_info["id"])
        if got is None:
            raise web.HTTPNotFound(text="no such cloud: it was replaced by a newer capture")
        return web.Response(body=got[1], content_type="application/octet-stream",
                            headers={"Cache-Control": "no-store"})  # fmt: skip

    studio.routes.append(web.get("/api/recon/cloud/{id}", cloud))

