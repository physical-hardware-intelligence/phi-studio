"""The 3D view's server side: two read-only routes for the SO-101 model, and one read-only command.

    GET /api/scene/model        robot_model.model_json()
    GET /api/scene/mesh/{name}  one binary STL, by a mesh name from that JSON
    cmd scene_arms              each arm's joint units, from robot-config.yaml (arm_units)

No token: the model and meshes are public CAD (Apache-2.0, see NOTICE) and these routes move
nothing. The Host check stays, as on every Studio route, so a DNS-rebinding page cannot read them
through a name that resolves to 127.0.0.1. Origin is not checked: a same-origin GET carries none.
scene_arms reads this Mac's rig files, so it is a WebSocket command, behind the token like the rest.

Not here yet, by design: point clouds from the cameras' depth (environment reconstruction) and
the real-arm backend. A future depth feature can add its own route beside these and draw into the
same three.js scene (web/src/scene/engine.ts), which is already in the robot's base frame.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from phi_studio import robot_model

if TYPE_CHECKING:
    from phi_studio.server import Client, Studio

# WHY immutable only with the hash: each mesh URL in the model carries a hash of the file
# (robot_model.model_json), so a changed mesh gets a new URL and a cached one can never be stale.
# A request without that hash, or with an old one, could be cached against the wrong file.
CACHE_FOREVER = "public, max-age=31536000, immutable"


def rig_units(studio: Studio) -> dict[str, Any]:
    """robot_model.arm_units for every arm robot-config.yaml names, keyed by Studio's arm name. An
    arm the config does not name reads degrees, the unit of Studio's rig contract (rig.py
    JointHealth), so it is left out. Reads the config and calibration files: run it off the loop."""
    import yaml

    from phi_studio import rigspec
    from phi_studio.files import root_order

    files = studio.files
    cal = files.roots.get("calibration")
    for key in root_order("robot-config.yaml"):
        r = files.roots.get(key)
        if r is None or not (r.path / "robot-config.yaml").is_file():
            continue
        try:
            spec = rigspec.parse((r.path / "robot-config.yaml").read_text())
        except (yaml.YAMLError, OSError):
            return {}
        out = {}
        for a in spec.arms:
            data = None
            path = a.calibration_path(cal.path) if cal else None
            if not a.use_degrees and path is not None and path.is_file():
                try:
                    data = json.loads(path.read_text())
                except (OSError, ValueError):
                    data = None
            out[a.key] = robot_model.arm_units(a.use_degrees, data)
        return out
    return {}


def register(studio: Studio) -> None:
    def refuse(request: web.Request) -> web.Response | None:
        if request.host not in studio.allowed_hosts:
            return web.Response(status=403, text=f"host {request.host!r} not allowed")
        return None

    async def model(request: web.Request) -> web.StreamResponse:
        bad = refuse(request)
        if bad is not None:
            return bad
        # WHY no-cache: the JSON names the mesh hashes, so it must be fresh after an update.
        return web.json_response(robot_model.model_json(), headers={"Cache-Control": "no-cache"})

    async def mesh(request: web.Request) -> web.StreamResponse:
        bad = refuse(request)
        if bad is not None:
            return bad
        name = request.match_info["name"]
        path = robot_model.mesh_file(name)
        if path is None:
            raise web.HTTPNotFound(text="no such mesh")
        current = robot_model.model_json()["meshes"][name]["url"].partition("?v=")[2]
        cache = CACHE_FOREVER if request.query.get("v") == current else "no-cache"
        return web.FileResponse(path, headers={"Content-Type": "model/stl",
                                               "Cache-Control": cache})  # fmt: skip

    async def scene_arms(client: Client, msg: dict[str, Any]) -> None:
        client.push({"type": "scene_arms", "arms": await asyncio.to_thread(rig_units, studio)})

    studio.handle("scene_arms", scene_arms, control=False)
    studio.routes.extend([
        web.get("/api/scene/model", model),
        web.get("/api/scene/mesh/{name}", mesh),
    ])  # fmt: skip
