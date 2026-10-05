"""The 3D view's two read-only routes: the SO-101 model as JSON, and its meshes.

    GET /api/scene/model        robot_model.model_json()
    GET /api/scene/mesh/{name}  one binary STL, by a mesh name from that JSON

No token: the model and meshes are public CAD (Apache-2.0, see NOTICE) and these routes move
nothing. The Host check stays, as on every Studio route, so a DNS-rebinding page cannot read them
through a name that resolves to 127.0.0.1. Origin is not checked: a same-origin GET carries none.

Point clouds from the cameras' depth (environment reconstruction) live in recon_api.py, with their
own token-checked route, and draw into the same three.js scene through web/src/scene/pointcloud.ts.
Not here yet: the real-arm backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from phi_studio import robot_model

if TYPE_CHECKING:
    from phi_studio.server import Studio

# WHY immutable: each mesh URL carries a hash of the file (robot_model.model_json), so a changed
# mesh gets a new URL and a cached one can never be stale.
CACHE_FOREVER = "public, max-age=31536000, immutable"


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
        path = robot_model.mesh_file(request.match_info["name"])
        if path is None:
            raise web.HTTPNotFound(text="no such mesh")
        return web.FileResponse(path, headers={"Content-Type": "model/stl",
                                               "Cache-Control": CACHE_FOREVER})  # fmt: skip

    studio.routes.extend([
        web.get("/api/scene/model", model),
        web.get("/api/scene/mesh/{name}", mesh),
    ])  # fmt: skip
