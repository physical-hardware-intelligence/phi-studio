"""The Evaluate page's commands. The record and its numbers live in evals.py; this file connects
them to the rig: it starts each trial's policy, attaches what the servos and cameras did, stamps
the eval with the rig's state, and keeps a blind eval blind.

Anything that changes a card or an eval needs control. Reading and exporting need none.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata as md
import time
from pathlib import Path
from typing import Any

from aiohttp import web

from phi_studio import evals as ev
from phi_studio.errors import Refusal

CLUB_CSV = Path("outputs") / "rollout_scores.csv"  # in the phi checkout (phi.utils.eval_rollouts)


class EvalAPI:
    def __init__(self, studio: Any) -> None:
        self.studio = studio

    # -- helpers ----------------------------------------------------------------------------------
    @property
    def store(self) -> ev.EvalStore:
        if self.studio.evals is None:
            raise Refusal("Evals need Studio's data folder.", "Start Studio with --data-dir.")
        return self.studio.evals  # type: ignore[no-any-return]

    def catalog(self) -> dict[str, str]:
        """The policies that can run on this rig now: id -> name."""
        pols = (self.studio.last.get("rig") or {}).get("policies") or []
        return {p["id"]: p["name"] for p in pols if p.get("available")}

    def club_csv(self) -> Path | None:
        rig = getattr(self.studio, "rig_dir", None)
        p = Path(rig) / CLUB_CSV if rig else None
        return p if p and p.is_file() else None

    def hz(self) -> float:
        rate = (self.studio.last.get("rig") or {}).get("hz")
        return float(rate) if isinstance(rate, int | float) and rate > 0 else 30.0

    def broadcast(self) -> None:
        self.studio._fanout(self.studio._eval_msg())

    def _call(self, fn: Any, *args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except ev.EvalError as e:
            raise Refusal(str(e)) from e
        except OSError as e:  # not saved, so not shown as saved
            raise Refusal(f"Could not save the eval: {e}",
                          f"Check {self.store.dir} is writable.") from e

    def stamps(self, policies: list[str]) -> dict[str, Any]:
        """The rig and software an eval ran on, so a number can be checked against the rig it came
        from. WHY calibration hashes: a policy run on a different calibration than it trained on
        reaches to the wrong place and looks exactly like a bad policy (phi docs/evaluation, hit
        on 2026-08-10); the club voids any number from a non-canonical calibration."""
        s = self.studio
        rig = s.last.get("rig") or {}
        cal_dir = rig.get("cal_dir")
        cals = {}
        for a in rig.get("arms") or []:
            entry: dict[str, Any] = {"id": a.get("id"), "file": a.get("file")}
            if cal_dir and a.get("file"):
                try:
                    data = (Path(cal_dir) / a["file"]).read_bytes()
                    entry["sha256"] = hashlib.sha256(data).hexdigest()[:16]
                except OSError:
                    entry["sha256"] = None
            cals[a.get("name", "?")] = entry
        ident = {x.get("name"): bool(x.get("ok")) for x in
                 (s.last.get("identity") or {}).get("arms", []) if isinstance(x, dict)}  # fmt: skip
        align = s.last.get("align_result") or {}
        cat = self.catalog()
        try:
            lerobot = md.version("lerobot")
        except md.PackageNotFoundError:
            lerobot = None
        try:
            studio_v = md.version("phi-studio")
        except md.PackageNotFoundError:
            studio_v = None
        return {"at": time.time(), "rig": s.spec.get("kind", "mock"),
                "calibration": cals, "identity_ok": ident,
                "camera_check": (s.camcheck or {}).get("status"),
                "align": {k: align.get(k) for k in ("verdict", "dataset", "at") if k in align},
                "versions": {"studio": studio_v, "lerobot": lerobot},
                "policies": {p: cat.get(p) for p in policies}}  # fmt: skip

    # -- commands ---------------------------------------------------------------------------------
    async def init(self, client: Any, msg: dict[str, Any]) -> None:
        client.push(await asyncio.to_thread(self.studio._eval_msg))

    async def card_save(self, client: Any, msg: dict[str, Any]) -> None:
        card = await asyncio.to_thread(self._call, self.store.save_card, msg.get("card"))
        client.push({"type": "eval_card_saved", "id": card["id"]})
        self.broadcast()

    async def card_dup(self, client: Any, msg: dict[str, Any]) -> None:
        card = await asyncio.to_thread(self._call, self.store.duplicate_card, msg.get("id"))
        client.push({"type": "eval_card_saved", "id": card["id"]})
        self.broadcast()

    async def card_ref(self, client: Any, msg: dict[str, Any]) -> None:
        """Keep the cameras' current pictures as a condition's reference."""
        frames = {k: f for k, f in self.studio.latest_frame.items()
                  if isinstance(f.get("jpeg"), bytes | bytearray)}  # fmt: skip
        want = msg.get("camera")
        if want is not None:
            frames = {k: f for k, f in frames.items() if k == want}
        if not frames:
            raise Refusal("No camera picture yet.", "Connect the rig so the cameras stream, "
                          "then take the reference again.")  # fmt: skip
        for key, f in frames.items():
            await asyncio.to_thread(self._call, self.store.save_ref, msg.get("id"),
                                    msg.get("condition"), key, bytes(f["jpeg"]))  # fmt: skip
        self.broadcast()

    async def begin(self, client: Any, msg: dict[str, Any]) -> None:
        store = self.store  # first: no data folder is the more basic problem
        if not self.studio.last.get("rig"):
            raise Refusal("The rig is still starting, so Studio does not know its policies yet.",
                          "Try again in a moment.")  # fmt: skip
        pols = msg.get("policies")
        stamps = await asyncio.to_thread(self.stamps, pols if isinstance(pols, list) else [])
        await asyncio.to_thread(
            self._call, store.begin, msg.get("card"), pols, msg.get("reps"),
            msg.get("blind"), msg.get("alpha", 0.05), self.catalog(),
            grouped=msg.get("grouped", False), stamps=stamps)  # fmt: skip
        self.broadcast()

    async def run(self, client: Any, msg: dict[str, Any]) -> None:
        """Start the next trial's policy. The window never names the policy, so a blind eval
        stays blind: the server looks it up from the schedule."""
        s, store = self.studio, self.store
        rec = store.current
        if rec is None:
            raise Refusal("No eval is running. Start one first.")
        slot = ev.next_slot(rec)
        if slot is None:
            raise Refusal("Every trial is done.", "End the eval to see who is better.")
        if s._unjudged():
            raise Refusal("The last trial is not scored yet.", "Score it, then run the next one.")
        if (s.camcheck or {}).get("status") == "unsure":
            # WHY: a policy reading the wrong camera is worse than none (camcheck)
            raise Refusal((s.camcheck or {}).get("message") or "The cameras are not recognised.",
                          (s.camcheck or {}).get("fix", ""))  # fmt: skip
        pol = ev.policy_of(rec, slot["alias"])
        if pol["id"] not in self.catalog():
            raise Refusal(f"Policy {slot['alias']} cannot run on this rig now.",
                          "Skip this trial, or end the eval.")  # fmt: skip
        s.to_worker({"cmd": "start", "activity": "policy", "policy": pol["id"],
                     "task": rec["card"]["task"], "limit_s": rec["card"]["limit_s"]})  # fmt: skip

    async def score(self, client: Any, msg: dict[str, Any]) -> None:
        s = self.studio
        run = s.run
        rid = msg.get("run_id")
        health = s.eval_health.summary(rid) if isinstance(rid, str) else {}
        stamp = {"camera_check": (s.camcheck or {}).get("status")}
        await asyncio.to_thread(self._call, self.store.score, msg.get("stage"),
                                msg.get("failures", []), msg.get("note", ""), run, rid,
                                health=health, hz=self.hz(), stamp=stamp)  # fmt: skip
        self.broadcast()

    async def void(self, client: Any, msg: dict[str, Any]) -> None:
        await asyncio.to_thread(self._call, self.store.void, msg.get("n"), msg.get("reason"))
        self.broadcast()

    async def skip(self, client: Any, msg: dict[str, Any]) -> None:
        await asyncio.to_thread(self._call, self.store.skip, msg.get("reason"))
        self.broadcast()

    async def undo(self, client: Any, msg: dict[str, Any]) -> None:
        await asyncio.to_thread(self._call, self.store.undo)
        self.broadcast()

    async def end(self, client: Any, msg: dict[str, Any]) -> None:
        await asyncio.to_thread(self._call, self.store.end)
        self.broadcast()

    async def export(self, client: Any, msg: dict[str, Any]) -> None:
        """CSV or a Markdown report, sent to the window that asked, which saves it as a file."""
        rid, fmt = msg.get("id"), msg.get("format")
        rec = self.store.current if rid is None else await asyncio.to_thread(self.store.record, rid)
        if rec is None:
            raise Refusal("There is no such eval.")
        if fmt not in ("csv", "md"):
            raise Refusal("Export as csv or md.")
        text = ev.to_csv(rec) if fmt == "csv" else ev.to_markdown(rec)
        client.push({"type": "eval_export", "id": rec["id"], "format": fmt, "text": text,
                     "name": f"eval-{rec['id']}.{fmt}"})  # fmt: skip

    async def import_club(self, client: Any, msg: dict[str, Any]) -> None:
        path = self.club_csv()
        if path is None:
            raise Refusal("The club's rollout_scores.csv was not found.",
                          "It lives in the phi checkout at outputs/rollout_scores.csv.")
        rec = await asyncio.to_thread(self._call, self.store.import_club, path)
        client.push({"type": "eval_imported", "id": rec["id"], **rec["imported"]})
        self.broadcast()

    # -- reference pictures over HTTP (an <img> cannot send headers: ?token=) ---------------------
    async def ref(self, request: web.Request) -> web.StreamResponse:
        data_api = getattr(self.studio, "data_api", None)
        why = data_api.refuse(request) if data_api else "not ready"
        if why:
            return web.Response(status=403, text=why)
        p = self.store.ref_path(request.match_info["card"], request.match_info["name"])
        if p is None:
            return web.Response(status=404, text="no such picture")
        return web.FileResponse(p, headers={"Cache-Control": "no-store"})


def register(studio: Any) -> None:
    api = EvalAPI(studio)
    studio.eval_api = api
    for cmd, fn, control in (
        ("eval_init", api.init, False),
        ("eval_export", api.export, False),
        ("eval_card_save", api.card_save, True),
        ("eval_card_dup", api.card_dup, True),
        ("eval_card_ref", api.card_ref, True),
        ("eval_begin", api.begin, True),
        ("eval_run", api.run, True),
        ("eval_score", api.score, True),
        ("eval_void", api.void, True),
        ("eval_skip", api.skip, True),
        ("eval_undo", api.undo, True),
        ("eval_end", api.end, True),
        ("eval_import", api.import_club, True),
    ):
        studio.handle(cmd, fn, control=control)
    studio.routes.append(web.get("/api/evals/ref/{card}/{name}", api.ref))
