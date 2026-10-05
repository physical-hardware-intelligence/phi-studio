"""The Train page's commands. Logic lives in train.py (forms, script, log) and hpc.py (ssh).

Anything that submits, cancels or writes needs control: settings, the cluster check (it makes the
run folder), submit, cancel, look-for-it, fetch and a run on this Mac. Reading needs none.
ssh runs in a worker thread; the event loop never waits on the network.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from phi_studio import hpc
from phi_studio.train import (
    POLICIES_060,
    PROBE,
    RUN_ID,
    FormError,
    Job,
    LogState,
    Settings,
    Store,
    fallback_policies,
    local_datasets,
    mac_command,
    new_run,
    plan,
    read_probe,
    sbatch_script,
    step_id,
    train_args,
)

MAX_SENT_POINTS = 600  # a chart is under 1 000 px wide; more points add bytes, not detail


class TrainAPI:
    def __init__(self, studio: Any, runner: hpc.Runner = hpc.run_process) -> None:
        self.studio = studio
        self.runner = runner
        self.store = Store(Path(studio.data_dir)) if studio.data_dir else None
        self.policies: dict[str, Any] | None = None
        self.probe: asyncio.Task[None] | None = None
        self.check: dict[str, Any] = {"running": False, "rows": [], "at": None}
        self.logs: dict[str, dict[str, Any]] = {}  # run id -> {"log": LogState, "read": {...}}
        self.fetches: dict[str, dict[str, Any]] = {}
        self.busy: set[str] = set()  # runs with a poll in flight, and "refresh"

    # -- helpers ----------------------------------------------------------------------------------
    def need_store(self) -> Store:
        if self.store is None:
            raise FormError({"form": "Training needs Studio's data folder. Start Studio with "
                                     "--data-dir."})  # fmt: skip
        return self.store

    def settings(self) -> Settings:
        return self.store.settings() if self.store else Settings()

    def cluster(self, s: Settings | None = None) -> hpc.Cluster:
        return hpc.Cluster(s or self.settings(), self.runner)

    def policy_types(self) -> tuple[str, ...]:
        p = self.policies or fallback_policies()
        return tuple(x["type"] for x in p["policies"]) or POLICIES_060

    def needs(self) -> dict[str, list[str]]:
        return (self.policies or {}).get("needs", {})

    def fail(self, client: Any, e: Exception, what: str) -> None:
        if isinstance(e, FormError):
            client.push({"type": "train_error", "what": what, "message": str(e),
                         "errors": e.errors})  # fmt: skip
            return
        r = getattr(e, "result", None)
        client.push({"type": "train_error", "what": what, "message": str(e),
                     "fix": getattr(e, "fix", "") or None,
                     "cmd": r.cmd if r else None, "output": r.shown() if r else None})  # fmt: skip

    def lookup(self, msg: dict[str, Any], where: str | None = None) -> tuple[str, dict[str, Any]]:
        """The run a message names, if Studio recorded it (and it ran `where`)."""
        rid = msg.get("run")
        run = self.need_store().run(rid) if isinstance(rid, str) and RUN_ID.match(rid) else None
        if run is None or (where and run["where"] != where):
            # WHY only recorded runs: Studio must never touch a job it did not submit.
            raise FormError({"run": "Studio has no such run. It acts only on runs it started."})
        return str(rid), run

    def mac_state(self, run: dict[str, Any]) -> str:
        assert self.store is not None
        d = self.store.runs_dir / run["id"]
        final = d / "train" / "checkpoints" / step_id(run["steps"], run["steps"])
        if (final / "pretrained_model").is_dir():
            return "COMPLETED"
        running = (self.studio.term_running or {}).get("command") or ""
        if str(d) in running:
            return "RUNNING"
        return "STOPPED" if (d / "train.log").exists() else "NOT STARTED"

    def view(self, run: dict[str, Any]) -> dict[str, Any]:
        state = self.mac_state(run) if run["where"] == "mac" else hpc.run_state(run)
        lg = self.logs.get(run["id"])
        progress = lg["log"].summary() if lg else None
        return {**{k: v for k, v in run.items() if k != "script"}, "state": state,
                "progress": progress, "fetch": self.fetches.get(run["id"])}  # fmt: skip

    def state_msg(self) -> dict[str, Any]:
        s = self.settings()
        runs = self.store.runs() if self.store else []
        return {"type": "train_state", "ready": self.store is not None,
                "settings": asdict(s), "defaults": asdict(Settings()),
                "runs": [self.view(r) for r in runs], "policies": self.policies,
                "datasets": local_datasets(), "check": self.check,
                "data_dir": str(self.store.runs_dir.parent) if self.store else None}  # fmt: skip

    def broadcast_runs(self) -> None:
        runs = self.store.runs() if self.store else []
        self.studio._fanout({"type": "train_runs", "runs": [self.view(r) for r in runs]})

    # -- commands ---------------------------------------------------------------------------------
    async def init(self, client: Any, msg: dict[str, Any]) -> None:
        msg_ = await asyncio.to_thread(self.state_msg)
        client.push(msg_)
        if self.policies is None and self.probe is None:
            self.probe = asyncio.get_running_loop().create_task(self._probe())

    async def _probe(self) -> None:
        """Ask the installed LeRobot for its policy types, in a child Python."""

        def go() -> dict[str, Any]:
            env = {**os.environ, "PYTHONNOUSERSITE": "1"}
            try:
                p = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                                   text=True, timeout=120, env=env)  # fmt: skip
                if p.returncode == 0:
                    return read_probe(p.stdout)
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
                pass
            return fallback_policies()

        self.policies = await asyncio.to_thread(go)
        self.studio._fanout({"type": "train_policies", "policies": self.policies})

    async def save_settings(self, client: Any, msg: dict[str, Any]) -> None:
        store = self.need_store()
        try:
            raw = msg.get("settings")
            s = Settings.from_dict(raw if isinstance(raw, dict) else {})
            s.remote_user = store.settings().remote_user  # learnt, never typed
            s.validate()
        except (FormError, TypeError) as e:
            self.fail(client, e if isinstance(e, FormError) else FormError({"form": str(e)}),
                      "settings")  # fmt: skip
            return
        await asyncio.to_thread(store.save_settings, s)
        self.studio._fanout({"type": "train_settings", "settings": asdict(s)})

    def _job(self, msg: dict[str, Any]) -> Job:
        return Job.from_dict(msg.get("job"), self.policy_types())

    async def preview(self, client: Any, msg: dict[str, Any]) -> None:
        try:
            job = self._job(msg)
            s = self.settings()
            out: dict[str, Any] = {"type": "train_preview", "ok": True, "run_id": job.run_id,
                                   "errors": {}}  # fmt: skip
            if job.where == "mac":
                d = self.need_store().runs_dir / job.run_id
                out["command"] = mac_command(job, d)
                out["output_dir"] = str(d / "train")
            else:
                try:
                    base = s.base()
                except FormError:
                    base = "/scratch/<you>/phi-studio"  # shown until the first check finds you
                run_dir = f"{base}/runs/{job.run_id}"
                out["command"] = "lerobot-train " + " ".join(
                    train_args(job, device="cuda", output_dir=f"{run_dir}/train", cluster=True,
                               workers=int(s.cpus)))  # fmt: skip
                out["script"] = sbatch_script(job, s, run_dir) if "<" not in base else None
                out["plan"] = plan(job, s)
                out["run_dir"] = run_dir
            client.push(out)
        except FormError as e:
            client.push({"type": "train_preview", "ok": False, "errors": e.errors})

    async def run_check(self, client: Any, msg: dict[str, Any]) -> None:
        if self.check["running"]:
            return
        store = self.need_store()
        try:
            job: Job | None = self._job(msg) if msg.get("job") else None
            if job and job.where != "cluster":
                job = None
        except FormError:
            job = None  # the check still runs, with a stand-in script
        s = store.settings()
        loop = asyncio.get_running_loop()
        self.check = {"running": True, "rows": [], "at": time.time()}
        self.studio._fanout({"type": "train_check", **self.check})

        def on_row(row: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(self._check_row, row)

        try:
            learnt = await asyncio.to_thread(hpc.run_checks, self.cluster(s), s, job,
                                             self.needs(), on_row)  # fmt: skip
            # WHY keep these: the window warns before Submit when the last check found too few
            # free submit slots. The server still checks again at submit time.
            self.check["limits"] = {k: learnt.get(k) for k in ("free", "submitted", "max_submit")}
            if learnt.get("user") and learnt["user"] != store.settings().remote_user:
                fresh = store.settings()
                fresh.remote_user = learnt["user"]
                await asyncio.to_thread(store.save_settings, fresh)
                self.studio._fanout({"type": "train_settings", "settings": asdict(fresh)})
        finally:
            await asyncio.sleep(0)  # WHY: let the last queued row land before "done"
            self.check = {**self.check, "running": False}
            self.studio._fanout({"type": "train_check", **self.check})

    def _check_row(self, row: dict[str, Any]) -> None:
        self.check = {**self.check, "rows": [*self.check["rows"], row]}
        self.studio._fanout({"type": "train_check_row", "row": row})

    async def submit(self, client: Any, msg: dict[str, Any]) -> None:
        try:
            store = self.need_store()
            job = self._job(msg)
            if job.where != "cluster":
                raise FormError({"where": "Use Run on this Mac for a local run."})
            if store.run(job.run_id):
                raise FormError({"name": "A run with this name and stamp exists. Change the name."})
            s = store.settings()
            run = new_run(job)
            sub = await asyncio.to_thread(hpc.submit, self.cluster(s), s, job, run, store.put)
        except (FormError, hpc.ClusterError) as e:
            self.fail(client, e, "submit")
            return
        client.push({"type": "train_submitted", "ok": sub.ok, "run_id": job.run_id,
                     "message": sub.message, "error": sub.run.get("error")})  # fmt: skip
        self.broadcast_runs()

    async def mac_start(self, client: Any, msg: dict[str, Any]) -> None:
        try:
            store = self.need_store()
            job = self._job(msg)
            if job.where != "mac":
                raise FormError({"where": "Use Submit for a cluster run."})
            if store.run(job.run_id):
                raise FormError({"name": "A run with this name and stamp exists. Change the name."})
            d = store.runs_dir / job.run_id
            d.mkdir(parents=True, exist_ok=False)
            run = new_run(job, local_dir=str(d))
            await asyncio.to_thread(store.put, run)
        except (FormError, OSError) as e:
            self.fail(client, e if isinstance(e, FormError) else FormError({"form": str(e)}), "mac")
            return
        client.push({"type": "train_mac_ready", "run_id": job.run_id,
                     "command": mac_command(job, d)})  # fmt: skip
        self.broadcast_runs()

    async def refresh(self, client: Any, msg: dict[str, Any]) -> None:
        if self.store is None or "refresh" in self.busy:
            return
        self.busy.add("refresh")
        try:
            runs = [r for r in self.store.runs() if r["where"] == "cluster"]
            states, _ = await asyncio.to_thread(hpc.refresh, self.cluster(), runs)
            if states:
                await asyncio.to_thread(self.store.update_jobs, states)
        except hpc.ClusterError as e:
            self.fail(client, e, "refresh")
        finally:
            self.busy.discard("refresh")
        self.broadcast_runs()

    async def poll(self, client: Any, msg: dict[str, Any]) -> None:
        """A run's log and chart: the next piece of the log since the last poll."""
        store = self.need_store()
        try:
            rid, run = self.lookup(msg)
        except FormError as e:
            self.fail(client, e, "poll")
            return
        if rid in self.busy:
            return
        self.busy.add(rid)
        try:
            lg = self.logs.setdefault(rid, {"log": LogState(run["log_freq"], run["steps"]),
                                            "read": {}, "done": set()})  # fmt: skip
            if run["where"] == "mac":
                await asyncio.to_thread(self._read_local, run, lg)
            else:
                await self._read_remote(rid, lg)
        except hpc.ClusterError as e:
            self.fail(client, e, "poll")
            return
        finally:
            self.busy.discard(rid)
        v = lg["log"].view()
        pts = v["points"]
        if len(pts) > MAX_SENT_POINTS:
            k = len(pts) / MAX_SENT_POINTS
            pts = [pts[int(i * k)] for i in range(MAX_SENT_POINTS - 1)] + [pts[-1]]
        fresh = store.run(rid) or run
        client.push({"type": "train_log", "run": rid, **v, "points": pts,
                     "view": self.view(fresh)})  # fmt: skip

    def _read_local(self, run: dict[str, Any], lg: dict[str, Any]) -> None:
        path = Path(run["local_dir"]) / "train.log"
        off = lg["read"].get("mac", 0)
        try:
            with open(path, "rb") as f:
                f.seek(off)
                data = f.read(hpc.LOG_CHUNK)
        except FileNotFoundError:
            return
        lg["read"]["mac"] = off + len(data)
        lg["log"].feed(data.decode(errors="replace"))

    async def _read_remote(self, rid: str, lg: dict[str, Any]) -> None:
        assert self.store is not None
        cluster = self.cluster()
        for _ in range(4):  # at most 4 ssh calls, about 4 MB of log, per poll
            run = self.store.run(rid)
            if run is None:
                return
            open_parts = [j for j in run.get("jobs", []) if j["id"] not in lg["done"]]
            part = next((j for j in open_parts if j.get("state") != "PENDING"), None)
            off = lg["read"].get(part["id"], 0) if part else 0
            log = part["log"] if part else None
            chunk, states, _ = await asyncio.to_thread(hpc.poll, cluster, run, log, off)
            if states:
                await asyncio.to_thread(self.store.update_jobs, states)
            if part is None:
                started = any(states.get(j["id"], {}).get("state", "PENDING") != "PENDING"
                              for j in open_parts)  # fmt: skip
                if started:
                    continue  # a part just started: read its log in this same poll
                return
            lg["read"][part["id"]] = off + len(chunk)
            lg["log"].feed(chunk.decode(errors="replace"))
            if len(chunk) >= hpc.LOG_CHUNK:
                continue  # more of this log waits
            if states.get(part["id"], {}).get("state", part.get("state")) in hpc.TERMINAL:
                lg["done"].add(part["id"])
                lg["log"].feed("\n")  # a last line with no newline
                continue  # the next part's log, in the same poll
            return

    async def cancel(self, client: Any, msg: dict[str, Any]) -> None:
        try:
            rid, run = self.lookup(msg, "cluster")
            r = await asyncio.to_thread(hpc.cancel, self.cluster(), run)
        except (FormError, hpc.ClusterError) as e:
            self.fail(client, e, "cancel")
            return
        client.push({"type": "train_cancelled", "run": rid, "ok": r is None or r.ok,
                     "cmd": r.cmd if r else None, "output": r.shown() if r else
                     "Nothing to cancel: every part had already ended."})  # fmt: skip
        await self.refresh(client, {})

    async def find(self, client: Any, msg: dict[str, Any]) -> None:
        store = self.need_store()
        try:
            rid, run = self.lookup(msg, "cluster")
            if not run.get("remote_dir"):
                raise FormError({"run": "This run never reached the cluster: its script was not "
                                        "written, so no job can exist."})  # fmt: skip
            found, r = await asyncio.to_thread(hpc.find, self.cluster(), run)
        except (FormError, hpc.ClusterError) as e:
            self.fail(client, e, "find")
            return
        new = hpc.adopt(run, found)
        if new:
            await asyncio.to_thread(store.put, run)
        client.push({"type": "train_found", "run": rid, "found": found, "new": new,
                     "cmd": r.cmd, "output": r.shown()})  # fmt: skip
        self.broadcast_runs()

    async def fetch(self, client: Any, msg: dict[str, Any]) -> None:
        store = self.need_store()
        s = store.settings()
        try:
            rid, run = self.lookup(msg, "cluster")
            if (self.fetches.get(rid) or {}).get("state") == "running":
                return
            remote = hpc.checkpoint_dir(run)
            total, _, _ = await asyncio.to_thread(hpc.remote_size, self.cluster(s), remote)
        except (FormError, hpc.ClusterError) as e:
            self.fail(client, e, "fetch")
            return
        loop = asyncio.get_running_loop()
        dest = store.models_dir / rid
        self._fetch(rid, {"state": "running", "bytes": 0, "total": total, "path": str(dest),
                          "message": None})  # fmt: skip
        last = [0.0]

        def progress(n: int) -> None:
            if time.monotonic() - last[0] >= 0.5:
                last[0] = time.monotonic()
                loop.call_soon_threadsafe(self._fetch, rid, {**self.fetches[rid], "bytes": n})

        rc, err = await asyncio.to_thread(hpc.rsync, s.host, remote, dest, total, progress)
        got = await asyncio.to_thread(hpc.local_bytes, dest)
        if rc == 0:
            run = store.run(rid) or run
            run["fetched"] = {"path": str(dest), "at": time.time(), "bytes": got}
            await asyncio.to_thread(store.put, run)
            self._fetch(rid, {"state": "done", "bytes": got, "total": total, "path": str(dest),
                              "message": None})  # fmt: skip
        else:
            self._fetch(rid, {"state": "error", "bytes": got, "total": total, "path": str(dest),
                              "message": f"rsync exited {rc}: {err[-800:]}"})  # fmt: skip
        self.broadcast_runs()

    def _fetch(self, rid: str, st: dict[str, Any]) -> None:
        self.fetches[rid] = st
        self.studio._fanout({"type": "train_fetch", "run": rid, **st})


def register(studio: Any) -> None:
    api = TrainAPI(studio)
    studio.train = api  # WHY: tests and the assistant reach the API through the Studio
    for cmd, fn, control in (
        ("train_init", api.init, False),
        ("train_preview", api.preview, False),
        ("train_refresh", api.refresh, False),
        ("train_poll", api.poll, False),
        ("train_settings", api.save_settings, True),
        ("train_check", api.run_check, True),  # mkdir -p of the run folder
        ("train_submit", api.submit, True),
        ("train_mac_start", api.mac_start, True),
        ("train_cancel", api.cancel, True),
        ("train_find", api.find, True),
        ("train_fetch", api.fetch, True),
    ):
        studio.handle(cmd, fn, control=control)
