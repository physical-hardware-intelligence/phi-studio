"""The Train page's commands. Logic lives in train.py (forms, script, log) and hpc.py (ssh).

Anything that submits, cancels or writes needs control: settings, the cluster check (it makes the
run folder), submit, finishing a broken chain, cancel, look-for-it, fetch and a run on this Mac.
Reading needs none.

ssh and rsync run on Train's own threads (POOL_PREFIX); the event loop never waits on the network.
A run's own commands go to the host and user recorded when it was submitted, never to whatever the
settings say now, so its job ids can only reach the cluster that made them.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeVar

from phi_studio import hpc
from phi_studio.errors import Refusal
from phi_studio.train import (
    HOST,
    POLICIES_060,
    PROBE,
    RUN_ID,
    USER,
    FormError,
    Job,
    LogState,
    Settings,
    Store,
    _hms,
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

log = logging.getLogger(__name__)
T = TypeVar("T")

MAX_SENT_POINTS = 600  # a chart is under 1 000 px wide; more points add bytes, not detail
# WHY own threads, as hub_api.py does: an ssh that hangs until its timeout holds a thread. On
# asyncio's shared default executor a few of them would starve every other to_thread in Studio.
POOL_PREFIX = "phi-train"
POOL_THREADS = 4
FETCH_THREADS = 2  # a fetch holds its thread for minutes; it never queues behind ssh calls
# WHY 0.85 [JUDGEMENT, not measured]: a part loads the dataset and the model before its first step.
# Studio has not timed that, so it keeps 15 percent of a part aside when it estimates whether one
# checkpoint fits.
FIT_SHARE = 0.85
# Kinds of change to one run that must not overlap: each reads the run, talks to the cluster, then
# writes the run back. Fetch only writes "fetched", so it has its own guard.
CHANGES = ("submit", "resubmit", "find", "cancel")


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
        self.changing: dict[str, str] = {}  # run id -> the change in flight (CHANGES)
        self.fetching: set[str] = set()
        self.rev = 0  # stamps every run view, so a window keeps the newest of two
        self.pool = ThreadPoolExecutor(POOL_THREADS, thread_name_prefix=POOL_PREFIX)
        self.fetcher = ThreadPoolExecutor(FETCH_THREADS, thread_name_prefix=POOL_PREFIX + "-fetch")

    @property
    def inflight(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {k: [] for k in CHANGES}
        for rid, kind in self.changing.items():
            out[kind].append(rid)
        return {**{k: sorted(v) for k, v in out.items()}, "fetch": sorted(self.fetching)}

    async def _on(self, pool: ThreadPoolExecutor, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(pool, functools.partial(fn, *args))

    # -- helpers ----------------------------------------------------------------------------------
    def need_store(self) -> Store:
        if self.store is None:
            raise FormError({"form": "Training needs Studio's data folder. Start Studio with "
                                     "--data-dir."})  # fmt: skip
        return self.store

    def settings(self) -> Settings:
        return self.store.settings() if self.store else Settings()

    def cluster(self, s: Settings | None = None) -> hpc.Cluster:
        """The cluster the settings name: for the check and for a new submit only."""
        return hpc.Cluster(s or self.settings(), self.runner)

    def run_cluster(self, run: dict[str, Any]) -> hpc.Cluster:
        """The cluster a run was submitted to, as recorded then."""
        host, user = run.get("host"), run.get("user")
        if not isinstance(host, str) or not HOST.match(host):
            raise Refusal("This run has no cluster host recorded, so Studio will not send its job "
                          "ids anywhere.", "Read its jobs with squeue on the cluster you submitted "
                          "it to.")  # fmt: skip
        if not isinstance(user, str) or not USER.match(user):
            raise Refusal("This run has no cluster user recorded, so Studio cannot act on it.")
        return hpc.Cluster(replace(self.settings(), host=host, remote_user=user), self.runner)

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

    def unexpected(self, client: Any, e: Exception, what: str, after: str) -> None:
        """An error no check foresaw: a plain message, never a dead handler."""
        log.exception("train %s failed", what)
        self.fail(client, Refusal(f"Studio stopped on an error it did not expect: "
                                  f"{type(e).__name__}: {e}.", after), what)  # fmt: skip

    def lookup(self, msg: dict[str, Any], where: str | None = None) -> tuple[str, dict[str, Any]]:
        """The run a message names, if Studio recorded it (and it ran `where`)."""
        rid = msg.get("run")
        run = self.need_store().run(rid) if isinstance(rid, str) and RUN_ID.match(rid) else None
        if run is None or (where and run["where"] != where):
            # WHY only recorded runs: Studio must never touch a job it did not submit.
            raise FormError({"run": "Studio has no such run. It acts only on runs it started."})
        return str(rid), run

    def claim(self, rid: str, kind: str) -> None:
        """One change to a run at a time. WHY: each reads the run, waits on ssh, then writes it
        back; two at once could submit a part twice or drop one."""
        now = self.changing.get(rid)
        if now:
            words = {"submit": "being submitted", "resubmit": "sending its remaining parts",
                     "find": "being looked for", "cancel": "being cancelled"}[now]  # fmt: skip
            raise Refusal(f"This run is {words} right now. Try again when that ends.")
        self.changing[rid] = kind
        self.broadcast_busy()

    def release(self, rid: str | None) -> None:
        if rid is not None and rid in self.changing:
            del self.changing[rid]
            self.broadcast_busy()

    def broadcast_busy(self) -> None:
        self.studio._fanout({"type": "train_busy", "busy": self.inflight})

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
        self.rev += 1
        return {**{k: v for k, v in run.items() if k != "script"}, "state": state,
                "progress": progress, "fetch": self.fetches.get(run["id"]),
                "remaining_from": hpc.missing_parts(run) if run.get("error") else None,
                "rev": self.rev}  # fmt: skip

    def state_msg(self) -> dict[str, Any]:
        s, dropped = self.store.settings_report() if self.store else (Settings(), [])
        runs = self.store.runs() if self.store else []
        return {"type": "train_state", "ready": self.store is not None,
                "settings": asdict(s), "defaults": asdict(Settings()), "settings_dropped": dropped,
                "runs": [self.view(r) for r in runs], "policies": self.policies,
                "datasets": local_datasets(), "check": self.check, "busy": self.inflight,
                "data_dir": str(self.store.runs_dir.parent) if self.store else None}  # fmt: skip

    def broadcast_runs(self) -> None:
        runs = self.store.runs() if self.store else []
        self.studio._fanout({"type": "train_runs", "runs": [self.view(r) for r in runs]})

    def checkpoint_fit(self, job: Job, s: Settings) -> dict[str, Any]:
        """Whether one checkpoint fits in one part, from the newest measured step rate of a cluster
        run with the same policy and batch size. None when no such run was measured."""
        part_s = _hms(s.time)
        runs = self.store.runs() if self.store else []
        def same(r: dict[str, Any]) -> bool:
            return (r.get("where") == "cluster" and r.get("policy") == job.policy_label
                    and r.get("batch_size") == job.batch_size
                    and isinstance(r.get("rate"), (int, float)) and r["rate"] > 0)  # fmt: skip

        hit = next((r for r in runs if same(r)), None)
        if hit is None:
            return {"rate": None, "from_run": None, "save_s": None, "part_s": part_s, "fits": None}
        save_s = job.save_freq / float(hit["rate"])
        return {"rate": hit["rate"], "from_run": hit["id"], "save_s": round(save_s),
                "part_s": part_s, "fits": save_s <= part_s * FIT_SHARE}

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
                                   text=True, timeout=120, env=env,
                                   stdin=subprocess.DEVNULL)  # fmt: skip
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
        self.studio._fanout({"type": "train_settings", "settings": asdict(s), "dropped": []})

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
                out["plan"] = {**plan(job, s), "fit": self.checkpoint_fit(job, s)}
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
            learnt = await self._on(self.pool, hpc.run_checks, self.cluster(s), s, job,
                                    self.needs(), on_row)  # fmt: skip
            # WHY keep these: the window warns before Submit when the last check found too few
            # free submit slots or a part longer than the partition allows. The server checks
            # both again at submit time.
            self.check["limits"] = {k: learnt.get(k) for k in
                                    ("free", "submitted", "max_submit", "partition", "max_time")}
            if learnt.get("user") and learnt["user"] != store.settings().remote_user:
                fresh = store.settings()
                fresh.remote_user = learnt["user"]
                await asyncio.to_thread(store.save_settings, fresh)
                self.studio._fanout({"type": "train_settings", "settings": asdict(fresh)})
        except Exception as e:  # noqa: BLE001  a check that dies must still end "running"
            self.unexpected(client, e, "check", "Run the check again.")
        finally:
            await asyncio.sleep(0)  # WHY: let the last queued row land before "done"
            self.check = {**self.check, "running": False}
            self.studio._fanout({"type": "train_check", **self.check})

    def _check_row(self, row: dict[str, Any]) -> None:
        self.check = {**self.check, "rows": [*self.check["rows"], row]}
        self.studio._fanout({"type": "train_check_row", "row": row})

    async def submit(self, client: Any, msg: dict[str, Any]) -> None:
        rid: str | None = None
        try:
            store = self.need_store()
            job = self._job(msg)
            if job.where != "cluster":
                raise FormError({"where": "Use Run on this Mac for a local run."})
            self.claim(job.run_id, "submit")
            rid = job.run_id
        except (FormError, Refusal) as e:
            self.fail(client, e, "submit")
            return
        try:
            if store.run(job.run_id):
                raise FormError({"name": "A run with this name and stamp exists. Change the name."})
            s = store.settings()
            run = new_run(job)
            sub = await self._on(self.pool, hpc.submit, self.cluster(s), s, job, run, store.put)
        except (FormError, Refusal, hpc.ClusterError) as e:
            self.fail(client, e, "submit")
            return
        except Exception as e:  # noqa: BLE001  never leave the window waiting on a dead handler
            self.unexpected(client, e, "submit", "Parts sent before the error are recorded. Use "
                                                 "Look for it before you submit this run again.")
            return
        finally:
            self.release(rid)
            self.broadcast_runs()
        client.push({"type": "train_submitted", "ok": sub.ok, "run_id": job.run_id,
                     "message": sub.message, "error": sub.run.get("error")})  # fmt: skip

    async def resubmit(self, client: Any, msg: dict[str, Any]) -> None:
        """After an sbatch error: look for the run's jobs by name first (a failed sbatch can still
        have made one), then submit only the parts that do not exist. The person clicks; Studio
        never does this by itself."""
        rid: str | None = None
        try:
            store = self.need_store()
            rid_, run = self.lookup(msg, "cluster")
            if not run.get("error") or not run.get("script_path"):
                raise Refusal("This run has no failed submit to finish.")
            cluster = self.run_cluster(run)
            self.claim(rid_, "resubmit")
            rid = rid_
        except (FormError, Refusal) as e:
            self.fail(client, e, "resubmit")
            return
        try:
            found, _ = await self._on(self.pool, hpc.find, cluster, run)
            fresh = await asyncio.to_thread(store.mutate, rid, lambda r: hpc.adopt(r, found))
            assert fresh is not None
            first = hpc.missing_parts(fresh)
            if first is None:
                await asyncio.to_thread(store.mutate, rid, lambda r: r.update(error=None))
                client.push({"type": "train_resubmitted", "ok": True, "run": rid,
                             "message": "Every part of this run exists on the cluster already, "
                                        "so nothing was submitted."})  # fmt: skip
                return
            need = int(fresh["parts_planned"]) - first + 1
            await self._on(self.pool, hpc.check_room, cluster, fresh["user"],
                           fresh["partition"], fresh["part_time"], need)  # fmt: skip
            sub = await self._on(self.pool, hpc.sbatch_parts, cluster, fresh, first, store.put)
        except (FormError, Refusal, hpc.ClusterError) as e:
            self.fail(client, e, "resubmit")
            return
        except Exception as e:  # noqa: BLE001
            self.unexpected(client, e, "resubmit", "Use Look for it before you try again.")
            return
        finally:
            self.release(rid)
            self.broadcast_runs()
        client.push({"type": "train_resubmitted", "ok": sub.ok, "run": rid, "message": sub.message,
                     "error": sub.run.get("error")})  # fmt: skip

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
            by_host: dict[str, list[dict[str, Any]]] = {}
            for r in self.store.runs():
                if r["where"] == "cluster" and isinstance(r.get("host"), str):
                    by_host.setdefault(r["host"], []).append(r)
            for runs in by_host.values():  # one ssh call per cluster; a run without a host: none
                try:
                    cluster = self.run_cluster(runs[0])
                except Refusal:
                    continue
                states, _ = await self._on(self.pool, hpc.refresh, cluster, runs)
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
                await self._read_remote(rid, run, lg)
        except (hpc.ClusterError, Refusal) as e:
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
        lg["log"].feed_bytes(data)

    async def _read_remote(self, rid: str, run: dict[str, Any], lg: dict[str, Any]) -> None:
        assert self.store is not None
        cluster = self.run_cluster(run)
        for _ in range(4):  # at most 4 ssh calls, about 4 MB of log, per poll
            run = self.store.run(rid) or run
            open_parts = [j for j in run.get("jobs", []) if j["id"] not in lg["done"]]
            part = next((j for j in open_parts if j.get("state") != "PENDING"), None)
            off = lg["read"].get(part["id"], 0) if part else 0
            log_path = part["log"] if part else None
            chunk, states, _ = await self._on(self.pool, hpc.poll, cluster, run, log_path, off)
            if states:
                await asyncio.to_thread(self.store.update_jobs, states)
            if part is None:
                started = any(states.get(j["id"], {}).get("state", "PENDING") != "PENDING"
                              for j in open_parts)  # fmt: skip
                if started:
                    continue  # a part just started: read its log in this same poll
                break
            lg["read"][part["id"]] = off + len(chunk)
            lg["log"].feed_bytes(chunk)
            if len(chunk) >= hpc.LOG_CHUNK:
                continue  # more of this log waits
            if states.get(part["id"], {}).get("state", part.get("state")) in hpc.TERMINAL:
                lg["done"].add(part["id"])
                lg["log"].end_part()
                continue  # the next part's log, in the same poll
            break
        rate = lg["log"].summary()["rate"]
        old = run.get("rate")
        if rate and (not isinstance(old, (int, float)) or abs(rate - old) > 0.05 * old):
            # Kept with the run, so a later preview can tell whether a checkpoint fits in a part.
            await asyncio.to_thread(self.store.mutate, rid,
                                    lambda r: r.update(rate=round(rate, 4)))  # fmt: skip

    async def cancel(self, client: Any, msg: dict[str, Any]) -> None:
        rid: str | None = None
        try:
            rid_, run = self.lookup(msg, "cluster")
            cluster = self.run_cluster(run)
            self.claim(rid_, "cancel")
            rid = rid_
            r = await self._on(self.pool, hpc.cancel, cluster, run)
        except (FormError, Refusal, hpc.ClusterError) as e:
            self.fail(client, e, "cancel")
            return
        finally:
            self.release(rid)
        client.push({"type": "train_cancelled", "run": rid, "ok": r is None or r.ok,
                     "cmd": r.cmd if r else None, "output": r.shown() if r else
                     "Nothing to cancel: every part had already ended."})  # fmt: skip
        await self.refresh(client, {})

    async def find(self, client: Any, msg: dict[str, Any]) -> None:
        store = self.need_store()
        rid: str | None = None
        try:
            rid_, run = self.lookup(msg, "cluster")
            if not run.get("remote_dir"):
                raise Refusal("This run never reached the cluster: its script was not written, so "
                              "no job can exist.")  # fmt: skip
            cluster = self.run_cluster(run)
            self.claim(rid_, "find")
            rid = rid_
            found, r = await self._on(self.pool, hpc.find, cluster, run)
            new = [0]
            # WHY mutate: adopt into the run as it is on disk now, not the copy read before ssh.
            await asyncio.to_thread(store.mutate, rid,
                                    lambda x: new.__setitem__(0, hpc.adopt(x, found)))
        except (FormError, Refusal, hpc.ClusterError) as e:
            self.fail(client, e, "find")
            return
        finally:
            self.release(rid)
        client.push({"type": "train_found", "run": rid, "found": found, "new": new[0],
                     "cmd": r.cmd, "output": r.shown()})  # fmt: skip
        self.broadcast_runs()

    async def fetch(self, client: Any, msg: dict[str, Any]) -> None:
        store = self.need_store()
        try:
            rid, run = self.lookup(msg, "cluster")
            if rid in self.fetching:
                return
            cluster = self.run_cluster(run)
            remote = hpc.checkpoint_dir(run)
        except (FormError, Refusal, hpc.ClusterError) as e:
            self.fail(client, e, "fetch")
            return
        self.fetching.add(rid)
        self.broadcast_busy()
        loop = asyncio.get_running_loop()
        dest = store.models_dir / rid
        # WHY a separate folder: rsync writes file by file, so a copy that fails halfway into the
        # real folder would leave half of a new checkpoint over the last good one.
        part = store.models_dir / f"{rid}.fetching"
        base = {"bytes": 0, "total": 0, "path": str(dest), "message": None}
        self._fetch(rid, {**base, "state": "running"})
        last = [0.0]
        end: dict[str, Any] | None = None

        def progress(n: int) -> None:
            if time.monotonic() - last[0] >= 0.5:
                last[0] = time.monotonic()
                loop.call_soon_threadsafe(self._fetch, rid, {**self.fetches[rid], "bytes": n})

        try:
            total, _, _ = await self._on(self.pool, hpc.remote_size, cluster, remote)
            base["total"] = total
            self._fetch(rid, {**base, "state": "running"})
            await asyncio.to_thread(part.mkdir, parents=True, exist_ok=True)
            rc, err = await self._on(self.fetcher, hpc.rsync, cluster.s.host, remote, part, total,
                                     progress)  # fmt: skip
            got = await asyncio.to_thread(hpc.local_bytes, part)
            if rc != 0:
                end = {**base, "state": "error", "bytes": got,
                       "message": f"rsync exited {rc}: {err[-800:]} The last good copy, if any, "
                                  "is unchanged."}  # fmt: skip
            else:
                old = await asyncio.to_thread(self._swap, part, dest)
                fetched = {"path": str(dest), "at": time.time(), "bytes": got, "replaced": old}
                await asyncio.to_thread(store.mutate, rid, lambda r: r.update(fetched=fetched))
                end = {**base, "state": "done", "bytes": got,
                       "message": f"The copy fetched before is kept at {old}." if old else None}
        except Exception as e:  # noqa: BLE001  a fetch must always end, never stay "running"
            log.info("train fetch %s failed: %s", rid, e)
            end = {**base, "state": "error", "message": f"{e} {getattr(e, 'fix', '')}".strip()}
        finally:
            self.fetching.discard(rid)
            self._fetch(rid, end or {**base, "state": "error",
                                     "message": "The fetch stopped before it finished."})
            self.broadcast_busy()
            self.broadcast_runs()

    @staticmethod
    def _swap(part: Path, dest: Path) -> str | None:
        """Put a finished copy in place. The one it replaces is renamed, never deleted."""
        old = None
        if dest.exists():
            moved = dest.with_name(f"{dest.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
            os.replace(dest, moved)
            old = str(moved)
        os.replace(part, dest)
        return old

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
        ("train_resubmit", api.resubmit, True),
        ("train_mac_start", api.mac_start, True),
        ("train_cancel", api.cancel, True),
        ("train_find", api.find, True),
        ("train_fetch", api.fetch, True),
    ):
        studio.handle(cmd, fn, control=control)
