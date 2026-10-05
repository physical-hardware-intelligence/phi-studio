"""The cluster, over ssh: checks, limits, submit, job states, logs, cancel and checkpoint fetch.

Rules for the login node, all enforced here: no python, no find, no du, no recursive grep. Only
squeue, sacct, sacctmgr, scontrol, sbatch, scancel, cat, ls, head, tail, test, mkdir, df and quota.

Every remote command is built from values train.py has checked, each quoted with shlex.quote, and
run with `ssh -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=15 -- <host> '<command>'`.
BatchMode means ssh never waits for a password or a Duo prompt: it fails, and Studio says so.

Studio never retries an sbatch. On Explorer an sbatch that printed an error has still created the
job, sometimes minutes later (phi group/sim/LESSONS.md, 2026-09-29), so a retry can run a job
twice. Studio stops, shows the error, and offers to look for the job by its name.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phi_studio.errors import Refusal
from phi_studio.train import (
    HOST,
    JOB_ID,
    MAX_PART_S,
    PARTITION,
    REMOTE_PATH,
    USER,
    Job,
    Settings,
    _hms,
    dist_names,
    hms,
    missing_for,
    part_name,
    sbatch_script,
)

SSH_OPTS = ("-o", "ControlPath=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
# WHY keepalives for rsync only: a copy can run for many minutes, and a network that drops without a
# reset would leave ssh waiting forever. 15 s x 4 unanswered probes ends it after about a minute.
RSYNC_SSH = (*SSH_OPTS, "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
RSYNC_IO_TIMEOUT_S = 120  # rsync's own --timeout: no data either way for this long ends the copy
RSYNC_STALL_S = 300.0  # no new bytes on disk for this long: Studio stops rsync itself
RSYNC_CAP_S = 6 * 3600.0  # the longest any one fetch may run, whatever its progress
MARK = "@@phi-studio@@"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}  # fmt: skip
LOG_CHUNK = 1_000_000  # bytes per log read; the next poll reads on from there
q = shlex.quote

# (argv, stdin, timeout) -> (returncode, stdout, stderr). Tests pass a fake.
Runner = Callable[[list[str], str | None, float], tuple[int, bytes, bytes]]


def run_process(argv: list[str], stdin: str | None, timeout: float) -> tuple[int, bytes, bytes]:
    # WHY DEVNULL with no input: ssh reads stdin, and Studio's own stdin is a terminal or nothing.
    feed: dict[str, Any] = ({"input": stdin.encode()} if stdin is not None
                            else {"stdin": subprocess.DEVNULL})  # fmt: skip
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout, **feed)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or b"", (e.stderr or b"") + f"\nNo answer in {timeout:.0f} s.".encode()
    except FileNotFoundError:
        return 127, b"", f"{argv[0]} is not installed on this Mac.".encode()


@dataclass
class Result:
    cmd: str  # the command line as a person would type it
    rc: int
    raw: bytes
    err: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    @property
    def out(self) -> str:
        return self.raw.decode(errors="replace")

    def shown(self, limit: int = 6000) -> str:
        text = (self.out + ("\n" if self.out and self.err else "") + self.err).strip()
        return text if len(text) <= limit else text[:limit] + "\n[cut]"


class ClusterError(RuntimeError):
    def __init__(self, message: str, result: Result | None = None, fix: str = "") -> None:
        super().__init__(message)
        self.result = result
        self.fix = fix


class Cluster:
    def __init__(self, settings: Settings, runner: Runner = run_process) -> None:
        self.s = settings
        self.runner = runner

    def ssh(self, remote: str, stdin: str | None = None, timeout: float = 60.0) -> Result:
        argv = ["ssh", *SSH_OPTS, "--", self.s.host, remote]
        rc, out, err = self.runner(argv, stdin, timeout)
        return Result(shlex.join(argv), rc, out, err.decode(errors="replace").strip())


def ssh_fix(r: Result) -> str:
    e = r.err.lower()
    if "permission denied" in e or "publickey" in e:
        return ("ssh wanted a password or a Duo prompt, which Studio cannot answer. Make the same "
                "ssh command work in Terminal with your key and no prompt.")  # fmt: skip
    if "could not resolve" in e:
        return "The host alias is not in ~/.ssh/config. Fix the alias in the settings."
    if r.rc == 124 or "timed out" in e:
        return "The cluster did not answer. Check the network or the VPN, then run the check again."
    return "Run the same command in Terminal to see what ssh needs."


# -- reading SLURM's answers -----------------------------------------------------------------------
def parse_kv(text: str) -> dict[str, str]:
    """scontrol's KEY=value pairs."""
    return dict(re.findall(r"(\w+)=(\S*)", text))


def slurm_seconds(v: str) -> int | None:
    """[days-]hours:minutes:seconds, or UNLIMITED (None)."""
    m = re.match(r"^(?:(\d+)-)?(\d+):(\d\d):(\d\d)$", v.strip())
    if not m:
        return None
    return int(m[1] or 0) * 86400 + int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])


@dataclass
class Limits:
    qos: str | None
    max_submit: int | None
    max_running: int | None
    submitted: int  # this user's jobs in the partition, pending or running
    running: int
    raw: str

    @property
    def free(self) -> int | None:
        return None if self.max_submit is None else max(0, self.max_submit - self.submitted)


def read_limits(
    cluster: Cluster, user: str, partition: str, qos: str | None
) -> tuple[Limits, Result]:
    if not USER.match(user) or not PARTITION.match(partition):
        raise ClusterError("Studio does not know your cluster user yet.")
    parts = []
    if qos:
        parts.append(f"sacctmgr -n -P show qos {q(qos)} "
                     "format=Name,MaxSubmitPU,MaxJobsPU,MaxWall,MaxTRESPU")  # fmt: skip
    parts += [f"echo {MARK}",
              f"sacctmgr -n -P show assoc user={q(user)} "
              "format=User,Account,Partition,QOS,MaxSubmit,MaxJobs",
              f"echo {MARK}", f"squeue -h -r -u {q(user)} -o '%i|%P|%T|%j'"]  # fmt: skip
    r = cluster.ssh("; ".join(parts))
    if not r.ok and not r.out:
        raise ClusterError("Could not read the SLURM limits.", r, ssh_fix(r))
    qos_txt, assoc_txt, queue_txt = (r.out.split(f"{MARK}\n", 2) + ["", ""])[:3]
    caps_submit: list[int] = []
    caps_run: list[int] = []
    for ln in qos_txt.splitlines():
        f = ln.split("|")
        if len(f) >= 3:
            caps_submit += [int(f[1])] if f[1].isdigit() else []
            caps_run += [int(f[2])] if f[2].isdigit() else []
    for ln in assoc_txt.splitlines():
        f = ln.split("|")
        if len(f) >= 6 and (not f[2] or f[2] == partition):
            caps_submit += [int(f[4])] if f[4].isdigit() else []
            caps_run += [int(f[5])] if f[5].isdigit() else []
    submitted = running = 0
    for ln in queue_txt.splitlines():
        f = ln.split("|")
        if len(f) >= 3 and partition in f[1].split(","):
            submitted += 1
            running += f[2] == "RUNNING"
    lim = Limits(qos, min(caps_submit) if caps_submit else None,
                 min(caps_run) if caps_run else None, submitted, running, r.out)  # fmt: skip
    return lim, r


def partition_info(cluster: Cluster, partition: str) -> tuple[dict[str, str], Result]:
    if not PARTITION.match(partition):
        raise ClusterError(f"Not a partition name: {partition}")
    r = cluster.ssh(f"scontrol show partition {q(partition)}")
    return (parse_kv(r.out) if r.ok else {}), r


# The checks after ssh, skipped together when ssh fails.
LATER = (("user", "Remote user"), ("partition", "Partition"), ("limits", "Submit limit"),
         ("env", "LeRobot in the env"), ("extras", "Packages the policy needs"),
         ("base", "Run folder"), ("test", "Test the job script"), ("space", "Space on scratch"))


# -- the checklist ---------------------------------------------------------------------------------
Row = dict[str, Any]


def _row(id: str, title: str, status: str, detail: str, r: Result | None = None,
         fix: str = "", cmds: list[Result] | None = None) -> Row:  # fmt: skip
    rs = cmds if cmds is not None else ([r] if r else [])
    return {"id": id, "title": title, "status": status, "detail": detail, "fix": fix or None,
            "runs": [{"cmd": x.cmd, "rc": x.rc, "output": x.shown()} for x in rs]}  # fmt: skip


def env_path(cluster: Cluster, s: Settings, home: str) -> tuple[str | None, Result | None]:
    """The env folder the setup activates: a path as given, or a conda env name looked up in
    ~/.conda/environments.txt (the list conda keeps of every env it made)."""
    name = s.env_name()
    if not name:
        return None, None
    if "/" in name:
        return (name if REMOTE_PATH.match(name) and ".." not in name else None), None
    r = cluster.ssh(f"cat {q(home + '/.conda/environments.txt')}")
    for ln in r.out.splitlines():
        ln = ln.strip()
        if ln.endswith(f"/envs/{name}") and REMOTE_PATH.match(ln) and ".." not in ln:
            return ln, r
    return None, r


def default_check_job(stamp: str) -> Job:
    return Job(where="cluster", dataset="lerobot/pusht", policy="act", steps=1000, save_freq=500,
               name="studio-check", stamp=stamp, parts=1)  # fmt: skip


def run_checks(cluster: Cluster, s: Settings, job: Job | None, needs: dict[str, list[str]],
               on_row: Callable[[Row], None]) -> dict[str, Any]:  # fmt: skip
    """Every check in order, each sent to on_row as it finishes. Returns what was learnt (user,
    home, free submit slots). Writes nothing but the base folder (mkdir -p)."""
    learnt: dict[str, Any] = {}
    rows: list[Row] = []

    def emit(row: Row) -> None:
        rows.append(row)
        on_row(row)

    r = cluster.ssh("echo connected")
    if not (r.ok and "connected" in r.out):
        emit(_row("ssh", "ssh reaches the cluster", "fail", f"ssh to {s.host} failed.", r,
                  ssh_fix(r)))  # fmt: skip
        for id, title in LATER:
            emit(_row(id, title, "skip", "Needs ssh first."))
        return learnt
    emit(_row("ssh", "ssh reaches the cluster", "pass",
              f"Connected to {s.host} with no password prompt. Studio cannot answer Duo or a "
              "password, so it needs this key-based login that does not prompt.", r))  # fmt: skip

    r = cluster.ssh('whoami && echo "$HOME"')
    lines = r.out.split()
    user, home = (lines + ["", ""])[:2]
    if not r.ok or not USER.match(user) or not REMOTE_PATH.match(home):
        emit(_row("user", "Remote user", "fail", "Could not read the user name.", r))
        return learnt
    learnt.update(user=user, home=home)
    s.remote_user = user
    emit(_row("user", "Remote user", "pass", f"{user}, home {home}.", r))

    info, r = partition_info(cluster, s.partition)
    max_s = slurm_seconds(info.get("MaxTime", "")) if info else None
    want = _hms(s.time)
    qos = info.get("QoS") if info and info.get("QoS") not in (None, "", "N/A") else None
    learnt.update(partition=s.partition, max_time=info.get("MaxTime") if info else None)
    if not info:
        emit(_row("partition", f"Partition {s.partition}", "fail", "The partition was not found.",
                  r, "Fix the partition in the settings."))  # fmt: skip
    elif max_s is not None and want > max_s:
        emit(_row("partition", f"Partition {s.partition}", "fail",
                  f"Each part asks for {s.time}, more than the limit of {info['MaxTime']}.", r,
                  f"Set the time per part to {info['MaxTime']} or less."))  # fmt: skip
    else:
        state = info.get("State", "?")
        emit(_row("partition", f"Partition {s.partition}", "pass" if state == "UP" else "warn",
                  f"State {state}, longest job {info.get('MaxTime', '?')}, QoS {qos or 'none'}. "
                  f"Each part asks for {s.time}.", r))  # fmt: skip

    try:
        lim, r = read_limits(cluster, user, s.partition, qos)
        need = job.parts if job else 1
        learnt.update(free=lim.free, submitted=lim.submitted, max_submit=lim.max_submit)
        what = (f"You have {lim.submitted} jobs submitted in {s.partition} ({lim.running} "
                f"running). The limit is {lim.max_submit} submitted and {lim.max_running} "
                f"running at once (QoS {qos}).")  # fmt: skip
        if lim.free is None:
            emit(_row("limits", "Submit limit", "warn", what + " No submit limit was found.", r))
        elif lim.free >= need:
            emit(_row("limits", "Submit limit", "pass",
                      f"{what} {lim.free} more can be submitted now.", r))  # fmt: skip
        else:
            emit(_row("limits", "Submit limit", "warn",
                      f"{what} {lim.free} more can be submitted now; this job needs {need}.", r,
                      "Wait for some of your jobs to finish. Studio cancels nothing it did not "
                      "submit."))  # fmt: skip
    except ClusterError as e:
        emit(_row("limits", "Submit limit", "fail", str(e), e.result, e.fix))

    env, r_env = env_path(cluster, s, home)
    if env is None:
        emit(_row("env", "LeRobot in the env", "fail",
                  f"Could not find the env {s.env_name() or '(none)'} the setup activates.", r_env,
                  "Check the activate line in the settings."))  # fmt: skip
        emit(_row("extras", "Packages the policy needs", "skip", "Needs the env first."))
    else:
        r = cluster.ssh(f"test -x {q(env + '/bin/lerobot-train')} && echo executable; "
                        f"head -n 1 {q(env + '/bin/lerobot-train')}")  # fmt: skip
        m = re.search(r"python(3\.\d+)", r.out)
        cmds = [x for x in (r_env, r) if x]
        if "executable" not in r.out:
            emit(_row("env", "LeRobot in the env", "fail",
                      f"{env}/bin/lerobot-train is missing or not executable.", cmds=cmds,
                      fix="Install LeRobot in that env, from a batch job, not the login node."))
            emit(_row("extras", "Packages the policy needs", "skip", "Needs lerobot-train."))
        else:
            emit(_row("env", "LeRobot in the env", "pass",
                      f"{env}/bin/lerobot-train is executable.", cmds=cmds))  # fmt: skip
            site = f"{env}/lib/python{m[1] if m else '3.12'}/site-packages"
            r = cluster.ssh(f"ls {q(site)}")
            have = dist_names(r.out)
            dist = [n for n in r.out.split()
                    if n.startswith("lerobot-") and n.endswith(".dist-info")]  # fmt: skip
            ver = dist[0][len("lerobot-") : -len(".dist-info")] if dist else None
            policy = job.policy if job and job.policy else "act"
            listing = Result(r.cmd, r.rc, f"{len(r.out.split())} entries".encode(), r.err)
            if not r.ok:
                emit(_row("extras", "Packages the policy needs", "fail",
                          f"Could not list {site}.", r))  # fmt: skip
            elif job and job.pretrained:
                emit(_row("extras", "Packages the policy needs", "warn",
                          f"LeRobot {ver or '?'} is installed. The policy type comes from "
                          f"{job.pretrained}, which Studio does not download to check.",
                          listing))  # fmt: skip
            elif not needs:
                emit(_row("extras", "Packages the policy needs", "warn",
                          f"LeRobot {ver or '?'} is installed. Studio could not read which "
                          "packages each policy needs, because LeRobot is not importable on this "
                          "Mac.", listing))  # fmt: skip
            else:
                miss = missing_for(policy, {k: set(v) for k, v in needs.items()}, have)
                head = f"LeRobot {ver or '?'} in the env."
                if not miss:
                    emit(_row("extras", f"Packages {policy} needs", "pass",
                              f"{head} Every package LeRobot lists for {policy} is installed.",
                              listing))  # fmt: skip
                else:
                    emit(_row("extras", f"Packages {policy} needs", "warn",
                              f"{head} LeRobot's install list for {policy} names "
                              f"{', '.join(miss)}, which the env does not have. Some are needed "
                              "only by part of a policy, so the run may still work.", listing,
                              "Install the missing packages from a batch job if the run fails "
                              "on an import."))  # fmt: skip

    try:
        base = s.base()
    except Exception as e:  # noqa: BLE001  FormError: no user and no folder
        emit(_row("base", "Run folder", "fail", str(e)))
        base = None
    if base:
        runs = f"{base}/runs"
        r = cluster.ssh(f"mkdir -p -- {q(runs)} && test -w {q(runs)} && echo writable")
        ok = r.ok and "writable" in r.out
        emit(_row("base", "Run folder", "pass" if ok else "fail",
                  f"{runs} exists and is writable." if ok else f"Cannot write to {runs}.", r,
                  "" if ok else "Pick a folder you own, under /scratch/<you>."))  # fmt: skip

        stand_in = job is None
        j = job or default_check_job(time.strftime("%Y%m%d-%H%M%S"))
        emit(test_only(cluster, s, j, f"{runs}/{j.run_id}", stand_in))

        r = cluster.ssh(f"df -h {q(base)}; quota -s")
        pct = re.search(r"(\d+)%\s+/\S*", r.out)
        full = int(pct[1]) if pct else None
        blocked = "not permitted" in (r.out + r.err).lower()
        detail = (f"The file system is {full}% full." if full is not None else "df gave no answer.")
        if blocked:
            detail += (" Explorer does not let you read your own quota, and df shows the whole "
                       "file system, not your share. A full quota fails a job with 'Disk quota "
                       "exceeded' and no other error.")  # fmt: skip
        status = "warn" if blocked or (full or 0) >= 95 else "pass"
        emit(_row("space", "Space on scratch", status, detail, r))
    else:
        emit(_row("test", "Test the job script", "skip", "Needs the run folder."))
        emit(_row("space", "Space on scratch", "skip", "Needs the run folder."))
    learnt["rows"] = rows
    return learnt


def test_only(cluster: Cluster, s: Settings, job: Job, run_dir: str, stand_in: bool) -> Row:
    """sbatch --test-only, which `man sbatch` on Explorer says submits nothing: "Validate the batch
    script and return an estimate of when a job would be scheduled to run ... No job is actually
    submitted." """
    title = "Test the job script"
    script = sbatch_script(job, s, run_dir)
    what = "a stand-in script (the form is not complete)" if stand_in else "this job's script"
    name = part_name(job.run_id, 1)
    base_cmd = f"sbatch --test-only --job-name={q(name)} --export=ALL,PHI_PART=1"
    r = cluster.ssh(base_cmd, stdin=script)
    text = r.out + r.err
    if r.ok and "to start at" in text:
        return _row("test", title, "pass", f"SLURM accepted {what}: {first(text, 'to start')}", r)
    if "QOSMaxSubmitJobPerUserLimit" not in text or not s.test_partition:
        return _row("test", title, "fail", f"SLURM refused {what}.", r,
                    "Read SLURM's answer below, fix the setting it names, and test again.")
    # The submit limit refuses even a test. Test on the other partition, within its time limit.
    info, r_p = partition_info(cluster, s.test_partition)
    cap = slurm_seconds(info.get("MaxTime", "")) if info else None
    t = hms(min(_hms(s.time), cap or MAX_PART_S))
    r2 = cluster.ssh(f"{base_cmd} --partition={q(s.test_partition)} --time={t}", stdin=script)
    text2 = r2.out + r2.err
    runs = [r, r_p, r2]
    if r2.ok and "to start at" in text2:
        return _row("test", title, "warn",
                    f"{s.partition} refused even a test: you are at your submit limit. On "
                    f"{s.test_partition} (time cut to {t}) SLURM accepted {what}: "
                    f"{first(text2, 'to start')}. The script and the request are valid; "
                    f"{s.partition} itself is untested until a slot frees.", cmds=runs)
    return _row("test", title, "fail", f"SLURM refused {what} on both partitions.", cmds=runs)


def first(text: str, needle: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if needle in ln), text.strip()[:200])


# -- submit, poll, cancel --------------------------------------------------------------------------
@dataclass
class Submitted:
    run: dict[str, Any]
    ok: bool
    message: str


def check_room(cluster: Cluster, user: str, partition: str, part_time: str, parts: int) -> None:
    """Refuse, in plain words, a submit the partition or the submit limit would refuse."""
    info, r = partition_info(cluster, partition)
    if not info:
        raise ClusterError(f"Partition {partition} was not found, so nothing was submitted.", r,
                           "Fix the partition in the settings, then run Check the cluster.")
    cap = slurm_seconds(info.get("MaxTime", ""))
    if cap is not None and _hms(part_time) > cap:
        raise Refusal(f"Not submitted. Each part asks for {part_time}, but partition {partition} "
                      f"allows at most {info['MaxTime']}.",
                      f"Set the time per part to {info['MaxTime']} or less in the settings.")
    qos = info.get("QoS") if info.get("QoS") not in (None, "", "N/A") else None
    if not USER.match(user):
        raise ClusterError("Run Check the cluster first, so Studio knows your user name.")
    lim, r = read_limits(cluster, user, partition, qos)
    if lim.free is None:
        raise ClusterError("Studio could not read your submit limit, so it did not submit.", r)
    if parts > lim.free:
        raise ClusterError(
            f"Not submitted. You have {lim.submitted} jobs submitted in {partition} and the "
            f"limit is {lim.max_submit}, so {lim.free} more fit. This run needs {parts}.", r,
            "Wait for some jobs to finish, or use fewer parts. Studio cancels nothing it did not "
            "submit.")  # fmt: skip


def submit(cluster: Cluster, s: Settings, job: Job, run: dict[str, Any],
           save: Callable[[dict[str, Any]], None]) -> Submitted:  # fmt: skip
    """Check the partition and the limit, write the script, then sbatch each part, each after the
    last with --dependency=afterany (a part that ends on its time limit counts as failed for
    afterok). Stops at the first error and never retries."""
    base = s.base()
    run_dir = f"{base}/runs/{job.run_id}"
    script = sbatch_script(job, s, run_dir)
    check_room(cluster, s.remote_user, s.partition, s.time, job.parts)
    path = f"{run_dir}/job.sbatch"
    # WHY noclobber: `>` then opens with O_EXCL, so two writers of one path cannot both succeed.
    # A separate `test ! -e` first would leave a gap between the test and the write.
    r = cluster.ssh(f"set -o noclobber && mkdir -p -- {q(run_dir + '/logs')} && cat > {q(path)}",
                    stdin=script)  # fmt: skip
    if not r.ok:
        raise ClusterError("Could not write the job script on the cluster. Nothing was submitted.",
                           r, ssh_fix(r) if r.rc == 255 else "")  # fmt: skip
    run.update(remote_dir=run_dir, script_path=path, script=script, host=s.host,
               user=s.remote_user, partition=s.partition, part_time=s.time)  # fmt: skip
    save(run)
    return sbatch_parts(cluster, run, 1, save)


def sbatch_parts(cluster: Cluster, run: dict[str, Any], first_part: int,
                 save: Callable[[dict[str, Any]], None]) -> Submitted:  # fmt: skip
    """sbatch parts first_part..parts_planned of a run whose script is written, the first after
    the run's newest recorded part. Stops at the first error and never retries."""
    path, run_dir, total = run["script_path"], run["remote_dir"], int(run["parts_planned"])
    if not REMOTE_PATH.match(path) or ".." in path:
        raise ClusterError(f"Not a script path Studio wrote: {path}")
    prev: str | None = run["jobs"][-1]["id"] if run.get("jobs") else None
    sent: list[str] = []
    run["error"] = None
    for k in range(first_part, total + 1):
        name = part_name(run["id"], k)
        dep = f" --dependency=afterany:{prev}" if prev else ""
        r = cluster.ssh(f"sbatch --parsable --job-name={q(name)} --export=ALL,PHI_PART={k}{dep} "
                        f"{q(path)}")  # fmt: skip
        jid = r.out.strip().split(";")[0] if r.ok else ""
        if not JOB_ID.match(jid):
            run["error"] = {"part": k, "cmd": r.cmd, "output": r.shown(),
                            "at": time.time()}  # fmt: skip
            save(run)
            done = ("" if k == first_part else f"Part {first_part} is queued. "
                    if k == first_part + 1 else f"Parts {first_part} to {k - 1} are queued. ")
            return Submitted(run, False,
                             f"sbatch failed for part {k} of {total}. {done}Studio did not "
                             "retry: on this cluster a failed sbatch can still create the job, "
                             "minutes later. Wait 3 minutes, then use Look for it.")  # fmt: skip
        run["jobs"].append({"part": k, "id": jid, "name": name, "state": "PENDING",
                            "log": f"{run_dir}/logs/{name}-{jid}.out"})  # fmt: skip
        save(run)
        sent.append(jid)
        prev = jid
    return Submitted(run, True, f"Submitted {len(sent)} part{'s' if len(sent) > 1 else ''}: "
                                f"{', '.join(sent)}.")  # fmt: skip


def missing_parts(run: dict[str, Any]) -> int | None:
    """The first part a run never got, after a failed sbatch, or None when every part exists."""
    have = {int(j["part"]) for j in run.get("jobs", [])}
    total = int(run["parts_planned"])
    first = next((k for k in range(1, total + 1) if k not in have), None)
    if first is not None and any(k > first for k in have):
        return None  # a gap in the middle: not a chain Studio can finish by adding parts
    return first


def _ids(run: dict[str, Any]) -> list[str]:
    return [str(j["id"]) for j in run.get("jobs", []) if JOB_ID.match(str(j.get("id")))]


def _states(text_q: str, text_a: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for ln in text_a.splitlines():  # sacct: every job it knows, ended or not
        f = ln.split("|")
        if len(f) >= 4 and JOB_ID.match(f[0]):
            out[f[0]] = {"state": f[1].split()[0] if f[1] else "UNKNOWN", "elapsed": f[2],
                         "exit": f[3], "reason": ""}  # fmt: skip
    for ln in text_q.splitlines():  # squeue: live, and the reason a job waits
        f = ln.split("|")
        if len(f) >= 4 and JOB_ID.match(f[0]):
            out[f[0]] = {**out.get(f[0], {"exit": ""}), "state": f[1], "elapsed": f[2],
                         "reason": f[3].strip("()") if f[1] == "PENDING" else ""}  # fmt: skip
    return out


def _state_cmd(ids: list[str]) -> str:
    joined = ",".join(ids)
    return (f"squeue -h -j {joined} -o '%i|%T|%M|%R' 2>/dev/null; echo {MARK}; "
            f"sacct -n -P -X -j {joined} -o JobIDRaw,State,Elapsed,ExitCode 2>/dev/null")


def apply_states(run: dict[str, Any], states: dict[str, dict[str, str]]) -> None:
    for j in run.get("jobs", []):
        st = states.get(str(j["id"]))
        if st:
            j.update(st)


def run_state(run: dict[str, Any]) -> str:
    """One word for the table: the most advanced part's state."""
    jobs = run.get("jobs", [])
    if not jobs:
        return "NOT SUBMITTED" if run.get("error") else "SUBMITTING"
    states = [j.get("state", "UNKNOWN") for j in jobs]
    if "RUNNING" in states:
        return "RUNNING"
    if all(st in TERMINAL for st in states):
        return "COMPLETED" if "COMPLETED" in states else states[-1]
    if any(st in TERMINAL for st in states) or "PENDING" in states:
        return "PENDING"
    return states[-1]


def refresh(
    cluster: Cluster, runs: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, str]], Result | None]:
    """Job states for every run not yet finished, in one ssh call."""
    ids = [i for run in runs if run_state(run) not in TERMINAL for i in _ids(run)]
    if not ids:
        return {}, None
    r = cluster.ssh(_state_cmd(ids))
    if r.rc == 255:
        raise ClusterError("Could not reach the cluster for job states.", r, ssh_fix(r))
    text_q, text_a = (r.out.split(f"{MARK}\n", 1) + [""])[:2]
    return _states(text_q, text_a), r


def poll(
    cluster: Cluster, run: dict[str, Any], log: str | None, offset: int
) -> tuple[bytes, dict[str, dict[str, str]], Result]:
    """States of a run's parts plus the next piece of one part's log, in one ssh call."""
    ids = _ids(run)
    cmd = _state_cmd(ids) if ids else f"echo {MARK}"
    if log:
        if not REMOTE_PATH.match(log) or ".." in log:
            raise ClusterError(f"Not a log path Studio wrote: {log}")
        start = int(offset) + 1
        cmd += f"; echo {MARK}; tail -c +{start} {q(log)} 2>/dev/null | head -c {LOG_CHUNK}"
    r = cluster.ssh(cmd)
    if r.rc == 255:
        raise ClusterError("Could not reach the cluster.", r, ssh_fix(r))
    mark = f"{MARK}\n".encode()
    pieces = r.raw.split(mark, 2)
    pieces += [b""] * (3 - len(pieces))
    states = _states(pieces[0].decode(errors="replace"), pieces[1].decode(errors="replace"))
    return (pieces[2] if log else b""), states, r


def cancel(cluster: Cluster, run: dict[str, Any]) -> Result | None:
    """scancel this run's parts that have not ended, last part first: with afterany, a part
    cancelled before the one after it would let that one start."""
    live = [str(j["id"]) for j in run.get("jobs", [])
            if JOB_ID.match(str(j["id"])) and j.get("state") not in TERMINAL]  # fmt: skip
    if not live:
        return None
    return cluster.ssh("scancel " + " ".join(reversed(live)))


def find(cluster: Cluster, run: dict[str, Any]) -> tuple[list[dict[str, str]], Result]:
    """Jobs on the cluster with this run's part names, from the last two days. The names hold the
    run's own time stamp, so they belong to this run and no other."""
    user = str(run.get("user", ""))
    if not USER.match(user):
        raise ClusterError("This run has no cluster user recorded.")
    names = ",".join(part_name(run["id"], k) for k in range(1, int(run["parts_planned"]) + 1))
    # WHY from the day before the run was made: sacct reads -S in the cluster's time zone, and a
    # whole day of margin covers any zone; a fixed "now-2days" misses a run older than that.
    made = float(run.get("created") or time.time())
    since = time.strftime("%Y-%m-%d", time.localtime(made - 86400))
    r = cluster.ssh(f"squeue -h -u {q(user)} -n {q(names)} -o '%i|%j|%T'; echo {MARK}; "
                    f"sacct -n -P -X -u {q(user)} -S {since} --name={q(names)} "
                    "-o JobIDRaw,JobName,State")  # fmt: skip
    if r.rc == 255:
        raise ClusterError("Could not reach the cluster.", r, ssh_fix(r))
    found: dict[str, dict[str, str]] = {}
    for ln in r.out.replace(f"{MARK}\n", "").splitlines():
        f = ln.split("|")
        if len(f) >= 3 and JOB_ID.match(f[0]) and f[1] in names.split(","):
            found.setdefault(f[0], {"id": f[0], "name": f[1], "state": f[2].split()[0]})
    return sorted(found.values(), key=lambda x: int(x["id"])), r


def adopt(run: dict[str, Any], found: list[dict[str, str]]) -> int:
    """Record found jobs as this run's parts. Returns how many were new."""
    have = {str(j["id"]) for j in run.get("jobs", [])}
    new = 0
    for f in found:
        if f["id"] in have:
            continue
        part = int(f["name"].rsplit("-p", 1)[1])
        log = f"{run['remote_dir']}/logs/{f['name']}-{f['id']}.out"
        run.setdefault("jobs", []).append({"part": part, "id": f["id"], "name": f["name"],
                                           "state": f["state"], "log": log})  # fmt: skip
        new += 1
    run["jobs"].sort(key=lambda j: (j["part"], int(j["id"])))
    return new


# -- fetch a checkpoint ----------------------------------------------------------------------------
def checkpoint_dir(run: dict[str, Any]) -> str:
    d = f"{run['remote_dir']}/train/checkpoints/last/pretrained_model"
    if not REMOTE_PATH.match(d) or ".." in d:
        raise ClusterError(f"Not a checkpoint path Studio wrote: {d}")
    return d


def remote_size(cluster: Cluster, path: str) -> tuple[int, list[str], Result]:
    """Bytes in a checkpoint folder, from `ls -lL` (follows the `last` link, reads no tree)."""
    r = cluster.ssh(f"ls -lL {q(path)}")
    if not r.ok:
        raise ClusterError("No checkpoint yet: the run has not saved one.", r,
                           "Wait for the first checkpoint (every save frequency steps).")
    total, names = 0, []
    for ln in r.out.splitlines():
        f = ln.split()
        if len(f) >= 9 and f[0].startswith("-") and f[4].isdigit():
            total += int(f[4])
            names.append(f[-1])
    return total, names, r


def local_bytes(d: Path) -> int:
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                n += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return n


def rsync(
    host: str,
    remote: str,
    local: Path,
    total: int,
    on_progress: Callable[[int], None],
    popen: Callable[..., Any] = subprocess.Popen,
) -> tuple[int, str]:
    """Copy a remote folder into `local`, reporting bytes on disk twice a second. WHY count the
    disk, not rsync's output: macOS ships openrsync, whose progress lines differ from rsync 3.
    Always ends: rsync's own --timeout, ssh keepalives, and Studio stops it after RSYNC_STALL_S
    with no new bytes or RSYNC_CAP_S in all."""
    if not HOST.match(host):
        return 2, f"Not an ssh host alias: {host}"
    local.mkdir(parents=True, exist_ok=True)
    argv = ["rsync", "-a", "--partial", f"--timeout={RSYNC_IO_TIMEOUT_S}",
            "-e", "ssh " + " ".join(RSYNC_SSH), "--", f"{host}:{remote}/", f"{local}/"]  # fmt: skip
    p = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    start = moved = time.monotonic()
    seen = -1
    why = ""
    while p.poll() is None:
        n = local_bytes(local)
        now = time.monotonic()
        if n != seen:
            seen, moved = n, now
        on_progress(min(total, n) if total else n)
        if now - moved > RSYNC_STALL_S:
            why = f"Studio stopped rsync: no data arrived for {RSYNC_STALL_S:.0f} s."
        elif now - start > RSYNC_CAP_S:
            why = f"Studio stopped rsync after {RSYNC_CAP_S / 3600:.0f} hours."
        if why:
            p.kill()
            break
        time.sleep(min(0.5, RSYNC_STALL_S / 4))
    err = p.stderr.read().decode(errors="replace") if p.stderr else ""
    on_progress(local_bytes(local))
    if why:
        return 124, f"{why} {err.strip()}".strip()
    return int(p.returncode if p.returncode is not None else 1), err.strip()
