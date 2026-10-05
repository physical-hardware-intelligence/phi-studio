"""Fixes from the review of Train: each test was written first and failed on the code before its
fix. All against the fake ssh; nothing here reaches a cluster."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from test_hpc import (
    STAMP,
    USER,
    FakeSSH,
    Window,
    assert_login_node_safe,
    cluster_rules,
    job,
    submitted_run,
)

from phi_studio import hpc, train
from phi_studio.errors import Refusal
from phi_studio.server import Studio
from phi_studio.train import FormError, LogState, Settings

RID = f"act-cubes-{STAMP}"
LS_ONE = "-rw-r--r-- 1 u g 10 Oct 4 17:00 m.safetensors\n"
FORM = {"where": "cluster", "dataset": "Parv-09/cubes", "policy": "act", "steps": 100_000,
        "batch_size": 8, "save_freq": 10_000, "name": "act-cubes", "stamp": STAMP, "parts": 3}


@pytest.fixture
def studio(tmp_path) -> Studio:
    st = Studio({"kind": "mock"}, 1, token="t0k", data_dir=tmp_path)
    st.train.policies = train.fallback_policies()  # type: ignore[attr-defined]
    return st


class HostSSH(FakeSSH):
    """FakeSSH that also records the host each command went to."""

    def __init__(self, rules: list[Any]) -> None:
        super().__init__(rules)
        self.hosts: list[str] = []

    def __call__(
        self, argv: list[str], stdin: str | None, timeout: float
    ) -> tuple[int, bytes, bytes]:
        self.hosts.append(argv[8])
        return super().__call__(argv, stdin, timeout)


def recorded_run(**kw: Any) -> dict[str, Any]:
    run = submitted_run()
    run.update(host="hpc", user=USER, partition="gpu", part_time="08:00:00", **kw)
    return run


# -- 1: a run's job ids go only to the host that run was submitted to ------------------------------
def test_per_run_actions_use_the_runs_own_host_not_the_settings(studio, monkeypatch) -> None:
    api = studio.train
    api.store.put(recorded_run())
    api.store.save_settings(Settings(host="othercluster", remote_user=USER))  # changed after submit
    m = hpc.MARK
    fake = HostSSH([(r"^squeue -h -j ", (0, f"10813301|RUNNING|0:05|d1011\n{m}\n{m}\n", "")),
                    (r"^squeue -h -u ", (0, f"{m}\n", "")), (r"^scancel", (0, "", "")),
                    (r"^ls -lL ", (0, "-rw-r--r-- 1 u g 10 Oct 4 17:00 model.safetensors\n", ""))])
    api.runner = fake
    rsynced: list[str] = []
    monkeypatch.setattr(hpc, "rsync", lambda host, *a, **k: (rsynced.append(host), (0, ""))[1])
    w = Window()

    async def go() -> None:
        await api.poll(w, {"run": RID})
        await api.refresh(w, {})
        await api.find(w, {"run": RID})
        await api.fetch(w, {"run": RID})
        await api.cancel(w, {"run": RID})

    asyncio.run(go())
    assert fake.hosts and set(fake.hosts) == {"hpc"}, fake.hosts
    assert rsynced == ["hpc"]
    assert fake.matching(r"^scancel") == ["scancel 10813303 10813302 10813301"]


def test_a_run_with_no_recorded_host_is_refused_and_sends_nothing(studio) -> None:
    api = studio.train
    run = recorded_run()
    del run["host"]
    api.store.put(run)
    fake = HostSSH([(r".", (0, "", ""))])
    api.runner = fake
    w = Window()
    asyncio.run(api.cancel(w, {"run": RID}))
    assert fake.calls == []
    assert "no cluster host recorded" in w.last("train_error")["message"]


# -- 2: a fetch always ends, and never mixes two checkpoints ---------------------------------------
def test_a_fetch_that_raises_ends_in_error_and_a_later_fetch_runs(studio, monkeypatch) -> None:
    api = studio.train
    api.store.put(recorded_run())
    api.runner = FakeSSH([(r"^ls -lL ", (0, LS_ONE, ""))])
    calls = []

    def boom(*a: Any, **k: Any) -> tuple[int, str]:
        calls.append(1)
        raise OSError("rsync vanished")

    monkeypatch.setattr(hpc, "rsync", boom)
    w = Window()
    for _ in range(2):
        try:
            asyncio.run(api.fetch(w, {"run": RID}))
        except OSError:
            pass
    assert api.fetches[RID]["state"] == "error"
    assert "rsync vanished" in api.fetches[RID]["message"]
    assert len(calls) == 2  # the first failure did not block the second fetch


def test_rsync_that_stalls_is_stopped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hpc, "RSYNC_STALL_S", 0.3, raising=False)

    class Stuck:
        returncode = None
        stderr = None
        killed = False

        def poll(self) -> int | None:
            return -9 if self.killed else None

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return -9

    p = Stuck()
    out: list[Any] = []
    t = threading.Thread(target=lambda: out.append(hpc.rsync(
        "hpc", "/scratch/u/r/x", tmp_path / "m", 100, lambda n: None, lambda *a, **k: p)),
        daemon=True)  # fmt: skip
    t.start()
    t.join(5)
    stopped = not t.is_alive()
    p.killed = True  # end the thread either way
    assert stopped, "rsync with no progress was never stopped"
    rc, err = out[0]
    assert rc != 0 and "no data" in err


def test_rsync_has_an_io_timeout_and_keepalives(tmp_path) -> None:
    seen: dict[str, Any] = {}

    class Done:
        returncode, stderr = 0, None

        def poll(self) -> int:
            return 0

    def popen(argv: list[str], **kw: Any) -> Done:
        seen.update(argv=argv, kw=kw)
        return Done()

    hpc.rsync("hpc", "/scratch/u/r/x", tmp_path / "m", 10, lambda n: None, popen)
    argv = seen["argv"]
    assert any(a.startswith("--timeout=") for a in argv)
    ssh = argv[argv.index("-e") + 1]
    assert "ServerAliveInterval=" in ssh and "ServerAliveCountMax=" in ssh
    assert seen["kw"].get("stdin") is not None  # DEVNULL: rsync never waits on Studio's stdin


def test_a_failed_refetch_keeps_the_last_good_copy(studio, monkeypatch, tmp_path) -> None:
    api = studio.train
    api.store.put(recorded_run())
    api.runner = FakeSSH([(r"^ls -lL ", (0, "-rw-r--r-- 1 u g 4 Oct 4 17:00 m.safetensors\n", ""))])
    good = api.store.models_dir / RID
    good.mkdir(parents=True)
    (good / "m.safetensors").write_text("good")

    def half(host: str, remote: str, local: Path, *a: Any, **k: Any) -> tuple[int, str]:
        local.mkdir(parents=True, exist_ok=True)
        (local / "m.safetensors").write_text("ha")
        return 23, "rsync: connection unexpectedly closed"

    monkeypatch.setattr(hpc, "rsync", half)
    asyncio.run(api.fetch(Window(), {"run": RID}))
    assert (good / "m.safetensors").read_text() == "good"
    assert api.fetches[RID]["state"] == "error"


# -- 3: submit checks the chosen partition's own time limit ----------------------------------------
def test_submit_refuses_a_part_longer_than_that_partitions_limit() -> None:
    fake = FakeSSH(cluster_rules())
    s = Settings(partition="gpu-short", time="08:00:00", remote_user=USER).validate()
    with pytest.raises(Refusal) as e:
        hpc.submit(hpc.Cluster(s, fake), s, job(), train.new_run(job()), lambda r: None)
    assert "gpu-short" in str(e.value) and "02:00:00" in str(e.value)
    assert not fake.matching(r"^(mkdir|sbatch)")


def test_the_settings_time_error_does_not_name_one_partition() -> None:
    with pytest.raises(FormError) as e:
        Settings(time="00:05:00").validate()
    assert "gpu" not in e.value.errors["time"]
    assert Settings(partition="long", time="24:00:00").validate().time == "24:00:00"


# -- 4: settings keep their types ------------------------------------------------------------------
def test_settings_refuse_values_of_the_wrong_type() -> None:
    with pytest.raises(FormError) as e:
        Settings(partition=5, test_partition=7, exclude=123, cpus=True, host=None,  # type: ignore[arg-type]
                 mem=64, gres=1).validate()  # type: ignore[arg-type]
    want = {"partition", "test_partition", "exclude", "cpus", "host", "mem", "gres"}
    assert want <= set(e.value.errors)


def test_stored_settings_keep_good_fields_and_name_the_dropped(tmp_path) -> None:
    store = train.Store(tmp_path)
    (tmp_path / "train_settings.json").write_text(json.dumps(
        {"host": "hpc2", "partition": 5, "time": "8h", "remote_user": USER}))
    s, dropped = store.settings_report()
    assert s.host == "hpc2" and s.remote_user == USER
    assert s.partition == "gpu" and s.time == "08:00:00"
    assert sorted(dropped) == ["partition", "time"]
    assert store.settings().host == "hpc2"


def test_an_unexpected_error_in_submit_becomes_a_plain_error(studio, monkeypatch) -> None:
    api = studio.train
    api.store.save_settings(Settings(remote_user=USER))
    api.runner = FakeSSH(cluster_rules())

    def broken(*a: Any, **k: Any) -> Any:
        raise KeyError("QoS")

    monkeypatch.setattr(hpc, "submit", broken)
    w = Window()
    asyncio.run(api.submit(w, {"job": FORM}))
    err = w.last("train_error")
    assert err["what"] == "submit" and "KeyError" in err["message"]
    assert RID not in api.inflight["submit"]


# -- 5 and 6: the log parser -----------------------------------------------------------------------
def test_a_new_part_clears_the_last_parts_end() -> None:
    st = LogState(log_freq=200, total=20_000)
    st.feed("phi-studio: starting at step 0\nphi-studio: lerobot-train exited with code 1, now\n")
    assert st.ended
    st.feed("phi-studio: resuming from step 2500\n")
    assert st.ended is None


def test_a_line_with_no_newline_is_capped() -> None:
    st = LogState(log_freq=1, total=10)
    for _ in range(2000):
        st.feed("x" * 1000)  # 2 MB with no newline: a stuck progress bar
    assert len(st.rest) <= 65536
    st.feed("\nINFO 2026-10-04 16:00:00 ot_train.py:606 step:1 smpl:1 ep:0 epch:0.00 loss:1.000 "
            "grdn:1.0 lr:1.0e-04\n")
    assert [p["step"] for p in st.points] == [1]


def test_a_truncated_last_line_at_a_parts_end_is_not_charted() -> None:
    st = LogState(log_freq=1, total=10)
    st.feed_bytes(b"INFO 2026-10-04 16:00:00 ot_train.py:606 step:1 smpl:1 ep:0 epch:0.00 "
                  b"loss:1.000 grdn:1.0 lr:1.0e-04\nINFO 2026-10-04 16:00:01 ot_train.py:606 "
                  b"step:2 smpl:2 ep:0 epch:0.00 loss:0.0")
    st.end_part()
    assert [p["step"] for p in st.points] == [1]
    assert st.tail[-1].endswith("loss:0.0")  # still shown in the log


def test_a_character_split_across_two_reads_decodes_once() -> None:
    st = LogState(log_freq=1, total=10)
    raw = "phi-studio: café\n".encode()
    st.feed_bytes(raw[:-2])
    st.feed_bytes(raw[-2:])
    assert st.tail[-1] == "phi-studio: café"


# -- 7: who is submitting is server state, sent to every window ------------------------------------
def test_busy_runs_are_broadcast_to_every_window(studio) -> None:
    api = studio.train
    api.store.save_settings(Settings(remote_user=USER))
    api.runner = FakeSSH(cluster_rules())
    sent: list[dict[str, Any]] = []
    studio._fanout = sent.append  # type: ignore[method-assign]
    asyncio.run(api.submit(Window(), {"job": FORM}))
    busy = [m["busy"] for m in sent if m["type"] == "train_busy"]
    assert busy[0]["submit"] == [RID] and busy[-1]["submit"] == []


# -- 8 and double submit: one change to a run at a time --------------------------------------------
def test_look_for_it_during_a_submit_never_drops_a_part(studio) -> None:
    api = studio.train
    api.store.save_settings(Settings(remote_user=USER))
    release = threading.Event()
    n = [10813300]

    def slow_sbatch(remote: str, stdin: str | None) -> tuple[int, str, str]:
        n[0] += 1
        if n[0] == 10813302:
            release.wait(5)
        return 0, f"{n[0]}\n", ""

    def squeue_names(remote: str, stdin: str | None) -> tuple[int, str, str]:
        for _ in range(100):  # answer once the submit has saved all three parts
            r = api.store.run(RID)
            if r and len(r["jobs"]) == 3:
                break
            time.sleep(0.05)
        return 0, f"10813301|{RID}-p1|RUNNING\n10813302|{RID}-p2|PENDING\n{hpc.MARK}\n", ""

    api.runner = FakeSSH([(r"^squeue -h -u ", squeue_names)] + cluster_rules(sbatch=slow_sbatch))
    w = Window()

    async def find_while_submitting() -> None:
        for _ in range(200):
            r = api.store.run(RID)
            if r and r["jobs"]:
                break
            await asyncio.sleep(0.01)
        task = asyncio.ensure_future(api.find(w, {"run": RID}))
        await asyncio.sleep(0.1)
        release.set()
        await task

    async def go() -> None:
        await asyncio.gather(api.submit(w, {"job": FORM}), find_while_submitting())

    asyncio.run(go())
    assert [j["id"] for j in api.store.run(RID)["jobs"]] == ["10813301", "10813302", "10813303"]


def test_two_submits_of_one_run_send_its_parts_once(studio) -> None:
    api = studio.train
    api.store.save_settings(Settings(remote_user=USER))
    fake = FakeSSH(cluster_rules())
    api.runner = fake
    a, b = Window(), Window()

    async def go() -> None:
        await asyncio.gather(api.submit(a, {"job": FORM}), api.submit(b, {"job": FORM}))

    asyncio.run(go())
    assert len(fake.matching(r"^sbatch --parsable")) == 3
    assert_login_node_safe(fake)


# -- gaps ------------------------------------------------------------------------------------------
def test_finish_a_broken_chain_looks_first_then_submits_only_the_missing_parts(studio) -> None:
    api = studio.train
    run = recorded_run()
    run["jobs"] = run["jobs"][:1]  # part 2's sbatch failed; part 3 was never sent
    run.update(error={"part": 2, "cmd": "sbatch", "output": "error"}, parts_planned=3,
               script_path=f"{run['remote_dir']}/job.sbatch")
    api.store.put(run)
    found = f"{hpc.MARK}\n10813301|{RID}-p1|RUNNING\n"  # the failed sbatch made no job
    fake = FakeSSH([(r"^squeue -h -u ", (0, found, ""))] + cluster_rules())
    api.runner = fake
    w = Window()
    asyncio.run(api.resubmit(w, {"run": RID}))
    assert w.last("train_resubmitted")["ok"], w.got
    sb = fake.matching(r"^sbatch --parsable")
    assert len(sb) == 2 and "--dependency=afterany:10813301" in sb[0] and f"{RID}-p2" in sb[0]
    assert "PHI_PART=3" in sb[1]
    assert fake.remote().index(fake.matching(r"^squeue -h -u ")[0]) < fake.remote().index(sb[0])
    fresh = api.store.run(RID)
    assert [j["part"] for j in fresh["jobs"]] == [1, 2, 3] and fresh["error"] is None
    assert_login_node_safe(fake)


def test_finish_a_broken_chain_adopts_a_part_the_failed_sbatch_made(studio) -> None:
    api = studio.train
    run = recorded_run()
    run["jobs"] = run["jobs"][:1]
    run.update(error={"part": 2}, parts_planned=2, script_path=f"{run['remote_dir']}/job.sbatch")
    api.store.put(run)
    found = f"{hpc.MARK}\n10813301|{RID}-p1|RUNNING\n10813302|{RID}-p2|PENDING\n"
    fake = FakeSSH([(r"^squeue -h -u ", (0, found, ""))] + cluster_rules())
    api.runner = fake
    w = Window()
    asyncio.run(api.resubmit(w, {"run": RID}))
    assert not fake.matching(r"^sbatch")
    assert "exists on the cluster already" in w.last("train_resubmitted")["message"]


def test_a_part_that_saves_no_checkpoint_cancels_the_waiting_parts(tmp_path) -> None:
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    base = tmp_path / "b"
    s = Settings(remote_base=str(base), env="module load x\nexport A=1").validate()
    j = job(parts=3, steps=6, save_freq=3)
    run_dir = f"{base}/runs/{j.run_id}"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls"
    for tool in ("srun", "scancel", "module"):
        (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> {calls}\n')
        (bin_ / tool).chmod(0o755)
    f = tmp_path / "job.sh"
    f.write_text(train.sbatch_script(j, s, run_dir))
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "SLURM_JOB_ID": "77", "PHI_PART": "1", "USER": "u",
           "HOME": str(tmp_path)}  # fmt: skip
    r = subprocess.run(["bash", str(f)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert train.STALLED in r.stdout
    lines = calls.read_text().splitlines()
    assert [ln for ln in lines if ln.startswith("scancel")] == [
        f"scancel --user=u --name={j.run_id}-p2 --state=PENDING",
        f"scancel --user=u --name={j.run_id}-p3 --state=PENDING"]
    st = LogState(log_freq=1, total=6)
    st.feed(r.stdout)
    assert st.stalled and train.STALLED in st.stalled


def test_preview_warns_when_one_checkpoint_does_not_fit_in_a_part(studio) -> None:
    api = studio.train
    api.store.save_settings(Settings(remote_user=USER, time="01:00:00"))
    w = Window()
    asyncio.run(api.preview(w, {"job": FORM}))
    assert w.last("train_preview")["plan"]["fit"]["rate"] is None  # nothing measured yet
    api.store.put(recorded_run(rate=2.0, batch_size=8, policy="act"))
    asyncio.run(api.preview(w, {"job": FORM}))
    fit = w.last("train_preview")["plan"]["fit"]
    assert fit["save_s"] == 5000 and fit["part_s"] == 3600 and fit["fits"] is False


@pytest.mark.parametrize("line", ["export HF_TOKEN=hf_abc", "export WANDB_API_KEY=x",
                                  "export A=1 db_password=x", "export MY_SECRET=x"])
def test_env_setup_refuses_tokens_and_says_where_they_go(line) -> None:
    with pytest.raises(FormError) as e:
        Settings(env=train.DEFAULT_ENV + "\n" + line).validate()
    assert "huggingface-cli login" in e.value.errors["env"]


def test_the_default_env_keeps_its_token_path() -> None:
    assert "HF_TOKEN_PATH" in train.DEFAULT_ENV
    Settings().validate()


def test_the_ssh_check_says_studio_cannot_answer_duo() -> None:
    rows: list[dict[str, Any]] = []
    hpc.run_checks(hpc.Cluster(Settings(), FakeSSH(cluster_rules())), Settings(), job(1), {},
                   rows.append)  # fmt: skip
    assert "cannot answer Duo" in rows[0]["detail"]


def test_ssh_never_reads_studios_own_stdin(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class P:
        returncode, stdout, stderr = 0, b"", b""

    def run(argv: list[str], **kw: Any) -> P:
        seen.clear()
        seen.update(kw)
        return P()

    monkeypatch.setattr(hpc.subprocess, "run", run)
    hpc.run_process(["ssh", "hpc", "true"], None, 5)
    assert seen["stdin"] is hpc.subprocess.DEVNULL
    hpc.run_process(["ssh", "hpc", "cat"], "script", 5)
    assert seen["input"] == b"script" and "stdin" not in seen


def test_refresh_sends_no_ids_for_a_run_without_a_host(studio) -> None:
    api = studio.train
    run = recorded_run()
    del run["host"]
    api.store.put(run)
    fake = HostSSH([(r".", (0, "", ""))])
    api.runner = fake
    asyncio.run(api.refresh(Window(), {}))
    assert fake.calls == []
