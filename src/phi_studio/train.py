"""Training with LeRobot 0.6.0: the job form, the lerobot-train flags, the SLURM script, and the
training log. Pure functions and small stores; nothing here talks to the cluster (hpc.py does).

Every value that reaches a command is checked against a pattern first, and every command is built
from those checked values with shlex. No text from a window reaches a shell unchecked.

Sources, LeRobot 0.6.0 (site-packages/lerobot):
  configs/train.py:92        resume needs --config_path; flags on the command line still override
  configs/train.py:260-266   push_to_hub with no repo_id raises
  configs/policies.py:70     push_to_hub defaults to True, so Studio always passes it
  configs/default.py:32-33   a local dataset is looked up under $HF_LEROBOT_HOME/<repo_id>
  common/train_utils.py:44   checkpoint folders are the step, zero padded to max(6, digits of steps)
  utils/logging_utils.py:190 the metrics line; utils/utils.py:64-68 the log line prefix
"""

from __future__ import annotations

import codecs
import json
import math
import os
import re
import shlex
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# -- limits the cluster sets -----------------------------------------------------------------------
# Read 2026-10-04 with `scontrol show partition gpu` and `sacctmgr show qos gpu`.
MAX_PART_S = 8 * 3600  # the gpu partition's MaxTime=08:00:00, the default part time
MAX_PARTS = 8  # the gpu QoS allows 8 submitted jobs per user (MaxSubmitPU)
MIN_PART_S = 600  # shorter parts spend most of their time loading the dataset and the model

# The policy types LeRobot 0.6.0 registers (PreTrainedConfig.get_known_choices() after
# `import lerobot.policies`). Studio asks the installed LeRobot first; this list is the fallback
# where LeRobot is missing, and tests/test_train.py checks it against the registry.
POLICIES_060 = ("act", "diffusion", "eo1", "evo1", "fastwam", "gaussian_actor", "groot",
                "lingbot_va", "molmoact2", "multi_task_dit", "pi0", "pi05", "pi0_fast", "smolvla",
                "tdmpc", "vla_jepa", "vqbet", "wall_x", "xvla")  # fmt: skip

# The cluster env setup that worked for LeRobot runs on Explorer, copied from
# /scratch/patodia.pa/phi-so101/configs/train_molmoact2_lang.sbatch (run 10766515, 10 000 steps),
# with $USER in place of the user name so it fits anyone's account.
DEFAULT_ENV = """\
module load anaconda3/2024.06
source activate lerobot-gpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
export HF_HOME=/scratch/$USER/phi-so101/hf HF_HUB_CACHE=/scratch/$USER/.hf/hub
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_LEROBOT_HOME=/scratch/$USER/phi-so101/lerobot-data"""

# -- patterns: what each value may be before it goes near a command --------------------------------
# WHY \Z, not $: in Python, $ also matches before a final newline, so "gpu:1\n" would pass.
HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}\Z")  # an ssh alias; never starts with "-"
USER = re.compile(r"^[a-z_][a-z0-9._-]{0,31}\Z")
REMOTE_PATH = re.compile(r"^(/[A-Za-z0-9._-]+){2,12}\Z")
PARTITION = re.compile(r"^[A-Za-z0-9_-]{1,32}\Z")
GRES = re.compile(r"^gpu(:[A-Za-z0-9_.-]{1,20})?(:[1-8])?\Z")
NODELIST = re.compile(r"^[A-Za-z0-9,\[\]-]{0,200}\Z")
TIME = re.compile(r"^(\d{1,2}):([0-5]\d):([0-5]\d)\Z")
MEM = re.compile(r"^[1-9]\d{0,4}[MG]\Z")
HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z")
STAMP = re.compile(r"^\d{8}-\d{6}\Z")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}-\d{8}-\d{6}\Z")
JOB_ID = re.compile(r"^\d{1,12}\Z")
POLICY = re.compile(r"^[a-z][a-z0-9_]{0,39}\Z")
# Lines allowed in the env setup: comments, module, (conda|source) activate, source <file>, export.
# WHY a whitelist: the setup lands in the job script; these forms set up an env and nothing else.
# No quotes, ;, |, &, <, >, backticks or parentheses, so no second command can hide in a line.
_VALUE = r"[A-Za-z0-9_./:@%+,=~${}-]*"
ENV_LINE = re.compile(
    r"^(#[^\n]*"
    r"|module (load|purge|use)( [A-Za-z0-9_./+-]+)*"
    r"|(source|conda) activate [A-Za-z0-9_./+-]+"
    r"|source [A-Za-z0-9_./$+-]+"
    rf"|export( [A-Za-z_][A-Za-z0-9_]*={_VALUE})+)\Z"
)


# Env names that look like a secret. WHY refuse them: the setup is written into a job script on the
# cluster's shared file system and shown on this page. A name ending in _PATH or _FILE points at a
# file the cluster's own login wrote (HF_TOKEN_PATH), which is the right place for a token.
_SECRET_NAME = re.compile(r"(?i)(TOKEN|KEY|SECRET|PASSWORD)")
_SECRET_OK = re.compile(r"(?i)_(PATH|FILE)\Z")
TOKEN_ADVICE = ("Tokens belong in the cluster's own login, never in Studio: run huggingface-cli "
                "login or wandb login once on the cluster, and the job finds them there.")


class FormError(ValueError):
    """Bad input: field -> message, for the form to show beside each field."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _hms(s: str) -> int:
    m = TIME.match(s)
    if not m:
        raise ValueError(s)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _int(v: Any, lo: int, hi: int) -> int | None:
    if isinstance(v, bool):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi and str(n) == str(v).strip() else None


# -- settings --------------------------------------------------------------------------------------
@dataclass
class Settings:
    host: str = "hpc"
    remote_base: str = ""  # empty: /scratch/<remote user>/phi-studio
    env: str = DEFAULT_ENV
    partition: str = "gpu"
    # WHY a second partition: while the user has 8 jobs submitted, the gpu QoS refuses even
    # `sbatch --test-only` (QOSMaxSubmitJobPerUserLimit, seen 2026-10-04). A test on gpu-short,
    # the same nodes with its own QoS, still checks the script and the request.
    test_partition: str = "gpu-short"
    gres: str = "gpu:1"
    # WHY these: d1025 is the one 16 GB GPU in gpu, c2204-c2207 are slow Zen 1 hosts that tripled
    # ACT step time (phi docs/hpc/explorer.md, "Partitions").
    exclude: str = "d1025,c2204,c2205,c2206,c2207"
    time: str = "08:00:00"
    cpus: int = 16
    mem: str = "64G"
    remote_user: str = ""  # learnt from the last check, never typed

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Settings:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> Settings:
        e: dict[str, str] = {}
        # WHY check types first: settings come from a window and from a file a person can edit,
        # and str(5) would pass a pattern while the 5 itself reached the script.
        for f in fields(self):
            if f.name != "cpus" and not isinstance(getattr(self, f.name), str):
                e[f.name] = "Must be text."
        if isinstance(self.cpus, bool) or not isinstance(self.cpus, (int, str)):
            e["cpus"] = "A whole number from 1 to 128."
        if e:
            raise FormError(e)
        if not HOST.match(self.host):
            e["host"] = ("An ssh host alias: letters, digits, dot, dash and underscore, "
                         "starting with a letter or digit.")  # fmt: skip
        rb = self.remote_base
        if rb and (not REMOTE_PATH.match(rb) or ".." in rb):
            e["remote_base"] = "An absolute folder with at least two parts, such as /scratch/you/x."
        lines = [ln.strip() for ln in self.env.splitlines() if ln.strip()]
        bad = [ln for ln in lines if not ENV_LINE.match(ln)]
        secret = [n for ln in lines if ln.startswith("export ")
                  for n in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", ln)
                  if _SECRET_NAME.search(n) and not _SECRET_OK.search(n)]  # fmt: skip
        if bad:
            e["env"] = (f"Studio does not allow this line: {bad[0][:80]}. Use module load, "
                        "source activate, source <file> or export NAME=value.")  # fmt: skip
        elif secret:
            e["env"] = f"Remove {secret[0]}. {TOKEN_ADVICE}"
        elif not any(ln.split()[0] in ("source", "conda", "module") for ln in lines):
            e["env"] = "Add the line that activates the env, such as source activate lerobot-gpu."
        if not PARTITION.match(self.partition):
            e["partition"] = "A partition name, such as gpu."
        if self.test_partition and not PARTITION.match(self.test_partition):
            e["test_partition"] = "A partition name, such as gpu-short, or leave empty."
        if not GRES.match(self.gres):
            e["gres"] = "Such as gpu:1 or gpu:h200:1."
        if not NODELIST.match(self.exclude):
            e["exclude"] = "Node names separated by commas, such as d1025,c2204."
        try:
            if _hms(self.time) < MIN_PART_S:
                e["time"] = ("At least 00:10:00. The longest a part may run is the partition's own "
                             "limit, which Check the cluster reads and Submit checks again.")
        except ValueError:
            e["time"] = "Hours, minutes and seconds, such as 08:00:00."
        if _int(self.cpus, 1, 128) is None:
            e["cpus"] = "A whole number from 1 to 128."
        if not MEM.match(self.mem):
            e["mem"] = "Such as 64G or 32000M."
        if self.remote_user and not USER.match(self.remote_user):
            e["remote_user"] = "Run Check the cluster again to learn your user name."
        if e:
            raise FormError(e)
        self.cpus = int(self.cpus)
        return self

    @classmethod
    def salvage(cls, raw: Any) -> tuple[Settings, list[str]]:
        """Settings from a stored file: the fields that pass, with every other one back at its
        default. Returns the names put back, so the window can say which."""
        d = raw if isinstance(raw, dict) else {}
        known = {f.name for f in fields(cls)}
        s = cls.from_dict(d)
        dropped: list[str] = []
        for _ in range(len(known)):  # each round puts back at least one field
            try:
                return s.validate(), sorted(dropped)
            except FormError as e:
                bad = [k for k in e.errors if k in known]
                if not bad:
                    break
                for k in bad:
                    setattr(s, k, getattr(cls(), k))
                    if k in d:
                        dropped.append(k)
        return cls().validate(), sorted(k for k in d if k in known)

    def base(self) -> str:
        """The run folder's parent on the cluster."""
        if self.remote_base:
            return self.remote_base
        if not USER.match(self.remote_user):
            raise FormError({"remote_base": "Run Check the cluster first, so Studio knows your "
                                            "user name, or type a folder."})  # fmt: skip
        return f"/scratch/{self.remote_user}/phi-studio"

    def env_lines(self) -> list[str]:
        return [ln.strip() for ln in self.env.splitlines() if ln.strip()]

    def env_name(self) -> str | None:
        """The env the setup activates: a name or a path, from its activate line."""
        for ln in self.env_lines():
            parts = ln.split()
            if len(parts) == 3 and parts[0] in ("source", "conda") and parts[1] == "activate":
                return parts[2]
        return None


# -- the job form ----------------------------------------------------------------------------------
@dataclass
class Job:
    where: str = "cluster"  # "cluster" or "mac"
    dataset: str = ""
    policy: str = "act"
    pretrained: str = ""  # a Hub policy to start from (--policy.path); empty: --policy.type
    steps: int = 100_000
    batch_size: int = 8
    save_freq: int = 10_000
    name: str = ""
    push: bool = False
    repo_id: str = ""
    wandb: bool = False
    parts: int = 1
    stamp: str = ""  # set once per form by the window, so the preview names the same run as submit

    @classmethod
    def from_dict(cls, d: Any, policies: tuple[str, ...] | list[str]) -> Job:
        if not isinstance(d, dict):
            raise FormError({"form": "Studio got no job form."})
        known = {f.name for f in fields(cls)}
        raw = {k: v for k, v in d.items() if k in known}
        e: dict[str, str] = {}
        job = cls()
        where = raw.get("where", "cluster")
        if where not in ("cluster", "mac"):
            e["where"] = "Cluster or this Mac."
        job.where = where
        for key in ("dataset", "pretrained", "name", "repo_id", "stamp", "policy"):
            v = raw.get(key, getattr(job, key))
            setattr(job, key, v.strip() if isinstance(v, str) else v)
        if not isinstance(job.dataset, str) or not HUB_ID.match(job.dataset):
            e["dataset"] = "A dataset id such as Parv-09/my_dataset: owner, slash, name."
        if job.pretrained:
            if not isinstance(job.pretrained, str) or not HUB_ID.match(job.pretrained):
                e["pretrained"] = "A Hub policy id such as lerobot/smolvla_base, or leave empty."
        elif not isinstance(job.policy, str) or not POLICY.match(job.policy) or (
            job.policy not in policies
        ):
            e["policy"] = "Pick a policy type from the list."
        if not isinstance(job.name, str) or not NAME.match(job.name):
            e["name"] = "Letters, digits, - and _, up to 40, starting with a letter or digit."
        steps = _int(raw.get("steps", job.steps), 1, 10_000_000)
        if steps is None:
            e["steps"] = "A whole number from 1 to 10 000 000."
        batch = _int(raw.get("batch_size", job.batch_size), 1, 4096)
        if batch is None:
            e["batch_size"] = "A whole number from 1 to 4096."
        save = _int(raw.get("save_freq", job.save_freq), 1, 10_000_000)
        if save is None:
            e["save_freq"] = "A whole number of steps."
        elif steps is not None and save > steps:
            e["save_freq"] = "No more than the number of steps."
        parts = _int(raw.get("parts", job.parts), 1, MAX_PARTS)
        if parts is None:
            e["parts"] = f"From 1 to {MAX_PARTS}: your submit limit."
        job.steps, job.batch_size = steps or 0, batch or 0
        job.save_freq, job.parts = save or 0, parts or 1
        job.push, job.wandb = raw.get("push") is True, raw.get("wandb") is True
        if job.push and (not isinstance(job.repo_id, str) or not HUB_ID.match(job.repo_id)):
            # WHY check here: LeRobot raises the same at start (configs/train.py:260-266), but
            # only once the job has waited in the queue for a GPU.
            e["repo_id"] = "Pushing needs the policy's Hub id, such as Parv-09/act_cubes."
        if not job.push:
            job.repo_id = ""
        if job.pretrained and "policy" not in e and job.policy not in policies:
            job.policy = ""  # unused with a pretrained policy; never kept unchecked
        if not isinstance(job.stamp, str) or not STAMP.match(job.stamp):
            e["stamp"] = "Reload the page: the form lost its run stamp."
        if e:
            raise FormError(e)
        return job

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.stamp}"

    @property
    def log_freq(self) -> int:
        """About a hundred points for a short run, LeRobot's default of one per 200 steps for a
        long one (configs/train.py:107)."""
        return min(200, max(1, self.steps // 100))

    @property
    def policy_label(self) -> str:
        return self.pretrained or self.policy


def step_id(step: int, total: int) -> str:
    """The checkpoint folder name LeRobot gives a step (common/train_utils.py:44-46)."""
    return f"{step:0{max(6, len(str(total)))}d}"


def train_args(
    job: Job, *, device: str, output_dir: str, cluster: bool, workers: int = 0
) -> list[str]:
    """lerobot-train flags for a new run. Each parses with LeRobot's TrainPipelineConfig
    (tests/test_train.py runs them through draccus and validate())."""
    a = [f"--dataset.repo_id={job.dataset}"]
    if cluster:
        # WHY pyav: torchcodec fails to load on Explorer (libnppicc.so.12); pyav decode is hidden
        # behind the GPU step (phi docs/hpc/explorer.md, "Video decoding").
        a.append("--dataset.video_backend=pyav")
    # WHY not both: LeRobot refuses --policy.path with --policy.type (configs/parser.py:221).
    a.append(f"--policy.path={job.pretrained}" if job.pretrained else f"--policy.type={job.policy}")
    a += [f"--policy.device={device}", f"--policy.push_to_hub={'true' if job.push else 'false'}"]
    if job.push:
        a.append(f"--policy.repo_id={job.repo_id}")
    a += [f"--batch_size={job.batch_size}", f"--steps={job.steps}", f"--save_freq={job.save_freq}",
          f"--log_freq={job.log_freq}"]  # fmt: skip
    if workers:
        a.append(f"--num_workers={workers}")
    a += [f"--output_dir={output_dir}", f"--job_name={job.name}",
          f"--wandb.enable={'true' if job.wandb else 'false'}"]  # fmt: skip
    return a


def resume_args(output_dir: str) -> list[str]:
    """Continue a run from its newest checkpoint (configs/train.py:87-92, 174-207)."""
    return [f"--config_path={output_dir}/checkpoints/last/pretrained_model/train_config.json",
            "--resume=true"]  # fmt: skip


def part_name(run_id: str, part: int) -> str:
    return f"{run_id}-p{part}"


def mac_command(job: Job, run_dir: Path) -> str:
    """The command typed into Studio's terminal. tee keeps a copy of the log, so the Train page can
    chart a run on this Mac the same way as one on the cluster."""
    args = train_args(job, device="mps", output_dir=str(run_dir / "train"), cluster=False)
    log = shlex.quote(str(run_dir / "train.log"))
    return f"{shlex.join(['lerobot-train', *args])} 2>&1 | tee -a {log}"


def sbatch_script(job: Job, s: Settings, run_dir: str) -> str:
    """One script for every part. Part 1 starts training; a later part resumes from the newest
    checkpoint, or exits at once if training already reached the last step."""
    if not REMOTE_PATH.match(run_dir) or ".." in run_dir:
        raise FormError({"remote_base": f"Not a usable folder: {run_dir}"})
    out = f"{run_dir}/train"
    final = step_id(job.steps, job.steps)
    flags = train_args(job, device="cuda", output_dir=out, cluster=True, workers=int(s.cpus))
    n = job.parts
    q = shlex.quote
    lines = [
        "#!/bin/bash",
        f"# Phi Studio run {job.run_id}: {job.policy_label} on {job.dataset}, {job.steps} steps,",
        f"# up to {n} part{'s' if n > 1 else ''} of {s.time}. Every part runs this same script.",
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --gres={s.gres}",
        f"#SBATCH --time={s.time}",
        f"#SBATCH --cpus-per-task={int(s.cpus)}",
        f"#SBATCH --mem={s.mem}",
    ]
    if s.exclude:
        lines.append(f"#SBATCH --exclude={s.exclude}")
    lines += [
        f"#SBATCH --output={run_dir}/logs/%x-%j.out",
        "",
        "set -u",
        f"OUT={q(out)}",
        'CONFIG="$OUT/checkpoints/last/pretrained_model/train_config.json"',
        f'FINAL="$OUT/checkpoints/{final}/pretrained_model"',
        "",
        "# The env, from Studio's settings",
        *s.env_lines(),
        "",
        f'echo "phi-studio: part ${{PHI_PART:-1}} of {n}, job $SLURM_JOB_ID on $(hostname),'
        ' $(date)"',
        'START_LAST=$(readlink "$OUT/checkpoints/last" 2>/dev/null || echo none)',
        'if [ -d "$FINAL" ]; then',
        f'  echo "phi-studio: training already reached step {job.steps}; nothing left to do"',
        "  exit 0",
        "fi",
        'if [ -f "$CONFIG" ]; then',
        '  LAST=$(basename "$(readlink "$OUT/checkpoints/last")")',
        '  echo "phi-studio: resuming from step $((10#$LAST))"',
        f"  srun lerobot-train {' '.join(q(a) for a in resume_args(out))}",
        "else",
        "  # WHY: a part that died before its first checkpoint leaves the folder behind, and",
        "  # lerobot-train refuses an existing output folder unless it resumes",
        "  # (configs/train.py:236-240).",
        '  if [ -e "$OUT" ]; then mv "$OUT" "$OUT.no-checkpoint-$SLURM_JOB_ID"; fi',
        '  echo "phi-studio: starting at step 0"',
        "  srun lerobot-train \\",
        *[f"    {q(a)} \\" for a in flags[:-1]],
        f"    {q(flags[-1])}",
        "fi",
        "RC=$?",
        'echo "phi-studio: lerobot-train exited with code $RC, $(date)"',
    ]
    if n > 1:
        lines += [
            "# Later parts wait with afterany, so each starts whatever this part did. Cancel",
            "# the ones still waiting, by their exact names, when they have nothing to do:",
            "# training is finished, or this part saved no checkpoint, so the next repeats it.",
            "cancel_later() {",
            f"  for ((k = ${{PHI_PART:-1}} + 1; k <= {n}; k++)); do",
            f'    scancel --user="$USER" --name={q(job.run_id)}-p"$k" --state=PENDING',
            "  done",
            "}",
            'END_LAST=$(readlink "$OUT/checkpoints/last" 2>/dev/null || echo none)',
            'if [ "$RC" = 0 ] && [ -d "$FINAL" ]; then',
            "  cancel_later",
            'elif [ "$END_LAST" = "$START_LAST" ]; then',
            f'  echo "{STALLED} (still at ${{START_LAST}}), so the next part would repeat it.'
            ' Cancelling the parts still waiting."',
            "  cancel_later",
            "fi",
        ]
    lines.append("exit $RC")
    return "\n".join(lines) + "\n"


def plan(job: Job, s: Settings) -> dict[str, Any]:
    t = _hms(s.time)
    return {"parts": job.parts, "part_time": s.time, "total_hours": round(job.parts * t / 3600, 2),
            "names": [part_name(job.run_id, k) for k in range(1, job.parts + 1)]}  # fmt: skip


# -- what each policy needs installed --------------------------------------------------------------
_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?[^;]*(?:;(.*))?$")
_EXTRA = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


def norm(name: str) -> str:
    """A distribution name as pip compares it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def extra_needs(requires: list[str]) -> dict[str, set[str]]:
    """LeRobot's extras, each to the distributions it pulls in, following lerobot[x] inside an
    extra. From the Requires-Dist lines of LeRobot's own metadata. Platform markers are ignored,
    so a package needed on only some machines is listed for all of them."""
    direct: dict[str, list[tuple[str, list[str]]]] = {}
    for r in requires:
        m = _REQ.match(r)
        if not m or not m[3]:
            continue
        ex = _EXTRA.search(m[3])
        if not ex:
            continue
        inner = [x.strip() for x in (m[2] or "").split(",") if x.strip()]
        direct.setdefault(norm(ex[1]), []).append((norm(m[1]), [norm(x) for x in inner]))

    def walk(extra: str, seen: set[str]) -> set[str]:
        if extra in seen:
            return set()
        seen.add(extra)
        out: set[str] = set()
        for name, inner in direct.get(extra, []):
            if name == "lerobot":
                for x in inner:
                    out |= walk(x, seen)
            else:
                out.add(name)
        return out

    return {x: walk(x, set()) for x in direct}


def policy_extra(policy: str, extras: set[str] | dict[str, Any]) -> str | None:
    """The extra a policy type installs with: smolvla -> smolvla, wall_x -> wallx, and the pi
    family (pi0, pi05, pi0_fast) -> pi, as LeRobot 0.6.0's metadata names them."""
    key = policy.replace("_", "").replace("-", "")
    for x in extras:
        if x.replace("-", "") == key:
            return str(x)
    if policy.startswith("pi0") and "pi" in extras:
        return "pi"
    return None


def missing_for(policy: str, needs: dict[str, set[str]], installed: set[str]) -> list[str]:
    want = set(needs.get("training", set()))
    ex = policy_extra(policy, needs)
    if ex:
        want |= needs[ex]
    return sorted(n for n in want if n not in installed)


def dist_names(listing: str) -> set[str]:
    """Distribution names from an `ls` of site-packages: foo_bar-1.2.dist-info -> foo-bar."""
    out = set()
    for ln in listing.split():
        if ln.endswith(".dist-info"):
            out.add(norm(ln[: -len(".dist-info")].rsplit("-", 1)[0]))
    return out


# The probe Studio runs in a child Python to ask the installed LeRobot. WHY a child: importing the
# policies pulls in torch, about 4 s and several hundred MB the server does not otherwise need.
PROBE = """
import json, importlib.metadata as md
import lerobot, lerobot.policies  # importing the package registers every policy config
from lerobot.configs.policies import PreTrainedConfig
print(json.dumps({"version": lerobot.__version__,
                  "policies": sorted(PreTrainedConfig.get_known_choices()),
                  "requires": md.requires("lerobot") or [],
                  "installed": sorted({(d.metadata["Name"] or "") for d in md.distributions()})}))
"""


def read_probe(out: str) -> dict[str, Any]:
    d = json.loads(out.strip().splitlines()[-1])
    needs = extra_needs(d["requires"])
    installed = {norm(n) for n in d["installed"] if n}
    pols = [p for p in d["policies"] if isinstance(p, str) and POLICY.match(p)]
    return {
        "source": f"LeRobot {d['version']} on this Mac",
        "version": d["version"],
        "policies": [{"type": p, "extra": policy_extra(p, needs),
                      "missing_here": missing_for(p, needs, installed)} for p in pols],  # fmt: skip
        "needs": {k: sorted(v) for k, v in needs.items()},
    }


def fallback_policies() -> dict[str, Any]:
    return {"source": "LeRobot 0.6.0's list (LeRobot is not importable here)", "version": None,
            "policies": [{"type": p, "extra": None, "missing_here": []} for p in POLICIES_060],
            "needs": {}}  # fmt: skip


# -- local datasets --------------------------------------------------------------------------------
def lerobot_home() -> Path:
    """$HF_LEROBOT_HOME, else $HF_HOME/lerobot, else ~/.cache/huggingface/lerobot (LeRobot's own
    rule, utils/constants.py:68-69)."""
    if os.environ.get("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    hf = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(hf).expanduser() / "lerobot"


def local_datasets(home: Path | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Datasets recorded on this Mac: <home>/<owner>/<name>/meta/info.json."""
    home = home or lerobot_home()
    out: list[dict[str, Any]] = []
    try:
        owners = sorted(p for p in home.iterdir() if p.is_dir())
    except OSError:
        return out
    for owner in owners:
        if owner.name in ("hub", "calibration") or owner.name.startswith("."):
            continue
        try:
            names = sorted(owner.iterdir())
        except OSError:
            continue
        for d in names:
            info = d / "meta" / "info.json"
            repo = f"{owner.name}/{d.name}"
            if not info.is_file() or not HUB_ID.match(repo):
                continue
            try:
                meta = json.loads(info.read_text())
            except (OSError, ValueError):
                meta = {}
            out.append({"id": repo, "episodes": meta.get("total_episodes"),
                        "frames": meta.get("total_frames"), "fps": meta.get("fps")})  # fmt: skip
            if len(out) >= limit:
                return out
    return out


# -- the training log ------------------------------------------------------------------------------
# utils/utils.py:64-68: "{level} {[PID: n] }{YYYY-mm-dd HH:MM:SS} {file:line, last 15} {message}";
# utils/logging_utils.py:190-201: "step:{big} smpl:{big} ep:{big} epch:{:.2f} loss:.. grdn:.. lr:.."
# WHY search, not match: on a terminal, tqdm's bar shares the line ("Training: 17%|..]INFO ...").
_METRIC = re.compile(
    r"(?:INFO|WARNING|DEBUG|ERROR|CRITICAL) (?:\[PID: \d+\] )?(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
    r" +\S+ step:(\d+(?:\.\d+)?)([KMBTQ]?) smpl:\S+ ep:\S+ epch:(\S+)(.*)$"
)
_PAIR = re.compile(r"([A-Za-z_/]+):(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
_CKPT = re.compile(r"Checkpoint policy after step (\d+)")
_ANCHOR = re.compile(r"phi-studio: (?:resuming from|starting at) step (\d+)")
_TOTAL = re.compile(r"cfg\.steps=(\d+)")
STALLED = "phi-studio: this part saved no new checkpoint"
_UNITS = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12, "Q": 10**15}
MAX_POINTS = 4000
# WHY a cap: a progress bar that never prints a newline (tqdm on a dead terminal) would otherwise
# grow the unfinished line, and the copy made on every read, without bound. A metrics line is under
# 400 characters; keeping the last 64 KiB keeps any line LeRobot writes whole.
MAX_PARTIAL_LINE = 64 * 1024


def _utf8() -> codecs.IncrementalDecoder:
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def _stamp(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


@dataclass
class LogState:
    """A run's log, fed in pieces as they arrive. LeRobot prints the step through
    format_big_number(precision=0) (utils/utils.py:102), so step 1 200 reads "1K". Studio keeps the
    exact step: LeRobot logs only when step % log_freq == 0 (scripts/lerobot_train.py:591), so each
    metrics line is the next multiple of log_freq after the last known step, checked against the
    printed value. "Checkpoint policy after step N" and the job script's own "resuming from step N"
    set the known step too."""

    log_freq: int
    total: int
    exact: int | None = None
    seg: int = 0  # one per start or resume, so the rate never spans a gap between parts
    points: list[dict[str, Any]] = field(default_factory=list)
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    rest: str = ""  # an unfinished last line, at most MAX_PARTIAL_LINE characters
    ended: str | None = None  # "End of training" or the job script's exit line
    stalled: str | None = None  # the job script's line for a part that saved no checkpoint
    decoder: codecs.IncrementalDecoder = field(default_factory=_utf8, repr=False)

    def feed_bytes(self, data: bytes) -> None:
        """Bytes as read. WHY a decoder that carries state: a read can end inside a character."""
        self.feed(self.decoder.decode(data))

    def end_part(self) -> None:
        """A part's log has ended. Its unfinished last line goes to the log view but not the chart:
        a job killed mid-write leaves a cut line, and "loss:0.0" from "loss:0.023" is not data."""
        tail = (self.rest + self.decoder.decode(b"", final=True)).strip()
        if tail:
            self.tail.append(tail[-600:])
        self.rest = ""
        self.decoder = _utf8()

    def feed(self, text: str) -> None:
        cut = max(text.rfind("\n"), text.rfind("\r"))
        if cut < 0:
            self.rest = (self.rest + text)[-MAX_PARTIAL_LINE:]
            return
        whole = self.rest + text[: cut + 1]
        self.rest = text[cut + 1 :][-MAX_PARTIAL_LINE:]
        for ln in re.split(r"[\r\n]", whole):
            if ln.strip():
                self.line(ln)

    def line(self, ln: str) -> None:
        if not ln.startswith("Training:") or "INFO" in ln:  # drop bare progress-bar redraws
            self.tail.append(ln[-600:])
        if m := _ANCHOR.search(ln):
            # A new part: the last part's end and stall no longer describe the run.
            self.exact, self.seg = int(m[1]), self.seg + 1
            self.ended = self.stalled = None
            return
        if STALLED in ln:
            self.stalled = ln.strip()[-300:]
            return
        if m := _CKPT.search(ln):
            self.exact = int(m[1])
            return
        if m := _TOTAL.search(ln):
            self.total = int(m[1])
            return
        if "End of training" in ln or "phi-studio: lerobot-train exited" in ln:
            self.ended = ln.strip()[-200:]
            return
        m = _METRIC.search(ln)
        if not m:
            return
        shown, unit = float(m[2]), _UNITS[m[3]]
        value = shown * unit
        # WHY the next multiple, not exact + log_freq: a checkpoint at step 2 500 with log_freq 200
        # is followed by the line for 2 600 (seen in a real Explorer log, job 9017370).
        lf = self.log_freq
        guess = (self.exact // lf + 1) * lf if self.exact is not None else None
        if unit == 1:
            step, sure = int(value), True
        elif guess is not None and abs(guess - value) <= unit / 2:
            step, sure = guess, True
        else:
            step, sure = int(value), False
        self.exact = step if sure else None
        p: dict[str, Any] = {"step": step, "sure": sure, "t": _stamp(m[1]), "seg": self.seg}
        for k, v in _PAIR.findall(m[5]):
            if k in ("loss", "lr", "grdn", "updt_s", "data_s"):
                p[k] = float(v)
        # WHY: a resumed part restarts from its last checkpoint and redoes the steps the part before
        # it logged after that checkpoint. Its values replace those, so the line never runs back.
        while sure and self.points and self.points[-1]["step"] >= step:
            self.points.pop()
        self.points.append(p)
        if len(self.points) > MAX_POINTS:  # thin the oldest half; the newest stay dense
            half = len(self.points) // 2
            self.points = self.points[:half:2] + self.points[half:]

    def summary(self) -> dict[str, Any]:
        last = self.points[-1] if self.points else None
        step = last["step"] if last else (self.exact or 0)
        rate = None
        seg = [p for p in self.points if last and p["seg"] == last["seg"]][-12:]
        if len(seg) >= 2 and seg[-1]["t"] > seg[0]["t"]:
            rate = (seg[-1]["step"] - seg[0]["step"]) / (seg[-1]["t"] - seg[0]["t"])
        left = max(0, self.total - step)
        eta = left / rate if rate and rate > 0 else None
        return {"step": step, "total": self.total, "rate": rate,
                "eta_s": None if eta is None or not math.isfinite(eta) else round(eta),
                "ended": self.ended, "stalled": self.stalled}  # fmt: skip

    def view(self) -> dict[str, Any]:
        pts = [{k: p[k] for k in ("step", "sure", "loss", "lr") if k in p} for p in self.points]
        return {**self.summary(), "points": pts, "tail": list(self.tail)}


# -- the stores ------------------------------------------------------------------------------------
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # WHY: a crash mid-write leaves the previous file, never half of one


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        # Keep the damaged file for a person to read; start clean rather than refuse to load.
        try:
            os.replace(path, path.with_suffix(path.suffix + f".damaged-{int(time.time())}"))
        except OSError:
            pass
        return default


class Store:
    """train_settings.json and train_jobs.json in Studio's data folder. Only runs Studio started
    are here, and only they may be cancelled."""

    def __init__(self, data_dir: Path) -> None:
        # WHY a thread lock: submit saves from a worker thread while the loop saves job states.
        self.lock = threading.Lock()
        self.settings_path = data_dir / "train_settings.json"
        self.jobs_path = data_dir / "train_jobs.json"
        self.runs_dir = data_dir / "train"
        self.models_dir = data_dir / "models"

    def settings(self) -> Settings:
        return self.settings_report()[0]

    def settings_report(self) -> tuple[Settings, list[str]]:
        """The stored settings, with any field that fails its check back at its default, and the
        names of those fields."""
        return Settings.salvage(read_json(self.settings_path, {}))

    def save_settings(self, s: Settings) -> None:
        write_json(self.settings_path, asdict(s))

    def runs(self) -> list[dict[str, Any]]:
        raw = read_json(self.jobs_path, {"runs": []})
        runs = raw.get("runs") if isinstance(raw, dict) else None
        return [r for r in runs or [] if isinstance(r, dict) and RUN_ID.match(str(r.get("id")))]

    def run(self, run_id: Any) -> dict[str, Any] | None:
        return next((r for r in self.runs() if r["id"] == run_id), None)

    def put(self, run: dict[str, Any]) -> None:
        """Add a run at the top, or replace it where it is."""
        with self.lock:
            runs = self.runs()
            i = next((i for i, r in enumerate(runs) if r["id"] == run["id"]), None)
            if i is None:
                runs.insert(0, run)
            else:
                runs[i] = run
            write_json(self.jobs_path, {"runs": runs})

    def mutate(self, run_id: str, fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any] | None:
        """Read one run, change it with fn and write it back, all under the lock. WHY: a change
        made from a copy read earlier would put back parts a submit saved in between."""
        with self.lock:
            runs = self.runs()
            run = next((r for r in runs if r["id"] == run_id), None)
            if run is None:
                return None
            fn(run)
            write_json(self.jobs_path, {"runs": runs})
            return run

    def update_jobs(self, states: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
        """Merge fresh job states into the runs on disk, read again under the lock, so a run
        saved meanwhile by a submit keeps its new parts."""
        with self.lock:
            runs = self.runs()
            for run in runs:
                for j in run.get("jobs", []):
                    if str(j.get("id")) in states:
                        j.update(states[str(j["id"])])
            write_json(self.jobs_path, {"runs": runs})
            return runs


def new_run(job: Job, **extra: Any) -> dict[str, Any]:
    return {"id": job.run_id, "name": job.name, "where": job.where, "created": time.time(),
            "dataset": job.dataset, "policy": job.policy_label, "steps": job.steps,
            "batch_size": job.batch_size, "save_freq": job.save_freq, "log_freq": job.log_freq,
            "parts_planned": job.parts if job.where == "cluster" else 1, "push": job.push,
            "repo_id": job.repo_id, "wandb": job.wandb, "jobs": [], "error": None,
            "fetched": None, **extra}  # fmt: skip
