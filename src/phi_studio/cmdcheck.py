"""Does a LeRobot command name each arm by its own port AND its own id?

LeRobot picks the calibration FILE by the id and the ARM by the port (robot.py:49-53). When the
motors disagree with the file, connect() asks for ENTER, and ENTER writes that file into the
servos (so_follower.py:99-123, so_leader.py:84-95). A port and an id that name two different arms
therefore put one arm's calibration into another arm: its angles are off by tens of degrees and
teleop drives joints into their stops (the 2026-10-05 fault, phi configs/calibration/README.md).

This module reads a command line the way LeRobot's CLI does and checks every (port, id) pair
against robot-config.yaml, which is the one place Studio records which arm is on which port.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phi_studio import rigspec
from phi_studio.rig import label

# The CLIs that connect arms (each calls robot.connect(), which may prompt). setup-motors and
# find-port never read a calibration file.
CONNECTS = ("lerobot-calibrate", "lerobot-teleoperate", "lerobot-record", "lerobot-replay",
            "lerobot-find-joint-limits", "lerobot-eval")  # fmt: skip
# so_follower.py:118-120 and so_leader.py:87-89, word for word up to the id
PROMPT = re.compile(r"Press ENTER to use provided calibration file associated with the id "
                    r"([^\s,]+), or type 'c'")  # fmt: skip
OK, DANGER, UNKNOWN = "ok", "danger", "unknown"


def same_port(a: str | None, b: str | None) -> bool:
    """macOS gives one USB device two names, /dev/tty.X and /dev/cu.X."""
    if not a or not b:
        return False
    return a.replace("/dev/cu.", "/dev/tty.") == b.replace("/dev/cu.", "/dev/tty.")


@dataclass(frozen=True)
class CmdArm:
    """One arm as a command names it."""

    section: str  # robot | teleop
    role: str  # follower | leader
    side: str | None
    port: str | None
    lerobot_id: str | None
    calibration_dir: str | None

    def file(self, root: Path) -> Path | None:
        if self.lerobot_id is None:
            return None
        if self.calibration_dir:
            return Path(self.calibration_dir) / f"{self.lerobot_id}.json"
        kind, folder = rigspec.FOLDER[self.role]
        return root / kind / folder / f"{self.lerobot_id}.json"


def _flags(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                k, v = a[2:], argv[i + 1]
                i += 1
            else:
                k, v = a[2:], "true"
            out[k] = v
        i += 1
    return out


def program(command: str) -> str | None:
    """The LeRobot CLI a command line runs: lerobot-teleoperate, also when run as
    `python -m lerobot.scripts.lerobot_teleoperate` or through a full path."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    for i, a in enumerate(argv):
        name = Path(a).name
        if name.startswith("lerobot-"):
            return name
        if a == "-m" and i + 1 < len(argv):
            mod = argv[i + 1].rsplit(".", 1)[-1]
            return mod.replace("_", "-") if mod.startswith("lerobot_") else None
    return None


def arms_in(command: str) -> list[CmdArm]:
    """Every arm a LeRobot command connects, from its --robot.* and --teleop.* flags."""
    try:
        flags = _flags(shlex.split(command))
    except ValueError:
        return []
    out = []
    for section, role in (("robot", "follower"), ("teleop", "leader")):
        typ = flags.get(f"{section}.type")
        if not typ:
            continue
        rid = flags.get(f"{section}.id")
        cal_dir = flags.get(f"{section}.calibration_dir")
        if typ in rigspec.BI_FOLLOWER + rigspec.BI_LEADER:
            for side in rigspec.SIDES:
                out.append(CmdArm(section, role, side,
                                  flags.get(f"{section}.{side}_arm_config.port"),
                                  f"{rid}_{side}" if rid else None, cal_dir))  # fmt: skip
        else:
            out.append(CmdArm(section, role, None, flags.get(f"{section}.port"), rid, cal_dir))
    return out


@dataclass(frozen=True)
class Verdict:
    level: str  # ok | danger | unknown
    message: str
    fix: str = ""
    arm: str | None = None  # the rig arm on the command's port, by Studio's key
    file: str | None = None

    def public(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message, "fix": self.fix, "arm": self.arm,
                "file": self.file}  # fmt: skip


def check_arm(c: CmdArm, spec: rigspec.RigSpec, root: Path) -> Verdict:
    """Whether `c`'s port and id name the same arm of the rig."""
    f = c.file(root)
    file = str(f) if f else None
    if not c.port:
        return Verdict(UNKNOWN, f"The command gives no port for the {c.role}.", arm=None, file=file)
    on_port = next((a for a in spec.arms if same_port(a.port, c.port)), None)
    by_id = next((a for a in spec.arms if c.lerobot_id and a.lerobot_id == c.lerobot_id
                  and a.role == c.role), None)  # fmt: skip
    if on_port is None:
        return Verdict(UNKNOWN, f"{c.port} is not an arm in robot-config.yaml, so Studio cannot "
                       "tell which arm it is.", "Run Detect arms on the Rig setup page, or use "
                       "the commands Studio shows.", file=file)  # fmt: skip
    who = label(on_port.key)
    if on_port.role != c.role:
        return Verdict(DANGER, f"{c.port} is the {who}, but the command uses it as the {c.role}.",
                       "Swap the ports in the command, or use the commands Studio shows.",
                       on_port.key, file)  # fmt: skip
    if c.lerobot_id != on_port.lerobot_id:
        other = f"the {label(by_id.key)}'s id" if by_id else "an id no arm in robot-config.yaml has"
        return Verdict(
            DANGER,
            f"{c.port} is the {who}, but the command calls it {c.lerobot_id or '(no id)'}, "
            f"{other}. "
            f"If LeRobot asks for ENTER, ENTER writes {Path(file).name if file else 'that file'} "
            f"into the {who}'s motors.",
            f"Use --{c.section}.id={on_port.lerobot_id} for this port"
            + (" (bimanual: the id without _left/_right)" if c.side else "") + ".",
            on_port.key, file)  # fmt: skip
    if f is not None and f.is_symlink():
        return Verdict(DANGER, f"{f.name} is a link to {f.resolve().name}: LeRobot reads and "
                       f"writes another file through it, so the {who} shares a calibration.",
                       "Fix it in Calibration files on the Rig setup page.", on_port.key,
                       file)  # fmt: skip
    return Verdict(OK, f"{c.port} is the {who} and {c.lerobot_id} is its id.", arm=on_port.key,
                   file=file)


def check(command: str, spec: rigspec.RigSpec, root: Path) -> list[Verdict]:
    """A verdict per arm the command connects; [] for a command that connects none."""
    prog = program(command)
    if prog is None or prog not in CONNECTS:
        return []
    return [check_arm(c, spec, root) for c in arms_in(command)]


def worst(verdicts: list[Verdict]) -> Verdict | None:
    order = {DANGER: 0, UNKNOWN: 1, OK: 2}
    return min(verdicts, key=lambda v: order[v.level]) if verdicts else None


def at_prompt(command: str | None, lerobot_id: str, spec: rigspec.RigSpec | None,
              root: Path) -> Verdict:  # fmt: skip
    """The verdict for LeRobot's "Press ENTER to use provided calibration file" prompt, which
    names the arm only by its id. ENTER writes that id's file into whichever arm the command put
    on that id's port."""
    if spec is None:
        return Verdict(UNKNOWN, f"Studio cannot read robot-config.yaml, so it cannot tell which "
                       f"arm {lerobot_id} is.")  # fmt: skip
    arms = [c for c in arms_in(command or "") if c.lerobot_id == lerobot_id]
    if not arms:
        return Verdict(UNKNOWN, f"Studio cannot tell which port {lerobot_id} is on in this "
                       "command.")  # fmt: skip
    return check_arm(arms[0], spec, root)
