"""Onboarding: say what this rig is, and write its robot-config.yaml from a few answers.

    cmd rig_status  (any window)  the config, the rig's name, each arm's port and calibration file,
                                  the cameras, calibration ids already on this Mac, and ports or ids
                                  found in configs/ports.local.sh
    cmd rig_write   (control)     write robot-config.yaml from onboarding's answers

WHY a whole-file write, when the rest of Studio edits single values (configedit.py): onboarding
creates the file. Run again, it writes the file from the answers once more, after a backup of the
old one to <data>/config-backups, keeping the camera settings it was not asked about.

The rig's display name lives in a top-level `studio:` block, which rigspec.parse and LeRobot ignore.
LeRobot ids stay what calibration files are named after, so an existing calibration keeps working.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from phi_studio import rigspec
from phi_studio.errors import Refusal
from phi_studio.files import lerobot_calibration_dir, root_order

if TYPE_CHECKING:
    from phi_studio.server import Client, Studio

CONFIG = "robot-config.yaml"
BACKUPS = "config-backups"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PORT_RE = re.compile(r"^(/dev/[\w.\-]+|COM\d{1,3})$")
CAMERA_ROLES = ("top", "front", "side", "wrist", "left_wrist", "right_wrist")
NAME_MAX = 60
# WHY 8: the worker's own step limit for arm joints (worker.py DEFAULT_MAX_STEP). Teleop at 30 Hz
# moves at most ~3 deg a frame (p99 on our recordings), so the clip only bites on a jump, which is
# its job.
STEP_LIMIT = 8.0
# Session states in which an arm may hold torque: no switching rigs then (rig_use).
HOLDING_STATES = ("ARMED", "MOVING", "STOPPED", "FAULT", "CALIBRATING")


def may_hold_torque(studio: Any) -> bool:
    """True when switching rigs would drop a powered arm. WHY telemetry too: an arm can hold torque
    in IDENTIFIED or READY, left on by an earlier session (worker.py _cmd_confirm)."""
    st = (studio.last.get("state") or {}).get("state", "DISCONNECTED")
    if st in HOLDING_STATES:
        return True
    arms = (studio.telemetry or {}).get("arms") or {}
    return any(isinstance(a, dict) and a.get("torque") for a in arms.values())
SLOTS = {
    "single": ("leader", "follower"),
    "bimanual": ("left_leader", "left_follower", "right_leader", "right_follower"),
}
PORT_VARS = {  # configs/ports.local.sh name -> onboarding slot
    "LEFT_LEADER_PORT": "left_leader",
    "LEFT_FOLLOWER_PORT": "left_follower",
    "RIGHT_LEADER_PORT": "right_leader",
    "RIGHT_FOLLOWER_PORT": "right_follower",
    "LEADER_PORT": "leader",
    "FOLLOWER_PORT": "follower",
}
ID_VARS = {
    "FOLLOWER_ID": ("single", "follower"),
    "LEADER_ID": ("single", "leader"),
    "BI_FOLLOWER_ID": ("bimanual", "follower"),
    "BI_LEADER_ID": ("bimanual", "leader"),
}


def slug(name: str) -> str:
    """Bench A -> bench_a: a LeRobot id from a display name."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s or "rig")[:40]


def config_target(studio: Studio) -> Path:
    """The config Studio reads (the copy the Files page and Checks read, files.root_order), or where
    onboarding puts a new one: the rig folder (--rig-dir) if Studio was given one, else Studio's own
    data folder. WHY not the code folder: installed, that is site-packages."""
    roots = studio.files.roots
    for key in root_order(CONFIG):
        r = roots.get(key)
        if r is not None and (r.path / CONFIG).is_file():
            return r.path / CONFIG
    rig_dir = getattr(studio, "rig_dir", None)
    if rig_dir:
        return Path(rig_dir) / CONFIG
    r = roots.get("studio") or roots.get("repo") or roots["code"]
    return r.path / CONFIG


def known_ids(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Calibration ids on this Mac, newest first. A bimanual id X counts when X_left and X_right
    exist."""
    out: dict[str, list[dict[str, Any]]] = {}
    for role, (kind, folder) in rigspec.FOLDER.items():
        files = {}
        d = root / kind / folder
        if d.is_dir():
            for p in d.glob("*.json"):
                if not p.name.startswith("._"):
                    files[p.stem] = p.stat().st_mtime
        out[role] = sorted(
            ({"id": k, "mtime": t} for k, t in files.items()), key=lambda r: -r["mtime"]
        )
        bi = []
        for k, t in files.items():
            if k.endswith("_left") and (base := k[: -len("_left")]) + "_right" in files:
                bi.append({"id": base, "mtime": max(t, files[base + "_right"])})
        out[f"bi_{role}"] = sorted(bi, key=lambda r: -r["mtime"])
    return out


def ports_hint(rig_dir: Path | None) -> dict[str, Any]:
    """Ports and ids from configs/ports.local.sh, if the rig folder has one. Plain `export
    NAME=value` lines only, with $NAME references to earlier lines resolved; anything else is
    ignored."""
    if rig_dir is None:
        return {"ports": {}, "ids": {}}
    path = Path(rig_dir) / "configs" / "ports.local.sh"
    try:
        text = path.read_text()
    except OSError:
        return {"ports": {}, "ids": {}}
    env: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if not m:
            continue
        try:
            words = shlex.split(m.group(2), comments=True)
        except ValueError:
            continue
        val = words[0] if words else ""
        env[m.group(1)] = re.sub(
            r"\$\{?([A-Z_][A-Z0-9_]*)\}?", lambda r: env.get(r.group(1), ""), val
        )
    ports = {slot: env[v] for v, slot in PORT_VARS.items() if PORT_RE.match(env.get(v, ""))}
    ids: dict[str, dict[str, str]] = {"single": {}, "bimanual": {}}
    for v, (layout, role) in ID_VARS.items():
        if ID_RE.match(env.get(v, "")):
            ids[layout][role] = env[v]
    return {"ports": ports, "ids": ids, "file": str(path)}


def read_config(path: Path) -> tuple[str | None, dict[str, Any]]:
    try:
        text = path.read_text()
    except OSError:
        return None, {}
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        data = {}
    return text, data if isinstance(data, dict) else {}


def status(studio: Studio) -> dict[str, Any]:
    path = config_target(studio)
    text, data = read_config(path)
    cal_root = lerobot_calibration_dir()
    rig_dir = getattr(studio, "rig_dir", None)
    out: dict[str, Any] = {
        "path": str(path),
        "exists": text is not None,
        "name": None,
        "layout": None,
        "arms": [],
        "cameras": [],
        "problems": [],
        "calibration_root": str(cal_root),
        "known": known_ids(cal_root),
        "hint": ports_hint(Path(rig_dir) if rig_dir else None),
    }
    if text is None:
        return out
    try:
        spec = rigspec.parse(text)
    except yaml.YAMLError as e:
        out["problems"] = [f"robot-config.yaml does not parse: {e}"]
        return out
    block = data.get("studio")
    studio_block: dict[str, Any] = block if isinstance(block, dict) else {}
    out["name"] = studio_block.get("name") if isinstance(studio_block.get("name"), str) else None
    out["layout"] = (
        "bimanual" if any(a.side for a in spec.arms) else ("single" if spec.arms else None)
    )
    for a in spec.arms:
        cal = a.calibration_path(cal_root)
        exists = bool(cal and cal.is_file())
        out["arms"].append(
            {
                "key": a.key,
                "role": a.role,
                "side": a.side,
                "port": a.port,
                "id": a.lerobot_id,
                "calibration": str(cal) if cal else None,
                "calibrated": exists,
                "calibrated_at": cal.stat().st_mtime if exists and cal else None,
            }
        )
    out["cameras"] = [
        {
            "key": c.key,
            "side": c.side,
            "feature": c.feature,
            "source": c.source,
            "width": c.fields.get("width"),
            "height": c.fields.get("height"),
            "fps": c.fields.get("fps"),
        }
        for c in spec.cameras
    ]
    out["problems"] = list(spec.problems)
    return out


# -- writing -------------------------------------------------------------------------------------
def _q(s: str) -> str:
    """A YAML double-quoted scalar (JSON strings are valid YAML)."""
    return json.dumps(s, ensure_ascii=False)


def check_answers(a: Any) -> dict[str, Any]:
    """Onboarding's answers, validated. Refuses anything LeRobot would choke on."""
    if not isinstance(a, dict):
        raise Refusal("The rig answers must be an object.")
    name = a.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > NAME_MAX:
        raise Refusal("Give the rig a name.", f"Up to {NAME_MAX} characters, such as Bench A.")
    layout = a.get("layout")
    if layout not in SLOTS:
        raise Refusal("Choose single arm or bimanual.")
    ids = a.get("ids") or {}
    follower, leader = ids.get("follower"), ids.get("leader")
    for what, v in (("follower", follower), ("leader", leader)):
        if not isinstance(v, str) or not ID_RE.match(v):
            raise Refusal(f"The {what} id must be letters, digits, _ . or - (up to 64).")
    if follower == leader:
        raise Refusal(
            "The follower and the leader need different ids.",
            "Their calibration files are named after them.",
        )
    ports = a.get("ports") or {}
    if not isinstance(ports, dict):
        raise Refusal("Ports must be an object.")
    clean: dict[str, str] = {}
    for slot in SLOTS[layout]:
        p = ports.get(slot)
        if p in (None, ""):
            continue
        if not isinstance(p, str) or not PORT_RE.match(p):
            raise Refusal(f"{p!r} is not a serial port name.")
        clean[slot] = p
    seen: dict[str, str] = {}
    for slot, p in clean.items():
        key = re.sub(r"^/dev/cu\.", "/dev/tty.", p)
        if key in seen:
            raise Refusal(
                f"{seen[key].replace('_', ' ')} and {slot.replace('_', ' ')} would share {p}.",
                "Each arm needs its own port.",
            )
        seen[key] = slot
    cams = a.get("cameras")
    clean_cams: dict[str, dict[str, Any]] = {}
    if cams is not None:
        if not isinstance(cams, dict):
            raise Refusal("Cameras must be an object.")
        used: dict[str, str] = {}
        for role, c in cams.items():
            if role not in CAMERA_ROLES:
                raise Refusal(f"{role!r} is not a camera role ({', '.join(CAMERA_ROLES)}).")
            if not isinstance(c, dict):
                raise Refusal(f"Camera {role} must be an object.")
            src = c.get("source")
            if isinstance(src, bool) or not (
                (isinstance(src, int) and 0 <= src < 64)
                or (isinstance(src, str) and re.fullmatch(r"/dev/video\d{1,2}", src))
            ):
                raise Refusal(f"{src!r} is not a camera number.")
            if str(src) in used:
                raise Refusal(f"{used[str(src)]} and {role} would both read camera {src}.")
            used[str(src)] = role
            dims = {}
            for k, lo, hi in (("width", 64, 4096), ("height", 48, 4096), ("fps", 1, 240)):
                v = c.get(k)
                if v is not None and (
                    isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi
                ):
                    raise Refusal(f"Camera {role}: {k} must be a whole number from {lo} to {hi}.")
                dims[k] = v
            clean_cams[role] = {"source": src, **dims}
    return {
        "name": name.strip(),
        "layout": layout,
        "ids": {"follower": follower, "leader": leader},
        "ports": clean,
        "cameras": clean_cams if cams is not None else None,
    }


def build_yaml(ans: dict[str, Any], keep_cameras: str | None = None) -> str:
    """robot-config.yaml for LeRobot 0.6.0 from checked answers. `keep_cameras`: a YAML block of
    camera settings to carry over when the answers do not set cameras."""
    bi = ans["layout"] == "bimanual"
    ports = ans["ports"]
    port = lambda slot: _q(ports[slot]) if slot in ports else '""  # find it in onboarding'  # noqa: E731
    lines = [
        "# This rig's settings for LeRobot 0.6.0, written by Phi Studio's onboarding.",
        "# Studio edits single values in place from now on and keeps your comments.",
        "studio:",
        f"  name: {_q(ans['name'])}",
        "robot:",
        f"  type: {'bi_so_follower' if bi else 'so101_follower'}",
        f"  id: {_q(ans['ids']['follower'])}",
    ]
    if bi:
        for side in ("left", "right"):
            lines += [
                f"  {side}_arm_config:",
                f"    port: {port(side + '_follower')}",
                f"    max_relative_target: {STEP_LIMIT}",
            ]
    else:
        lines += [f"  port: {port('follower')}", f"  max_relative_target: {STEP_LIMIT}"]
    cams = ans.get("cameras")
    if cams:
        lines.append("  cameras:")
        for role, c in cams.items():
            src = c["source"]
            lines += [
                f"    {role}:",
                "      type: opencv",
                f"      index_or_path: {src if isinstance(src, int) else _q(src)}",
            ]
            for k in ("width", "height", "fps"):
                if c.get(k) is not None:
                    lines.append(f"      {k}: {c[k]}")
    elif keep_cameras:
        lines.append(keep_cameras.rstrip())
    lines += [
        "teleop:",
        f"  type: {'bi_so_leader' if bi else 'so101_leader'}",
        f"  id: {_q(ans['ids']['leader'])}",
    ]
    if bi:
        for side in ("left", "right"):
            lines += [f"  {side}_arm_config:", f"    port: {port(side + '_leader')}"]
    else:
        lines.append(f"  port: {port('leader')}")
    return "\n".join(lines) + "\n"


def existing_cameras(data: dict[str, Any]) -> str | None:
    """The robot's top-level camera settings from an old config, as a YAML block to carry over."""
    got = data.get("robot")
    robot: dict[str, Any] = got if isinstance(got, dict) else {}
    cams = robot.get("cameras")
    if not isinstance(cams, dict) or not cams:
        return None
    body = yaml.safe_dump({"cameras": cams}, sort_keys=False, default_flow_style=False)
    return "\n".join("  " + ln for ln in body.rstrip().splitlines())


def write(path: Path, ans: dict[str, Any], backup_dir: Path) -> Path | None:
    """Write the config atomically; back up an existing one first. Returns the backup's path."""
    old_text, old_data = read_config(path)
    keep = existing_cameras(old_data) if ans.get("cameras") is None else None
    text = build_yaml(ans, keep)
    spec = rigspec.parse(text)  # never write a file Studio itself cannot read
    hard = [p for p in spec.problems if "no id" in p or "not a LeRobot" in p]
    if hard:
        raise Refusal("Studio would write a config it cannot use.", "; ".join(hard))
    backup = None
    if old_text is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"robot-config.{time.strftime('%Y%m%d-%H%M%S')}.yaml"
        backup.write_text(old_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return backup


class OnboardAPI:
    def __init__(self, studio: Studio) -> None:
        self.studio = studio

    def data_dir(self) -> Path:
        return self.studio.data_dir or Path.home() / ".cache" / "phi" / "studio"

    async def rig_status(self, client: Client, msg: dict[str, Any]) -> None:
        client.push({"type": "rig_status", **(await asyncio.to_thread(status, self.studio))})

    async def rig_write(self, client: Client, msg: dict[str, Any]) -> None:
        ans = check_answers(msg.get("rig"))
        path = config_target(self.studio)
        async with self.studio.config_lock:
            backup = await asyncio.to_thread(write, path, ans, self.data_dir() / BACKUPS)
        self.studio._note("state", f"Onboarding wrote {path.name} for {ans['name']}")
        client.push(
            {
                "type": "rig_saved",
                "path": str(path),
                "backup": str(backup) if backup else None,
                "ref": msg.get("ref"),
            }
        )
        st = await asyncio.to_thread(status, self.studio)
        self.studio._fanout({"type": "rig_status", **st})
        try:  # Set up, Checks and Files read the config through the files index
            self.studio._fanout({"type": "files", **self.studio.files.index()})
        except Exception:  # the write stands; those pages re-read on their own
            pass

    async def rig_use(self, client: Client, msg: dict[str, Any]) -> None:
        """Switch between the simulated arms and the real ones in robot-config.yaml without a
        restart. Refused while any arm may hold torque; a connected rig is disconnected first."""
        hardware = msg.get("hardware") is True
        st = (self.studio.last.get("state") or {}).get("state", "DISCONNECTED")
        if may_hold_torque(self.studio):
            raise Refusal("An arm may be holding torque.", "Release torque, then disconnect.")
        rec = (self.studio.telemetry or {}).get("recording") or {}
        if rec and not rec.get("finished", True):
            raise Refusal("A recording is still being written.", "Try again in a few seconds.")
        if hardware:
            path = config_target(self.studio)
            if not path.is_file():
                raise Refusal("There is no robot-config.yaml yet.", "Set up the rig first.")
            spec = rigspec.parse(path.read_text())
            missing = [a.key for a in spec.arms if not a.port]
            if not spec.arms or missing:
                raise Refusal(f"Every arm needs a port in {path.name}"
                              + (f": {', '.join(missing)} has none." if missing else "."),
                              "Edit the rig.")  # fmt: skip
            new: dict[str, Any] = {"kind": "lerobot", "config": str(path)}
        else:
            layout = (await asyncio.to_thread(status, self.studio)).get("layout")
            new = {"kind": "mock", "pairs": 2 if layout == "bimanual" else 1}
        if st != "DISCONNECTED":
            self.studio.to_worker({"cmd": "disconnect"})
            for _ in range(40):  # the worker answers within a tick or two
                await asyncio.sleep(0.05)
                if (self.studio.last.get("state") or {}).get("state") == "DISCONNECTED":
                    break
            else:
                raise Refusal("The rig did not disconnect.", "Disconnect it, then switch.")
        await asyncio.to_thread(self.studio.switch_rig, new)
        self.studio._note("state", "Using the real arms" if hardware else "Using simulated arms")
        st2 = await asyncio.to_thread(status, self.studio)
        self.studio._fanout({"type": "rig_status", **st2})


def register(studio: Studio) -> None:
    api = OnboardAPI(studio)
    studio.handle("rig_status", api.rig_status, control=False)
    studio.handle("rig_write", api.rig_write, control=True)
    studio.handle("rig_use", api.rig_use, control=True)
