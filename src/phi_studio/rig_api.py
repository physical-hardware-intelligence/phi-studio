"""Set up's Detect arms step, server side: find the real arms plugged into this Mac, tell which is
which, and write that into robot-config.yaml and the calibration folder.

    cmd rig_scan    (control)  ping every USB serial board, read each arm's registers, match them
                               to calibration files (detect.scan); pushes rig_scan
    cmd rig_motion  (control)  watch every healthy arm for MOTION_S while a person moves one by
                               hand; pushes rig_motion with ticks moved per port
    cmd rig_check   (control)  POWERED: hold each motor of one arm still with torque on, one at a
                               time, polling the whole bus (detect.check_arm); pushes rig_check
                               per motor, then with done
    cmd rig_stop    (control)  stop a process that holds an arm's port (SIGTERM, then SIGKILL),
                               only when a fresh lsof still shows it holding one; rescans
    cmd rig_cal_prepare  (control)  before lerobot-calibrate runs for one arm in the terminal: back
                               up its file, and turn a symlinked file into a copy, because LeRobot
                               writes through a link into the file it points at (robot.py:170)
    cmd rig_cal_verify   (control)  after it ends: read that arm's registers and compare them to its
                               file; pushes rig_cal_verified
    cmd rig_cal_from_motors (control)  write that arm's file from its registers, only when they hold
                               a finished calibration (identity.unfinished)
    cmd rig_motors  (control)  ping every id at every STS3215 rate on one port (detect.find_motors):
                               finds a motor given a wrong id or rate, or none; pushes rig_motors
    cmd rig_calfiles*          the calibration folder: list (any window), archive, restore,
                               install a shared set (control); calfiles.py
    cmd rig_apply   (control)  rescan, check the chosen roles and sides, write any missing
                               calibration file from the arm's own registers, replace robot: and
                               teleop: in robot-config.yaml (backup first, every other line kept),
                               then switch the worker to the real arms; pushes rig_applied

Scanning opens each port read-only, so it runs only when no worker holds the real arms (the mock
rig, or real arms disconnected) and no LeRobot command in the terminal may hold a port.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from phi_studio import calfiles, configedit, detect, rigspec
from phi_studio.errors import Refusal
from phi_studio.identity import (
    Calibration,
    load_calibration,
    match_fingerprint,
    save_calibration,
    unfinished,
)
from phi_studio.onboard_api import STEP_LIMIT, config_target, may_hold_torque
from phi_studio.setup_api import BACKUPS, data_dir, lerobot_busy

if TYPE_CHECKING:
    from phi_studio.server import Client, Studio

MOTION_S = 3.0
CHECK_S = 3.0  # per motor: tonight's faulty motor broke the bus within 1.7 s
DEFAULT_IDS = {"follower": "phi_follower", "leader": "phi_leader"}


class RigError(Refusal):
    """A request Studio refuses; the message is shown as is."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie is gone for our purposes: its ports are closed
        import subprocess

        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,  # noqa: S603, S607
                            text=True, timeout=5).stdout.strip()  # fmt: skip
        return bool(st) and not st.startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def stop_process(pid: int, wait_s: float = 2.0, kill: Any = os.kill, alive: Any = _alive,
                 sleep: Any = time.sleep) -> bool:
    """SIGINT (with SIGCONT, so a process paused with Ctrl-Z can act on it), then SIGTERM, then
    SIGKILL, each after `wait_s` if it is still there. True when it is gone. WHY SIGINT first:
    lerobot-teleoperate turns torque off only on KeyboardInterrupt (lerobot_teleoperate.py
    274-280); SIGTERM ends it with the follower still powered."""

    def gone_within(s: float) -> bool:
        for _ in range(max(1, int(s / 0.1))):
            if not alive(pid):
                return True
            sleep(0.1)
        return not alive(pid)

    for sigs, wait in (((signal.SIGINT, signal.SIGCONT), wait_s), ((signal.SIGTERM,), wait_s),
                       ((signal.SIGKILL,), 1.0)):  # fmt: skip
        try:
            for sig in sigs:
                kill(pid, sig)
        except ProcessLookupError:
            return True
        if gone_within(wait):
            return True
    return False


def _section(role: str, rid: str, arms: list[dict[str, Any]]) -> str:
    """The robot: or teleop: section for these arms, in LeRobot 0.6.0's own keys."""
    name = "robot" if role == "follower" else "teleop"
    bi = len(arms) == 2
    typ = ("bi_so_follower" if bi else "so101_follower") if role == "follower" else (
        "bi_so_leader" if bi else "so101_leader")
    out = [f"{name}:  # written by Studio's Detect arms, {time.strftime('%Y-%m-%d')}",
           f"  type: {typ}", f"  id: {rid}"]  # fmt: skip
    # WHY the step limit on followers only: a bi_so_leader takes no per-arm options (rigspec), and
    # the limit is what stops a follower jumping to a leader that disagrees with it.
    step = [f"max_relative_target: {STEP_LIMIT}"] if role == "follower" else []
    if bi:
        for a in sorted(arms, key=lambda a: a["side"]):
            out += [f"  {a['side']}_arm_config:", f"    port: {a['port']}"]
            out += [f"    {s}" for s in step]
    else:
        out.append(f"  port: {arms[0]['port']}")
        out += [f"  {s}" for s in step]
    return "\n".join(out) + "\n"


def check_assignment(arms: Any, found: dict[str, detect.Found]) -> list[dict[str, Any]]:
    """The arms to write, each {port, role, side}: one leader and one follower, or two of each
    with one left and one right. Raises RigError naming what is wrong."""
    if not isinstance(arms, list) or not arms:
        raise RigError("Choose a role for each arm first.")
    out = []
    for a in arms:
        if not isinstance(a, dict) or a.get("role") not in ("leader", "follower"):
            raise RigError("Each arm needs a role: leader or follower.")
        port = a.get("port")
        f = found.get(port) if isinstance(port, str) else None
        if f is None:
            raise RigError(f"{port} is not an arm Studio found. Scan again.")
        if f.problem:
            raise RigError(f"The arm on {port} cannot be used yet: {f.problem}")
        side = a.get("side")
        if side not in (None, *rigspec.SIDES):
            raise RigError(f"{side!r} is not a side; use left or right.")
        out.append({"port": port, "role": a["role"], "side": side})
    if len({a["port"] for a in out}) != len(out):
        raise RigError("The same port is chosen twice.")
    by = {r: [a for a in out if a["role"] == r] for r in ("leader", "follower")}
    n = len(by["leader"])
    if n != len(by["follower"]) or n not in (1, 2):
        raise RigError("Studio drives one leader with one follower, or two of each. These are "
                       f"{len(by['leader'])} leaders and {len(by['follower'])} "
                       "followers.")  # fmt: skip
    for role, group in by.items():
        sides = sorted(str(a["side"]) for a in group)
        if n == 2 and sides != ["left", "right"]:
            raise RigError(f"With two pairs, one {role} must be left and one right.")
        if n == 1:
            group[0]["side"] = None
    return out


def existing_id(role: str, arms: list[dict[str, Any]], found: dict[str, detect.Found],
                library: dict[str, Calibration],
                prefer: str | None = None) -> str | None:  # fmt: skip
    """An id whose files already hold these exact arms (<id>.json, or <id>_left/_right.json for a
    pair), so applying keeps the files LeRobot commands already use. WHY: an arm can match more
    than one file (a symlink, an old copy); only an id that fits every arm of the role is safe."""
    kind, folder = rigspec.FOLDER[role]
    prefix = f"{kind}/{folder}/"
    names = {k[len(prefix):] for k in library if k.startswith(prefix)}
    group = [a for a in arms if a["role"] == role]
    ok = []
    for name in sorted(names):
        base = name
        if group[0]["side"]:
            if not name.endswith(f"_{group[0]['side']}"):
                continue
            base = name[: -len(group[0]["side"]) - 1]
        if all(library.get(prefix + (f"{base}_{a['side']}" if a["side"] else base))
               == found[a["port"]].registers for a in group):  # fmt: skip
            ok.append(base)
    if prefer in ok:
        return prefer
    return ok[0] if ok else None


def distinct_ids(ids: dict[str, str]) -> dict[str, str]:
    """Never one id for both roles. WHY: LeRobot would accept it (the files live in different
    folders), but Studio's worker keys calibrations by id alone (worker._save_run,
    mock.calibration_files) and onboard_api.check_answers refuses it. The leader gets its own id,
    and _write_calibrations writes that id's files from the leader's own registers."""
    if ids["follower"] != ids["leader"]:
        return ids
    return {**ids, "leader": f"{ids['leader']}_leader"}


class RigApi:
    def __init__(self, studio: Studio) -> None:
        self.studio = studio
        self.lock = asyncio.Lock()  # one scan, motion watch or apply at a time
        self.found: dict[str, detect.Found] = {}

    @property
    def cal_root(self) -> Path:
        from phi_studio.files import lerobot_calibration_dir

        return Path(self.studio.spec.get("cal_root") or lerobot_calibration_dir())

    async def _may_open_ports(self) -> None:
        state = (self.studio.last.get("state") or {}).get("state")
        if self.studio.spec.get("kind") != "mock" and state not in (None, "DISCONNECTED"):
            raise RigError("The real arms are connected, so Studio's robot worker holds their "
                           "ports.", "Disconnect the rig first.")  # fmt: skip
        busy = await lerobot_busy(self.studio)
        if busy:
            raise RigError(f"The terminal is running {busy}, which may hold the arms' ports.",
                           "Stop it first.")  # fmt: skip

    def _config_view(self) -> dict[str, Any]:
        """Which port robot-config.yaml gives which arm now, so the page can start from it."""
        ids = dict(DEFAULT_IDS)
        try:
            spec = rigspec.parse(config_target(self.studio).read_text())
        except (Refusal, OSError, ValueError):
            return {"arms": [], "ids": ids}
        for role, sec in (("follower", spec.robot), ("leader", spec.teleop)):
            if isinstance(sec.get("id"), str) and sec["id"]:
                ids[role] = sec["id"]
        return {"arms": [{"key": a.key, "role": a.role, "side": a.side, "port": a.port}
                         for a in spec.arms], "ids": ids}  # fmt: skip

    def _ours(self) -> set[int]:
        """Studio's own pids: the server and its robot worker. WHY: the worker is a separate
        process, and it must never be reported as a holder or offered to Stop."""
        pid = getattr(getattr(self.studio, "proc", None), "pid", None)
        return {os.getpid()} | ({pid} if isinstance(pid, int) else set())

    def _holders(self, ports: list[str]) -> dict[str, list[detect.Holder]]:
        return detect.port_holders(ports, me=self._ours())

    async def _scan(self) -> list[detect.Found]:
        ports = await asyncio.to_thread(detect.serial_ports)
        found = await asyncio.to_thread(detect.scan, ports, self.cal_root, holders=self._holders)
        self.found = {f.port: f for f in found}
        return found

    async def scan(self, client: Client, msg: dict[str, Any]) -> None:
        async with self.lock:
            await self._may_open_ports()
            found = await self._scan()
        client.push({"type": "rig_scan", "arms": [f.public() for f in found],
                     "config": self._config_view(), "at": time.time()})  # fmt: skip

    async def stop_holder(self, client: Client, msg: dict[str, Any]) -> None:
        pid = msg.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise RigError("Which process? Scan again and use its Stop button.")
        async with self.lock:
            ports = [d for d, _ in await asyncio.to_thread(detect.serial_ports)]
            try:
                held = await asyncio.to_thread(self._holders, ports)
            except OSError as e:
                raise RigError(f"Could not check which processes hold the ports: {e}") from e
            if pid in self._ours() or not any(h.pid == pid for hs in held.values() for h in hs):
                # WHY re-check: the pid comes from the page and may be stale or reused; Studio
                # stops only a process that holds an arm's port right now.
                raise RigError(f"Process {pid} no longer holds an arm's port.", "Scan again.")
            gone = await asyncio.to_thread(stop_process, pid)
            if not gone:
                raise RigError(f"Process {pid} did not stop.", f"Run kill -9 {pid} in a terminal.")
            found = await self._scan()
        client.push({"type": "rig_scan", "arms": [f.public() for f in found],
                     "config": self._config_view(), "at": time.time()})  # fmt: skip

    async def check(self, client: Client, msg: dict[str, Any]) -> None:
        port = msg.get("port")
        async with self.lock:
            await self._may_open_ports()
            f = self.found.get(port) if isinstance(port, str) else None
            if f is None:
                raise RigError("Scan for arms first, then check one of them.")
            if f.problem:
                raise RigError(f"Fix this arm first: {f.problem}")
            try:  # WHY again: the scan may be minutes old, and a powered motor on a shared line
                held = await asyncio.to_thread(self._holders, [f.port])  # corrupts both
            except OSError as e:
                raise RigError(f"Could not check which processes hold {f.port}: {e}") from e
            if held.get(f.port):
                h = held[f.port][0]
                raise RigError(f"{h.name} (process {h.pid}) has {f.port} open now.",
                               "Stop it, then scan again.")  # fmt: skip
            loop = asyncio.get_running_loop()
            push = client.push

            def each(r: detect.MotorCheck) -> None:
                loop.call_soon_threadsafe(push, {"type": "rig_check", "port": port,
                                                 "motor": r.public(), "done": False})  # fmt: skip

            push({"type": "rig_check", "port": port, "motor": None, "done": False,
                  "seconds": CHECK_S})  # fmt: skip
            try:
                out = await asyncio.to_thread(detect.check_arm, f.port, CHECK_S, on_motor=each)
            except OSError as e:
                push({"type": "rig_check", "port": port, "done": True, "motors": []})
                raise RigError(f"Could not check {port}: {e}") from e
        bad = [r.id for r in out if not r.ok]
        self.studio._note("state", f"Motor check {port}: " + (
            f"motor {', '.join(map(str, bad))} failed" if bad else "all six fine"))
        client.push({"type": "rig_check", "port": port, "done": True,
                     "motors": [r.public() for r in out]})  # fmt: skip

    async def motion(self, client: Client, msg: dict[str, Any]) -> None:
        async with self.lock:
            await self._may_open_ports()
            healthy = [f for f in self.found.values() if f.problem is None]
            if not healthy:
                raise RigError("Scan for arms first.")
            client.push({"type": "rig_motion", "watching": True, "seconds": MOTION_S})
            try:
                moved = await asyncio.to_thread(detect.watch_motion, healthy, MOTION_S)
            finally:
                client.push({"type": "rig_motion", "watching": False})
        client.push({"type": "rig_motion", "watching": False, "moved": moved, "at": time.time()})

    # -- calibration in the terminal, one arm at a time -------------------------------------------
    def _others(self, arm: rigspec.ArmSpec) -> dict[str, str]:
        """The rig's other arms by their calibration file, as kind/folder/id (detect.scan's
        names), so a port holding another arm's registers is caught."""
        out = {}
        try:
            spec = rigspec.parse(config_target(self.studio).read_text())
        except (OSError, ValueError, Refusal, yaml.YAMLError):
            return {}
        for b in spec.arms:
            if b.key != arm.key and b.lerobot_id:
                kind, folder = rigspec.FOLDER[b.role]
                out[f"{kind}/{folder}/{b.lerobot_id}"] = b.key
        return out

    def _arm(self, msg: dict[str, Any]) -> tuple[rigspec.ArmSpec, Path]:
        """The arm `msg["arm"]` names in robot-config.yaml, and its calibration file."""
        try:
            spec = rigspec.parse(config_target(self.studio).read_text())
        except (OSError, ValueError) as e:
            raise RigError(f"Could not read robot-config.yaml: {e}", "Set up the rig first.") from e
        arm = next((a for a in spec.arms if a.key == msg.get("arm")), None)
        if arm is None:
            raise RigError(f"robot-config.yaml has no arm {msg.get('arm')!r}.")
        path = arm.calibration_path(self.cal_root)
        if path is None or not arm.port:
            name = arm.key.replace("_", " ")
            raise RigError(f"{name} has no id or no port in robot-config.yaml.",
                           "Run Detect arms on the Rig setup page.")  # fmt: skip
        return arm, path

    async def cal_prepare(self, client: Client, msg: dict[str, Any]) -> None:
        arm, path = self._arm(msg)
        unlinked, kept = await asyncio.to_thread(_own_copy, path, self.cal_root)
        client.push({"type": "rig_cal_prepared", "arm": arm.key, "path": str(path),
                     "unlinked": unlinked, "backup": str(kept) if kept else None})  # fmt: skip

    def _verdict(self, arm: rigspec.ArmSpec, path: Path, f: detect.Found) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "rig_cal_verified", "arm": arm.key, "path": str(path),
                               "file": path.is_file(), "exact": False, "max_deg": None,
                               "worst_joint": None, "problem": f.problem,
                               "unfinished": None, "at": time.time()}  # fmt: skip
        if f.registers is None:
            out["problem"] = out["problem"] or "Could not read its calibration registers."
            return out
        others = self._others(arm)
        kind, folder = rigspec.FOLDER[arm.role]
        own = f"{kind}/{folder}/{arm.lerobot_id}"
        # WHY not when it matches its own file too: one file can be a copy of another
        swap = None if own in f.matches else next((m for m in f.matches if m in others), None)
        if swap is not None and not out["problem"]:
            # WHY a problem, not a difference: saving would write that arm's calibration into
            # this arm's file, and the next Identify would call the swap correct.
            other = others[swap].replace("_", " ")
            out["problem"] = (f"These motors hold the {other}'s calibration ({swap.split('/')[-1]}"
                              f".json) exactly: are the {other}'s and this arm's cables swapped?")
        out["unfinished"] = unfinished(f.registers)
        if out["file"]:
            try:
                d = match_fingerprint(f.registers, {"file": load_calibration(path)})[0].distance
            except (OSError, ValueError, KeyError, TypeError) as e:
                out["problem"] = f"Its file cannot be read: {e}"
                return out
            out |= {"exact": d.exact, "max_deg": round(d.max_deg, 2), "worst_joint": d.worst_joint}
        return out

    async def _read_arm(self, arm: rigspec.ArmSpec) -> detect.Found:
        await self._may_open_ports()
        ports = dict(await asyncio.to_thread(detect.serial_ports))
        if arm.port not in ports:
            raise RigError(f"Nothing is plugged in at {arm.port}.",
                           "Plug the arm in, or run Detect arms to find its port.")  # fmt: skip
        found = await asyncio.to_thread(detect.scan, [(arm.port, ports[arm.port])], self.cal_root,
                                        holders=self._holders)  # fmt: skip
        if not found:
            raise RigError(f"No motor answers at {arm.port}.", "Check the arm's power.")
        return found[0]

    async def cal_verify(self, client: Client, msg: dict[str, Any]) -> None:
        arm, path = self._arm(msg)
        async with self.lock:
            f = await self._read_arm(arm)
        client.push(self._verdict(arm, path, f))

    async def cal_from_motors(self, client: Client, msg: dict[str, Any]) -> None:
        arm, path = self._arm(msg)
        async with self.lock:
            f = await self._read_arm(arm)
            v = self._verdict(arm, path, f)
            if v["problem"]:
                raise RigError(f"Did not save: {v['problem']}")
            assert f.registers is not None  # else _verdict names a problem
            why = unfinished(f.registers)
            if why:
                raise RigError(f"Did not save: {why}.", "Calibrate this arm again.")
            regs = f.registers
            await asyncio.to_thread(_save_own, regs, path, self.cal_root)
        client.push(self._verdict(arm, path, f))

    async def apply(self, client: Client, msg: dict[str, Any]) -> None:
        async with self.lock:
            state = (self.studio.last.get("state") or {}).get("state", "DISCONNECTED")
            if may_hold_torque(self.studio):
                raise RigError("An arm may be holding torque.", "Release torque, then disconnect.")
            await self._may_open_ports()
            await self._scan()  # the registers as they are now, not at the first scan
            arms = check_assignment(msg.get("arms"), self.found)
            for a in arms:  # before any write: a file from unfinished registers is a wrong zero
                regs = self.found[a["port"]].registers
                why = unfinished(regs) if regs is not None else None
                if why:
                    raise RigError(f"{detect_name(a)}: {why}.",
                                   "Calibrate this arm on the Calibrate page, then apply again.")
            library = await asyncio.to_thread(detect.calibration_library, self.cal_root)
            ids = self._config_view()["ids"]
            for role in ("follower", "leader"):
                v = msg.get("ids", {}).get(role) if isinstance(msg.get("ids"), dict) else None
                if isinstance(v, str) and v.strip():
                    ids[role] = v.strip()
                else:
                    ids[role] = existing_id(role, arms, self.found, library,
                                            prefer=ids[role]) or ids[role]  # fmt: skip
            for v in ids.values():
                if not v or not all(c.isalnum() or c in "_-" for c in v):
                    raise RigError(f"{v!r} is not a usable id: letters, digits, _ and - only.")
            ids = distinct_ids(ids)
            files = await asyncio.to_thread(self._write_calibrations, arms, ids)
            path = config_target(self.studio)
            sections = {
                "robot": _section("follower", ids["follower"],
                                  [a for a in arms if a["role"] == "follower"]),
                "teleop": _section("leader", ids["leader"],
                                   [a for a in arms if a["role"] == "leader"]),
            }  # fmt: skip
            async with self.studio.config_lock:
                backup = await asyncio.to_thread(self._write_config, path, sections)
            if state != "DISCONNECTED":
                self.studio.to_worker({"cmd": "disconnect"})
                await asyncio.sleep(0.3)
            await asyncio.to_thread(self.studio.switch_rig, {"kind": "lerobot",
                                                             "config": str(path)})  # fmt: skip
        n = len(arms) // 2
        what = "bimanual" if n == 2 else "single-arm"
        note = f"Detect arms wrote a {what} rig to {path.name}"
        self.studio._note("state", note + (f"; backup {backup}" if backup else ""))
        client.push({"type": "rig_applied", "backup": str(backup) if backup else None,
                     "files": files, "bimanual": n == 2, "ids": ids, "config": str(path),
                     "at": time.time()})  # fmt: skip
        try:
            self.studio._fanout({"type": "files", **self.studio.files.index()})
        except Exception:
            pass

    def _write_config(self, path: Path, sections: dict[str, str]) -> Path | None:
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(sections.values()))
            return None
        return configedit.replace_sections(path, sections, data_dir(self.studio) / BACKUPS)

    def _write_calibrations(self, arms: list[dict[str, Any]],
                            ids: dict[str, str]) -> list[dict[str, Any]]:  # fmt: skip
        """Each arm's file, from the registers it holds now. WHY its own registers and never
        another arm's file: homing offsets record where each servo's magnet sat when that arm was
        calibrated, so they differ arm to arm. Copying a file onto another arm drove it into its
        stops on 2026-10-05."""
        out = []
        for a in arms:
            f = self.found[a["port"]]
            assert f.registers is not None  # check_assignment refused arms with a problem
            kind, folder = rigspec.FOLDER[a["role"]]
            rid = ids[a["role"]] + (f"_{a['side']}" if a["side"] else "")
            target = self.cal_root / kind / folder / f"{rid}.json"
            if target.is_file():
                try:
                    same = load_calibration(target) == f.registers
                except (ValueError, KeyError, TypeError):
                    same = False
                if same:
                    out.append({"port": a["port"], "file": str(target), "action": "unchanged"})
                    continue
            kept = _save_own(f.registers, target, self.cal_root)
            out.append({"port": a["port"], "file": str(target),
                        "action": "replaced" if kept else "written", "from": f.match,
                        "backup": str(kept) if kept else None})  # fmt: skip
        return out


    async def find_motors(self, client: Client, msg: dict[str, Any]) -> None:
        """Every motor a port reaches, at any id and any rate. Read-only (pings), about 14 s."""
        port = msg.get("port")
        if not isinstance(port, str) or not port:
            raise RigError("Choose an arm's port.")
        async with self.lock:
            await self._may_open_ports()
            try:
                held = await asyncio.to_thread(self._holders, [port])
            except OSError as e:
                raise RigError(f"Could not check which processes hold {port}: {e}") from e
            if held.get(port):
                h = held[port][0]
                raise RigError(f"{h.name} (process {h.pid}) has {port} open.", "Stop it first.")
            loop, push = asyncio.get_running_loop(), client.push

            def rate(baud: int) -> None:
                loop.call_soon_threadsafe(push, {"type": "rig_motors", "port": port,
                                                 "baud": baud, "done": False})  # fmt: skip

            try:
                out = await asyncio.to_thread(detect.find_motors, port, on_rate=rate)
            except OSError as e:
                raise RigError(f"Could not open {port}: {e}", "Check its USB cable.") from e
        push({"type": "rig_motors", "port": port, "done": True, **out.public(),
              "setup": self._setup_commands(port)})  # fmt: skip

    def _setup_commands(self, port: str) -> list[dict[str, str]]:
        """lerobot-setup-motors for this port: as the arm robot-config.yaml says it is, or as
        either role when the port is not in it (or Detect matched no file)."""
        spec = self._spec()
        arm = next((a for a in spec.arms if a.port == port), None) if spec else None
        f = self.found.get(port)
        roles = [arm.role] if arm else [f.role] if f and f.role else ["follower", "leader"]
        out = []
        for role in roles:
            dev = "robot" if role == "follower" else "teleop"
            typ = arm.single_type if arm else ("so101_follower" if role == "follower"
                                               else "so101_leader")  # fmt: skip
            out.append({"role": role, "cmd": f"lerobot-setup-motors --{dev}.type={typ} "
                        f"--{dev}.port={port}"})  # fmt: skip
        return out

    # -- the calibration folder (calfiles.py) ---------------------------------------------------
    def _spec(self) -> rigspec.RigSpec | None:
        try:
            return rigspec.parse(config_target(self.studio).read_text())
        except (OSError, ValueError, yaml.YAMLError):
            return None

    def _source(self, msg: dict[str, Any]) -> Path | None:
        src = msg.get("source")
        if isinstance(src, str) and src.strip():
            return Path(src.strip()).expanduser()
        found = calfiles.shared_candidates(getattr(self.studio, "rig_dir", None))
        return found[0] if found else None

    def _calfiles_view(self, msg: dict[str, Any]) -> dict[str, Any]:
        spec, root, source = self._spec(), self.cal_root, self._source(msg)
        motors = {p: f.matches for p, f in self.found.items() if f.matches}
        files = calfiles.inventory(root, spec, motors)
        rows = calfiles.shared(source, root, spec) if source and source.is_dir() else []
        cands = calfiles.shared_candidates(getattr(self.studio, "rig_dir", None))
        return {"type": "rig_calfiles", "root": str(root), "files": [f.public() for f in files],
                "archives": calfiles.archives(root), "config": spec is not None,
                "shared": {"source": str(source) if source else None,
                           "exists": bool(source and source.is_dir()), "rows": rows,
                           "candidates": [str(c) for c in cands]},
                "scanned": bool(self.found)}  # fmt: skip

    async def calfiles(self, client: Client, msg: dict[str, Any]) -> None:
        client.push(await asyncio.to_thread(self._calfiles_view, msg))

    async def _may_change_files(self) -> None:
        busy = await lerobot_busy(self.studio)
        if busy:  # WHY: lerobot-calibrate writes its file when it ends; a move now would race it
            raise RigError(f"The terminal is running {busy}, which may read or write these files.",
                           "Let it finish first.")  # fmt: skip

    async def calfiles_archive(self, client: Client, msg: dict[str, Any]) -> None:
        rels = msg.get("files")
        if not isinstance(rels, list) or not rels or not all(isinstance(r, str) for r in rels):
            raise RigError("Choose the files to move aside.")
        await self._may_change_files()
        try:
            dest = await asyncio.to_thread(calfiles.archive, self.cal_root, rels, self._spec())
        except ValueError as e:
            raise RigError(str(e)) from e
        self.studio._note("state", f"Moved {len(rels)} calibration files to {dest}")
        client.push({**self._calfiles_view(msg), "done": {"action": "archived", "files": rels,
                                                          "archive": dest.name}})  # fmt: skip

    async def calfiles_restore(self, client: Client, msg: dict[str, Any]) -> None:
        name = msg.get("name")
        if not isinstance(name, str):
            raise RigError("Choose an archive to restore.")
        await self._may_change_files()
        try:
            out = await asyncio.to_thread(calfiles.restore, self.cal_root, name)
        except ValueError as e:
            raise RigError(str(e)) from e
        self.studio._note("state", f"Restored calibration archive {name}")
        client.push({**self._calfiles_view(msg), "done": {"action": "restored", **out}})

    async def calfiles_install(self, client: Client, msg: dict[str, Any]) -> None:
        source, rels = self._source(msg), msg.get("files")
        if source is None or not source.is_dir():
            raise RigError("There is no shared calibration folder.",
                           "Get the phi repo's configs/calibration (git pull), or give its path.")
        if not isinstance(rels, list) or not rels or not all(isinstance(r, str) for r in rels):
            raise RigError("Choose the files to install.")
        await self._may_change_files()
        try:
            out = await asyncio.to_thread(calfiles.install, source, self.cal_root, rels)
        except ValueError as e:
            raise RigError(str(e)) from e
        self.studio._note("state", f"Installed calibration files from {source}: {out['copied']}")
        client.push({**self._calfiles_view(msg), "done": {"action": "installed", **out}})


def detect_name(a: dict[str, Any]) -> str:
    """'Left follower' for an assigned arm."""
    return (f"{a['side']} {a['role']}" if a.get("side") else str(a["role"])).capitalize()


def _own_copy(path: Path, root: Path) -> tuple[bool, Path | None]:
    """Make `path` a file of its own and back it up. Returns (was a link, backup). WHY: LeRobot
    saves with open(path, "w") (robot.py:170), which writes through a symlink into the file it
    points at, so recalibrating phi_bi_left.json would rewrite phi_follower.json too."""
    unlinked = path.is_symlink()
    if unlinked:
        data = path.read_bytes() if path.exists() else None
        path.unlink()
        if data is not None:
            path.write_bytes(data)
    return unlinked, calfiles.backup(root, path)


def _save_own(cal: Calibration, path: Path, root: Path) -> Path | None:
    """Write `cal` to `path`, the old file copied to the archive first. Returns that copy."""
    kept = calfiles.backup(root, path)
    if path.is_symlink():
        path.unlink()  # replace the link itself, never the file it points at
    save_calibration(cal, path)
    return kept


def register(studio: Studio) -> None:
    api = RigApi(studio)
    studio.rig_api = api  # type: ignore[attr-defined]
    # WHY control for a read-only scan too: it opens the ports, and only one window may.
    studio.handle("rig_scan", api.scan, control=True)
    studio.handle("rig_motion", api.motion, control=True)
    studio.handle("rig_stop", api.stop_holder, control=True)
    studio.handle("rig_check", api.check, control=True)
    studio.handle("rig_apply", api.apply, control=True)
    studio.handle("rig_cal_prepare", api.cal_prepare, control=True)
    studio.handle("rig_cal_verify", api.cal_verify, control=True)
    studio.handle("rig_cal_from_motors", api.cal_from_motors, control=True)
    studio.handle("rig_motors", api.find_motors, control=True)
    studio.handle("rig_calfiles", api.calfiles, control=False)  # reads files only
    studio.handle("rig_calfiles_archive", api.calfiles_archive, control=True)
    studio.handle("rig_calfiles_restore", api.calfiles_restore, control=True)
    studio.handle("rig_calfiles_install", api.calfiles_install, control=True)
