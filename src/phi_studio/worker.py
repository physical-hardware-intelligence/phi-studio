"""The robot worker: the only code that touches the rig's buses.

`RigWorker` is pure control logic driven one `tick()` at a time, so tests run it
deterministically on the mock rig. `run_worker` wraps it in the threads of the real worker
process (ARCHITECTURE ADR-S2).

Rules it enforces, whatever the UI sends:
  * torque only after every arm's identity is confirmed (session state READY)
  * torque only on an arm whose registers match its own calibration file (a swap is refused)
  * enabling torque first writes goal = present position, so the arm holds where it is
  * goals only while MOVING, at most `max_step` ahead of the present position
  * Stop and heartbeat loss freeze followers at their present position; torque stays on (ADR-S5)
  * any bus error or servo fault bit freezes every reachable follower and faults the session, and
    so does any error the worker did not expect: a bug must stop the rig, not kill the worker
  * a fault clears only with torque off, since clearing re-checks every arm's identity
  * calibration runs only with every arm's torque off, never on an arm whose cable is swapped, and
    restores the old registers if it ends any way other than Save
  * recording only writes what teleop already does: it reads, it never moves an arm, and it ends
    with the session's motion (a stop, a fault, a lost heartbeat)
  * auto-calibration turns torque on its own follower only, writes goal = present first, presses
    into the stops only through a lowered torque limit, pauses on Stop or a lost heartbeat, and
    writes registers only with that arm's torque off (a new homing under torque would jump it)
  * a policy drives followers only while MOVING, through the same clip as teleop, and stops itself
    at its time limit
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phi_studio import autocal, recorder, rigspec
from phi_studio.identity import Calibration, JointCal, match_fingerprint, save_calibration
from phi_studio.policy import Policy, PolicyInfo, catalog, is_finite_number
from phi_studio.rig import BUS_ERRORS, JOINTS, ArmBus, JointHealth, label, labels
from phi_studio.session import IllegalTransition, Session, State

# WHY these step limits: LeRobot's max_relative_target defaults to None (config_so_follower.py:36),
# and a gripper motor burned from over-tightening (RECORDING_DAY.md). The clip is measured from the
# present position, so it bounds how far the goal leads the arm, not the arm's speed: the speed
# comes from the servo's position loop on that lead (unmeasured). Placeholder values until a
# hardware session with Parv sets them (GAP LEDGER).
DEFAULT_MAX_STEP = {j: 8.0 for j in JOINTS} | {"gripper": 5.0}
HEARTBEAT_TIMEOUT_S = 1.0  # Franka stops a hold-to-run link silent for over 1 s (research/02)
STOP_REASONS = {"user", "window closed", "control moved"}  # what a stop message may claim
# LeRobot records no range for wrist_roll and fixes it at 0..4095 (so_follower.py:135-143).
FULL_TURN = "wrist_roll"
FULL_RANGE = (0, 4095)
POLICY_LIMIT_S = (1.0, 600.0)
REC_LIMITS = {"episodes": (1, 500), "episode_s": (1.0, 600.0), "reset_s": (0.0, 300.0)}
FULL_TORQUE = 1000  # Torque_Limit's top: what a swept joint goes back to
FINISH_S = 15.0  # how long a closing worker waits for a recording to finalize (usually < 1 s)


@dataclass
class CalRun:
    """One arm's calibration in progress. By hand: middle -> ranges -> review. Auto: auto-middle ->
    auto (the sweep) -> auto-review, or auto-failed. Then save or cancel."""

    arm: ArmBus
    old: Calibration  # registers before calibration, written back on cancel
    step: str = "middle"
    written: bool = False  # the servo's registers have changed
    homings: dict[str, int] = field(default_factory=dict)
    pos: dict[str, int] = field(default_factory=dict)
    mins: dict[str, int] = field(default_factory=dict)
    maxes: dict[str, int] = field(default_factory=dict)
    new: Calibration | None = None
    joints: list[str] = field(default_factory=list)  # auto: the joints asked for
    auto: autocal.AutoCal | None = None
    limited: str | None = None  # auto: the joint whose torque limit is lowered now
    torque: int | None = None  # auto: Torque_Limit for a swept joint; None: autocal.sweep_limit
    waiting: bool = False  # auto: holding for another arm's shoulder-pan turn

    def view(self) -> dict[str, Any]:
        joints = {}
        found = self.auto.found if self.auto else {}
        for j, p in self.pos.items():
            lo, hi = (FULL_RANGE if j == FULL_TURN else found.get(j)
                      or (self.mins.get(j, p), self.maxes.get(j, p)))  # fmt: skip
            joints[j] = {"min": lo, "pos": p, "max": hi, "fixed": j == FULL_TURN}
        return {"arm": self.arm.name, "role": self.arm.role, "step": self.step, "joints": joints,
                "old": {j: c._asdict() for j, c in self.old.items()},
                "new": ({j: c._asdict() for j, c in self.new.items()}
                        if self.new else None),
                "auto": ({**self.auto.view(), "waiting": self.waiting}
                         if self.auto else None)}  # fmt: skip


@dataclass
class PolicyRun:
    info: PolicyInfo
    policy: Policy
    task: str
    limit_s: float
    t0: float
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # one per episode
    # Wall clock, so an eval can refuse runs that started before it began.
    started_at: float = field(default_factory=time.time)
    queue: deque[dict[str, dict[str, float]]] = field(default_factory=deque)
    step: int = 0
    episode_s: float = 0.0
    chunk_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    action: dict[str, dict[str, float]] = field(default_factory=dict)
    running: bool = True
    ended: str | None = None

    def view(self) -> dict[str, Any]:
        ms = sorted(self.chunk_ms)
        return {"id": self.info.id, "name": self.info.name, "task": self.task,
                "run_id": self.run_id, "started_at": self.started_at,
                "limit_s": self.limit_s, "episode_s": round(self.episode_s, 2), "step": self.step,
                "chunk": self.policy.chunk, "running": self.running, "ended": self.ended,
                "chunk_ms": round(self.chunk_ms[-1], 3) if ms else 0.0,
                "chunk_ms_p50": round(ms[len(ms) // 2], 3) if ms else 0.0,
                "chunk_ms_max": round(ms[-1], 3) if ms else 0.0,
                "action": {f: {j: round(v, 2) for j, v in g.items()}
                           for f, g in self.action.items()}}  # fmt: skip


def pair_arms(arms: list[ArmBus]) -> list[tuple[ArmBus, ArmBus]]:
    """(leader, follower) pairs by side, left with left, as LeRobot's bi_so_* classes pair them
    (bi_so_follower.py:128-138). Arms without a side fall back to the name: `x_leader` drives
    `x_follower`."""
    by_name = {a.name: a for a in arms}
    pairs = []
    for f in (a for a in arms if a.role == "follower"):
        side = getattr(f, "side", None)
        if side:
            lead = next(
                (a for a in arms if a.role == "leader" and getattr(a, "side", None) == side), None
            )
        else:
            lead = by_name.get(f.name.replace("follower", "leader"))
        if lead is not None and lead.role == "leader":
            pairs.append((lead, f))
    return pairs


def calibration_file(a: ArmBus) -> str:
    """The arm's file under a LeRobot calibration root (robot.py:49-53, teleoperator.py:47-53)."""
    kind, folder = rigspec.FOLDER[a.role]
    return f"{kind}/{folder}/{a.calibration_id}.json"


def _calibrated(a: ArmBus) -> bool:
    """False for a real arm with no calibration file: it has no degrees to read (hardware.py
    read_positions), so the loop skips it instead of faulting the whole rig. Mock arms always
    have one."""
    return getattr(a, "calibration", True) is not None


def _identity_problem(a: dict[str, Any]) -> tuple[str, str]:
    """(message, fix) for an arm whose registers do not match its own calibration file."""
    name, match = label(a["name"]), a["match"]
    if a.get("error"):
        return (f"{name} is not answering: {a['error']}",
                "Check its power and USB cable, then Connect again.")  # fmt: skip
    if match is None:
        return (f"{name}: no calibration files to compare with",
                "Calibrate the arm, or point Studio at the calibration directory.")  # fmt: skip
    if a["exact"]:
        return (f"{name} holds the calibration in {match}.json: are the cables swapped?",
                f"Swap the USB cables of {name} and the arm {match}.json belongs to, then read "
                "again.")  # fmt: skip
    if not a.get("calibrated", True):
        return (f"{name} has no calibration file ({a['file']})",
                "Calibrate this arm. Until then Studio does not read its position.")  # fmt: skip
    return (f"{name} does not match its calibration file exactly",
            f"Nearest is {match}.json, {a['max_deg']:.1f} deg off on {label(a['worst_joint'])}. "
            "Calibrate this arm, or connect the arm that file belongs to.")  # fmt: skip


class RigWorker:
    def __init__(
        self,
        rig: Any,
        send: Callable[[dict[str, Any]], None],
        clock: Callable[[], float] = time.monotonic,
        loop_hz: float = 30.0,
        calibrations: dict[str, Calibration] | None = None,
        health_every: int = 10,
        cal_dir: Path | None = None,
        policies: list[PolicyInfo] | None = None,
    ) -> None:
        self.rig, self.send, self.clock, self.loop_hz = rig, send, clock, loop_hz
        self.arms: list[ArmBus] = list(rig.arms)
        self.followers = [a for a in self.arms if a.role == "follower"]
        self.pairs = pair_arms(self.arms)
        self.calibrations = calibrations if calibrations is not None else rig.calibration_files()
        self._reload_files = calibrations is None  # tests hand in fixed calibrations
        self.health_every = health_every
        self.max_step = dict(DEFAULT_MAX_STEP)
        self.cal_dir = cal_dir
        self.mock = all(hasattr(a, "inject") for a in self.arms)
        found = policies if policies is not None else catalog(self.mock)
        self.policies = {p.id: p for p in found}
        self.cal: CalRun | None = None
        self.autos: list[CalRun] = []  # an auto-calibration: every arm it sweeps, at once
        self.pan_turn: str | None = None  # the one arm sweeping its shoulder pan now
        # Calibrations that ended under torque (a fault, a disconnect): their old registers go back
        # once that arm's torque is off (_release_all), never under torque.
        self.restores: list[CalRun] = []
        self.run: PolicyRun | None = None
        self.rec: recorder.Recording | None = None  # the recording, kept after it ends for its view
        self.session = Session(on_change=self._on_state)
        self.last_heartbeat = clock()
        self.dead: set[str] = set()  # arms whose bus stopped answering
        self.torque: dict[str, bool] = {a.name: False for a in self.arms}
        self.health: dict[str, dict[str, JointHealth]] = {}
        self.identity: list[dict[str, Any]] = []
        self._ticks: deque[float] = deque(maxlen=int(loop_hz * 2))
        self._cost_ms: deque[float] = deque(maxlen=int(loop_hz * 10))
        self._n = 0

    # -- commands from the UI ---------------------------------------------------------------------
    def handle(self, msg: dict[str, Any]) -> None:
        cmd = msg.get("cmd", "")
        fn = getattr(self, f"_cmd_{cmd}", None)
        if fn is None:
            self.error(f"unknown command {cmd!r}")
            return
        try:
            fn(msg)
        except IllegalTransition as e:
            self.error(str(e))
        except BUS_ERRORS as e:
            self._fault(f"A bus stopped answering during {cmd}: {e}")
        except Exception as e:  # a bug: stop the rig and say so, do not kill the worker
            if self.session.state is State.DISCONNECTED:
                self.error(f"Studio hit an internal error running {cmd}: {e!r}")
            else:
                self._fault(f"Studio hit an internal error running {cmd}: {e!r}")

    def error(self, message: str, fix: str = "") -> None:
        self.send({"type": "error", "message": message, "fix": fix})

    def _on_state(self, s: Session) -> None:
        if self.run is not None and self.run.running and s.state is not State.MOVING:
            self.run.running = False
            self.run.ended = s.stop_reason or ("fault" if s.state is State.FAULT else "stopped")
        if self.rec is not None and self.rec.phase != "done" and s.state is not State.MOVING:
            self._rec_end()  # recording follows teleop: whatever stopped the arms stops it too
        self.send({"type": "state", **s.snapshot()})

    def describe(self) -> dict[str, Any]:
        """What this rig is, for the UI: sent once at start and replayed to every new window."""
        return {"type": "rig", "mock": self.mock,
                "arms": [{"name": a.name, "role": a.role, "side": getattr(a, "side", None),
                          "id": a.calibration_id, "file": calibration_file(a)} for a in self.arms],
                "bimanual": any(getattr(a, "side", None) for a in self.arms),
                "policies": [p.public() for p in self.policies.values()],
                "cal_dir": str(self.cal_dir) if self.cal_dir else None}  # fmt: skip

    def _cmd_heartbeat(self, msg: dict[str, Any]) -> None:
        self.last_heartbeat = self.clock()

    def _cmd_connect(self, msg: dict[str, Any]) -> None:
        self.session.connected()
        self.dead.clear()
        self._identify()

    def _cmd_identify(self, msg: dict[str, Any]) -> None:
        if self.cal is not None or self.autos:
            self.error("Finish or cancel the calibration first.")
            return
        self._identify()

    def _cmd_confirm(self, msg: dict[str, Any]) -> None:
        bad = [a for a in self.identity if not a["ok"]]
        if bad and self.session.state is State.IDENTIFIED:
            self.error(*_identity_problem(bad[0]))
            return
        self.session.confirmed()
        if any(self.torque[f.name] for f in self.followers):
            # Torque left on by an earlier session: the arm is holding, so say so.
            self._freeze()
            self.session.armed()

    def _cmd_arm(self, msg: dict[str, Any]) -> None:
        self.session.armed()  # raises unless READY, before any torque write
        for f in self.followers:
            # WHY: a servo may drive to the goal it last stored once torque is on, and the arm may
            # have been moved by hand since (bench test 21). LeRobot does not do this:
            # enable_torque writes only Torque_Enable and Lock (feetech.py:302-305).
            f.write_goals(f.read_positions())
            self._set_torque(f, True)
        self.last_heartbeat = self.clock()

    def _cmd_start(self, msg: dict[str, Any]) -> None:
        activity = msg.get("activity", "teleop")
        if activity == "policy":
            self._start_policy(msg)
            return
        if activity != "teleop":
            self.error(f"{activity} is not available yet")
            return
        if not self.pairs:
            self.error("teleop needs a leader and a follower", "connect a leader arm")
            return
        self.session.started(activity)

    def _start_policy(self, msg: dict[str, Any]) -> None:
        pid = msg.get("policy")
        info = self.policies.get(pid) if isinstance(pid, str) else None
        if info is None or info.factory is None or not info.available:
            why = info.note if info else f"no policy named {msg.get('policy')!r}"
            self.error(f"{info.name if info else 'That policy'} cannot run here", why)
            return
        limit, task = msg.get("limit_s", 30.0), msg.get("task", "")
        lo, hi = POLICY_LIMIT_S
        if not is_finite_number(limit) or not lo <= limit <= hi:
            self.error(f"The time limit must be between {lo:.0f} and {hi:.0f} seconds.")
            return
        if not isinstance(task, str) or len(task) > 300:
            self.error("The task must be text of at most 300 characters.")
            return
        if not self.followers:
            self.error("A policy needs at least one follower arm.")
            return
        policy = info.factory(self.loop_hz)
        self.session.started("policy")  # raises unless ARMED: after a stop, resume first
        start = {f.name: f.read_positions() for f in self.followers}
        policy.reset(start, task)
        self.run = PolicyRun(info, policy, task, float(limit), t0=self.clock())

    def _policy_goals(
        self, now: float, pos: dict[str, dict[str, float]]
    ) -> list[tuple[ArmBus, dict[str, float]]]:
        """This tick's clipped goal per follower, or none once the time limit is reached."""
        r = self.run
        assert r is not None
        r.episode_s = now - r.t0
        if r.episode_s >= r.limit_s:
            self.session.stopped("time limit")
            self._freeze()
            return []
        if not r.queue:
            # WHY inline: the mock policy takes microseconds. A real policy must not run here.
            t = time.perf_counter()
            state = {f.name: pos[f.name] for f in self.followers}
            r.queue.extend(r.policy.infer(r.episode_s, state))
            r.chunk_ms.append((time.perf_counter() - t) * 1e3)
        r.action = r.queue.popleft()
        r.step += 1
        return [(f, self._clip(r.action[f.name], pos[f.name], f)) for f in self.followers]

    # -- recording (recorder.py): episodes of teleop into a LeRobot dataset ------------------------
    def _cmd_rec_start(self, msg: dict[str, Any]) -> None:
        if self.session.state is not State.MOVING or self.session.activity != "teleop":
            self.error("Recording needs teleop running.", "Enable torque and start teleop first.")
            return
        if self.rec is not None and self.rec.phase != "done":
            self.error("A recording is running.")
            return
        nums: dict[str, float] = {}
        for k, (lo, hi) in REC_LIMITS.items():
            v = msg.get(k)
            if not isinstance(v, int | float) or not is_finite_number(v) or not lo <= v <= hi:
                self.error(f"{k.replace('_', ' ')} must be between {lo:g} and {hi:g}.")
                return
            nums[k] = float(v)
        task = msg.get("task", "")
        if not isinstance(task, str) or not task.strip() or len(task) > 300:
            self.error("Say the task in a few words (at most 300 characters).",
                       "A policy trained on this data is given it as its instruction.")  # fmt: skip
            return
        try:
            repo = recorder.check_repo_id(msg.get("repo_id"))
            home = recorder.dataset_home()
        except (ValueError, ImportError) as e:
            self.error(str(e))
            return
        resume = msg.get("resume") is True
        if not resume:
            repo = recorder.stamped(repo)
        root = home / repo
        if resume and not (root / "meta" / "info.json").is_file():
            self.error(f"There is no dataset {repo} to add to.")
            return
        raw = [a.name for p in self.pairs for a in p if getattr(a, "use_degrees", True) is False]
        if raw:
            # WHY refuse: Studio reads degrees (hardware.motor_table); lerobot-record with this
            # config records -100..100, and one rig's datasets would disagree on units.
            self.error(f"robot-config.yaml sets use_degrees: false for {labels(raw)}, and Studio "
                       "records in degrees.",
                       "Set use_degrees: true (LeRobot's default), or record with lerobot-record "
                       "in the terminal.")  # fmt: skip
            return
        full = recorder.low_disk(home)
        if full:
            self.error(full, "Free some space, or move older datasets off this Mac.")
            return
        shapes: dict[str, tuple[int, int, int]] = {}
        for cam in self.rig.cameras:
            try:
                img = cam.read_latest()[0]
            except ConnectionError as e:
                self.error(f"Camera {cam.key} has no picture: {e}", "Plug it in, or remove it.")
                return
            h, w, c = (int(x) for x in img.shape)
            shapes[str(cam.key)] = (h, w, c)
        names, state_cols, action_cols = recorder.columns(self.pairs)
        spec = recorder.RecSpec(repo, root, int(round(self.loop_hz)), task.strip(),
                                int(nums["episodes"]), nums["episode_s"], nums["reset_s"],
                                resume=resume)  # fmt: skip
        robot_type = "bi_so_follower" if len(self.pairs) > 1 else "so_follower"
        feats = recorder.features(names, shapes)
        if resume:
            why = recorder.resume_problem(root, feats, spec.fps)
            if why:
                self.error(f"Cannot add to {repo}. {why}", "Record a new dataset instead.")
                return
        ring_shapes: dict[str, tuple[int, ...]] = dict(shapes)
        writer = recorder.WriterProcess(spec, feats, robot_type, self.send, ring_shapes)
        writer.start()
        self.rec = recorder.Recording(spec, writer, names, state_cols, action_cols,
                                      list(self.rig.cameras), t0=self.clock())  # fmt: skip

    def _rec_now(self, step: str) -> recorder.Recording | None:
        r = self.rec
        if r is None or r.phase == "done":
            self.error("No recording is running.")
            return None
        return r

    def _cmd_rec_next(self, msg: dict[str, Any]) -> None:
        """Get on with it: start now (warm-up), keep this take (record), end the reset."""
        r = self._rec_now("next")
        if r is None:
            return
        now = self.clock()
        if r.phase == "warmup":
            r.t0 = min(r.t0, now - r.spec.warmup_s)  # as soon as the writer is ready
        elif r.phase == "paused":
            self._rec_take(now)
        elif r.phase == "record":
            if r.take is None or len(r.take.state) < 2:
                self._rec_take(now)  # nothing to keep yet: start it again
            else:
                self._rec_keep(now)
        elif r.phase == "reset":
            self._rec_after_reset(now)

    def _cmd_rec_redo(self, msg: dict[str, Any]) -> None:
        """Throw the take away and record it again: the one running, or the one in the reset."""
        r = self._rec_now("redo")
        if r is None:
            return
        if r.phase == "paused":
            self._rec_take(self.clock())
            return
        if r.phase == "record" or (r.phase == "reset" and r.waiting):
            r.writer.discard()
            if r.take is not None:
                r.take.discarded = True
            r.takes.pop(r.episode, None)
            r.waiting = False
            self._rec_take(self.clock())

    def _cmd_rec_stop(self, msg: dict[str, Any]) -> None:
        if self._rec_now("stop") is not None:
            self._rec_end()

    def _rec_take(self, now: float) -> None:
        r = self.rec
        assert r is not None
        r.phase, r.t0, r.why = "record", now, None
        r.take = recorder.Take(r.episode, now, dropped0=r.writer.dropped)

    def _rec_pause(self, why: str) -> None:
        """A camera went dark: drop this take, keep teleop, wait for Resume (or Stop)."""
        r = self.rec
        assert r is not None
        if r.phase == "record":
            r.writer.discard()
            if r.take is not None:
                r.take.discarded = True
        r.phase, r.t0, r.why = "paused", self.clock(), why
        self.send({"type": "rec_paused", "why": why})

    def _rec_keep(self, now: float) -> None:
        """The take is over and kept: its health now, saved when the reset ends (or now, if it was
        the last)."""
        r = self.rec
        assert r is not None and r.take is not None
        take, dropped = r.take, r.writer.dropped - r.take.dropped0

        def check() -> None:  # WHY a thread: an analysis of a long take must not stall the loop
            h = take.health(r.spec.fps, r.names, dropped)
            if not take.discarded:
                r.takes[take.index] = h
                self.send({"type": "rec_take", **h})

        threading.Thread(target=check, name="phi-take-health", daemon=True).start()
        if r.episode + 1 >= r.spec.episodes:
            r.writer.save()
            r.writer.finish()
            r.phase, r.t0, r.take = "done", now, None
            return
        r.phase, r.t0, r.waiting = "reset", now, True

    def _rec_after_reset(self, now: float) -> None:
        r = self.rec
        assert r is not None
        if r.waiting:
            r.writer.save()
            r.waiting = False
        r.episode += 1
        self._rec_take(now)

    def _rec_end(self) -> None:
        """Stop: a take in progress is dropped, one waiting in the reset is kept, then finalize."""
        r = self.rec
        if r is None or r.phase == "done":
            return
        if r.phase == "record":
            r.writer.discard()
        elif r.phase == "reset" and r.waiting:
            r.writer.save()
        r.writer.finish()
        r.phase, r.t0, r.take, r.waiting = "done", self.clock(), None, False

    def _rec_tick(self, now: float, pos: dict[str, dict[str, float]]) -> None:
        r = self.rec
        assert r is not None
        if r.writer.error:
            # WHY end it: a dead writer drops every frame, and takes would look recorded.
            # Teleop goes on; the error is in the recording's view (Recording.view).
            r.phase, r.t0, r.take, r.waiting = "done", now, None, False
            r.why = r.writer.error
            return
        if r.phase == "warmup" and now - r.t0 >= r.spec.warmup_s and r.writer.ready.is_set():
            self._rec_take(now)  # WHY wait for ready: the writer process takes a moment to start
        elif r.phase == "reset" and now - r.t0 >= r.spec.reset_s:
            self._rec_after_reset(now)
        if r.phase != "record" or r.take is None:
            return
        state = np.array([pos[a][j] for a, j in r.state_cols], np.float32)
        action = np.array([pos[a][j] for a, j in r.action_cols], np.float32)
        frame: dict[str, Any] = {"observation.state": state, "action": action, "task": r.spec.task}
        for cam in r.cameras:
            try:
                img, t, _ = cam.read_latest()
                r.seen[cam.key] = (img, t)
            except ConnectionError:  # reopening (RealCamera retries): its last picture, if fresh
                img, t = r.seen.get(cam.key, (None, -1e9))
            if img is None or now - t > recorder.CAMERA_GRACE_S:
                self._rec_pause(f"Camera {cam.key} has had no picture for "
                                f"{recorder.CAMERA_GRACE_S:g} s.")  # fmt: skip
                return
            if now - t > recorder.LATE_S:
                r.take.late += 1
            frame[f"observation.images.{cam.key}"] = img
        r.writer.frame(frame)
        r.take.state.append(state)
        r.take.action.append(action)
        r.take.ticks.append(now)
        if len(r.take.state) >= r.target:
            self._rec_keep(now)

    # -- calibration: LeRobot so_follower.py:126-157, one step per command ------------------------
    def _cmd_cal_start(self, msg: dict[str, Any]) -> None:
        a = next((x for x in self.arms if x.name == msg.get("arm")), None)
        if a is None:
            self.error(f"There is no arm named {msg.get('arm')!r}.")
            return
        if self.autos:
            self.error("An auto-calibration is running.", "Finish or cancel it first.")
            return
        holding = [n for n, on in self.torque.items() if on]
        if holding:
            self.error(f"Turn torque off before calibrating: {labels(holding)} "
                       f"{'holds' if len(holding) == 1 else 'hold'} torque.",
                       "Support the arm, then use Torque off.")  # fmt: skip
            return
        mine = next((x for x in self.identity if x["name"] == a.name), None)
        if mine and mine["exact"] and mine["match"] != a.calibration_id:
            self.error(f"{label(a.name)} holds the calibration in {mine['match']}.json: are the "
                       "cables swapped?",
                       f"Fix the cables first. Calibrating now would write {label(a.name)}'s "
                       "calibration into the other arm's servos and hide the swap.")  # fmt: skip
            return
        self.session.calibration_started()  # raises unless IDENTIFIED or READY
        a.set_torque(False)  # LeRobot disables torque first (so_follower.py:127)
        self.cal = CalRun(a, old=a.read_calibration())

    # -- auto-calibration (autocal.py): every arm asked for at once, leaders too -------------------
    def _cmd_autocal_start(self, msg: dict[str, Any]) -> None:
        """Step 1: the same checks as calibrating by hand, for each arm; then wait for the middle
        pose. arms: names (default every arm); torque stays off until autocal_go."""
        names = msg.get("arms") or [a.name for a in self.arms]
        arms = [a for a in self.arms if a.name in names] if isinstance(names, list) else []
        if not arms or len(arms) != len(set(names)):
            self.error(f"Arms must be some of {', '.join(a.name for a in self.arms)}.")
            return
        if self.cal is not None or self.autos:
            self.error("A calibration is running.", "Finish or cancel it first.")
            return
        holding = [n for n, on in self.torque.items() if on]
        if holding:
            self.error(f"Turn torque off before calibrating: {labels(holding)} "
                       f"{'holds' if len(holding) == 1 else 'hold'} torque.",
                       "Support the arm, then use Torque off.")  # fmt: skip
            return
        for a in arms:
            mine = next((x for x in self.identity if x["name"] == a.name), None)
            if mine and mine["exact"] and mine["match"] != a.calibration_id:
                self.error(f"{label(a.name)} holds the calibration in {mine['match']}.json: are "
                           "the cables swapped?", "Fix the cables first.")  # fmt: skip
                return
        self.session.calibration_started()  # raises unless IDENTIFIED or READY
        runs = []
        for a in arms:
            a.set_torque(False)
            runs.append(CalRun(a, old=a.read_calibration(), step="auto-middle"))
        self.autos = runs

    def _cmd_autocal_go(self, msg: dict[str, Any]) -> None:
        """Every arm is in the middle pose: home them, hold them, sweep. joints: which ones
        (default all; wrist roll is homed, never swept). torque: Torque_Limit for swept joints."""
        if not self.autos or any(c.step != "auto-middle" for c in self.autos):
            self.error("Auto-calibration is not waiting for the middle pose.")
            return
        if self.clock() - self.last_heartbeat > HEARTBEAT_TIMEOUT_S:
            self.error("This window is not sending heartbeats, so Studio will not move the arms.",
                       "Reload the window, then start again.")  # fmt: skip
            return
        joints = msg.get("joints") or list(JOINTS)
        if not isinstance(joints, list) or any(j not in JOINTS for j in joints):
            self.error(f"Joints must be some of {', '.join(JOINTS)}.")
            return
        keep = msg.get("arms")  # the person may leave some arms out at the middle pose
        if keep is not None:
            if not isinstance(keep, list) or not keep or any(
                k not in {c.arm.name for c in self.autos} for k in keep
            ):
                self.error("Pick at least one of the arms being calibrated.")
                return
            self.autos = [c for c in self.autos if c.arm.name in keep]  # nothing written yet
        torque = msg.get("torque")
        if torque is not None and (not isinstance(torque, int) or isinstance(torque, bool)
                                   or not 100 <= torque <= 800):  # fmt: skip
            self.error("The sweep torque must be between 100 and 800 (of 1000).")
            return
        order = [j for j in JOINTS if j in joints]
        homed = list(JOINTS) if len(order) == len(JOINTS) else order  # roll's homing: middle pose
        now = self.clock()
        for c in self.autos:
            c.joints, c.torque = order, torque
            c.written = True  # before the write: one that fails halfway still changed registers
            c.homings = c.arm.set_half_turn_homings(homed)
            raw = c.arm.read_raw_positions()
            c.pos = raw
            c.arm.write_raw_goals(raw)  # goal = present before torque, as for teleop (_cmd_arm)
            self._set_torque(c.arm, True)
            c.auto = autocal.AutoCal([j for j in order if j != FULL_TURN], raw, now,
                                     expected_deg=autocal.expected_travel())  # fmt: skip
            c.step = "auto"
        self.last_heartbeat = now

    def _cmd_autocal_resume(self, msg: dict[str, Any]) -> None:
        paused = [c for c in self.autos if c.auto is not None and c.auto.state == "paused"]
        if not paused:
            self.error("Auto-calibration is not paused.")
            return
        now = self.last_heartbeat = self.clock()
        for c in paused:
            assert c.auto is not None
            c.auto.resume(c.arm.read_raw_positions(), now)

    def _cmd_autocal_save(self, msg: dict[str, Any]) -> None:
        """Write what every arm found, torque off first. An arm that failed gets its old
        registers back instead."""
        if not self.autos or any(c.step not in ("auto-review", "auto-failed") for c in self.autos):
            self.error("Auto-calibration has not finished.")
            return
        runs, self.autos, self.pan_turn = self.autos, [], None
        for c in runs:
            if self.torque[c.arm.name]:
                # WHY torque off first: a new homing written under torque moves the goal the servo
                # holds by the homing's change, and the arm jumps. The page asks to support it.
                self._set_torque(c.arm, False)
            if c.step == "auto-failed" or c.new is None:
                self._restore(c)
                continue
            self._save_run(c)
        self.session.calibration_ended()
        self._identify()

    def _cmd_autocal_cancel(self, msg: dict[str, Any]) -> None:
        if not self.autos:
            self.error("No auto-calibration is running.")
            return
        runs, self.autos, self.pan_turn = self.autos, [], None
        for c in runs:
            if self.torque[c.arm.name]:
                self._set_torque(c.arm, False)  # the person supports the arm (the page asks)
            if c.written:
                self._restore(c)
        self.session.calibration_ended()
        self._identify()

    def _autocal_abort(self) -> None:
        """A fault or a disconnect: arms holding torque keep holding, and get their old registers
        back once their torque goes off; the rest get them back now."""
        runs, self.autos, self.pan_turn = self.autos, [], None
        for c in runs:
            if not c.written:
                continue
            if self.torque.get(c.arm.name):
                self.restores.append(c)
            else:
                self._restore(c)

    def _autocal_tick(self, now: float) -> None:
        """One sweep step for every arm. Shoulder pans take turns, so two followers side by side
        never swing into each other."""
        stale = now - self.last_heartbeat > HEARTBEAT_TIMEOUT_S
        for c in self.autos:
            e = c.auto
            if c.step != "auto" or e is None:
                continue
            if stale and e.state == "running":
                e.pause("This window stopped answering, so the sweep paused.")
            panning = e.state == "running" and e.current is not None \
                and e.current.joint == "shoulder_pan"  # fmt: skip
            if panning and self.pan_turn is None:
                self.pan_turn = c.arm.name
            c.pos = c.arm.read_raw_positions()
            # WHY the goals first: a joint leaving the sweep gets its full torque back only once
            # its goal is where it stands, so it never presses into a stop at full strength.
            wait = panning and self.pan_turn != c.arm.name
            c.waiting = wait
            c.arm.write_raw_goals(e.step(c.pos, now, advance=not wait))
            j = e.current.joint if e.state == "running" and e.current and not wait else None
            if j != c.limited:
                limits = {c.limited: FULL_TORQUE} if c.limited else {}
                if j:
                    limits[j] = autocal.sweep_limit(j, c.torque)
                c.arm.set_torque_limits(limits)
                c.limited = j
            still_panning = e.state in ("running", "paused") and e.current is not None \
                and e.current.joint == "shoulder_pan"  # fmt: skip
            if self.pan_turn == c.arm.name and not still_panning:
                self.pan_turn = None
            if e.state == "done":
                c.new = dict(c.old)
                for jj in c.homings:
                    if jj in e.found:
                        homing, lo, hi = e.range(jj, c.homings[jj])
                    elif jj == FULL_TURN:
                        homing, (lo, hi) = c.homings[jj], FULL_RANGE
                    else:
                        continue
                    c.new[jj] = JointCal(c.old[jj].id, 0, homing, lo, hi)  # drive_mode 0
                c.step = "auto-review"
            elif e.state == "failed":
                c.step = "auto-failed"

    def _cal_at(self, step: str) -> CalRun | None:
        if self.cal is None:
            self.error("No calibration is running.", "Start one from the Calibrate page.")
            return None
        if self.cal.step != step:
            self.error(f"Calibration is at the {self.cal.step} step, not {step}.")
            return None
        return self.cal

    def _cmd_cal_middle(self, msg: dict[str, Any]) -> None:
        c = self._cal_at("middle")
        if c is None:
            return
        c.written = True  # before the write: a write that fails halfway still changed registers
        c.homings = c.arm.set_half_turn_homings()
        c.pos = c.arm.read_raw_positions()
        c.mins = {j: v for j, v in c.pos.items() if j != FULL_TURN}
        c.maxes = dict(c.mins)
        c.step = "ranges"

    def _cmd_cal_finish(self, msg: dict[str, Any]) -> None:
        c = self._cal_at("ranges")
        if c is None:
            return
        still = [j for j in c.mins if c.mins[j] == c.maxes[j]]
        if still:  # LeRobot raises here (motors_bus.py:844-846); Studio lets the user carry on
            self.error(f"These joints did not move: {', '.join(still)}.",
                       "Move each one through its full range, then finish.")  # fmt: skip
            return
        c.new = {}
        for j, old in c.old.items():
            lo, hi = FULL_RANGE if j == FULL_TURN else (c.mins[j], c.maxes[j])
            c.new[j] = JointCal(old.id, 0, c.homings[j], lo, hi)  # drive_mode 0, as LeRobot
        c.step = "review"

    def _cmd_cal_save(self, msg: dict[str, Any]) -> None:
        c = self._cal_at("review")
        if c is None or c.new is None:
            return
        self._save_run(c)
        self.cal = None
        self.session.calibration_ended()
        self._identify()

    def _save_run(self, c: CalRun) -> None:
        """Write a finished calibration to the servos and its file, and say so."""
        assert c.new is not None
        c.arm.write_calibration(c.new)
        self.calibrations[c.arm.calibration_id] = c.new
        path = None
        if self.cal_dir is not None:
            # WHY the arm's own file first: Identify re-reads that one (hardware.py
            # reload_calibration), so a save anywhere else would be undone by the next Identify.
            path = getattr(c.arm, "cal_path", None) or self.cal_dir / calibration_file(c.arm)
            try:
                # WHY a backup for a real arm: the file it replaces may be the only record of a
                # calibration someone made by hand.
                save_calibration(c.new, path, backup=not self.mock)
            except OSError as e:  # not a bus error: the registers are written
                self.error(f"Wrote {c.arm.name}'s registers but could not save {path}: {e}",
                           "Fix the folder's space or permissions, then calibrate again: the next "
                           "Identify reads the old file.")  # fmt: skip
                path = None
        self.send({"type": "calibrated", "arm": c.arm.name, "path": str(path) if path else None})

    def _cmd_cal_cancel(self, msg: dict[str, Any]) -> None:
        if self.cal is None:
            self.error("No calibration is running.")
            return
        self._cal_abort()
        self.session.calibration_ended()
        self._identify()

    def _cal_abort(self) -> None:
        """Drop the calibration in progress and write back the registers it changed."""
        c, self.cal = self.cal, None
        if c is None or not c.written:
            return
        self._restore(c)

    def _restore(self, c: CalRun) -> None:
        try:
            if c.limited:
                c.arm.set_torque_limits({c.limited: FULL_TORQUE})
            c.arm.write_calibration(c.old)
        except BUS_ERRORS as e:
            self.error(f"Could not restore {c.arm.name}'s calibration registers: {e}",
                       "They no longer match its file. Calibrate it again once it "
                       "answers.")  # fmt: skip

    def _cmd_stop(self, msg: dict[str, Any]) -> None:
        reason = msg.get("reason", "user")
        for c in self.autos:
            if c.auto is not None:
                c.auto.pause("Stopped")  # holds where it is; Resume goes on, Cancel ends it
        if self.session.state in (State.MOVING, State.ARMED):
            self.session.stopped(reason if reason in STOP_REASONS else "user")
            self._freeze()
        elif self.session.state is State.FAULT:
            self._freeze()  # already frozen; Stop still acts while torque is on

    def _cmd_resume(self, msg: dict[str, Any]) -> None:
        self.session.resumed()
        self.last_heartbeat = self.clock()

    def _cmd_release(self, msg: dict[str, Any]) -> None:
        # WHY allowed in FAULT: an overloaded gripper keeps squeezing while torque is on.
        # WHY allowed before arming when an arm holds torque: an arm can come up holding it, such as
        # a leader left on by another program. LeRobot turns leader torque off on connect
        # (so_leader.py:137).
        st = self.session.state
        stays = st is State.FAULT or (
            st in (State.IDENTIFIED, State.READY) and any(self.torque.values())
        )
        if not stays and st not in (State.ARMED, State.STOPPED):
            self.session.released()  # raises with the reason
        silent = self._release_all()
        if silent:
            self.error(f"{', '.join(silent)} did not answer, so its torque may still be on",
                       "Support the arm and cut its power, then reconnect it.")  # fmt: skip
        if not stays:
            self.session.released()

    def _cmd_clear(self, msg: dict[str, Any]) -> None:
        holding = [n for n, on in self.torque.items() if on]
        if holding and self.session.state is State.FAULT:
            self.error(f"Turn torque off before clearing: {', '.join(holding)} still holds",
                       "Clearing re-checks every arm, so no arm may hold torque. Support each "
                       "arm, then use Torque off.")  # fmt: skip
            return
        self.session.cleared()
        self.dead.clear()
        self._identify()

    def _cmd_disconnect(self, msg: dict[str, Any]) -> None:
        self._rec_end()
        self._cal_abort()
        self._autocal_abort()
        self._release_all()
        # WHY keep the run: Disconnect is how the scene gets reset, and an ended eval episode must
        # stay judgeable after it. Dropping the run lost it from the eval record with no message.
        if self.run is not None and self.run.running:
            self.run.running, self.run.ended = False, "disconnected"
        self.health.clear()  # a reading from a rig that is gone would stay on screen for good
        self.identity = []
        self.send({"type": "identity", "arms": []})  # so no window keeps showing a rig that is gone
        release = getattr(self.rig, "release_ports", None)
        if release is not None:  # real arms: free the ports while disconnected (hardware.py)
            release()
        self.session.disconnected()

    def _cmd_inject(self, msg: dict[str, Any]) -> None:
        """Mock rig only: inject a fault so the UI's error paths can be exercised."""
        target = next((a for a in self.arms if a.name == msg.get("arm")), None)
        kinds = {"overload", "overheat", "voltage", "unplug", "replug", "clear", "swap", "hand",
                 "still"}  # fmt: skip
        if target is None or not hasattr(target, "inject"):
            self.error("fault injection works only on the mock rig")
            return
        if msg.get("kind") not in kinds:
            self.error(f"unknown fault {msg.get('kind')!r}", f"one of {', '.join(sorted(kinds))}")
            return
        if msg["kind"] == "swap":
            self.rig.swap_cables(target.name)
            return
        target.inject(msg["kind"], joint=msg.get("joint"))

    def shutdown(self, finish_s: float = FINISH_S) -> None:
        """The server is gone or Studio is closing: nobody can stop an arm from now on, so freeze
        it and release torque (LeRobot itself releases torque on disconnect,
        config_so_follower.py:31); put calibrations back; and let a recording finalize, or its
        dataset cannot be read."""
        self._freeze()
        self._rec_end()
        self._autocal_abort()  # restored as each arm's torque goes off, just below
        self._cal_abort()
        self._release_all()
        if self.rec is not None:
            self.rec.writer.done.wait(finish_s)

    # -- internals --------------------------------------------------------------------------------
    def _identify(self) -> None:
        reload = getattr(self.rig, "reload_calibrations", None)
        if self._reload_files and reload is not None:
            # WHY every Identify: lerobot-calibrate in the terminal writes files after Studio read
            # them at start, and a stale copy would call a freshly calibrated arm a mismatch.
            self.calibrations = reload()
        out = []
        for a in self.arms:
            try:
                regs = a.read_calibration()
                self.torque[a.name] = a.read_torque()  # the servo's word, not what Studio sent
            except BUS_ERRORS as e:
                # WHY go on: one unplugged arm used to fault the connect, so the page could not
                # say which of four arms was the silent one.
                self.dead.add(a.name)
                out.append({"name": a.name, "role": a.role, "port": getattr(a, "port", None),
                            "serial": getattr(a, "serial", None), "expected": a.calibration_id,
                            "file": calibration_file(a), "match": None, "max_deg": None,
                            "worst_joint": None, "exact": False, "calibrated": _calibrated(a),
                            "error": str(e), "ok": False})  # fmt: skip
                continue
            self.dead.discard(a.name)
            cals = dict(self.calibrations)
            own = getattr(a, "calibration", None)
            if own is not None:
                # WHY: files are keyed by name alone, so a leader's phi_bi_left.json hid the
                # follower's phi_bi_left.json; the arm's own file comes from its own folder.
                cals[a.calibration_id] = own
            ranked = match_fingerprint(regs, cals) if cals else []
            # WHY prefer its own among exact matches: phi_follower.json can be a copy of
            # phi_bi_left.json, and a tie must not read as swapped cables.
            best = next((m for m in ranked if m.distance.exact and m.name == a.calibration_id),
                        ranked[0] if ranked else None)  # fmt: skip
            exact = bool(best and best.distance.exact)
            out.append({
                "name": a.name, "role": a.role,
                "port": getattr(a, "port", f"mock://{a.name}"),
                "serial": getattr(a, "serial", f"MOCK-{a.name}"),
                "expected": a.calibration_id,
                "file": calibration_file(a),
                "match": best.name if best else None,
                "max_deg": round(best.distance.max_deg, 2) if best else None,
                "worst_joint": best.distance.worst_joint if best else None,
                "exact": exact,
                "calibrated": _calibrated(a),
                "ok": exact and best is not None and best.name == a.calibration_id,
            })  # fmt: skip
        self.identity = out
        self.send({"type": "identity", "arms": out})  # before the state, so the UI has the matches
        self.session.identified()

    def _set_torque(self, a: ArmBus, on: bool) -> None:
        a.set_torque(on)
        self.torque[a.name] = on

    def _release_all(self) -> list[str]:
        """Torque off on every arm that may hold it, leaders too and dead ones included (a
        replugged arm answers again). Returns the arms that did not answer."""
        silent = []
        for a in self.arms:
            if not self.torque[a.name]:
                continue
            try:
                self._set_torque(a, False)
            except BUS_ERRORS:
                silent.append(a.name)
                continue
            for c in [c for c in self.restores if c.arm is a]:
                self.restores.remove(c)
                self._restore(c)
        return silent

    def _freeze(self) -> None:
        """Goal = present position on every reachable follower with torque on. Torque stays on.
        A follower that does not answer is marked dead and faults the session."""
        lost = None
        # Leaders too while an auto-calibration drives them: they hold torque then.
        driven = [c.arm for c in self.autos if c.arm.role == "leader"]
        for f in [*self.followers, *driven]:
            # WHY skip limp followers: a goal write may enable torque on some Feetech firmware
            # (bench test 22), and a limp arm has nothing to hold.
            if f.name in self.dead or not self.torque[f.name]:
                continue
            try:
                f.write_goals(f.read_positions())
            except BUS_ERRORS as e:
                self.dead.add(f.name)
                lost = lost or f"{label(f.name)} is not answering: {e}"
        if lost and self.session.state is not State.FAULT:
            self.session.faulted(lost)

    def _fault(self, why: str) -> None:
        self.session.faulted(why)
        self._freeze()
        self._cal_abort()
        self._autocal_abort()

    def _limit(self, arm: ArmBus | None) -> dict[str, float]:
        """Per joint, the largest step one tick may take: Studio's own, or robot-config.yaml's
        max_relative_target where that is stricter. WHY never looser: Studio's step is per 30 Hz
        tick, LeRobot's per 60 fps step, so a config value is already twice as fast here."""
        cfg = getattr(arm, "max_step", None)
        if isinstance(cfg, (int, float)) and not isinstance(cfg, bool):
            return {j: min(v, float(cfg)) for j, v in self.max_step.items()}
        if isinstance(cfg, dict):
            return {j: min(v, float(cfg.get(j, v))) for j, v in self.max_step.items()}
        return self.max_step

    def _clip(self, goal: dict[str, float], present: dict[str, float],
              arm: ArmBus | None = None) -> dict[str, float]:  # fmt: skip
        step = self._limit(arm)
        return {j: present[j] + max(-step[j], min(step[j], goal[j] - present[j])) for j in goal}

    def tick(self) -> None:
        """One control cycle: heartbeat, read, act, health, publish."""
        t_start, now = time.perf_counter(), self.clock()
        self._ticks.append(now)
        self._n += 1
        st = self.session.state
        if st in (State.ARMED, State.MOVING) and now - self.last_heartbeat > HEARTBEAT_TIMEOUT_S:
            self.session.heartbeat_lost()
            self._freeze()
        if st is State.DISCONNECTED:
            self._publish({})  # so torque badges and the Stop button reflect the release
            return
        pos: dict[str, dict[str, float]] = {}
        current = None
        try:
            for a in self.arms:
                if a.name not in self.dead and _calibrated(a):
                    current = a
                    pos[a.name] = a.read_positions()
            if self.session.may_move and self.session.activity == "teleop":
                for lead, fol in self.pairs:
                    current = fol
                    fol.write_goals(self._clip(pos[lead.name], pos[fol.name], fol))
                if self.rec is not None and self.rec.phase != "done":
                    self._rec_tick(now, pos)
            elif self.session.may_move and self.session.activity == "policy" and self.run:
                for fol, goal in self._policy_goals(now, pos):
                    current = fol
                    fol.write_goals(goal)
            if self.autos:
                current = self.autos[0].arm
                self._autocal_tick(now)
            if self.cal is not None and self.cal.step == "ranges":
                c = self.cal
                current = c.arm
                c.pos = c.arm.read_raw_positions()
                for j in c.mins:
                    c.mins[j], c.maxes[j] = min(c.mins[j], c.pos[j]), max(c.maxes[j], c.pos[j])
            if self._n % self.health_every == 0:
                for a in self.arms:
                    if a.name not in self.dead:
                        current = a
                        self.health[a.name] = a.read_health()
                self._check_health()
        except BUS_ERRORS as e:
            if current is not None:
                self.dead.add(current.name)
            where = "A bus"
            if current is not None:
                where = f"{label(current.name)} on {getattr(current, 'port', 'its port')}"
            self._fault(f"{where} is not answering: {e}")
        except Exception as e:  # a bug: stop the rig and say so, do not kill the worker
            self._fault(f"Studio hit an internal error in the control loop: {e!r}")
        self._cost_ms.append((time.perf_counter() - t_start) * 1e3)
        self._publish(pos)

    def _check_health(self) -> None:
        if self.session.state is State.FAULT:
            return
        for name, joints in self.health.items():
            for j, h in joints.items():
                if h.faults:
                    self._fault(f"{label(name)} {label(j)}: {', '.join(h.faults)}")
                    return

    def _publish(self, pos: dict[str, dict[str, float]]) -> None:
        span = self._ticks[-1] - self._ticks[0] if len(self._ticks) > 1 else 0.0
        costs = sorted(self._cost_ms)
        arms: dict[str, Any] = {}
        for a in self.arms:
            h = self.health.get(a.name, {})
            arms[a.name] = {
                "role": a.role,
                "online": a.name not in self.dead,
                "torque": self.torque[a.name],
                "pos": {j: round(v, 2) for j, v in pos.get(a.name, {}).items()},
                "health": {j: {"load": round(x.load_pct, 1), "temp": round(x.temperature_c, 1),
                               "volt": round(x.voltage_v, 2), "faults": x.faults}
                           for j, x in h.items()},
            }  # fmt: skip
        self.send({
            "type": "telemetry", "t": self.clock(), "arms": arms,
            "calibration": self.cal.view() if self.cal else None,
            "autocal": [c.view() for c in self.autos] or None,
            "recording": self.rec.view(self.clock()) if self.rec else None,
            "policy": self.run.view() if self.run else None,
            "loop": {"hz": round((len(self._ticks) - 1) / span, 1) if span else 0.0,
                     "p50_ms": round(costs[len(costs) // 2], 2) if costs else 0.0,
                     "p99_ms": round(costs[int(len(costs) * 0.99)], 2) if costs else 0.0},
        })  # fmt: skip


# -- the worker process --------------------------------------------------------------------------
class Outbox:
    """Worker -> server messages. `put` never blocks, so a slow or stuck server cannot stall the bus
    loop or its heartbeat watchdog; one sender thread does the blocking Pipe writes. As in the
    server's Client, the newest telemetry and the newest frame per camera win, and every event is
    kept, in order."""

    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        self._send = send
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._events: deque[dict[str, Any]] = deque()
        self._telemetry: dict[str, Any] | None = None
        self._frames: dict[str, dict[str, Any]] = {}

    def put(self, msg: dict[str, Any]) -> None:
        with self._lock:
            kind = msg.get("type")
            if kind == "telemetry":
                self._telemetry = msg
            elif kind == "frame":
                self._frames[msg["key"]] = msg
            else:
                self._events.append(msg)
        self._wake.set()

    def _take(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
            if self._telemetry is not None:
                out.append(self._telemetry)
                self._telemetry = None
            out.extend(self._frames.values())
            self._frames = {}
        return out

    def run(self, done: threading.Event) -> None:
        while not done.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            for m in self._take():
                self._send(m)


def build_rig(spec: dict[str, Any]) -> Any:
    if spec.get("kind", "mock") == "mock":
        from phi_studio.mock import mock_rig

        cams = spec.get("cameras")  # None: the mock's own set for that many pairs
        pairs = int(spec.get("pairs", 1))
        return mock_rig(pairs=pairs, cameras=None if cams is None else tuple(cams))
    if spec.get("kind") == "lerobot":
        # Real arms over LeRobot's Feetech bus, from robot-config.yaml (hardware.py). Connecting
        # reads and configures; it never turns torque on.
        from phi_studio.hardware import build_rig as hardware_rig

        return hardware_rig(Path(spec["config"]))
    raise ValueError(f"rig kind {spec.get('kind')!r} is not available")


def encode_jpeg(frame: Any, quality: int = 80) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def run_worker(conn: Any, spec: dict[str, Any]) -> None:
    """Entry point of the spawned worker process. `conn` is one end of a multiprocessing Pipe.

    Threads: a reader that queues commands, one publisher per camera, the outbox sender, and the
    bus loop, which is the only thread that calls the rig's arms (one owner per bus, ADR-S4).
    """
    import queue

    def send(msg: dict[str, Any]) -> None:  # only the outbox thread calls this after startup
        try:
            conn.send(msg)
        except (BrokenPipeError, OSError):
            pass

    try:
        rig = build_rig(spec)
    except Exception as e:  # report, do not die silently
        send({"type": "error", "message": f"could not build the rig: {e}", "fix": ""})
        return
    hz = float(spec.get("hz", 30))
    outbox = Outbox(send)
    # WHY a separate directory for the mock: its calibrations must never land in LeRobot's own
    # calibration directory, where a real arm's file would be overwritten.
    data = spec.get("data_dir")
    mock = spec.get("kind", "mock") == "mock"
    if mock:
        cal_dir = Path(data) / "mock-calibration" if data else None
    else:
        # A real arm's calibration goes where LeRobot reads it, so lerobot-record uses that file.
        from phi_studio.files import lerobot_calibration_dir

        cal_dir = lerobot_calibration_dir()
    w = RigWorker(rig, outbox.put, loop_hz=hz, cal_dir=cal_dir)
    inbox: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    done = threading.Event()

    def reader() -> None:
        while not done.is_set():
            try:
                inbox.put(conn.recv())
            except (EOFError, OSError):
                inbox.put({"cmd": "__exit__"})
                return

    def camera(cam: Any, fps: float) -> None:
        period, nxt = 1.0 / fps, time.monotonic()
        while not done.is_set():
            try:
                frame, t, seq = cam.read_latest()
                outbox.put({"type": "frame", "key": cam.key, "t": t, "seq": seq,
                      "w": int(frame.shape[1]), "h": int(frame.shape[0]),
                      "jpeg": encode_jpeg(frame)})  # fmt: skip
            except ConnectionError as e:
                outbox.put({"type": "camera", "key": cam.key, "online": False, "message": str(e)})
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=outbox.run, args=(done,), daemon=True).start()
    for cam in rig.cameras:
        threading.Thread(target=camera, args=(cam, float(spec.get("preview_fps", 15))),
                         daemon=True).start()  # fmt: skip
    outbox.put(w.describe())
    outbox.put({"type": "state", **w.session.snapshot()})
    period, nxt = 1.0 / hz, time.monotonic()
    try:
        while True:
            while True:
                try:
                    msg = inbox.get_nowait()
                except queue.Empty:
                    break
                if msg.get("cmd") == "__exit__":
                    return
                w.handle(msg)
            w.tick()
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -period:  # fell behind by a whole cycle: resynchronise, do not burst
                nxt = time.monotonic()
            time.sleep(max(0.0, delay))
    finally:
        done.set()
        w.shutdown()
        release = getattr(rig, "release_ports", None)
        if release is not None:
            release()
        for cam in rig.cameras:
            cam.close()
