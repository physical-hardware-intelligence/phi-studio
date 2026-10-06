"""Holds ENTER at LeRobot's "use provided calibration file" prompt until it is known to be safe.

At that prompt ENTER writes the id's file into the servos of whatever arm is on the command's port
(so_follower.py:115-124, so_leader.py:84-95). Studio watches the terminal's output for the prompt,
asks cmdcheck whether that port and that id are the same arm, and drops a bare ENTER typed into
the panel when they are not or Studio cannot tell. c (recalibrate) passes only when Studio cannot
tell: when the port and the id name different arms, c calibrates the arm on the port and saves it
as the other arm's file, which is the same fault the other way round.

While armed, the keys typed are kept here and echoed by Studio, not sent: the line LeRobot reads
is then exactly what passes the gate (a "y" typed before a held ENTER would otherwise stay in the
tty and turn a later "c" into "yc", which LeRobot takes as ENTER).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from phi_studio.cmdcheck import DANGER, OK, PROMPT, Verdict

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
KEYS = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|O.|.)?")  # arrow and function keys, whole
TAIL = 4096  # the prompt fits in one line; this only has to survive a split read
ERASE = "\b \b"


class PromptGuard:
    def __init__(self, verdict_for: Callable[[str], Verdict | None],
                 echo: Callable[[str], None] = lambda s: None) -> None:  # fmt: skip
        self.verdict_for = verdict_for  # lerobot id -> Verdict, or None: not LeRobot's prompt
        self.echo = echo  # shows text on the panels without sending it to the shell
        self.tail = ""
        self.armed: Verdict | None = None
        self.armed_id: str | None = None  # the lerobot id the prompt names
        self.standing: Verdict | None = None  # the whole command's verdict, when DANGER
        self.typed = ""

    def feed(self, data: bytes) -> Verdict | None:
        """Terminal output. Returns the verdict when a new prompt has just appeared."""
        self.tail = (self.tail + ANSI.sub("", data.decode("utf-8", "replace")))[-TAIL:]
        m = PROMPT.search(self.tail)
        if m is None:
            return None
        self.tail = self.tail[m.end():]  # WHY cut: the same prompt must not arm twice
        v = self.verdict_for(m.group(1))
        if v is None:
            return None
        self._clear_typed()
        self.armed_id, self.armed = m.group(1), v
        return v

    def recheck(self) -> Verdict | None:
        """The verdict again, once Studio knows the running command. WHY: a prompt can print
        before the half-second poll has seen the command, and then the first verdict knew no
        ports. Returns the new verdict, or None if no prompt is waiting (or it was not LeRobot's,
        and the guard is disarmed)."""
        if self.armed is None or self.armed_id is None:
            return None
        v = self.verdict_for(self.armed_id)
        if v is None:
            self.disarm()
            return None
        self.armed = v
        return v

    def stand(self, v: Verdict | None) -> None:
        """The running command's own verdict. A DANGER command holds ENTER from its start, so an
        ENTER typed ahead while it connects (the tty keeps it for input()) is held too."""
        self.standing = v if v is not None and v.level == DANGER else None
        if self.armed is None and self.standing is not None:
            self.armed = self.standing

    def gate(self, data: str) -> tuple[str, Verdict | None]:
        """Input from the panel. Returns what to send to the shell, and the verdict if an ENTER
        was held back."""
        if self.armed is None:
            return data, None
        data = KEYS.sub("", data)  # WHY: the shell sees only the line that passes
        for i, ch in enumerate(data):
            if ch == "\x03":
                # WHY stay armed: lerobot-rollout catches the first SIGINT (utils/process.py:57)
                # and its input() keeps waiting; the guard ends when the command does (reset)
                self._clear_typed()
                return "\x03", None
            if ch in "\r\n":
                recal = self.typed.strip().lower() == "c" and self.armed.level != DANGER
                if recal or self.armed.level == OK:
                    line = self.typed
                    self._clear_typed()
                    self.disarm()
                    return line + data[i:], None
                self._clear_typed()
                return "", self.armed
            if ch in "\x7f\b":
                if self.typed:
                    self.typed = self.typed[:-1]
                    self.echo(ERASE)
            elif ch == "\x15":  # Ctrl-U
                self._clear_typed()
            elif ch.isprintable():
                self.typed += ch
                self.echo(ch)
        return "", None

    def allow(self) -> bool:
        """The user read the warning and wants ENTER anyway. True if one was waiting. Never for a
        port and id that name different arms: fix the command instead."""
        if self.armed is None or self.armed_id is None or self.armed.level == DANGER:
            return False
        self._clear_typed()
        self.disarm()
        return True

    def _clear_typed(self) -> None:
        if self.typed:
            self.echo(ERASE * len(self.typed))
        self.typed = ""

    def disarm(self) -> None:
        """This prompt is answered. A DANGER command stays held for its next prompt."""
        self.armed, self.armed_id, self.typed = self.standing, None, ""

    def reset(self) -> None:
        """The command ended: its prompt is gone."""
        self.standing = None
        self.disarm()
        self.tail = ""
