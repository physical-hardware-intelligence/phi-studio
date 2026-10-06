"""Holds ENTER at LeRobot's "use provided calibration file" prompt until it is known to be safe.

At that prompt ENTER writes the id's file into the servos of whatever arm is on the command's port
(so_follower.py:115-124, so_leader.py:84-95). Studio watches the terminal's output for the prompt,
asks cmdcheck whether that port and that id are the same arm, and drops a bare ENTER typed into
the panel when they are not or Studio cannot tell. Ctrl-C always passes. c (recalibrate) passes
only when Studio cannot tell: when the port and the id name different arms, c calibrates the arm
on the port and saves it as the other arm's file, which is the same fault the other way round.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from phi_studio.cmdcheck import DANGER, OK, PROMPT, Verdict

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
TAIL = 4096  # the prompt fits in one line; this only has to survive a split read


class PromptGuard:
    def __init__(self, verdict_for: Callable[[str], Verdict]) -> None:
        self.verdict_for = verdict_for  # lerobot id -> Verdict
        self.tail = ""
        self.armed: Verdict | None = None
        self.armed_id: str | None = None  # the lerobot id the prompt names
        self.typed = ""

    def feed(self, data: bytes) -> Verdict | None:
        """Terminal output. Returns the verdict when a new prompt has just appeared."""
        self.tail = (self.tail + ANSI.sub("", data.decode("utf-8", "replace")))[-TAIL:]
        m = PROMPT.search(self.tail)
        if m is None:
            return None
        self.tail = self.tail[m.end():]  # WHY cut: the same prompt must not arm twice
        self.armed_id = m.group(1)
        self.armed, self.typed = self.verdict_for(self.armed_id), ""
        return self.armed

    def recheck(self) -> Verdict | None:
        """The verdict again, once Studio knows the running command. WHY: a prompt can print
        before the half-second poll has seen the command, and then the first verdict knew no
        ports. Returns the new verdict, or None if no prompt is waiting."""
        if self.armed is None or self.armed_id is None:
            return None
        self.armed = self.verdict_for(self.armed_id)
        return self.armed

    def gate(self, data: str) -> tuple[str, Verdict | None]:
        """Input from the panel. Returns what to send to the shell, and the verdict if an ENTER
        was held back."""
        if self.armed is None:
            return data, None
        out = ""
        for i, ch in enumerate(data):
            if ch == "\x03":  # Ctrl-C ends the command, so nothing is written
                self.disarm()
                return out + data[i:], None
            if ch in "\r\n":
                recal = self.typed.strip().lower() == "c" and self.armed.level != DANGER
                if recal or self.armed.level == OK:
                    self.disarm()
                    return out + data[i:], None
                held = self.armed
                self.typed = ""  # WHY clear: the shell's line is still empty, nothing was sent
                return out, held
            if ch in "\x7f\b":
                self.typed = self.typed[:-1]
            elif ch.isprintable():
                self.typed += ch
            out += ch
        return out, None

    def allow(self) -> bool:
        """The user read the warning and wants ENTER anyway. True if one was waiting. Never for a
        port and id that name different arms: fix the command instead."""
        if self.armed is None or self.armed.level == DANGER:
            return False
        self.disarm()
        return True

    def disarm(self) -> None:
        self.armed, self.armed_id, self.typed = None, None, ""

    def reset(self) -> None:
        """The command ended: its prompt is gone."""
        self.disarm()
        self.tail = ""
