"""The terminal panel: the user's own shell on this Mac, in the env Studio runs in.

It runs inside a pseudo-terminal, so LeRobot's prompts ("press ENTER"), colours and progress bars
behave as they do in Terminal.app. Studio keeps one shell; every window sees its output, and the
server lets only the window with control type into it, so two windows never type at once.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCROLLBACK = 256 * 1024  # bytes of output replayed to a window that opens the terminal later

# WHY a helper process: the shell needs the pty as its controlling terminal, or Ctrl-C and job
# control do not work. os.login_tty does that (setsid + TIOCSCTTY), but must run in the child
# between fork and exec; doing it in Studio's own forked child is unsafe because Studio has
# threads. A fresh single-threaded Python does it and then becomes the shell.
_HELPER = "import os, sys; os.login_tty(0); os.execvpe(sys.argv[1], sys.argv[1:], os.environ)"


def env_activation(prefix: Path) -> str:
    """Shell lines that put Studio's own Python env first on PATH, through conda when it can."""
    q = shlex.quote(str(prefix))
    if (prefix / "conda-meta").is_dir():
        return (
            "if typeset -f conda >/dev/null 2>&1; then conda activate " + q + " 2>/dev/null"
            " || export PATH=" + shlex.quote(f"{prefix}/bin") + ':"$PATH"; '
            "else export PATH=" + shlex.quote(f"{prefix}/bin") + ':"$PATH"; fi'
        )
    return "export PATH=" + shlex.quote(f"{prefix}/bin") + ':"$PATH"'


def zsh_dotdir(home: Path, prefix: Path, cwd: Path) -> Path:
    """A ZDOTDIR whose startup files read the user's own, then activate the env and cd.
    WHY: zsh has no --rcfile; pointing ZDOTDIR at wrappers is how editors add to a user's shell
    without editing their files."""
    d = Path(tempfile.mkdtemp(prefix="phi-studio-zsh-"))
    h = shlex.quote(str(home))
    for name in (".zshenv", ".zprofile"):
        (d / name).write_text(f'[ -f {h}/{name} ] && source {h}/{name}\n')
    (d / ".zshrc").write_text(
        f"[ -f {h}/.zshrc ] && source {h}/.zshrc\n"
        f"ZDOTDIR={h}  # .zlogin and anything later read your own files\n"
        f"{env_activation(prefix)}\n"
        "export PYTHONNOUSERSITE=1  # the phi env's rule: no ~/.local packages\n"
        f"cd {shlex.quote(str(cwd))}\n"
    )
    return d


class Terminal:
    """One shell in a pseudo-terminal. Output goes to `on_output` subscribers and a scrollback."""

    def __init__(self, cwd: Path, shell: str | None = None, prefix: Path | None = None) -> None:
        self.cwd = cwd
        self.shell = shell or os.environ.get("SHELL") or "/bin/zsh"
        self.prefix = prefix or Path(sys.prefix)
        self.proc: subprocess.Popen[bytes] | None = None
        self.fd: int | None = None
        self.dotdir: Path | None = None
        self.scrollback: deque[bytes] = deque()
        self.size = 0
        self.listeners: set[Callable[[bytes], None]] = set()
        self.on_exit: Callable[[], None] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.cols, self.rows = 100, 30
        self.pending = b""
        self.writing = False

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self.alive:
            return
        self.close()
        self.loop = loop
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        env.pop("ZDOTDIR", None)
        argv = [self.shell, "-l", "-i"]
        if Path(self.shell).name == "zsh":
            self.dotdir = zsh_dotdir(Path.home(), self.prefix, self.cwd)
            env["ZDOTDIR"] = str(self.dotdir)
        else:  # any other shell: no startup hook, so put the env on PATH directly
            env["PATH"] = f"{self.prefix}/bin:{env.get('PATH', '')}"
            env["PYTHONNOUSERSITE"] = "1"
        master, slave = os.openpty()
        self._winsize(master)
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-c", _HELPER, *argv], stdin=slave, stdout=slave, stderr=slave,
                cwd=self.cwd, env=env, close_fds=True,
            )  # fmt: skip
        finally:
            os.close(slave)
        os.set_blocking(master, False)
        self.fd = master
        loop.add_reader(master, self._readable)

    def _winsize(self, fd: int) -> None:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))

    def _readable(self) -> None:
        assert self.fd is not None
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return
        except OSError:  # EIO: the shell exited and the pty closed
            data = b""
        if not data:
            self._ended()
            return
        self.scrollback.append(data)
        self.size += len(data)
        while self.size > SCROLLBACK and len(self.scrollback) > 1:
            self.size -= len(self.scrollback.popleft())
        for fn in list(self.listeners):
            fn(data)

    def _ended(self) -> None:
        if self.loop is not None and self.fd is not None:
            self.loop.remove_reader(self.fd)
        msg = b"\r\n[The shell exited. Press Restart to open a new one.]\r\n"
        self.scrollback.append(msg)
        for fn in list(self.listeners):
            fn(msg)
        if self.on_exit:
            self.on_exit()

    def history(self) -> bytes:
        return b"".join(self.scrollback)

    def write(self, data: bytes) -> None:
        if self.fd is None or not self.alive:
            raise OSError("The shell is not running.")
        # WHY queue: a command that is not reading its input fills the pty buffer, and a blocking
        # write would freeze Studio's event loop; the rest goes out when the pty is writable.
        self.pending += data
        self._flush()

    def _flush(self) -> None:
        if self.fd is None:
            return
        try:
            n = os.write(self.fd, self.pending) if self.pending else 0
        except BlockingIOError:
            n = 0
        except OSError:
            self.pending = b""
            return
        self.pending = self.pending[n:]
        if self.loop is not None:
            if self.pending and not self.writing:
                self.loop.add_writer(self.fd, self._flush)
                self.writing = True
            elif not self.pending and self.writing:
                self.loop.remove_writer(self.fd)
                self.writing = False

    def resize(self, cols: int, rows: int) -> None:
        self.cols, self.rows = max(2, min(cols, 1000)), max(2, min(rows, 1000))
        if self.fd is not None:
            self._winsize(self.fd)  # the kernel sends SIGWINCH to the foreground job

    def running(self) -> dict[str, Any] | None:
        """The command in the foreground, or None at the prompt. WHY the terminal's foreground
        process group: it changes exactly when the shell hands the terminal to a command."""
        if self.fd is None or self.proc is None or not self.alive:
            return None
        try:
            pgid = os.tcgetpgrp(self.fd)
        except OSError:
            return None
        # WHY <= 0 is idle: until the shell has read its startup files it has not yet made itself
        # the foreground group, and tcgetpgrp says 0 (no group), not "a command is running".
        if pgid <= 0 or pgid == self.proc.pid:
            return None
        out = ""
        for _ in range(3):  # a job that has forked but not yet exec'd shows no command line
            try:
                out = subprocess.run(["ps", "-o", "args=", "-p", str(pgid)], capture_output=True,
                                     text=True, timeout=2).stdout.strip()  # fmt: skip
            except (OSError, subprocess.SubprocessError):
                out = ""
            if out:
                break
            time.sleep(0.05)
        return {"pgid": pgid, "command": out}

    def interrupt(self) -> None:
        self.write(b"\x03")  # Ctrl-C, through the pty, as a keyboard would

    def close(self) -> None:
        if self.loop is not None and self.fd is not None:
            self.loop.remove_reader(self.fd)
            if self.writing:
                self.loop.remove_writer(self.fd)
        self.pending, self.writing = b"", False
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGHUP)  # the shell's session: it and its jobs
            except OSError:
                pass
            try:
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        if self.fd is not None:
            os.close(self.fd)
        if self.dotdir is not None:
            shutil.rmtree(self.dotdir, ignore_errors=True)
        self.proc, self.fd, self.dotdir = None, None, None
