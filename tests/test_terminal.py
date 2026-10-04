"""The terminal panel's shell: a real pty, real input, Ctrl-C, and the zsh wrapper."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from phi_studio.terminal import Terminal, env_activation, zsh_dotdir


async def until(term: Terminal, seen: list[bytes], want: bytes, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while want not in b"".join(seen):
        if time.monotonic() > end:
            raise AssertionError(f"{want!r} never came; got {b''.join(seen)[-400:]!r}")
        await asyncio.sleep(0.05)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_a_shell_runs_commands_and_ctrl_c_stops_one(tmp_path: Path) -> None:
    async def go() -> None:
        term = Terminal(tmp_path, shell="/bin/sh")
        seen: list[bytes] = []
        term.listeners.add(seen.append)
        term.start(asyncio.get_running_loop())
        try:
            term.write(b"echo hi-$((1+2)); pwd\n")
            await until(term, seen, b"hi-3")
            await until(term, seen, str(tmp_path.resolve()).encode())
            assert term.running() is None  # back at the prompt
            term.write(b"sleep 30\n")
            for _ in range(100):
                r = term.running()
                if r:
                    break
                await asyncio.sleep(0.05)
            assert r is not None and "sleep 30" in r["command"]
            term.interrupt()
            for _ in range(100):
                if term.running() is None:
                    break
                await asyncio.sleep(0.05)
            assert term.running() is None  # Ctrl-C reached the job: the pty is its terminal
            assert b"hi-3" in term.history()
        finally:
            term.close()
        assert not term.alive

    run(go())


def test_the_window_size_reaches_the_shell(tmp_path: Path) -> None:
    async def go() -> None:
        term = Terminal(tmp_path, shell="/bin/sh")
        seen: list[bytes] = []
        term.listeners.add(seen.append)
        term.start(asyncio.get_running_loop())
        try:
            term.resize(123, 45)
            term.write(b"stty size\n")
            await until(term, seen, b"45 123")
        finally:
            term.close()

    run(go())


def test_exit_is_reported_and_writing_after_it_fails(tmp_path: Path) -> None:
    async def go() -> None:
        term = Terminal(tmp_path, shell="/bin/sh")
        seen: list[bytes] = []
        ended = asyncio.Event()
        term.listeners.add(seen.append)
        term.on_exit = ended.set
        term.start(asyncio.get_running_loop())
        try:
            term.write(b"exit\n")
            await asyncio.wait_for(ended.wait(), 10)
            await until(term, seen, b"shell exited")
            for _ in range(50):
                if not term.alive:
                    break
                await asyncio.sleep(0.05)
            with pytest.raises(OSError):
                term.write(b"echo no\n")
        finally:
            term.close()

    run(go())


def test_scrollback_keeps_only_the_newest_output(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("phi_studio.terminal.SCROLLBACK", 1000)

    async def go() -> None:
        term = Terminal(tmp_path, shell="/bin/sh")
        seen: list[bytes] = []
        term.listeners.add(seen.append)
        term.start(asyncio.get_running_loop())
        try:
            term.write(b"i=0; while [ $i -lt 300 ]; do echo line-$i; i=$((i+1)); done; echo DONE\n")
            await until(term, seen, b"DONE\r\n")
            await asyncio.sleep(0.2)
            assert term.size <= 1000 + 65536 and b"line-299" in term.history()
            assert b"line-0\r\n" not in term.history()
        finally:
            term.close()

    run(go())


def test_env_activation_uses_conda_only_for_a_conda_env(tmp_path: Path) -> None:
    plain = tmp_path / "venv"
    plain.mkdir()
    assert env_activation(plain) == f"export PATH={plain}/bin:\"$PATH\""
    conda = tmp_path / "envs" / "phi"
    (conda / "conda-meta").mkdir(parents=True)
    line = env_activation(conda)
    assert f"conda activate {conda}" in line and f"export PATH={conda}/bin" in line


def test_the_zsh_wrapper_reads_the_users_files_then_the_env(tmp_path: Path) -> None:
    home, prefix, cwd = tmp_path / "home", tmp_path / "env", tmp_path / "work"
    for d in (home, prefix, cwd):
        d.mkdir()
    d = zsh_dotdir(home, prefix, cwd)
    try:
        rc = (d / ".zshrc").read_text()
        assert rc.index(f"source {home}/.zshrc") < rc.index(f"export PATH={prefix}/bin")
        assert f"cd {cwd}" in rc and f"ZDOTDIR={home}" in rc
        assert f"source {home}/.zprofile" in (d / ".zprofile").read_text()
    finally:
        shutil.rmtree(d)


@pytest.mark.skipif(not Path("/bin/zsh").exists(), reason="needs zsh")
def test_zsh_starts_with_the_users_rc_in_the_env(tmp_path: Path, monkeypatch: Any) -> None:
    home, prefix, cwd = tmp_path / "home", tmp_path / "env", tmp_path / "work"
    for d in (home, prefix / "bin", cwd):
        d.mkdir(parents=True)
    (home / ".zshrc").write_text("export FROM_USER_RC=yes\n")
    monkeypatch.setenv("HOME", str(home))

    async def go() -> None:
        term = Terminal(cwd, shell="/bin/zsh", prefix=prefix)
        seen: list[bytes] = []
        term.listeners.add(seen.append)
        term.start(asyncio.get_running_loop())
        try:
            term.write(b'echo "rc=$FROM_USER_RC first=${PATH%%:*} here=$PWD"\n')
            await until(term, seen, b"rc=yes")
            out = b"".join(seen).decode(errors="replace")
            assert f"first={prefix}/bin" in out and f"here={cwd.resolve()}" in out
            assert term.dotdir is not None and term.dotdir.is_dir()
            dotdir = term.dotdir
        finally:
            term.close()
        assert not dotdir.exists()  # the wrapper folder is removed with the shell

    run(go())


def test_helper_gives_the_shell_a_controlling_terminal(tmp_path: Path) -> None:
    # Without a controlling terminal, `tty` still names the pty, but the shell's job control
    # (and so Ctrl-C to a running command) does not work; the Ctrl-C test above covers that.
    async def go() -> None:
        term = Terminal(tmp_path, shell="/bin/sh")
        seen: list[bytes] = []
        term.listeners.add(seen.append)
        term.start(asyncio.get_running_loop())
        try:
            assert term.proc is not None
            term.write(b"ps -o tty= -p $$\n")
            await until(term, seen, b"ttys")
            assert os.getpgid(term.proc.pid) == term.proc.pid  # its own session leader
        finally:
            term.close()

    if sys.platform != "darwin":
        pytest.skip("tty names checked for macOS")
    run(go())
