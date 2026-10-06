"""ENTER at LeRobot's calibration prompt is held unless the port and id name one arm."""

from __future__ import annotations

from phi_studio.cmdcheck import Verdict
from phi_studio.promptguard import PromptGuard

PROMPT = (b"\x1b[32mPress ENTER to use provided calibration file associated with the id "
          b"phi_bi_follower_right, or type 'c' and press ENTER to run calibration: \x1b[0m")


def guard(level: str | None) -> tuple[PromptGuard, list[str]]:
    asked: list[str] = []
    shown: list[str] = []

    def verdict(i: str) -> Verdict | None:
        asked.append(i)
        return Verdict(level, f"{i} is {level}") if level else None

    g = PromptGuard(verdict, shown.append)
    g.shown = shown  # type: ignore[attr-defined]
    return g, asked


def screen(g: PromptGuard) -> str:
    """What the panel shows of the local echo, after backspaces."""
    out = ""
    for ch in "".join(g.shown).replace("\b \b", "\x7f"):  # type: ignore[attr-defined]
        out = out[:-1] if ch == "\x7f" else out + ch
    return out


def test_enter_passes_when_no_prompt_is_showing() -> None:
    g, _ = guard("danger")
    assert g.gate("ls\r") == ("ls\r", None)


def test_a_prompt_split_across_reads_is_found_once() -> None:
    g, asked = guard("danger")
    assert g.feed(PROMPT[:40]) is None
    v = g.feed(PROMPT[40:])
    assert v is not None and asked == ["phi_bi_follower_right"]
    assert g.feed(b"\r\n") is None  # the same prompt does not arm twice


def test_enter_is_held_for_a_dangerous_or_unknown_arm() -> None:
    for level in ("danger", "unknown"):
        g, _ = guard(level)
        g.feed(PROMPT)
        sent, held = g.gate("\r")
        assert sent == "" and held is not None and held.level == level
        sent, held = g.gate("y\r")  # anything but c also writes the file (so_follower.py:121)
        assert sent == "" and held is not None


def test_keys_typed_at_the_prompt_reach_the_shell_only_with_the_enter_that_passes() -> None:
    """Review finding: "y" sent ahead stayed in the tty, so a later "c" was read as "yc"."""
    g, _ = guard("unknown")
    g.feed(PROMPT)
    assert g.gate("y") == ("", None) and screen(g) == "y"  # echoed, not sent
    assert g.gate("\r")[0] == "" and screen(g) == ""  # held: the answer is wiped
    assert g.gate("c\r") == ("c\r", None)  # the shell's line is exactly "c"
    g, _ = guard("unknown")
    g.feed(PROMPT)
    assert g.gate("c\x15\r")[0] == ""  # Ctrl-U clears it here too
    g.gate("x\x1b[A\x7fc")  # arrows dropped, backspace edits
    assert g.gate("\r") == ("c\r", None)


def test_c_and_a_safe_arm_pass() -> None:
    g, _ = guard("unknown")
    g.feed(PROMPT)
    assert g.gate("c") == ("", None)
    assert g.gate("\r") == ("c\r", None) and g.armed is None  # recalibrate: later ENTERs pass
    assert g.gate("\r") == ("\r", None)
    g, _ = guard("danger")
    g.feed(PROMPT)
    assert g.gate("c\r") == ("", g.armed)  # c on a crossed port and id saves the wrong file
    g, _ = guard("ok")
    g.feed(PROMPT)
    assert g.gate("\r") == ("\r", None)


def test_ctrl_c_passes_but_the_prompt_stays_held_until_the_command_ends() -> None:
    """lerobot-rollout catches the first SIGINT (utils/process.py:57) and keeps waiting."""
    g, _ = guard("danger")
    g.feed(PROMPT)
    assert g.gate("y\x03") == ("\x03", None) and g.armed is not None
    assert g.gate("\r")[0] == ""
    g.reset()
    assert g.gate("\r") == ("\r", None)


def test_a_dangerous_command_holds_enter_typed_ahead_and_between_prompts() -> None:
    g, _ = guard("ok")
    g.stand(Verdict("danger", "left leader crossed"))
    assert g.gate("\r") == ("", g.standing)  # before any prompt: typeahead while it connects
    g.feed(PROMPT)  # this arm is fine
    assert g.gate("\r") == ("\r", None)
    assert g.armed is g.standing and g.gate("\r")[0] == ""  # held again for the next prompt
    g.stand(Verdict("unknown", "x"))  # only DANGER stands
    g.reset()
    assert g.armed is None


def test_prompt_text_from_another_program_does_not_arm() -> None:
    g, asked = guard(None)
    assert g.feed(PROMPT) is None and g.armed is None and asked


def test_backspace_edits_the_typed_answer() -> None:
    g, _ = guard("unknown")
    g.feed(PROMPT)
    g.gate("x\x7fc")
    assert g.gate("\r") == ("c\r", None)


def test_allow_releases_one_held_enter_and_reset_forgets_the_prompt() -> None:
    g, _ = guard("unknown")
    assert not g.allow()
    g.feed(PROMPT)
    g.gate("\r")
    assert g.allow() and g.armed is None
    g, _ = guard("danger")
    g.feed(PROMPT)
    assert not g.allow() and g.armed is not None  # a crossed port and id: fix the command
    g.feed(PROMPT[:50])
    g.reset()
    assert g.feed(PROMPT[50:]) is None
