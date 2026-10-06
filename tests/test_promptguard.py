"""ENTER at LeRobot's calibration prompt is held unless the port and id name one arm."""

from __future__ import annotations

from phi_studio.cmdcheck import Verdict
from phi_studio.promptguard import PromptGuard

PROMPT = (b"\x1b[32mPress ENTER to use provided calibration file associated with the id "
          b"phi_bi_follower_right, or type 'c' and press ENTER to run calibration: \x1b[0m")


def guard(level: str) -> tuple[PromptGuard, list[str]]:
    asked: list[str] = []

    def verdict(i: str) -> Verdict:
        asked.append(i)
        return Verdict(level, f"{i} is {level}")

    return PromptGuard(verdict), asked


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
        assert sent == "y" and held is not None


def test_c_ctrl_c_and_a_safe_arm_pass() -> None:
    g, _ = guard("unknown")
    g.feed(PROMPT)
    assert g.gate("c") == ("c", None)
    assert g.gate("\r") == ("\r", None) and g.armed is None  # recalibrate: later ENTERs pass
    g, _ = guard("danger")
    g.feed(PROMPT)
    assert g.gate("c\r") == ("c", g.armed)  # c on a crossed port and id saves the wrong file
    g.feed(PROMPT)
    assert g.gate("\x03") == ("\x03", None) and g.armed is None
    g, _ = guard("ok")
    g.feed(PROMPT)
    assert g.gate("\r") == ("\r", None)


def test_backspace_edits_the_typed_answer() -> None:
    g, _ = guard("unknown")
    g.feed(PROMPT)
    g.gate("x\x7fc")
    assert g.gate("\r") == ("\r", None)


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
