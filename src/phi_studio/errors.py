"""The error a feature raises to refuse a request. The server shows its message as written, with
no traceback, because the person can act on it."""

from __future__ import annotations


class Refusal(ValueError):
    """A request Studio refuses, for a reason the person can fix."""

    def __init__(self, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix
