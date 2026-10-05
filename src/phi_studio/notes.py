"""Notes and issues on datasets, episodes and moments, kept in one SQLite file in Studio's data
folder.

A note pins to a dataset, optionally an episode, optionally a time range in it, optionally an arm
and
joint. Kinds:
  note    an observation
  issue   a problem someone should fix (open until resolved)
  bad     this episode should not be trained on
  good    a reference episode worth keeping
Automatic flags from analysis.py are not stored; dismissing one is (table `dismissed`), so a flag a
person
judged harmless stays hidden on every Mac that shares the file.

WHY SQLite: notes are written by several windows and read with filters (one episode, all open
issues);
one file, atomic writes, no server. WHY not in the dataset: datasets are LeRobot's format, often a
Hub
download, and notes must survive a re-download.
"""

from __future__ import annotations

import builtins
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from phi_studio.errors import Refusal

KINDS = ("note", "issue", "bad", "good")
SEVERITIES = ("info", "warn", "error")
STATUSES = ("open", "resolved")
TEXT_MAX = 4000

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  author TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  severity TEXT NOT NULL DEFAULT 'info',
  status TEXT NOT NULL DEFAULT 'open',
  dataset TEXT NOT NULL,
  dataset_name TEXT NOT NULL DEFAULT '',
  episode INTEGER,
  t0 REAL,
  t1 REAL,
  arm TEXT,
  joint TEXT,
  tags TEXT NOT NULL DEFAULT '[]',
  text TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'manual'
);
CREATE INDEX IF NOT EXISTS notes_where ON notes (dataset, episode);
CREATE TABLE IF NOT EXISTS dismissed (
  dataset TEXT NOT NULL,
  episode INTEGER NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  at REAL NOT NULL,
  author TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (dataset, episode, kind, key)
);
"""

COLUMNS = (
    "id",
    "created",
    "updated",
    "author",
    "kind",
    "severity",
    "status",
    "dataset",
    "dataset_name",
    "episode",
    "t0",
    "t1",
    "arm",
    "joint",
    "tags",
    "text",
    "source",
)


def _num(v: Any, what: str, lo: float = 0.0, hi: float = 1e7) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= float(v) <= hi:
        raise Refusal(f"{what} must be a number from {lo:g} to {hi:g}.")
    return float(v)


def _str(v: Any, what: str, n: int = 200, allowed: tuple[str, ...] | None = None) -> str | None:
    if v is None or v == "":
        return None
    if not isinstance(v, str) or len(v) > n:
        raise Refusal(f"{what} must be text of at most {n} characters.")
    if allowed is not None and v not in allowed:
        raise Refusal(f"{what} must be one of {', '.join(allowed)}.")
    return v


class NoteStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _row(r: tuple[Any, ...]) -> dict[str, Any]:
        d = dict(zip(COLUMNS, r, strict=True))
        d["tags"] = json.loads(d["tags"] or "[]")
        return d

    def save(self, note: dict[str, Any], author: str = "") -> dict[str, Any]:
        """Create (no id) or update (id) one note. Returns it as stored."""
        if not isinstance(note, dict):
            raise Refusal("A note must be an object.")
        now = time.time()
        t0 = _num(note.get("t0"), "Start time")
        t1 = _num(note.get("t1"), "End time")
        if t0 is not None and t1 is not None and t1 < t0:
            t0, t1 = t1, t0  # a range drawn right to left
        fields = {
            "kind": _str(note.get("kind", "note"), "Kind", allowed=KINDS) or "note",
            "severity": _str(note.get("severity", "info"), "Severity", allowed=SEVERITIES)
            or "info",
            "status": _str(note.get("status", "open"), "Status", allowed=STATUSES) or "open",
            "dataset": _str(note.get("dataset"), "Dataset", 64),
            "dataset_name": _str(note.get("dataset_name"), "Dataset name", 300) or "",
            "episode": None
            if note.get("episode") is None
            else int(_num(note.get("episode"), "Episode") or 0),
            "t0": t0,
            "t1": t1,
            "arm": _str(note.get("arm"), "Arm", 20),
            "joint": _str(note.get("joint"), "Joint", 40),
            "text": (_str(note.get("text"), "Text", TEXT_MAX) or "").strip(),
            "source": _str(note.get("source", "manual"), "Source", 20, ("manual", "flag", "live"))
            or "manual",
        }
        tags = note.get("tags") or []
        if (
            not isinstance(tags, list)
            or len(tags) > 20
            or not all(isinstance(t, str) and 0 < len(t) <= 40 for t in tags)
        ):
            raise Refusal("Tags must be a list of at most 20 short words.")
        if fields["dataset"] is None:
            raise Refusal("A note needs a dataset.")
        if not fields["text"] and fields["kind"] in ("note", "issue"):
            raise Refusal("Write what you noticed.", "An empty note cannot be found again later.")
        with self._lock:
            nid = note.get("id")
            if nid:
                cur = self._db.execute("SELECT id FROM notes WHERE id = ?", (nid,)).fetchone()
                if cur is None:
                    raise Refusal(
                        "That note no longer exists.", "Another window may have deleted it."
                    )
                sets = ", ".join(f"{k} = ?" for k in fields) + ", tags = ?, updated = ?"
                self._db.execute(
                    f"UPDATE notes SET {sets} WHERE id = ?",
                    (*fields.values(), json.dumps(sorted(set(tags))), now, nid),
                )
            else:
                nid = secrets.token_hex(6)
                self._db.execute(
                    f"INSERT INTO notes ({', '.join(COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(COLUMNS))})",
                    (
                        nid,
                        now,
                        now,
                        author[:80],
                        fields["kind"],
                        fields["severity"],
                        fields["status"],
                        fields["dataset"],
                        fields["dataset_name"],
                        fields["episode"],
                        fields["t0"],
                        fields["t1"],
                        fields["arm"],
                        fields["joint"],
                        json.dumps(sorted(set(tags))),
                        fields["text"],
                        fields["source"],
                    ),
                )
            row = self._db.execute(
                f"SELECT {', '.join(COLUMNS)} FROM notes WHERE id = ?", (nid,)
            ).fetchone()
        return self._row(row)

    def delete(self, nid: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM notes WHERE id = ?", (nid,))

    def list(
        self,
        dataset: str | None = None,
        episode: int | None = None,
        status: str | None = None,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        q = f"SELECT {', '.join(COLUMNS)} FROM notes"
        args: builtins.list[Any] = []
        where = []
        if dataset:
            where.append("dataset = ?")
            args.append(dataset)
        if episode is not None:
            where.append("episode = ?")
            args.append(int(episode))
        if status:
            where.append("status = ?")
            args.append(status)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY created DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            return [self._row(r) for r in self._db.execute(q, args).fetchall()]

    # -- automatic flags a person judged harmless --------------------------------------------------
    @staticmethod
    def flag_key(flag: dict[str, Any]) -> str:
        return f"{flag.get('arm') or ''}|{flag.get('joint') or ''}|{flag.get('t0')}"

    def dismiss(
        self, dataset: str, episode: int, kind: str, key: str, author: str = "", undo: bool = False
    ) -> None:
        with self._lock:
            if undo:
                self._db.execute(
                    "DELETE FROM dismissed WHERE dataset=? AND episode=? AND kind=? AND key=?",
                    (dataset, int(episode), kind, key),
                )
            else:
                self._db.execute(
                    "INSERT OR REPLACE INTO dismissed VALUES (?, ?, ?, ?, ?, ?)",
                    (dataset, int(episode), kind[:40], key[:200], time.time(), author[:80]),
                )

    def dismissed(self, dataset: str) -> set[tuple[int, str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT episode, kind, key FROM dismissed WHERE dataset = ?", (dataset,)
            ).fetchall()
        return {(int(e), k, key) for e, k, key in rows}

    def excluded(self, dataset: str) -> builtins.list[int]:
        """Episodes marked bad: the ones a training run should leave out."""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT episode FROM notes WHERE dataset = ? AND kind = 'bad' "
                "AND episode IS NOT NULL "
                "AND status = 'open'",
                (dataset,),
            ).fetchall()
        return sorted(int(r[0]) for r in rows)
