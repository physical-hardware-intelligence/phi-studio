"""The calibration folder as a whole: which file each arm uses, and the rest.

LeRobot reads exactly one file per arm, <root>/<kind>/<folder>/<id>.json (robot.py:49-56). Every
other file in that folder is never read, but it is one wrong id away from being written into an
arm's servos (cmdcheck.py). Laptops collect them: old ids, a teammate's copies, links, .bak files.
On 2026-10-05 one Mac had six follower files and eight leader files, and two arms held another
arm's calibration.

This module lists every file with what it is, moves the ones no arm uses into an archive folder
(never deletes; each move can be undone), and installs a shared set, such as the phi repo's
configs/calibration, that everyone uses.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phi_studio import rigspec
from phi_studio.identity import Calibration, load_calibration, match_fingerprint, unfinished

FOLDERS = tuple(f"{k}/{f}" for k, f in rigspec.FOLDER.values())  # the two LeRobot reads for SO arms
ARCHIVE = "calibration-archive"  # next to the calibration folder, which LeRobot never reads


def archive_root(root: Path) -> Path:
    return root.parent / ARCHIVE


@dataclass
class CalFile:
    rel: str  # robots/so_follower/phi_follower.json
    used_by: list[str] = field(default_factory=list)  # rig arm keys whose file this is
    link: str | None = None  # a symlink's target, as written
    error: str | None = None  # why LeRobot could not load it
    unfinished: str | None = None  # identity.unfinished: registers no finished run leaves
    same_as: list[str] = field(default_factory=list)  # other files with exactly these numbers
    on_ports: list[str] = field(default_factory=list)  # ports whose motors hold it exactly
    mtime: float = 0.0
    junk: str | None = None  # why it is not a calibration at all: ._ file, backup

    @property
    def id(self) -> str:
        return Path(self.rel).name.removesuffix(".json")

    @property
    def folder(self) -> str:
        return str(Path(self.rel).parent)

    def public(self) -> dict[str, Any]:
        return {"rel": self.rel, "id": self.id, "folder": self.folder, "used_by": self.used_by,
                "link": self.link, "error": self.error, "unfinished": self.unfinished,
                "same_as": self.same_as, "on_ports": self.on_ports, "mtime": self.mtime,
                "junk": self.junk, "unused": not self.used_by}  # fmt: skip


def _used(spec: rigspec.RigSpec | None, root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in spec.arms if spec else ():
        p = a.calibration_path(root)
        if p is not None and not a.calibration_dir:  # a calibration_dir is outside this folder
            out.setdefault(str(p.relative_to(root)), []).append(a.key)
    return out


def inventory(root: Path, spec: rigspec.RigSpec | None,
              motors: dict[str, list[str]] | None = None) -> list[CalFile]:  # fmt: skip
    """Every file in the folders LeRobot reads. `motors`: port -> the kind/folder/id names its
    registers match exactly (detect.Found.matches), from the last Detect arms."""
    used = _used(spec, root)
    files: list[CalFile] = []
    numbers: dict[str, Calibration] = {}
    for folder in FOLDERS:
        d = root / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_dir():
                continue
            rel = f"{folder}/{p.name}"
            f = CalFile(rel, used_by=used.get(rel, []))
            try:
                f.mtime = p.lstat().st_mtime
            except OSError:
                pass
            if p.name.startswith("._"):
                f.junk = "a macOS metadata file, not a calibration"
            elif ".json.bak" in p.name or p.suffix != ".json":
                f.junk = "a backup or other file; LeRobot never reads it"
            if p.is_symlink():
                f.link = str(p.readlink())
            if f.junk is None:
                try:
                    cal = load_calibration(p)
                    numbers[rel] = cal
                    f.unfinished = unfinished(cal)
                except (OSError, ValueError, KeyError, TypeError) as e:
                    f.error = f"LeRobot cannot load it: {e}"
            files.append(f)
    by_rel = {f.rel: f for f in files}
    for rel, cal in numbers.items():
        f = by_rel[rel]
        f.same_as = [o for o, c in numbers.items() if o != rel and c == cal]
        name = rel.removesuffix(".json")
        f.on_ports = sorted(p for p, names in (motors or {}).items() if name in names)
    return files


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _fresh(root: Path, when: str | None) -> Path:
    """A new archive folder. WHY never reuse one: two moves in one second would put two versions
    of one file at one path, and the second would overwrite the first."""
    base = archive_root(root) / (when or stamp())
    d, n = base, 1
    while d.exists():
        n += 1
        d = base.with_name(f"{base.name}-{n}")
    return d


def backup(root: Path, path: Path) -> Path | None:
    """A copy of `path` in a new archive folder, before something overwrites it. WHY not
    <name>.bak next to it: those piled up in the folder LeRobot reads. A file outside `root` (a
    calibration_dir override) keeps the old sibling backup."""
    if not path.is_file():
        return None
    try:
        rel = path.relative_to(root)
    except ValueError:
        dest = path.with_name(f"{path.name}.bak-{stamp()}")
    else:
        dest = _fresh(root, None) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return dest


def archive(root: Path, rels: list[str], spec: rigspec.RigSpec | None,
            when: str | None = None) -> Path:  # fmt: skip
    """Moves `rels` into <archive>/<when>/, keeping their folders. Refuses a file an arm uses,
    or anything outside the folders LeRobot reads."""
    used = _used(spec, root)
    for rel in rels:
        if not any(rel.startswith(f"{f}/") for f in FOLDERS) or "/../" in f"/{rel}" or "\\" in rel:
            raise ValueError(f"{rel} is not in the calibration folder.")
        if rel in used:
            raise ValueError(f"{rel} is the file the {', '.join(used[rel])} uses.")
        p = root / rel
        if not (p.exists() or p.is_symlink()):
            raise ValueError(f"{rel} is not there any more.")
    dest = _fresh(root, when)
    for rel in rels:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root / rel), str(target))  # a link moves as a link
    return dest


def archives(root: Path) -> list[dict[str, Any]]:
    """Each archive folder, newest first, with the files it holds."""
    a = archive_root(root)
    out = []
    for d in sorted((p for p in a.iterdir() if p.is_dir()), reverse=True) if a.is_dir() else []:
        rels = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() or p.is_symlink())
        out.append({"name": d.name, "files": rels})
    return out


def restore(root: Path, name: str) -> dict[str, Any]:
    """Moves an archive's files back. A file there now goes to a new archive first, so a restore
    can be undone too."""
    d = archive_root(root) / name
    if "/" in name or name.startswith(".") or not d.is_dir():
        raise ValueError(f"There is no archive {name}.")
    rels = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() or p.is_symlink())
    clash = [r for r in rels if (root / r).exists() or (root / r).is_symlink()]
    aside = None
    if clash:
        aside = _fresh(root, None)
        for r in clash:
            (aside / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / r), str(aside / r))
    for r in rels:
        (root / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d / r), str(root / r))
    shutil.rmtree(d, ignore_errors=True)
    return {"restored": rels, "archived": clash, "archive": aside.name if aside else None}


def shared(source: Path, root: Path) -> list[dict[str, Any]]:
    """Each file of a shared set against this Mac's copy: new, same, or how far it differs."""
    out = []
    for folder in FOLDERS:
        d = source / folder
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            if p.name.startswith("._"):
                continue
            rel = f"{folder}/{p.name}"
            row: dict[str, Any] = {"rel": rel, "id": p.stem, "state": "new", "max_deg": None,
                                   "worst_joint": None, "error": None}  # fmt: skip
            try:
                theirs = load_calibration(p)
            except (OSError, ValueError, KeyError, TypeError) as e:
                row |= {"state": "broken", "error": str(e)}
                out.append(row)
                continue
            local = root / rel
            if local.is_symlink():
                row["state"] = "link"  # installing replaces the link with the shared file
            elif local.is_file():
                try:
                    d_ = match_fingerprint(load_calibration(local), {"x": theirs})[0].distance
                    row |= {"state": "same" if d_.exact else "differs",
                            "max_deg": round(d_.max_deg, 1), "worst_joint": d_.worst_joint}
                except (OSError, ValueError, KeyError, TypeError) as e:
                    row |= {"state": "differs", "error": f"this Mac's copy cannot be read: {e}"}
            out.append(row)
    return out


def install(source: Path, root: Path, rels: list[str], when: str | None = None) -> dict[str, Any]:
    """Copies `rels` from the shared set. A file or link it replaces goes to the archive first,
    so Restore brings it back."""
    rows = {r["rel"]: r for r in shared(source, root)}
    for rel in rels:
        if rel not in rows or rows[rel]["state"] == "broken":
            raise ValueError(f"{rel} is not a calibration file in {source}.")
    replaced = [r for r in rels if rows[r]["state"] in ("differs", "link")]
    moved = None
    if replaced:
        moved = _fresh(root, when)
        for rel in replaced:
            (moved / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / rel), str(moved / rel))
    copied = []
    for rel in rels:
        if rows[rel]["state"] == "same":
            continue
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, root / rel)
        copied.append(rel)
    return {"copied": copied, "archived": replaced, "archive": moved.name if moved else None}


def shared_candidates(rig_dir: Path | None) -> list[Path]:
    """Where a shared set may be: the phi repo's configs/calibration (phi 00d92de)."""
    out = []
    for base in (rig_dir, Path.home() / "phi"):
        if base is not None and (Path(base) / "configs" / "calibration").is_dir():
            p = (Path(base) / "configs" / "calibration").resolve()
            if p not in out:
                out.append(p)
    return out
