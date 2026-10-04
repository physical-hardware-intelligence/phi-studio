"""Comment-preserving edits to robot-config.yaml: set single values at key paths, nothing more.

PyYAML loses comments on a round trip and ruamel is not installed, so Studio edits the text.
PyYAML's composer gives every node its exact start and end offset in the file, so an edit replaces
only the characters of one value, or inserts one new line; every other byte (comments, blank
lines, key order) stays as written. Anything this cannot do safely is refused with a
ConfigEditError, and every result is parsed again and compared with the intended data before it
is returned.
"""

from __future__ import annotations

import copy
import math
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from phi_studio.rigspec import CAMERA_SOURCE, ArmSpec, CameraSpec, RigSpec

KeyPath = tuple[str, ...]
STR_TAG = "tag:yaml.org,2002:str"
MERGE_TAG = "tag:yaml.org,2002:merge"
_COMMENT_GAP = re.compile(r" +(?=#)")  # the spaces between a value and its trailing comment


class ConfigEditError(ValueError):
    """robot-config.yaml cannot be edited safely; the message says why in plain English."""


def _emit(value: Any, name: str) -> str:
    """`value` as YAML text that reads back as the same value and type, in a block mapping or
    inside a flow mapping such as `{type: opencv, index_or_path: ...}`."""
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ConfigEditError(f"{name}: Studio writes text, numbers, true/false or null here, "
                              f"not a {type(value).__name__}.")  # fmt: skip
    if isinstance(value, str) and len(value.splitlines()) > 1:
        raise ConfigEditError(f"{name}: Studio writes one-line values only.")
    # WHY dump inside a flow list: PyYAML then quotes whatever a flow mapping would misread
    # (commas, braces, `: `) and whatever would read back as another type ('8', 'true'); text
    # that is safe inside a flow mapping is safe in a block mapping too.
    out = yaml.safe_dump([value], default_flow_style=True, width=math.inf, allow_unicode=True)
    out = out.strip()
    if not (out.startswith("[") and out.endswith("]")) or len(out.splitlines()) > 1:
        raise ConfigEditError(f"{name}: Studio cannot write {value!r} on one line.")
    return out[1:-1]


def _no_anchor(node: Any, name: str, anchored: set[int]) -> None:
    if node.start_mark.index in anchored:
        raise ConfigEditError(f"{name} goes through a YAML anchor or alias (& or *), so one edit "
                              "could change several places. Edit it by hand.")  # fmt: skip


def _child(node: Any, key: str, name: str) -> tuple[Any, Any] | None:
    """The (key node, value node) pair for `key` in a mapping node, or None."""
    if any(k.tag == MERGE_TAG for k, _ in node.value):
        raise ConfigEditError(f"{name} goes through a section with a YAML merge key (<<), so its "
                              "values may come from elsewhere. Edit it by hand.")  # fmt: skip
    hits = [(k, v) for k, v in node.value
            if isinstance(k, yaml.ScalarNode) and k.tag == STR_TAG and k.value == key]  # fmt: skip
    if len(hits) > 1:
        raise ConfigEditError(f"{name}: {key} appears twice in the same section and YAML keeps "
                              "only the last one. Fix the file by hand.")  # fmt: skip
    return hits[0] if hits else None


def _last_offset(node: Any) -> int:
    """Where the last value inside a block collection ends. WHY not the node's own end: a block
    mapping ends at the next token, past any comments and blank lines that follow it."""
    blocks = (yaml.MappingNode, yaml.SequenceNode)
    while isinstance(node, blocks) and not node.flow_style and node.value:
        node = node.value[-1][1] if isinstance(node, yaml.MappingNode) else node.value[-1]
    return int(node.end_mark.index)


def _insert(text: str, parent: Any, key: str, value: Any, name: str) -> tuple[int, int, str]:
    """An edit that adds `key: value` as the last entry of an existing mapping."""
    entry = f"{_emit(key, name)}: {_emit(value, name)}"
    if parent.flow_style:
        if parent.value:
            at = parent.value[-1][1].end_mark.index
            return at, at, ", " + entry
        at = parent.end_mark.index - 1  # inside the closing brace of {}
        return at, at, entry
    indent = " " * parent.value[0][0].start_mark.column  # a block mapping has at least one key
    nl = "\r\n" if "\r\n" in text else "\n"
    at = _last_offset(parent)
    if text[at - 1 : at] not in ("\n", "\r"):  # WHY: a block scalar already ends at a line start
        eol = text.find("\n", at)
        if eol == -1:
            return len(text), len(text), nl + indent + entry + nl
        at = eol + 1
    return at, at, indent + entry + nl


def _plan(
    text: str, root: Any, path: KeyPath, value: Any, anchored: set[int]
) -> tuple[int, int, str]:
    """The edit for one change as (start, end, replacement), offsets into `text`."""
    name = ".".join(path)
    node: Any = root
    parent: Any = None
    for depth, key in enumerate(path):
        here = ".".join(path[:depth])
        if isinstance(node, yaml.SequenceNode):
            raise ConfigEditError(f"{name}: {here} is a list, and Studio does not edit inside "
                                  "lists.")  # fmt: skip
        if not isinstance(node, yaml.MappingNode):
            raise ConfigEditError(f"{name}: robot-config.yaml has no {here} section to put it in. "
                                  "Add the section by hand first.")  # fmt: skip
        _no_anchor(node, name, anchored)
        hit = _child(node, key, name)
        if hit is None:
            if depth == len(path) - 1:
                return _insert(text, node, key, value, name)
            missing = ".".join(path[: depth + 1])
            raise ConfigEditError(f"{name}: robot-config.yaml has no {missing} section to put it "
                                  "in. Add the section by hand first.")  # fmt: skip
        _no_anchor(hit[0], name, anchored)
        parent, node = node, hit[1]
    _no_anchor(node, name, anchored)
    if not isinstance(node, yaml.ScalarNode):
        raise ConfigEditError(f"{name} is a section or a list, not a single value; Studio sets "
                              "single values only.")  # fmt: skip
    if node.style in ("|", ">") or node.start_mark.line != node.end_mark.line:
        raise ConfigEditError(f"{name} spans more than one line; Studio edits one-line values "
                              "only. Edit it by hand.")  # fmt: skip
    new = _emit(value, name)
    start, end = node.start_mark.index, node.end_mark.index
    if start == end and text[start - 1 : start] == ":":
        new = " " + new  # an empty value: `port:` becomes `port: <value>`
    gap = None if parent.flow_style else _COMMENT_GAP.match(text, end)
    if gap:
        # WHY: keep a trailing comment in its column, so a column of aligned comments stays aligned.
        pad = len(gap.group()) - (len(new) - (end - start))
        new, end = new + " " * max(pad, 1), gap.end()
    return start, end, new


def _same(a: Any, b: Any) -> bool:
    """Equal data with equal types. WHY not ==: in Python 1 == True and 1 == 1.0, and a port
    that turned into a number must count as a change."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(map(_same, a, b))
    return bool(a == b or (a != a and b != b))  # NaN reads back as NaN


def set_values(text: str, changes: Mapping[KeyPath, Any]) -> str:
    """`text` with each key path set to its value and every other byte unchanged. A key missing
    from an existing mapping is added at its end. Raises ConfigEditError for anything it cannot
    edit safely, and when the result does not read back as the original data with exactly these
    changes applied."""
    if "\t" in text:
        line = text[: text.index("\t")].count("\n") + 1
        raise ConfigEditError(f"robot-config.yaml has a tab on line {line}; Studio edits only "
                              "files that use spaces. Replace the tab with spaces.")  # fmt: skip
    try:
        before = yaml.safe_load(text)
        root = yaml.compose(text)
        # WHY start offsets: an alias composes to the anchored node itself, so this one set
        # catches both ends of an anchor/alias pair.
        anchored = {e.start_mark.index for e in yaml.parse(text)
                    if isinstance(e, yaml.NodeEvent) and e.anchor}  # fmt: skip
    except yaml.YAMLError as e:
        raise ConfigEditError(f"robot-config.yaml is not valid YAML: {e}") from e
    if not isinstance(root, yaml.MappingNode) or not isinstance(before, dict):
        raise ConfigEditError("robot-config.yaml is not a mapping of sections, so Studio cannot "
                              "edit it.")  # fmt: skip
    edits = []
    for seq, (path, value) in enumerate(changes.items()):
        if not path or not all(isinstance(k, str) and k for k in path):
            raise ConfigEditError(f"{path!r} is not a key path.")
        start, end, new = _plan(text, root, tuple(path), value, anchored)
        # WHY this order at one offset: deeper keys first, then as requested, so two keys added
        # after the same line keep their nesting and the caller's order.
        edits.append(((start, -len(path), seq), end, new))
    edits.sort()
    for prev, cur in zip(edits, edits[1:], strict=False):
        if prev[1] > cur[0][0]:
            raise ConfigEditError("Two of the changes touch the same place in robot-config.yaml; "
                                  "make them one at a time.")  # fmt: skip
    out = text
    for (start, _, _), end, new in reversed(edits):
        out = out[:start] + new + out[end:]
    want = copy.deepcopy(before)
    for path, value in changes.items():
        d = want
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    try:
        after = yaml.safe_load(out)
    except yaml.YAMLError:
        after = None
    if not _same(after, want):
        raise ConfigEditError("Studio's edit did not read back as intended, so robot-config.yaml "
                              "was left as it was. Edit it by hand.")  # fmt: skip
    return out


def write_values(path: Path, changes: Mapping[KeyPath, Any], backup_dir: Path) -> Path:
    """Apply `changes` to the file at `path` (see set_values) and return the backup of the
    original. Order: edit in memory, back up, then replace atomically, so a refused edit writes
    nothing and a crash leaves the old file or the new one, never half of one."""
    path = path.resolve()  # WHY: os.replace on a symlink would swap the link for a plain file
    raw = path.read_bytes()
    try:
        # WHY bytes, not read_text: text mode turns \r\n into \n and would rewrite every line.
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ConfigEditError(f"{path.name} is not UTF-8 text, so Studio cannot edit it.") from e
    new = set_values(text, changes)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.name}.{datetime.now():%Y%m%d-%H%M%S-%f}.bak"
    backup.write_bytes(raw)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, path.stat().st_mode & 0o7777)  # WHY: mkstemp creates the file as 0600
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return backup


# -- paths for rigspec's objects -----------------------------------------------------------------


def port_path(arm: ArmSpec) -> KeyPath:
    """Where `arm`'s port lives, the way LeRobot 0.6.0 names it: `<section>.port`, or
    `<section>.<side>_arm_config.port` on a bimanual rig (config_bi_so_follower.py:30-31). An old
    flat `left_arm_port` is not written to; set_values refuses the missing section instead."""
    if arm.side:
        return (arm.section, f"{arm.side}_arm_config", "port")
    return (arm.section, "port")


def camera_source_path(cam: CameraSpec, spec: RigSpec) -> KeyPath:
    """Where the field that picks `cam`'s device lives (CAMERA_SOURCE: index_or_path for opencv,
    serial_number_or_name for intelrealsense). WHY `spec`: a camera with no side comes from
    robot.cameras or, when that holds none, from the top-level cameras: block (rigspec.parse), and
    only the parsed robot section says which."""
    field = CAMERA_SOURCE.get(str(cam.fields.get("type")))
    if field is None:
        kinds = " and ".join(CAMERA_SOURCE)
        raise ConfigEditError(f"Camera {cam.key} has type {cam.fields.get('type')!r}; Studio "
                              f"knows which field picks the device only for {kinds} "
                              "cameras.")  # fmt: skip
    if cam.side:
        return ("robot", f"{cam.side}_arm_config", "cameras", cam.key, field)
    on_robot = spec.robot.get("cameras")
    if isinstance(on_robot, dict) and any(isinstance(v, dict) for v in on_robot.values()):
        return ("robot", "cameras", cam.key, field)
    return ("cameras", cam.key, field)
