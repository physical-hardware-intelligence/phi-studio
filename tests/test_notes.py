"""The notes store: create, update, validate, filter, dismiss flags, and the training exclusion
list."""

from __future__ import annotations

import pytest

from phi_studio.errors import Refusal
from phi_studio.notes import NoteStore


@pytest.fixture
def store(tmp_path):
    s = NoteStore(tmp_path / "studio.db")
    yield s
    s.close()


def test_create_update_list_delete(store) -> None:
    n = store.save(
        {
            "dataset": "abc",
            "episode": 3,
            "kind": "issue",
            "severity": "warn",
            "text": " gripper slips ",
            "t0": 5.5,
            "t1": 2.0,
            "tags": ["grip", "grip"],
        },
        author="yash",
    )
    assert n["text"] == "gripper slips" and n["author"] == "yash" and n["tags"] == ["grip"]
    assert (n["t0"], n["t1"]) == (2.0, 5.5)  # a backwards range is put in order
    u = store.save({**n, "status": "resolved"})
    assert u["id"] == n["id"] and u["status"] == "resolved" and u["updated"] >= n["updated"]
    store.save({"dataset": "abc", "episode": 4, "kind": "note", "text": "ok"})
    store.save({"dataset": "zzz", "kind": "note", "text": "other"})
    assert len(store.list("abc")) == 2 and len(store.list("abc", 3)) == 1
    assert len(store.list(status="resolved")) == 1 and len(store.list()) == 3
    store.delete(n["id"])
    assert len(store.list("abc")) == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "note", "text": "x"},  # no dataset
        {"dataset": "a", "kind": "rumour", "text": "x"},  # unknown kind
        {"dataset": "a", "kind": "note", "text": ""},  # empty note
        {"dataset": "a", "kind": "note", "text": "x", "t0": "soon"},  # not a number
        {"dataset": "a", "kind": "note", "text": "x", "t0": True},  # a bool is not a time
        {"dataset": "a", "kind": "note", "text": "x" * 5000},  # too long
        {"dataset": "a", "kind": "note", "text": "x", "tags": "grip"},  # tags must be a list
    ],
)
def test_refusals(store, bad) -> None:
    with pytest.raises(Refusal):
        store.save(bad)


def test_update_of_a_deleted_note(store) -> None:
    n = store.save({"dataset": "a", "kind": "note", "text": "x"})
    store.delete(n["id"])
    with pytest.raises(Refusal):
        store.save({**n, "text": "y"})


def test_bad_and_good_need_no_text_and_bad_excludes(store) -> None:
    store.save({"dataset": "a", "episode": 7, "kind": "bad"})
    store.save({"dataset": "a", "episode": 2, "kind": "bad"})
    store.save({"dataset": "a", "episode": 2, "kind": "bad"})  # twice: still one exclusion
    store.save({"dataset": "a", "episode": 1, "kind": "good"})
    r = store.save({"dataset": "a", "episode": 9, "kind": "bad"})
    store.save({**r, "status": "resolved"})  # a resolved exclusion no longer excludes
    assert store.excluded("a") == [2, 7] and store.excluded("b") == []


def test_dismissals(store) -> None:
    flag = {"kind": "tracking", "arm": None, "joint": "wrist_roll", "t0": 2.9}
    key = NoteStore.flag_key(flag)
    store.dismiss("a", 44, "tracking", key, author="yash")
    store.dismiss("a", 44, "tracking", key)  # idempotent
    assert store.dismissed("a") == {(44, "tracking", key)}
    store.dismiss("a", 44, "tracking", key, undo=True)
    assert store.dismissed("a") == set()


def test_survives_reopen(tmp_path) -> None:
    s = NoteStore(tmp_path / "db.sqlite")
    s.save({"dataset": "a", "kind": "note", "text": "kept"})
    s.close()
    s2 = NoteStore(tmp_path / "db.sqlite")
    assert s2.list("a")[0]["text"] == "kept"
    s2.close()
