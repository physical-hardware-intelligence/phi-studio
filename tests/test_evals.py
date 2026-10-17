"""Evals (evals.py): task cards locked at the start, a seeded blind schedule of bundles, milestone
scores, rig-fault voids, the numbers, and every change on disk before the call returns."""

from __future__ import annotations

import json
import math
from itertools import count
from pathlib import Path

import pytest

from phi_studio import evals as ev
from phi_studio.evals import EvalError, EvalStore, wilson

CATALOG = {"act-a": "ACT recovery", "act-b": "ACT position holdout", "dp": "Diffusion"}
_ids = count(1)


def card_raw(**kw) -> dict:
    c = ev.blank_card()
    c.update(name="Cube to bin", task="Put the red cube in the white bin",
             success="The cube ends inside the bin", limit_s=20.0,
             conditions=[{"label": "Trained spot", "axis": "train", "control": True},
                         {"label": "Spot 4 (held out)", "axis": "held-out position"},
                         {"label": "Spot 5 (held out)", "axis": "held-out position"}])  # fmt: skip
    c.update(kw)
    return c


def store(tmp: Path, clock: list[float] | None = None) -> EvalStore:
    t = clock or [1000.0]

    def now() -> float:
        t[0] += 1.0
        return t[0]

    return EvalStore(tmp, now=now)


def started(tmp: Path, policies=("act-a", "act-b"), reps=2, **kw) -> EvalStore:
    s = store(tmp)
    card = s.save_card(card_raw())
    s.begin(card["id"], list(policies), reps, kw.pop("blind", True), kw.pop("alpha", 0.05),
            CATALOG, seed=kw.pop("seed", 42), **kw)  # fmt: skip
    return s


def ran(s: EvalStore, rid: str | None = None, **kw) -> dict:
    """The worker's view of an ended run of the next trial's policy (worker.PolicyRun.view)."""
    rec = s.current
    assert rec is not None
    slot = ev.next_slot(rec)
    assert slot is not None
    pol = ev.policy_of(rec, slot["alias"])
    return {"run_id": rid or f"run{next(_ids)}", "id": pol["id"], "task": rec["card"]["task"],
            "limit_s": rec["card"]["limit_s"], "started_at": rec["started_at"] + 0.5,
            "running": False, "episode_s": 6.0, "chunk": 10, "chunk_ms_p50": 3.0,
            "chunk_ms_max": 5.0, "step": 180, "clipped": 0, "ended": "time limit",
            **kw}  # fmt: skip


def score(s: EvalStore, stage: int, failures=(), **kw) -> dict:
    run = ran(s, **kw)
    return s.score(stage, list(failures), "", run, run["run_id"])


# -- cards -----------------------------------------------------------------------------------------
def test_a_card_is_checked_and_keeps_condition_ids_across_edits(tmp_path: Path) -> None:
    s = store(tmp_path)
    c = s.save_card(card_raw())
    assert [x["id"] for x in c["conditions"]] == ["c1", "c2", "c3"]
    assert c["rubric"][-1]["points"] == 1.0 and c["conditions"][0]["control"] is True
    raw = {**c, "conditions": [c["conditions"][2], {"label": "New spot", "axis": "distractors"}]}
    c2 = s.save_card(raw)
    assert [x["id"] for x in c2["conditions"]] == ["c3", "c1"]  # c3 kept, the new one takes c1


@pytest.mark.parametrize("patch,word", [
    ({"name": ""}, "Name"),
    ({"rubric": [{"label": "a", "points": 0.5}, {"label": "b", "points": 0.4}]}, "more than"),
    ({"rubric": [{"label": "a", "points": 0.5}]}, "worth 1"),
    ({"failures": ["x", "x"]}, "different"),
    ({"conditions": []}, "1 to"),
    ({"conditions": [{"label": "same"}, {"label": "same"}]}, "both called"),
    ({"conditions": [{"label": "a", "axis": "vibes"}]}, "pick what it tests"),
    ({"limit_s": 0}, "time limit"),
])  # fmt: skip
def test_bad_cards_are_refused_in_plain_words(tmp_path: Path, patch: dict, word: str) -> None:
    with pytest.raises(EvalError, match=word):
        store(tmp_path).save_card(card_raw(**patch))


def test_starting_an_eval_locks_its_card_and_a_copy_is_editable(tmp_path: Path) -> None:
    s = started(tmp_path)
    cid = s.current["card"]["id"]  # type: ignore[index]
    assert s.card(cid)["locked_at"]  # type: ignore[index]
    with pytest.raises(EvalError, match="locked"):
        s.save_card({**s.card(cid), "name": "Changed"})  # type: ignore[dict-item]
    dup = s.duplicate_card(cid)
    assert dup["id"] != cid and not dup["locked_at"] and dup["name"].endswith("(copy)")
    s.save_card({**dup, "name": "Changed"})


def test_reference_pictures_can_be_added_after_lock_but_not_replaced(tmp_path: Path) -> None:
    s = store(tmp_path)
    c = s.save_card(card_raw())
    name = s.save_ref(c["id"], "c1", "top", b"\xff\xd8jpeg")
    assert s.ref_path(c["id"], name).read_bytes() == b"\xff\xd8jpeg"  # type: ignore[union-attr]
    s.begin(c["id"], ["act-a"], 1, False, 0.05, CATALOG, seed=1)
    s.save_ref(c["id"], "c2", "top", b"\xff\xd8two")  # c2 had none: allowed
    with pytest.raises(EvalError, match="already has"):
        s.save_ref(c["id"], "c1", "top", b"\xff\xd8new")
    assert s.ref_path(c["id"], "../x.jpg") is None


# -- the schedule ----------------------------------------------------------------------------------
def test_the_schedule_is_seeded_balanced_and_opens_with_the_controls(tmp_path: Path) -> None:
    conds = ev.check_card(card_raw())["conditions"]
    sched = ev.make_schedule(conds, ["A", "B", "C"], 4, False, __import__("random").Random(7))
    again = ev.make_schedule(conds, ["A", "B", "C"], 4, False, __import__("random").Random(7))
    assert sched == again and len(sched) == 12
    assert sched[0]["c"] == "c1"  # the trained-on control first: a failure there points at the rig
    assert all(sorted(b["order"]) == ["A", "B", "C"] for b in sched)
    assert {b["c"] for b in sched[:3]} == {"c1", "c2", "c3"}  # each round holds every condition
    assert len({tuple(b["order"]) for b in sched}) > 1  # the order within a bundle varies
    grouped = ev.make_schedule(conds, ["A", "B"], 3, True, __import__("random").Random(7))
    assert [b["c"] for b in grouped][:3] == ["c1"] * 3


def test_begin_refuses_bad_requests(tmp_path: Path) -> None:
    s = store(tmp_path)
    c = s.save_card(card_raw())
    for args, word in [((c["id"], [], 2, True, 0.05), "Choose 1"),
                       ((c["id"], ["act-a", "act-a"], 2, True, 0.05), "once"),
                       ((c["id"], ["nope"], 2, True, 0.05), "can run on this rig"),
                       ((c["id"], ["act-a"], 0, True, 0.05), "Bundles"),
                       ((c["id"], ["act-a"], 2, True, 0.5), "false-alarm"),
                       (("missing", ["act-a"], 2, True, 0.05), "task card")]:  # fmt: skip
        with pytest.raises(EvalError, match=word):
            s.begin(*args, CATALOG)


def test_a_blind_eval_shows_letters_until_it_ends(tmp_path: Path) -> None:
    s = started(tmp_path, stamps={"policies": {"act-a": "sha"}, "studio": "x"})
    pub = ev.public(s.current)  # type: ignore[arg-type]
    assert pub["policies"] == [{"alias": "A"}, {"alias": "B"}]
    assert "policies" not in pub["stamps"] and pub["stamps"]["studio"] == "x"
    assert "ACT" not in ev.to_csv(s.current)  # type: ignore[arg-type]
    rec = s.end()
    assert {p["id"] for p in ev.public(rec)["policies"]} == {"act-a", "act-b"}


# -- scoring ---------------------------------------------------------------------------------------
def test_every_score_is_on_disk_and_a_restart_resumes(tmp_path: Path) -> None:
    s = started(tmp_path)
    score(s, 5)
    rid = s.current["id"]  # type: ignore[index]
    on_disk = json.loads((tmp_path / "evals" / f"{rid}.json").read_text())
    assert len(on_disk["trials"]) == 1 and on_disk["trials"][0]["success"] is True
    again = EvalStore(tmp_path)
    assert again.current is not None and again.current["id"] == rid
    assert ev.next_slot(again.current) == ev.next_slot(s.current)  # type: ignore[arg-type]


def test_scores_follow_the_schedule_and_milestones(tmp_path: Path) -> None:
    s = started(tmp_path)
    first = ev.next_slot(s.current)  # type: ignore[arg-type]
    rec = score(s, 2, ["Slipped or dropped"])
    t = rec["trials"][0]
    assert (t["b"], t["alias"], t["c"]) == (first["b"], first["alias"], first["c"])  # type: ignore[index]
    assert t["score"] == 0.4 and not t["success"] and t["failures"] == ["Slipped or dropped"]
    assert ev.next_slot(rec) != first


def test_only_this_trials_run_can_be_scored_and_only_once(tmp_path: Path) -> None:
    s = started(tmp_path)
    run = ran(s)
    with pytest.raises(EvalError, match="not this trial"):
        s.score(5, [], "", {**run, "id": "dp"}, run["run_id"])
    with pytest.raises(EvalError, match="before this eval"):
        s.score(5, [], "", {**run, "started_at": 0.0}, run["run_id"])
    with pytest.raises(EvalError, match="still running"):
        s.score(5, [], "", {**run, "running": True}, run["run_id"])
    with pytest.raises(EvalError, match="newer run"):
        s.score(5, [], "", run, "someone-else")
    with pytest.raises(EvalError, match="from the card"):
        s.score(2, ["Made up"], "", run, run["run_id"])
    s.score(5, [], "", run, run["run_id"])
    assert not s.awaiting(run)
    with pytest.raises(EvalError, match="already scored|not this trial"):
        s.score(5, [], "", run, run["run_id"])


def test_awaiting_is_true_only_for_an_ended_unscored_run_of_the_next_trial(tmp_path: Path) -> None:
    s = started(tmp_path)
    run = ran(s)
    assert s.awaiting(run) and not s.awaiting({**run, "running": True})
    assert not s.awaiting({**run, "id": "dp"}) and not s.awaiting(None)


def test_a_voided_trial_runs_again_and_counts_for_nothing(tmp_path: Path) -> None:
    s = started(tmp_path)
    slot = ev.next_slot(s.current)  # type: ignore[arg-type]
    rec = score(s, 0)
    rec = s.void(1, "Servo overload on the gripper")
    again = ev.next_slot(rec)
    assert again is not None and (again["b"], again["alias"]) == (slot["b"], slot["alias"])  # type: ignore[index]
    assert again["repeat"] is True
    assert ev.summarize(rec)["valid_trials"] == 0 and ev.summarize(rec)["voided"] == 1
    rec = score(s, 5)  # the repeat
    assert rec["trials"][1]["b"] == slot["b"] and ev.summarize(rec)["valid_trials"] == 1  # type: ignore[index]
    with pytest.raises(EvalError, match="already void"):
        s.void(1, "twice")


def test_skip_and_undo(tmp_path: Path) -> None:
    s = started(tmp_path)
    first = ev.next_slot(s.current)  # type: ignore[arg-type]
    s.skip("The bin was missing")
    assert ev.next_slot(s.current) != first  # type: ignore[arg-type]
    rec = s.undo()  # undoes the skip
    assert ev.next_slot(rec) == first and not rec["skipped"]
    score(s, 3)
    s.void(1, "camera")
    rec = s.undo()  # undoes the void, keeps the score
    assert len(rec["trials"]) == 1 and rec["trials"][0]["void"] is None
    rec = s.undo()  # undoes the score
    assert rec["trials"] == []
    with pytest.raises(EvalError, match="nothing to undo"):
        s.undo()


def test_a_failed_save_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    s = started(tmp_path)
    before = json.dumps(s.current)

    def boom(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(EvalStore, "_write", staticmethod(boom))
    with pytest.raises(OSError):
        score(s, 5)
    assert json.dumps(s.current) == before


# -- the numbers -----------------------------------------------------------------------------------
def play(s: EvalStore, wins: dict[str, int]) -> dict:
    """Run every remaining trial: policy id -> milestone it always reaches."""
    rec = s.current
    while (slot := ev.next_slot(rec)) is not None:  # type: ignore[arg-type]
        pid = ev.policy_of(rec, slot["alias"])["id"]  # type: ignore[arg-type]
        rec = score(s, wins[pid])
    return rec  # type: ignore[return-value]


def test_a_clear_winner_is_decided_and_named_after_the_eval(tmp_path: Path) -> None:
    s = started(tmp_path, reps=6)  # 3 conditions x 6 = 18 bundles
    rec = play(s, {"act-a": 5, "act-b": 1})
    summ = ev.summarize(rec)
    pair = summ["pairs"][0]
    winner = next(p["alias"] for p in rec["policies"] if p["id"] == "act-a")
    assert pair["decision"]["state"] == ("a" if pair["a"] == winner else "b")
    assert pair["decision"]["at_bundle"] <= 18
    assert summ["policies"][winner]["rate"] == 1.0 and summ["complete"]
    assert summ["policies"][winner]["by_axis"]["held-out position"]["n"] == 12
    md = ev.to_markdown(s.end())
    assert "ACT recovery" in md and "decided" in md and "cannot show" in md


def test_equal_policies_stay_undecided_with_an_estimate(tmp_path: Path) -> None:
    s = started(tmp_path, reps=4)
    rec = s.current
    flip = 0
    while ev.next_slot(rec) is not None and len(rec["trials"]) < 10:  # type: ignore[index, arg-type]
        flip += 1
        rec = score(s, 5 if flip % 3 else 0)
    summ = ev.summarize(rec)
    assert summ["pairs"][0]["decision"]["state"] == "continue"
    assert 0.0 <= summ["pairs"][0]["p_a_better"] <= 1.0


def test_three_policies_get_letters(tmp_path: Path) -> None:
    s = started(tmp_path, policies=("act-a", "act-b", "dp"), reps=8)
    rec = play(s, {"act-a": 5, "act-b": 0, "dp": 5})
    summ = ev.summarize(rec)
    names = {p["alias"]: p["id"] for p in rec["policies"]}
    by_id = {names[a]: lt for a, lt in summ["letters"].items()}
    assert by_id["act-a"] == by_id["dp"] != by_id["act-b"]
    assert summ["comparisons"] == 3


# -- what the rig did ------------------------------------------------------------------------------
def tel(run_id: str, running=True, episode_s=2.0, faults=(), load=10.0, hz=30.0, p99=12.0) -> dict:
    return {"policy": {"run_id": run_id, "running": running, "episode_s": episode_s},
            "loop": {"hz": hz, "p99_ms": p99},
            "arms": {"follower": {"role": "follower", "health": {
                "gripper": {"load": -load, "temp": 40.0, "volt": 11.9, "faults": list(faults)}}},
                     "leader": {"role": "leader", "health": {
                "gripper": {"load": 99.0, "temp": 70.0, "volt": 4.0,
                            "faults": ["overload"]}}}}}  # fmt: skip


def test_run_health_gathers_the_followers_readings_during_the_run_only() -> None:
    h = ev.RunHealth()
    h.feed(tel("r1", load=30.0))
    h.feed(tel("r1", load=55.0, faults=["overload"]))
    h.camera("wrist", False)
    h.feed(tel("r1", running=False, load=99.0, faults=["overheat"]))  # after the end: ignored
    h.camera("top", False)
    s = h.summary("r1")
    assert s["max_load_pct"] == {"follower": 55.0}
    assert s["faults"] == {"follower": {"gripper": ["overload"]}}
    assert s["cameras_lost"] == ["wrist"] and h.summary("other") == {}
    h.feed(tel("r2", episode_s=0.5, hz=5.0))  # first second: the loop rate is still settling
    assert h.summary("r2")["min_hz"] is None and h.summary("r2")["faults"] == {}


def test_suspect_names_rig_faults_and_policy_warnings() -> None:
    health = {"faults": {"follower": {"gripper": ["overload"]}}, "cameras_lost": ["top"],
              "min_hz": 12.0, "max_p99_ms": 250.0}  # fmt: skip
    metrics = {"ended": "heartbeat", "chunk": 10, "chunk_ms_max": 900.0, "step": 100,
               "clipped": 40, "boundary_jump_deg": 69.6}  # fmt: skip
    out = ev.suspect(health, metrics, 30.0)
    joined = " ".join(out["void"])
    for w in ("overload", "Camera top", "slowed to 12", "stalled", "lost its link"):
        assert w in joined
    assert len(out["warn"]) == 3 and "70°" in out["warn"][2]
    assert ev.suspect({}, {"ended": "time limit"}, 30.0) == {"void": [], "warn": []}
    assert ev.suspect({}, {"ended": "user"}, 30.0)["void"] == []  # a safety stop is the policy's


def test_a_scored_trial_carries_its_health_and_suggested_void(tmp_path: Path) -> None:
    s = started(tmp_path)
    run = ran(s, ended="disconnected")
    rec = s.score(1, [], "", run, run["run_id"], health={"faults": {}, "cameras_lost": ["wrist"]})
    t = rec["trials"][0]
    assert any("wrist" in r for r in t["suspect"]["void"])
    assert t["metrics"]["ended"] == "disconnected"
    assert t["void"] is None  # suggested, never applied without the person


# -- import, export, listing -----------------------------------------------------------------------
CLUB = """timestamp,model,episode,split,object,container,stage,score,grasp_approach,hesitated,notes
2026-08-12T10:00:00+00:00,u/act_rec,0,heldout,red cube (25mm),white bin,5,1.0,top,n,
2026-08-12T10:05:00+00:00,u/act_rec,1,heldout,red cube (25mm),white bin,1,0.2,side,y,tapped it
2026-08-12T10:10:00+00:00,u/dp,0,heldout,red cube (25mm),white bin,1,0.2,top,n,
2026-08-12T10:15:00+00:00,u/dp,5,train,red cube (25mm),cardboard box,5,1.0,top,n,
2026-08-12T10:20:00+00:00,u/dp,6,train,red cube (25mm),cardboard box,9,0.55,top,n,odd score
"""


def test_the_clubs_csv_imports_as_an_unpaired_ended_eval(tmp_path: Path) -> None:
    f = tmp_path / "rollout_scores.csv"
    f.write_text(CLUB)
    s = store(tmp_path)
    rec = s.import_club(f)
    assert rec["imported"] == {"from": str(f), "rows": 4, "kept": 4}  # the 0.55 row is left out
    assert rec["paired"] is False and rec["ended_at"] and s.current is None
    summ = ev.summarize(rec)
    assert not summ["paired"] and summ["pairs"][0]["decision"] is None
    assert {c["axis"] for c in rec["card"]["conditions"]} == {"held-out position", "train"}
    tapped = next(t for t in rec["trials"] if "tapped" in t["note"])
    assert tapped["stage"] == 1 and "Hesitated or oscillated" in tapped["failures"]
    assert s.card(rec["card"]["id"])["locked_at"]  # type: ignore[index]
    listed = s.past()
    assert listed[0]["imported"]
    assert {r["policy"] for r in listed[0]["results"]} == {"u/act_rec", "u/dp"}
    bad = tmp_path / "other.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(EvalError, match="not the club"):
        s.import_club(bad)


def test_a_one_policy_report_does_not_mention_comparisons(tmp_path: Path) -> None:
    s = started(tmp_path, policies=("act-a",), blind=False)
    score(s, 5)
    md = ev.to_markdown(s.end())
    assert "one policy, so no comparison" in md and "## Comparisons" not in md


def test_csv_export_has_one_row_per_trial(tmp_path: Path) -> None:
    s = started(tmp_path, blind=False)
    score(s, 5)
    score(s, 2, ["Missed the grasp"])
    s.void(2, "camera dropped")
    rows = ev.to_csv(s.current).strip().splitlines()  # type: ignore[arg-type]
    assert len(rows) == 3 and rows[0].startswith("eval,card,trial")
    assert "camera dropped" in rows[2] and "Missed the grasp" in rows[2]


def test_damaged_files_are_skipped_and_old_records_still_list(tmp_path: Path) -> None:
    d = tmp_path / "evals"
    d.mkdir()
    (d / "broken.json").write_text("{not json")
    (d / "odd.json").write_text(json.dumps({"schema": 2, "id": "x"}))
    (d / "20260801-100000.json").write_text(json.dumps({
        "id": "20260801-100000", "policy": "mock-reach", "task": "cube", "planned": 5,
        "limit_s": 30.0, "started_at": 5.0, "ended_at": 9.0,
        "episodes": [{"outcome": "success"}, {"outcome": "failure"}]}))  # fmt: skip
    s = EvalStore(tmp_path)
    assert s.current is None
    old = s.past()
    assert len(old) == 1 and old[0]["schema"] == 1 and old[0]["k"] == 1 and old[0]["n"] == 2


def test_wilson_is_still_importable_from_evals() -> None:
    lo, hi = wilson(8, 10)
    assert math.isclose(lo, 0.49, abs_tol=0.01) and math.isclose(hi, 0.94, abs_tol=0.01)
