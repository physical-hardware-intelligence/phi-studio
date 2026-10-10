"""Eval statistics (evalstats.py): exact checks, published numbers, and the error rate of the
sequential test proved by enumerating every outcome."""

from __future__ import annotations

import itertools
import random

import pytest

from phi_studio import evalstats as es


def test_wilson_matches_the_published_small_sample_example() -> None:
    lo, hi = es.wilson(8, 10)  # Brown, Cai and DasGupta's regime: Wilson [0.49, 0.94]
    assert (round(lo, 2), round(hi, 2)) == (0.49, 0.94)
    assert es.wilson(0, 0) == (0.0, 1.0)
    assert es.wilson(0, 10)[0] == 0.0 and es.wilson(10, 10)[1] == 1.0


def test_binomial_tail_and_p_values_are_exact() -> None:
    assert es.binom_cdf_half(0, 3) == 1 / 8 and es.binom_cdf_half(1, 3) == 4 / 8
    assert es.binom_cdf_half(-1, 3) == 0.0 and es.binom_cdf_half(3, 3) == 1.0
    assert es.two_sided_p(0, 5) == 2 / 32 and es.two_sided_p(5, 5) == 2 / 32
    assert es.two_sided_p(2, 4) == 1.0


def test_paired_posterior_is_symmetric_and_grows_with_wins() -> None:
    assert es.p_better_paired(0, 0) == 0.5
    for a, b in [(3, 1), (7, 2), (0, 4)]:
        assert es.p_better_paired(a, b) + es.p_better_paired(b, a) == pytest.approx(1.0)
    assert es.p_better_paired(5, 0) == pytest.approx(1 - 1 / 64)
    assert es.p_better_paired(6, 1) > es.p_better_paired(4, 1) > 0.5


def test_unpaired_posterior_reproduces_kress_gazit_and_a_monte_carlo_check() -> None:
    # arXiv 2409.09491: A 15/18 vs B 11/17 gives P(B > A) = 0.11 (0.112 when recomputed)
    p_b = es.p_better_unpaired(11, 17, 15, 18)
    assert p_b == pytest.approx(0.112, abs=0.002)
    assert es.p_better_unpaired(15, 18, 11, 17) == pytest.approx(1 - p_b)
    rng = random.Random(7)
    draws = 100_000
    hits = sum(rng.betavariate(12, 7) > rng.betavariate(16, 4) for _ in range(draws))
    assert p_b == pytest.approx(hits / draws, abs=0.005)


def all_paths(m: int):
    return itertools.product((0, 1), repeat=m)


@pytest.mark.parametrize("max_pairs", [1, 5, 10, 14])
def test_the_dynamic_program_matches_enumeration(max_pairs: int) -> None:
    pl = es.plan(0.05, max_pairs)
    hits = 0
    for path in all_paths(max_pairs):  # each path: who won each one-sided bundle (1 = A)
        w = 0
        for m, x in enumerate(path, 1):
            w += x
            if w <= pl.lower[m - 1] or w >= pl.upper[m - 1]:
                hits += 1
                break
    assert es.crossing(pl.lower, pl.upper) == pytest.approx(hits / 2**max_pairs)
    assert pl.achieved == pytest.approx(hits / 2**max_pairs)


@pytest.mark.parametrize("alpha,max_pairs", [(0.05, 20), (0.05, 60), (0.01, 40), (0.10, 100)])
def test_a_plan_never_exceeds_its_false_alarm_rate(alpha: float, max_pairs: int) -> None:
    pl = es.plan(alpha, max_pairs)
    assert 0.0 < pl.achieved <= alpha
    # and is the most permissive single threshold: the next candidate would break alpha
    looser = min(es.two_sided_p(w, m) for m in range(1, max_pairs + 1) for w in range((m + 1) // 2)
                 if es.two_sided_p(w, m) > pl.nominal)  # fmt: skip
    if looser <= alpha:
        assert es.crossing(*es._bounds(looser, max_pairs)) > alpha


def test_stopping_early_is_safe_and_peeking_at_a_fixed_test_is_not() -> None:
    """WHY the plan exists: rejecting at the first fixed-size p < 0.05 over 40 looks makes the
    false-alarm rate far above 5% (STEP); the plan's per-look threshold keeps it at most 5%."""
    m = 40
    naive = es.crossing(*es._bounds(0.05, m))
    assert naive > 0.10
    assert es.plan(0.05, m).achieved <= 0.05


def test_decide_finds_a_clear_winner_and_ignores_ties() -> None:
    pl = es.plan(0.05, 30)
    a_better = [(True, False)] * 12
    d = es.decide(a_better, pl, complete=False)
    assert d["state"] == "a" and 5 <= d["at_bundle"] <= 12  # type: ignore[operator]
    ties = [(True, True), (False, False)] * 10
    assert es.decide(ties, pl, complete=False)["state"] == "continue"
    assert es.decide(ties, pl, complete=True)["state"] == "none"
    mixed = [(True, False), (False, True)] * 6
    d = es.decide(mixed, pl, complete=False)
    assert d["state"] == "continue" and d["pairs"] == 12 and d["wins_a"] == 6
    b_better = [(False, True)] * 12
    assert es.decide(b_better, pl, complete=False)["state"] == "b"


def test_sample_size_matches_the_standard_formula() -> None:
    # Two-sided 5%, power 0.8 (the table in concepts/robot-policy-evaluation.md)
    assert es.n_two_proportions(0.5, 0.9) == 20
    assert es.n_two_proportions(0.5, 0.8) == 39
    assert es.n_two_proportions(0.6, 0.7) == 356
    assert es.detectable_gap(20) == 0.4 and es.detectable_gap(5) is None
    assert es.pairs_needed(0.5) is None
    assert es.pairs_needed(0.9) < es.pairs_needed(0.7)  # type: ignore[operator]


def test_bootstrap_interval_is_seeded_and_brackets_the_mean() -> None:
    xs = [0.2, 0.2, 1.0, 0.4, 1.0, 0.2, 0.7]
    m, lo, hi = es.mean_ci(xs)
    assert lo is not None and hi is not None and lo <= m <= hi
    assert es.mean_ci(xs) == (m, lo, hi)
    assert es.mean_ci([0.5])[1] is None


def test_letters_follow_piepho() -> None:
    order = ["A", "C", "B"]  # best first
    assert es.letters(order, []) == {"A": "a", "C": "a", "B": "a"}
    assert es.letters(order, [("A", "B")]) == {"A": "a", "C": "ab", "B": "b"}
    full = es.letters(order, [("A", "B"), ("A", "C"), ("C", "B")])
    assert len(set(full.values())) == 3 and all(len(v) == 1 for v in full.values())
