"""Statistics for policy evals: what a few real trials can and cannot say.

Pure Python (math and the standard library), so the server and its tests need nothing extra. The
Φ wiki page concepts/robot-policy-evaluation.md lists every source.

Every comparison here assumes the bundle design: a bundle is one start condition with every policy
run once from it, in a random order (TRI LBM, arXiv 2507.05331; DeepMind Gemini Robotics,
2503.20020). Two policies are then compared bundle by bundle, so drift over a day and a hard
condition hit both alike.

  * Wilson score interval for one success rate. Clopper-Pearson is too conservative at small n
    (Brown, Cai and DasGupta 2001).
  * Which of two policies is better, from the bundles where exactly one succeeded. Bundles where
    both succeeded or both failed say nothing about which is better (McNemar's test); under "no
    difference" each one-sided bundle is a fair coin.
  * When to stop. Looking at a fixed-size test after every bundle and stopping at the first small
    p-value inflates the false-alarm rate (STEP, arXiv 2503.10966). Here the test fixes the error
    rate and the largest number of bundles in advance, then uses one per-look threshold chosen so
    that the exact chance of ever declaring a difference between two equal policies, at any look
    up to that maximum, is at most the error rate (a Pocock-style repeated test, computed exactly
    by dynamic programming, not by simulation). Own implementation: STEP's code is CC BY-NC and
    Studio is Apache-2.0. STEP tunes its boundary for power; this one is simpler and also valid.
  * P(A is better) with a uniform prior (Kress-Gazit et al., arXiv 2409.09491).
  * Compact letter display for three or more policies (Piepho 2004), as TRI reports.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from statistics import NormalDist

MAX_PAIRS = 200  # the longest plan: no small lab runs more bundles per pair (TRI ran 50)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials: every p the score test does not reject
    at level z. (0, 1) when there are no trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@cache
def _cdf_row(n: int) -> tuple[float, ...]:
    """P(X <= k) for k = 0..n, X ~ Binomial(n, 1/2): whole-number sums, one division each."""
    out, acc = [], 0
    for i in range(n + 1):
        acc += math.comb(n, i)
        out.append(acc / 2**n)
    return tuple(out)


def binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), exact."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return _cdf_row(n)[k]


def two_sided_p(wins: int, pairs: int) -> float:
    """Exact two-sided p-value for `wins` of `pairs` fair coin flips."""
    return min(1.0, 2.0 * binom_cdf_half(min(wins, pairs - wins), pairs))


# -- which of two policies is better ---------------------------------------------------------------
def p_better_paired(wins_a: int, wins_b: int) -> float:
    """P(A's success rate is above B's), from the bundles where exactly one succeeded.

    A Dirichlet(1, 1, 1, 1) prior over the four outcomes of a bundle gives P(A wins | one wins) a
    Beta(1 + wins_a, 1 + wins_b) posterior, and A's rate is above B's exactly when that is above
    1/2. For whole numbers the Beta tail is a binomial sum:
    P(theta > 1/2) = P(Binomial(wins_a + wins_b + 1, 1/2) <= wins_a)."""
    return binom_cdf_half(wins_a, wins_a + wins_b + 1)


def _lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def p_better_unpaired(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """P(p_A > p_B) for independent uniform-prior Beta posteriors, exact for whole-number counts
    (the closed form for P(Beta_B > Beta_A), summed over B's first parameter). For trials that
    were not run in bundles: a fair comparison only if both ran under the same conditions."""
    a1, b1 = 1 + k_a, 1 + n_a - k_a
    a2, b2 = 1 + k_b, 1 + n_b - k_b
    b_wins = sum(math.exp(_lbeta(a1 + i, b1 + b2) - math.log(b2 + i) - _lbeta(1 + i, b2)
                          - _lbeta(a1, b1)) for i in range(a2))  # fmt: skip
    return min(1.0, max(0.0, 1.0 - b_wins))


# -- when to stop: the sequential plan -------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    """A sequential test fixed before the first bundle. lower[m - 1] and upper[m - 1] are the
    boundaries after m one-sided bundles: B is better when A's wins <= lower, A when >= upper."""

    alpha: float  # the false-alarm rate asked for
    max_pairs: int  # the most one-sided bundles the plan looks at
    nominal: float  # the exact two-sided p-value that decides at any single look
    achieved: float  # the exact chance that two equal policies are ever called different
    lower: tuple[int, ...]
    upper: tuple[int, ...]


def _bounds(nominal: float, max_pairs: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lower, upper = [], []
    for m in range(1, max_pairs + 1):
        lo = -1  # the largest losing count that already decides; -1: none yet
        for w in range(0, (m + 1) // 2):
            if two_sided_p(w, m) <= nominal:
                lo = w
            else:
                break
        lower.append(lo)
        upper.append(m - lo if lo >= 0 else m + 1)  # symmetric: the same evidence for either side
    return tuple(lower), tuple(upper)


def crossing(lower: Sequence[int], upper: Sequence[int], p: float = 0.5) -> float:
    """Exact chance that A's wins cross a boundary within len(lower) one-sided bundles, when each
    is A's with probability p. Dynamic programming over the counts still inside the boundaries."""
    alive = {0: 1.0}
    hit = 0.0
    for lo, hi in zip(lower, upper, strict=True):
        step: dict[int, float] = {}
        for w, pr in alive.items():
            step[w + 1] = step.get(w + 1, 0.0) + pr * p
            step[w] = step.get(w, 0.0) + pr * (1.0 - p)
        alive = {}
        for w, pr in step.items():
            if w <= lo or w >= hi:
                hit += pr
            else:
                alive[w] = pr
    return hit


@lru_cache(maxsize=64)
def plan(alpha: float, max_pairs: int) -> Plan:
    """The most permissive single per-look threshold whose exact overall false-alarm rate is at
    most alpha. Stopping before max_pairs, for any reason, can only lower that rate."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 1 <= max_pairs <= MAX_PAIRS:
        raise ValueError(f"max_pairs must be between 1 and {MAX_PAIRS}")
    candidates = sorted({two_sided_p(w, m) for m in range(1, max_pairs + 1)
                         for w in range((m + 1) // 2) if two_sided_p(w, m) <= alpha})  # fmt: skip
    best, best_rate = 0.0, 0.0
    lo, hi = 0, len(candidates) - 1
    while lo <= hi:  # the overall rate grows with the threshold: binary search the candidates
        mid = (lo + hi) // 2
        rate = crossing(*_bounds(candidates[mid], max_pairs))
        if rate <= alpha:
            best, best_rate, lo = candidates[mid], rate, mid + 1
        else:
            hi = mid - 1
    lower, upper = _bounds(best, max_pairs)  # best 0.0: no look can decide
    return Plan(alpha, max_pairs, best, best_rate, lower, upper)


def decide(outcomes: Iterable[tuple[bool, bool]], p: Plan, complete: bool) -> dict[str, object]:
    """Walk the bundles in the order they ran: (A succeeded, B succeeded) each. Returns the state:
    "a" or "b" (that policy is better, decided at `at_bundle`), "none" (the plan ended with no
    difference shown: not proof that there is none), or "continue". `complete`: every planned
    bundle has run."""
    wins = pairs = bundles = 0
    for a, b in outcomes:
        bundles += 1
        if a == b:
            continue
        pairs += 1
        wins += int(a)
        if pairs > p.max_pairs:
            break
        if wins >= p.upper[pairs - 1]:
            return {"state": "a", "at_bundle": bundles, "pairs": pairs, "wins_a": wins,
                    "wins_b": pairs - wins}  # fmt: skip
        if wins <= p.lower[pairs - 1]:
            return {"state": "b", "at_bundle": bundles, "pairs": pairs, "wins_a": wins,
                    "wins_b": pairs - wins}  # fmt: skip
    state = "none" if complete or pairs >= p.max_pairs else "continue"
    return {"state": state, "at_bundle": None, "pairs": pairs, "wins_a": wins,
            "wins_b": pairs - wins, "bundles": bundles}  # fmt: skip


# -- planning aids (approximate, and labelled so) --------------------------------------------------
def _z(q: float) -> float:
    return NormalDist().inv_cdf(q)


def pairs_needed(theta: float, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """About how many one-sided bundles a fixed-size test needs to tell `theta` (the share of
    one-sided bundles A wins) from 1/2 with this power. Normal approximation to the one-sample
    proportion test; None when theta is 1/2, where no number is enough."""
    if abs(theta - 0.5) < 1e-9:
        return None
    n = ((_z(1 - alpha / 2) * 0.5 + _z(power) * math.sqrt(theta * (1 - theta)))
         / abs(theta - 0.5)) ** 2  # fmt: skip
    return math.ceil(n)


def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Trials per policy to tell success rates p1 and p2 apart (two-sided, normal approximation,
    unpaired). Pairing in bundles can only need fewer."""
    pbar = (p1 + p2) / 2
    num = (_z(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar))
           + _z(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))  # fmt: skip
    return math.ceil((num / (p1 - p2)) ** 2)


def detectable_gap(n: int, base: float = 0.5, alpha: float = 0.05,
                   power: float = 0.8) -> float | None:  # fmt: skip
    """The smallest gap above `base` (in steps of 0.01) that n trials per policy can detect."""
    for d in range(1, 100):
        p2 = base + d / 100
        if p2 >= 1.0:
            return None
        if n_two_proportions(base, p2, alpha, power) <= n:
            return d / 100
    return None


def mean_ci(values: Sequence[float], reps: int = 2000,
            seed: int = 0) -> tuple[float, float | None, float | None]:  # fmt: skip
    """Mean with a 95% percentile-bootstrap interval, seeded so the same scores always give the
    same interval. No interval below two values."""
    n = len(values)
    if n == 0:
        return (math.nan, None, None)
    mean = sum(values) / n
    if n < 2:
        return (mean, None, None)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(reps))
    return (mean, means[int(0.025 * reps)], means[min(reps - 1, int(0.975 * reps))])


def letters(order: Sequence[str], different: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Compact letter display (Piepho 2004, insert and absorb): two policies that share a letter
    were not shown to differ. `order` is best first; the first group gets "a"."""
    groups: list[set[str]] = [set(order)]
    for a, b in different:
        split: list[set[str]] = []
        for g in groups:
            split += [g - {a}, g - {b}] if a in g and b in g else [g]
        groups = [g for i, g in enumerate(split)
                  if not any(g < h or (g == h and j < i) for j, h in enumerate(split) if j != i)]
    rank = {p: i for i, p in enumerate(order)}
    groups.sort(key=lambda g: min(rank[x] for x in g))
    out = {p: "" for p in order}
    for i, g in enumerate(groups):
        for p in g:
            out[p] += chr(ord("a") + i)
    return out

