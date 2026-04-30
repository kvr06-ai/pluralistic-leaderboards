"""Unit tests for algorithms.py."""

from __future__ import annotations

import numpy as np
import pytest

from pluralistic_leaderboards.algorithms import (
    approximately_stable_lottery,
    bradley_terry_ranking,
    committee_selection_iterated_rounding,
    kendall_tau_distance,
    mallows_mixture_sample,
    mallows_sample,
    make_mallows_mixture_oracle,
    positions_of,
    ranking_via_committee_monotonicity,
    ranking_via_geometric_checkpoints,
)
from pluralistic_leaderboards.evaluation import (
    estimated_unsatisfied_probability,
    rank_of_committee_under_lottery,
    stability_approximation_factor,
    stability_curve,
    user_prefers_alternative_to_committee,
)


# ---------------------------------------------------------------------------
# Mallows sampler
# ---------------------------------------------------------------------------


def test_kendall_tau_distance_self_zero():
    perm = np.array([3, 1, 0, 2])
    assert kendall_tau_distance(perm, perm) == 0


def test_kendall_tau_distance_symmetric():
    a = np.array([0, 1, 2, 3])
    b = np.array([3, 2, 1, 0])
    # Reversed permutation -> max distance = m*(m-1)/2 = 6.
    assert kendall_tau_distance(a, b) == 6


def test_mallows_phi_zero_returns_center():
    rng = np.random.default_rng(0)
    center = np.array([2, 0, 3, 1, 4])
    samples = mallows_sample(center, phi=0.0, n_samples=50, rng=rng)
    for s in samples:
        np.testing.assert_array_equal(s, center)


def test_mallows_phi_one_is_uniform_on_average():
    rng = np.random.default_rng(0)
    m = 4
    center = np.arange(m)
    n = 5000
    samples = mallows_sample(center, phi=1.0, n_samples=n, rng=rng)
    # Each candidate should appear in each position with frequency ~ 1/m.
    pos = positions_of(samples)
    for c in range(m):
        for p in range(m):
            freq = (pos[:, c] == p).mean()
            assert abs(freq - 1.0 / m) < 0.05, f"c={c} p={p} freq={freq:.3f}"


def test_mallows_low_phi_concentrates_near_center():
    rng = np.random.default_rng(0)
    center = np.arange(8)
    samples_low = mallows_sample(center, phi=0.1, n_samples=300, rng=rng)
    samples_high = mallows_sample(center, phi=0.9, n_samples=300, rng=rng)
    d_low = np.mean([kendall_tau_distance(s, center) for s in samples_low])
    d_high = np.mean([kendall_tau_distance(s, center) for s in samples_high])
    assert d_low < d_high


def test_mallows_mixture_weights_respected():
    rng = np.random.default_rng(0)
    m = 5
    centers = [np.arange(m), np.arange(m)[::-1].copy()]
    samples = mallows_mixture_sample(centers, phi=0.0, weights=np.array([0.7, 0.3]), n_samples=2000, rng=rng)
    # phi=0 -> all rows are exactly one of the centers.
    matches_c0 = np.all(samples == centers[0], axis=1).mean()
    matches_c1 = np.all(samples == centers[1], axis=1).mean()
    assert abs(matches_c0 - 0.7) < 0.05
    assert abs(matches_c1 - 0.3) < 0.05


# ---------------------------------------------------------------------------
# Local stability evaluation
# ---------------------------------------------------------------------------


def test_user_prefers_alternative_to_committee_basic():
    # User 0 ranks: 2, 0, 1, 3  -> top is 2.
    # User 1 ranks: 1, 3, 0, 2.
    rankings = np.array([[2, 0, 1, 3], [1, 3, 0, 2]])
    # Committee = {3}. Alt = 0.
    # User 0: pos(0)=1, pos(3)=3 -> alt better -> True.
    # User 1: pos(0)=2, pos(3)=1 -> alt worse -> False.
    out = user_prefers_alternative_to_committee(rankings, np.array([3]), 0)
    np.testing.assert_array_equal(out, [True, False])


def test_stability_check_perfect_stable_committee():
    # Construct a setup where the top-1 of every user is in the committee.
    rng = np.random.default_rng(1)
    m = 6
    n = 200
    # All users have candidate 0 as their top.
    samples = []
    for _ in range(n):
        s = np.arange(m)
        rng.shuffle(s[1:])  # shuffle non-top
        samples.append(s)
    rankings = np.stack(samples)
    # Committee = {0} should be perfectly stable: no alternative is preferred
    # by anyone since everyone's top is in the committee.
    gamma = stability_approximation_factor(rankings, np.array([0]), m=m)
    assert gamma == 0.0


def test_stability_check_unstable_committee():
    # Two equally-sized factions; pick a committee that ignores faction A.
    rng = np.random.default_rng(2)
    m = 4
    n = 100
    # Half users prefer 0, half prefer 1.
    half = n // 2
    samples = np.empty((n, m), dtype=np.int64)
    for i in range(half):
        s = np.arange(m)
        # Force 0 to top.
        s = np.array([0, 1, 2, 3])
        samples[i] = s
    for i in range(half, n):
        samples[i] = np.array([1, 0, 2, 3])
    # Committee = {2, 3} ignores both factions.
    W = np.array([2, 3])
    gamma = stability_approximation_factor(samples, W, m=m)
    # Pr_user[0 >_i {2,3}] = 1.0, * k=2 -> gamma = 2.
    assert abs(gamma - 2.0) < 1e-9


def test_stability_curve_shape():
    rng = np.random.default_rng(0)
    m = 6
    rankings = mallows_sample(np.arange(m), phi=0.3, n_samples=200, rng=rng)
    # An identity ranking should be very stable.
    pi = np.arange(m)
    curve = stability_curve(rankings, pi, m)
    assert curve.shape == (m - 1,)
    assert (curve <= 1.5).all(), f"identity ranking on low-phi Mallows should be near-stable, got {curve}"


# ---------------------------------------------------------------------------
# Bradley-Terry baseline
# ---------------------------------------------------------------------------


def test_bradley_terry_clear_winner():
    # Candidate 0 beats everyone, candidate m-1 loses to everyone.
    m = 5
    battles = []
    for j in range(1, m):
        battles.append((0, j, 100))
    for i in range(1, m - 1):
        battles.append((i, m - 1, 100))
    battles_arr = np.array(battles, dtype=np.int64)
    ranking, theta = bradley_terry_ranking(battles_arr, m=m, return_scores=True)
    assert ranking[0] == 0
    assert ranking[-1] == m - 1
    # Scores should be monotonically decreasing.
    sorted_theta = theta[ranking]
    assert np.all(np.diff(sorted_theta) <= 1e-9)


def test_bradley_terry_with_weights():
    m = 3
    # Without weights: 1 beats 2 once, 2 beats 1 once -> tie.
    battles = np.array([[1, 2, 1], [2, 1, 1], [0, 1, 5], [0, 2, 5]], dtype=np.int64)
    ranking_unweighted = bradley_terry_ranking(battles, m=m)
    assert ranking_unweighted[0] == 0
    # Up-weight the (1,2,1) battle 100x and down-weight (2,1,1) -> 1 should beat 2.
    weights = np.array([100.0, 0.01, 1.0, 1.0])
    ranking_weighted = bradley_terry_ranking(battles, m=m, weights=weights)
    assert ranking_weighted[0] == 0
    assert ranking_weighted[1] == 1


# ---------------------------------------------------------------------------
# Stable lottery + Algorithms 1, 2, 3 end-to-end on synthetic
# ---------------------------------------------------------------------------


def test_stable_lottery_returns_valid_distribution():
    rng = np.random.default_rng(0)
    m = 6
    rankings = mallows_sample(np.arange(m), phi=0.3, n_samples=200, rng=rng)
    lot = approximately_stable_lottery(rankings, k=2, epsilon=0.1, rng=rng, n_iters=10)
    assert len(lot.committees) > 0
    assert abs(lot.weights.sum() - 1.0) < 1e-9
    for S in lot.committees:
        assert S.shape == (2,)
        assert len(set(S.tolist())) == 2


def test_algorithm_1_runs_and_returns_committee_subset():
    rng = np.random.default_rng(0)
    m = 8
    centers = [np.array([0, 1, 2, 3, 4, 5, 6, 7]),
               np.array([4, 5, 6, 7, 0, 1, 2, 3])]
    oracle = make_mallows_mixture_oracle(centers, phi=0.3, weights=np.array([0.5, 0.5]))
    res = committee_selection_iterated_rounding(
        oracle, m=m, k=4, rng=rng, epsilon=0.1, n_users_per_round=200, n_mwu_iters=10
    )
    assert res.committee.size <= 4
    assert res.committee.size >= 1
    assert all(0 <= c < m for c in res.committee)
    assert len(set(res.committee.tolist())) == res.committee.size


def test_algorithm_3_returns_full_ranking():
    rng = np.random.default_rng(0)
    m = 6
    centers = [np.array([0, 1, 2, 3, 4, 5]),
               np.array([3, 4, 5, 0, 1, 2])]
    oracle = make_mallows_mixture_oracle(centers, phi=0.3, weights=np.array([0.5, 0.5]))
    pi = ranking_via_committee_monotonicity(
        oracle, m=m, rng=rng, epsilon=0.1, n_users_per_round=150, n_mwu_iters=10
    )
    assert pi.shape == (m,)
    assert sorted(pi.tolist()) == list(range(m))


def test_algorithm_3_stability_better_than_random_on_low_phi():
    """With a Mallows mixture (two clear factions), Algorithm 3 should produce
    a top-2 prefix that includes representatives of both factions.
    """
    rng = np.random.default_rng(7)
    m = 6
    centers = [np.array([0, 1, 2, 3, 4, 5]),
               np.array([5, 4, 3, 2, 1, 0])]
    oracle = make_mallows_mixture_oracle(centers, phi=0.1, weights=np.array([0.5, 0.5]))
    pi = ranking_via_committee_monotonicity(
        oracle, m=m, rng=rng, epsilon=0.1, n_users_per_round=300, n_mwu_iters=15
    )
    eval_users = oracle(2000, np.random.default_rng(99))
    gamma_top2 = stability_approximation_factor(eval_users, pi[:2], m=m)
    # The top-1 of each faction is 0 and 5. A stable top-2 covers both -> gamma should be small.
    # A "Bradley-Terry-style" ranking would put two faction-A models together
    # and have gamma ~= 1.0; we want strictly better than that.
    assert gamma_top2 < 0.7, f"top-2 gamma = {gamma_top2:.3f} should be small"


def test_algorithm_2_runs_and_returns_full_ranking():
    rng = np.random.default_rng(0)
    m = 6
    centers = [np.array([0, 1, 2, 3, 4, 5])]
    oracle = make_mallows_mixture_oracle(centers, phi=0.3, weights=np.array([1.0]))
    pi = ranking_via_geometric_checkpoints(
        oracle, m=m, rng=rng, growth_factor=2.0, epsilon=0.1, n_users_per_round=150, n_mwu_iters=10
    )
    assert pi.shape == (m,)
    assert sorted(pi.tolist()) == list(range(m))


def test_unsatisfied_probability_known():
    rankings = np.array([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 0, 1, 3],
    ])
    # Committee = {0, 1}: user 0 top is 0 (in), user 1 top is 1 (in), user 2 top is 2 (out)
    # -> 1/3 unsatisfied.
    p = estimated_unsatisfied_probability(rankings, np.array([0, 1]))
    assert abs(p - 1.0 / 3) < 1e-9


def test_rank_of_committee_under_lottery_basic():
    rankings = np.array([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
    ])
    S = np.array([0])
    lottery = [np.array([0]), np.array([1])]
    # User 0: top in S=0 is pos 0; in S'=0 also pos 0 (tie -> True), in S'=1 pos 1 (S beats) -> 2/2 = 1.
    # User 1: top in S=0 is pos 1; in S'=0 pos 1 (tie -> True), in S'=1 pos 0 (S' beats) -> 1/2.
    ranks = rank_of_committee_under_lottery(rankings, S, lottery)
    np.testing.assert_array_almost_equal(ranks, [1.0, 0.5])
