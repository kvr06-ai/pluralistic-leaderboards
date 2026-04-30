"""Local stability checks and rank/satisfaction estimators.

The paper defines (Section 2):

    A committee W_k of size k is gamma-approximately locally stable iff
        max_{a not in W_k}  Pr_{i ~ D}[ a >_i W_k ]  <=  gamma / k.
    The committee is locally stable when gamma <= 1.

The "stability ratio" of W_k is the smallest gamma satisfying the above
condition, equivalently:
    gamma_hat(W_k) = k * max_{a not in W_k} Pr_{i ~ D}[ a >_i W_k ].

A ranking pi is gamma-approximately locally stable iff for every k, the prefix
W_k^pi is gamma-approximately locally stable.
"""

from __future__ import annotations

import numpy as np

from .algorithms import positions_of


def user_prefers_alternative_to_committee(
    user_rankings: np.ndarray, committee: np.ndarray, alternative: int
) -> np.ndarray:
    """For each user, does ``alternative`` rank above every member of ``committee``?

    Returns a boolean (N,) array.
    """
    if alternative in committee.tolist():
        return np.zeros(user_rankings.shape[0], dtype=bool)
    pos = positions_of(user_rankings)
    if committee.size == 0:
        return np.ones(user_rankings.shape[0], dtype=bool)
    best_in_committee = pos[:, committee].min(axis=1)
    alt_pos = pos[:, alternative]
    return alt_pos < best_in_committee


def stability_approximation_factor(
    user_rankings: np.ndarray, committee: np.ndarray, m: int
) -> float:
    """Empirical stability ratio gamma_hat = k * max_{a not in W_k} Pr[a >_i W_k]."""
    if committee.size == 0:
        return float("inf")
    k = committee.size
    pos = positions_of(user_rankings)
    best_in_committee = pos[:, committee].min(axis=1)
    outside = np.array([c for c in range(m) if c not in committee.tolist()], dtype=np.int64)
    if outside.size == 0:
        return 0.0
    # For each outside candidate compute Pr_{i}[ a's position < best in W ].
    alt_positions = pos[:, outside]  # (N, |outside|)
    prefers = alt_positions < best_in_committee[:, None]  # (N, |outside|)
    fractions = prefers.mean(axis=0)  # (|outside|,)
    return float(k * fractions.max())


def estimated_unsatisfied_probability(
    user_rankings: np.ndarray, committee: np.ndarray
) -> float:
    """Fraction of users for whom the committee fails to contain their top-1.

    This matches the paper's "1/k unsatisfied" target -- a strict-stability
    ranking should have at most 1/k unsatisfied users at every prefix size k.
    """
    if committee.size == 0:
        return 1.0
    pos = positions_of(user_rankings)
    user_top = np.argmin(pos, axis=1)
    in_committee = np.isin(user_top, committee)
    return float((~in_committee).mean())


def rank_of_committee_under_lottery(
    user_rankings: np.ndarray, committee: np.ndarray, lottery_committees: list[np.ndarray]
) -> np.ndarray:
    """Algorithm 5 implementation, deterministic over a finite lottery.

    Rank(i; S, Delta) := Pr_{S' ~ Delta}[user i prefers S to S' (or ties)] where
    "prefers S to S'" means user i's top-1 in S is at least as good as their
    top-1 in S'.

    Returns
    -------
    (N,) float array of ranks in [0, 1].
    """
    if committee.size == 0:
        return np.zeros(user_rankings.shape[0])
    pos = positions_of(user_rankings)
    top_in_S = pos[:, committee].min(axis=1)
    counts = np.zeros(user_rankings.shape[0], dtype=np.float64)
    for S_prime in lottery_committees:
        if S_prime.size == 0:
            counts += 1.0
            continue
        top_in_Sp = pos[:, S_prime].min(axis=1)
        counts += (top_in_S <= top_in_Sp).astype(np.float64)
    return counts / len(lottery_committees)


def stability_curve(
    user_rankings: np.ndarray, ranking: np.ndarray, m: int
) -> np.ndarray:
    """Stability ratio gamma_hat for every prefix W_k^pi, k = 1..m-1.

    Returns
    -------
    (m-1,) float array. gamma[k-1] is the stability ratio for the top-k prefix.
    """
    out = np.empty(m - 1, dtype=np.float64)
    for k in range(1, m):
        prefix = ranking[:k]
        out[k - 1] = stability_approximation_factor(user_rankings, prefix, m)
    return out
