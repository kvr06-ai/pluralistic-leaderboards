"""Pluralistic Leaderboards: independent reference implementation.

Reference paper:
    Haghtalab, N., Procaccia, A. D., Shao, H., Wang, S. L., and Yang, K.
    "Pluralistic Leaderboards." January 2026.
    https://procaccia.info/wp-content/uploads/2026/01/leaderboard.pdf

This package implements:
    - Algorithm 1: Committee Selection via Iterated Rounding.
    - Algorithm 2: Ranking via Geometric Checkpoints.
    - Algorithm 3: Ranking via Committee Monotonicity (single-addition decomposition).
    - Bradley-Terry MLE baseline (Chiang et al., 2024 style).
    - Mallows mixture sampler.
    - Local stability evaluation (Aziz et al., 2017b).

Notation follows the paper:
    - C = {1, ..., m}: set of competing models.
    - W_k subset C of size k: a committee.
    - W_k pi_k: prefix-k of ranking pi.
    - gamma-approximate local stability:
        max_{a not in W_k} Pr_{i ~ D}[a >_i W_k] <= gamma / k.

This is an independent reference implementation. It is not officially
affiliated with the paper authors.
"""

from .algorithms import (
    bradley_terry_ranking,
    committee_selection_iterated_rounding,
    mallows_sample,
    mallows_mixture_sample,
    ranking_via_committee_monotonicity,
    ranking_via_geometric_checkpoints,
)
from .evaluation import (
    estimated_unsatisfied_probability,
    rank_of_committee_under_lottery,
    stability_approximation_factor,
    user_prefers_alternative_to_committee,
)

__all__ = [
    "bradley_terry_ranking",
    "committee_selection_iterated_rounding",
    "estimated_unsatisfied_probability",
    "mallows_mixture_sample",
    "mallows_sample",
    "rank_of_committee_under_lottery",
    "ranking_via_committee_monotonicity",
    "ranking_via_geometric_checkpoints",
    "stability_approximation_factor",
    "user_prefers_alternative_to_committee",
]

__version__ = "0.1.0"
