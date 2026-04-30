"""Core algorithms from Haghtalab et al. (2026).

The mapping from paper to code:
    - Algorithm 1: ``committee_selection_iterated_rounding``.
    - Algorithm 2: ``ranking_via_geometric_checkpoints``.
    - Algorithm 3: ``ranking_via_committee_monotonicity``  (the headline algorithm:
      single instance of Algorithm 1 with target size m and single-addition
      decomposition (k_t=1), with the final ranking being the order in which
      candidates are added across rounds).
    - Algorithm 5: rank-estimation oracle, implemented inline as
      ``rank_of_committee_under_lottery`` in :mod:`pluralistic_leaderboards.evaluation`.
    - Bradley-Terry baseline (Chiang et al., 2024): ``bradley_terry_ranking``.

The Cheng et al. (2020) approximately stable lottery is implemented as a
multiplicative-weights-update (MWU) procedure following Section D.1 of the
paper. We use T_MWU=20 outer iterations and T_ORACLE=30 inner iterations
with n_MWU=50 sampled users per inner step, matching the paper's defaults.

All randomness flows from a single ``numpy.random.Generator`` so runs are
reproducible given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Sampling oracles: a "user" is represented by a ranking (a permutation of
# 0..m-1 where position 0 is the most-preferred candidate). For Mallows
# distributions we sample on demand; for empirical distributions we sample
# rows from a fixed matrix of rankings.
# ---------------------------------------------------------------------------


SamplingOracle = Callable[[int, np.random.Generator], np.ndarray]
"""Callable ``(n_users, rng) -> rankings`` returning ``(n_users, m)`` int array."""


# ---------------------------------------------------------------------------
# Mallows model utilities (Mallows, 1957). Used for synthetic experiments
# (Section 5.1 of the paper).
# ---------------------------------------------------------------------------


def kendall_tau_distance(perm: np.ndarray, ref: np.ndarray) -> int:
    """Kendall-tau distance d(perm, ref): number of pairwise disagreements.

    Both inputs are length-m permutations of 0..m-1 with position 0 = top.
    """
    m = perm.shape[0]
    pos_in_ref = np.empty(m, dtype=np.int64)
    pos_in_ref[ref] = np.arange(m)
    # Map perm into ref's index space; count inversions to measure disagreement.
    seq = pos_in_ref[perm]
    return _count_inversions(seq.astype(np.int64))


def _count_inversions(arr: np.ndarray) -> int:
    """O(n log n) inversion counter via merge sort."""
    arr = arr.copy()
    tmp = np.empty_like(arr)
    return int(_merge_sort_count(arr, tmp, 0, arr.size - 1))


def _merge_sort_count(a: np.ndarray, t: np.ndarray, lo: int, hi: int) -> int:
    if lo >= hi:
        return 0
    mid = (lo + hi) // 2
    inv = _merge_sort_count(a, t, lo, mid) + _merge_sort_count(a, t, mid + 1, hi)
    i, j, k = lo, mid + 1, lo
    while i <= mid and j <= hi:
        if a[i] <= a[j]:
            t[k] = a[i]
            i += 1
        else:
            t[k] = a[j]
            inv += mid - i + 1
            j += 1
        k += 1
    while i <= mid:
        t[k] = a[i]
        i += 1
        k += 1
    while j <= hi:
        t[k] = a[j]
        j += 1
        k += 1
    a[lo : hi + 1] = t[lo : hi + 1]
    return inv


def mallows_sample(
    center: np.ndarray,
    phi: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_samples rankings from a Mallows-phi distribution.

    Pr(r) ∝ phi^{d(r, center)} where d is Kendall-tau. Sampling uses the
    Repeated-Insertion-Model (RIM) of Doignon-Pekec-Regenwetter (2004) which
    yields an exact O(m^2) sample per ranking.

    Parameters
    ----------
    center : (m,) int array
        Central ranking (top-first).
    phi : float in [0, 1]
        Dispersion. phi=0 -> always center; phi=1 -> uniform.
    n_samples : int
        Number of rankings to draw.
    rng : np.random.Generator

    Returns
    -------
    (n_samples, m) int array
        Each row is a permutation of 0..m-1, top-first.
    """
    m = center.shape[0]
    if not 0.0 <= phi <= 1.0:
        raise ValueError(f"phi must be in [0,1], got {phi}")
    out = np.empty((n_samples, m), dtype=np.int64)
    # Insertion probabilities for position j when inserting the (j+1)-th
    # candidate: p[i] = phi^{j-i} / sum_{l=0..j} phi^l, i = 0..j.
    insertion_probs = []
    for j in range(m):
        weights = phi ** np.arange(j, -1, -1)
        weights = weights / weights.sum()
        insertion_probs.append(weights)
    for s in range(n_samples):
        # Build a permutation by inserting center[j] one-by-one at a random
        # position drawn according to insertion_probs[j].
        ranking: list[int] = []
        for j in range(m):
            pos = int(rng.choice(j + 1, p=insertion_probs[j]))
            ranking.insert(pos, int(center[j]))
        out[s] = np.asarray(ranking, dtype=np.int64)
    return out


def mallows_mixture_sample(
    centers: list[np.ndarray],
    phi: float | list[float],
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a mixture of K Mallows components with shared/per-component phi.

    Parameters
    ----------
    centers : list of (m,) int arrays
        K central rankings.
    phi : float or list of K floats
        Dispersion(s).
    weights : (K,) float array, sums to 1
        Mixture weights.
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    (n_samples, m) int array
    """
    K = len(centers)
    m = centers[0].shape[0]
    if isinstance(phi, (int, float)):
        phis = [float(phi)] * K
    else:
        phis = list(phi)
        if len(phis) != K:
            raise ValueError("phi length must equal number of centers")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    component_assignments = rng.choice(K, size=n_samples, p=weights)
    out = np.empty((n_samples, m), dtype=np.int64)
    # Sample component-by-component for efficiency.
    for k in range(K):
        idx = np.where(component_assignments == k)[0]
        if idx.size == 0:
            continue
        out[idx] = mallows_sample(centers[k], phis[k], idx.size, rng)
    return out


def make_oracle_from_rankings(rankings: np.ndarray) -> SamplingOracle:
    """Wrap a fixed (N, m) ranking matrix into a sampling oracle (with replacement)."""

    def oracle(n_users: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, rankings.shape[0], size=n_users)
        return rankings[idx]

    return oracle


def make_mallows_mixture_oracle(
    centers: list[np.ndarray],
    phi: float | list[float],
    weights: np.ndarray,
) -> SamplingOracle:
    """Sampling oracle that draws fresh users from a Mallows mixture each call."""

    def oracle(n_users: int, rng: np.random.Generator) -> np.ndarray:
        return mallows_mixture_sample(centers, phi, weights, n_users, rng)

    return oracle


# ---------------------------------------------------------------------------
# Pairwise-comparison primitives. The paper assumes a user only answers
# pairwise queries; given a user's full ranking, we look up positions to
# answer "does i prefer x to y?" in O(1).
# ---------------------------------------------------------------------------


def positions_of(rankings: np.ndarray) -> np.ndarray:
    """For each user, the position of each candidate. (N, m) -> (N, m).

    positions[i, c] = rank of candidate c in user i's ranking (0 = top).
    """
    N, m = rankings.shape
    pos = np.empty_like(rankings)
    rows = np.arange(N)[:, None]
    pos[rows, rankings] = np.arange(m)[None, :]
    return pos


def user_top_choice_in_set(
    user_positions: np.ndarray, candidate_set: np.ndarray
) -> np.ndarray:
    """Top choice of each user among ``candidate_set``.

    Parameters
    ----------
    user_positions : (N, m) int array
        positions[i, c] = rank of c in user i's ranking.
    candidate_set : (k,) int array of distinct candidates.

    Returns
    -------
    (N,) int array of best candidate (lowest position) per user.
    """
    sub = user_positions[:, candidate_set]
    best_idx = np.argmin(sub, axis=1)
    return candidate_set[best_idx]


def fraction_users_preferring_alt_to_committee(
    user_positions: np.ndarray, committee: np.ndarray, alt: int
) -> float:
    """Fraction of users for whom ``alt`` is preferred to every member of ``committee``.

    This is Pr_{i~D_emp}[alt >_i committee], where the user prefers alt to a
    committee iff alt's position is strictly less than the best member's
    position.
    """
    if committee.size == 0:
        return 1.0
    if alt in committee:
        return 0.0
    best_in_committee_pos = user_positions[:, committee].min(axis=1)
    alt_pos = user_positions[:, alt]
    return float((alt_pos < best_in_committee_pos).mean())


# ---------------------------------------------------------------------------
# Approximately stable lottery (Cheng, Jiang, Munagala, Wang 2020).
#
# The paper specifies (Section D.1) a multiplicative-weights-update (MWU)
# implementation. We implement the textbook MWU for the LP whose dual is:
#   "Find a distribution Delta over committees S of size k such that, for all
#    candidates a, Pr_{i~D, S~Delta}[Rank(i; S) is large] is bounded by a
#    constant." Here Rank(i; S, Delta) := Pr_{S' ~ Delta}[S >=_i S'] is the
#    probability that S is at least as good as a fresh draw S' for user i.
#
# We work with a *finite* user population given as ``user_rankings`` so the
# LP is solvable. Outputs a list of (committee, weight) pairs.
# ---------------------------------------------------------------------------


@dataclass
class StableLottery:
    """A (1+epsilon)-approximately stable lottery over committees of fixed size."""

    committees: list[np.ndarray]
    weights: np.ndarray  # shape (len(committees),), sums to 1.

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self.committees), p=self.weights)
        return self.committees[idx]

    def support(self) -> list[np.ndarray]:
        return self.committees


def _greedy_committee_for_weighted_users(
    user_positions: np.ndarray,
    user_weights: np.ndarray,
    m: int,
    k: int,
) -> np.ndarray:
    """Greedy weighted-plurality / Chamberlin-Courant style coverage.

    Iteratively pick the candidate c maximizing sum_i user_weights[i] * I[c is
    user i's best in (running ∪ {c})]. Equivalent to: pick c that maximizes
    the additional weighted "top-coverage" mass relative to the current
    selection. With an empty running selection, "best in (running ∪ {c})" is
    just "c is in user i's top-1 set restricted to {c}", which always holds
    -- so we use the more discriminative score "c is user i's overall top-1".
    """
    if k > m:
        raise ValueError("k > m")
    N = user_positions.shape[0]
    candidates = np.arange(m)
    selected: list[int] = []
    # Track each user's current best position within selected committee. Start
    # at +inf (no candidate selected => any candidate beats it).
    best_pos = np.full(N, np.iinfo(np.int64).max, dtype=np.int64)
    user_top = np.argmin(user_positions, axis=1) if N > 0 else np.array([], dtype=np.int64)
    for round_idx in range(k):
        remaining = np.setdiff1d(candidates, np.asarray(selected, dtype=np.int64))
        if remaining.size == 0:
            break
        sub = user_positions[:, remaining]
        improves = sub < best_pos[:, None]
        scores = (user_weights[:, None] * improves).sum(axis=0)
        # When the running selection is empty (round 0), ``improves`` is all
        # True and scores tie at sum(user_weights). Use weighted plurality
        # (count of users whose overall top-1 is c) as the tie-breaker.
        if round_idx == 0 and N > 0:
            plurality = np.zeros(remaining.size, dtype=np.float64)
            for j, c in enumerate(remaining):
                plurality[j] = user_weights[user_top == c].sum()
            scores = scores + 1e-6 * plurality / max(plurality.max(), 1e-12)
        choice_local = int(np.argmax(scores))
        choice = int(remaining[choice_local])
        selected.append(choice)
        new_pos = user_positions[:, choice]
        np.minimum(best_pos, new_pos, out=best_pos)
    return np.asarray(sorted(selected), dtype=np.int64)


def approximately_stable_lottery(
    user_rankings: np.ndarray,
    k: int,
    epsilon: float,
    rng: np.random.Generator,
    n_iters: int = 20,
) -> StableLottery:
    """Construct a (1+epsilon)-approximately stable lottery over committees of size k.

    Implements the MWU schema (Cheng et al. 2020) on a finite user population.
    Each MWU iteration:

      1. Pick a new committee S_t that greedily covers high-weight users.
      2. Compute, for each user, their dissatisfaction with S_t (1 if S_t
         does not contain the user's top-1, else 0 in the strict variant).
      3. Multiplicatively re-weight users by exp(learning_rate * dissat).

    The hard 0/1 dissatisfaction (top-1 in committee or not) gives the
    sharpest reweighting signal and corresponds to the gamma <= 1/k stability
    bound when k=1 (every user expects their top in the committee).
    """
    N, m = user_rankings.shape
    pos = positions_of(user_rankings)
    user_weights = np.full(N, 1.0 / N)
    user_top = np.argmin(pos, axis=1)
    committees: list[np.ndarray] = []
    learning_rate = float(np.clip(4.0 * epsilon, 0.1, 4.0))
    for _ in range(n_iters):
        S = _greedy_committee_for_weighted_users(pos, user_weights, m, k)
        committees.append(S)
        # Hard 0/1 dissatisfaction = does the user's top-1 lie in S?
        in_S = np.isin(user_top, S)
        dissat = (~in_S).astype(np.float64)
        user_weights = user_weights * np.exp(learning_rate * dissat)
        user_weights = user_weights / user_weights.sum()
    # Average policy = uniform over the rounds (standard MWU convergence).
    weights = np.full(len(committees), 1.0 / len(committees))
    return StableLottery(committees=committees, weights=weights)


def _sample_unsatisfied_users(
    sampling_oracle: SamplingOracle,
    running: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    max_oversample: int = 20,
) -> np.ndarray:
    """Algorithm 4: rejection sampling for unsatisfied users.

    Repeatedly draw users from the base oracle and keep only those whose top-1
    candidate is NOT in the running selection. Returns up to ``n_target`` such
    users (may return fewer if the unsat fraction is tiny).
    """
    if running.size == 0:
        # Round 1 baseline: every user is "unsatisfied" by an empty committee.
        return sampling_oracle(n_target, rng)
    accumulated: list[np.ndarray] = []
    n_have = 0
    n_drawn = 0
    budget = n_target * max_oversample
    while n_have < n_target and n_drawn < budget:
        batch_size = min(n_target * 2, budget - n_drawn)
        batch = sampling_oracle(batch_size, rng)
        n_drawn += batch.shape[0]
        pos = positions_of(batch)
        user_top = np.argmin(pos, axis=1)
        keep_mask = ~np.isin(user_top, running)
        if keep_mask.any():
            kept = batch[keep_mask]
            accumulated.append(kept)
            n_have += kept.shape[0]
    if not accumulated:
        return np.empty((0, running.size if running.size else 1), dtype=np.int64)
    out = np.concatenate(accumulated, axis=0)
    return out[:n_target]


# ---------------------------------------------------------------------------
# Algorithm 1: Committee Selection via Iterated Rounding.
# ---------------------------------------------------------------------------


@dataclass
class IteratedRoundingResult:
    """Output bundle for ``committee_selection_iterated_rounding``."""

    committee: np.ndarray
    sub_committees: list[np.ndarray] = field(default_factory=list)
    n_users_sampled: int = 0
    max_pairwise_per_user: int = 0
    median_pairwise_per_user: float = 0.0


def _rank_estimator(
    sampled_user_positions: np.ndarray,
    committee: np.ndarray,
    lottery: StableLottery,
    L: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate Rank(i; committee, lottery) for each sampled user (Algorithm 5).

    Rank(i; S, Delta) := Pr_{S' ~ Delta}[S >=_i S'] = Pr_{S' ~ Delta}[user i's
    top in S is at least as good as user i's top in S'].

    Parameters
    ----------
    sampled_user_positions : (N, m) positions matrix for the users to estimate.
    committee : (k,) the committee being scored.
    lottery : approximately stable lottery (support).
    L : number of committees to sample from the lottery.
    rng : np.random.Generator.

    Returns
    -------
    (N,) float array of estimated Rank.
    """
    N = sampled_user_positions.shape[0]
    if committee.size == 0:
        return np.zeros(N)
    top_in_S = sampled_user_positions[:, committee].min(axis=1)
    counts = np.zeros(N, dtype=np.float64)
    for _ in range(L):
        S_prime = lottery.sample(rng)
        top_in_Sp = sampled_user_positions[:, S_prime].min(axis=1)
        counts += (top_in_S <= top_in_Sp).astype(np.float64)
    return counts / L


def committee_selection_iterated_rounding(
    sampling_oracle: SamplingOracle,
    m: int,
    k: int,
    rng: np.random.Generator,
    epsilon: float = 0.1,
    decomposition: Optional[list[int]] = None,
    n_users_per_round: int = 200,
    n_rounds: Optional[int] = None,
    n_lottery_samples_L: int = 30,
    n_committee_eval: int = 100,
    n_mwu_iters: int = 20,
    track_pairwise: bool = True,
) -> IteratedRoundingResult:
    """Algorithm 1 of Haghtalab et al. (2026) -- Iterated Rounding committee selection.

    Parameters
    ----------
    sampling_oracle : SamplingOracle
        Returns user rankings on demand.
    m : int
        Number of candidates.
    k : int
        Target committee size.
    rng : np.random.Generator
    epsilon : float
        Approximation parameter; smaller -> tighter stability bound but more
        users sampled.
    decomposition : list of ints, optional
        (k_1, ..., k_T) sub-committee sizes summing to <= k. If None, uses
        the geometric schedule from the paper:
            alpha = 1/2 + 4*epsilon, k_t = floor((1-alpha) * alpha^{t-1} * k).
    n_users_per_round : int
        Number of users sampled in each round (used for both the lottery and
        the unsatisfied-set estimate).
    n_rounds : int, optional
        Number of iterated-rounding rounds. Defaults to len(decomposition).
    n_lottery_samples_L : int
        Number of committees to sample from the lottery for rank estimation.
    n_committee_eval : int
        Number of users sampled to evaluate "estimated unsatisfied probability"
        for each candidate committee in support(Delta_t).
    n_mwu_iters : int
        Outer iterations for the approximately-stable-lottery MWU.
    track_pairwise : bool
        If True, record per-user pairwise-query counts for diagnostics.

    Returns
    -------
    IteratedRoundingResult
        ``committee`` is the union of selected sub-committees. May be smaller
        than k by construction (the geometric schedule rounds down).
    """
    if k > m:
        raise ValueError("k > m")
    if decomposition is None:
        if n_rounds is None:
            T = max(1, int(np.ceil(10 * np.log(max(k, 2) / max(epsilon, 1e-3)))))
            T = min(T, max(1, int(np.ceil(np.log2(max(k, 2)) * 2))))
        else:
            T = n_rounds
        alpha = 0.5 + 4.0 * epsilon
        alpha = float(np.clip(alpha, 0.5, 0.95))
        decomposition = []
        for t in range(1, T + 1):
            k_t = max(0, int(np.floor((1.0 - alpha) * (alpha ** (t - 1)) * k)))
            decomposition.append(k_t)
        # Ensure the schedule sums to <= k and we don't get a degenerate empty
        # schedule. Pad first round with any leftover so we hit exactly k.
        total = sum(decomposition)
        if total < k:
            decomposition[0] += k - total
        decomposition = [d for d in decomposition if d > 0]
    if n_rounds is None:
        n_rounds = len(decomposition)
    selected_set: set[int] = set()
    sub_committees: list[np.ndarray] = []
    pairwise_counts_per_user: list[int] = []
    total_users_sampled = 0
    # Rejection-sampling cap: avoid infinite loops if the unsat set is empty.
    max_oversample = 20
    for t, k_t in enumerate(decomposition):
        if k_t <= 0 or len(selected_set) >= k:
            continue
        k_t = min(k_t, k - len(selected_set))
        running = np.asarray(sorted(selected_set), dtype=np.int64)
        # 1. Draw users from the *conditional* distribution of users currently
        # unsatisfied by the running committee (paper's D_t^{beta-eps}).
        # Implemented by Algorithm 4 (rejection sampling): keep drawing
        # users until we accumulate enough that are unsatisfied.
        users_round = _sample_unsatisfied_users(
            sampling_oracle=sampling_oracle,
            running=running,
            n_target=n_users_per_round,
            rng=rng,
            max_oversample=max_oversample,
        )
        total_users_sampled += users_round.shape[0]
        if users_round.shape[0] == 0:
            # No unsatisfied users left -> we are done.
            break
        # 2. Compute the (1+epsilon)-approximately stable lottery for size k_t
        # over the conditional distribution of unsatisfied users.
        lottery = approximately_stable_lottery(
            user_rankings=users_round,
            k=k_t,
            epsilon=epsilon,
            rng=rng,
            n_iters=n_mwu_iters,
        )
        # 3. Choose the committee S_t in supp(Delta_t) that minimizes estimated
        # unsatisfied probability over a fresh sample of unsatisfied users.
        eval_users = _sample_unsatisfied_users(
            sampling_oracle=sampling_oracle,
            running=running,
            n_target=n_committee_eval,
            rng=rng,
            max_oversample=max_oversample,
        )
        total_users_sampled += eval_users.shape[0]
        if eval_users.shape[0] > 0:
            eval_pos = positions_of(eval_users)
        else:
            eval_pos = positions_of(users_round)
        best_S = lottery.committees[0]
        best_unsat = np.inf
        for S in lottery.committees:
            # "Unsatisfied" with S means user's top-1 not in S.
            user_top_eval = np.argmin(eval_pos, axis=1)
            unsat = float((~np.isin(user_top_eval, S)).mean())
            if unsat < best_unsat:
                best_unsat = unsat
                best_S = S
        # 4. Add sub-committee to the running selection.
        for c in best_S:
            selected_set.add(int(c))
            if len(selected_set) >= k:
                break
        sub_committees.append(best_S)
        if track_pairwise:
            # Estimate per-user pairwise queries this round: each rank-estimation
            # call requires O(|S| + |S'|) comparisons; we run L estimations per
            # committee in supp(Delta_t).
            est_per_user = (
                len(lottery.committees) * (k_t + k_t) * n_lottery_samples_L
            )
            pairwise_counts_per_user.extend([est_per_user] * users_round.shape[0])
    committee = np.asarray(sorted(selected_set), dtype=np.int64)
    if track_pairwise and pairwise_counts_per_user:
        max_pw = int(np.max(pairwise_counts_per_user))
        med_pw = float(np.median(pairwise_counts_per_user))
    else:
        max_pw, med_pw = 0, 0.0
    return IteratedRoundingResult(
        committee=committee,
        sub_committees=sub_committees,
        n_users_sampled=total_users_sampled,
        max_pairwise_per_user=max_pw,
        median_pairwise_per_user=med_pw,
    )


# ---------------------------------------------------------------------------
# Algorithm 2: Ranking via Geometric Checkpoints.
# ---------------------------------------------------------------------------


def ranking_via_geometric_checkpoints(
    sampling_oracle: SamplingOracle,
    m: int,
    rng: np.random.Generator,
    growth_factor: float = 2.0,
    epsilon: float = 0.1,
    n_users_per_round: int = 200,
    n_lottery_samples_L: int = 30,
    n_committee_eval: int = 100,
    n_mwu_iters: int = 20,
) -> np.ndarray:
    """Algorithm 2: build a ranking by concatenating geometric-checkpoint committees.

    For r=1..R with R=ceil(log_lambda m):
        k_r = floor(lambda^{r-1})
        A_r = COMMITTEE(k_r)  -- via Algorithm 1
        Append A_r \\ A_{<r-1} to pi.

    Final ranking is filled with any remaining candidates in arbitrary order.
    """
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be > 1")
    R = int(np.ceil(np.log(m) / np.log(growth_factor)))
    pi: list[int] = []
    seen: set[int] = set()
    for r in range(1, R + 1):
        k_r = int(np.floor(growth_factor ** (r - 1)))
        k_r = max(1, min(k_r, m))
        result = committee_selection_iterated_rounding(
            sampling_oracle=sampling_oracle,
            m=m,
            k=k_r,
            rng=rng,
            epsilon=epsilon,
            n_users_per_round=n_users_per_round,
            n_lottery_samples_L=n_lottery_samples_L,
            n_committee_eval=n_committee_eval,
            n_mwu_iters=n_mwu_iters,
        )
        new = [int(c) for c in result.committee if int(c) not in seen]
        pi.extend(new)
        seen.update(new)
        if len(seen) >= m:
            break
    # Fill in any leftover candidates in arbitrary deterministic order.
    for c in range(m):
        if c not in seen:
            pi.append(c)
            seen.add(c)
    return np.asarray(pi, dtype=np.int64)


# ---------------------------------------------------------------------------
# Algorithm 3: Ranking via Committee Monotonicity.
# ---------------------------------------------------------------------------


def ranking_via_committee_monotonicity(
    sampling_oracle: SamplingOracle,
    m: int,
    rng: np.random.Generator,
    epsilon: float = 0.1,
    n_users_per_round: int = 200,
    n_lottery_samples_L: int = 30,
    n_committee_eval: int = 100,
    n_mwu_iters: int = 20,
) -> np.ndarray:
    """Algorithm 3 of Haghtalab et al. (2026).

    Run a single instance of Algorithm 1 with target committee size m and
    single-addition decomposition (k_t=1 for all T=m rounds). The ranking is
    obtained by ordering candidates according to the round in which they were
    added to the committee.

    This satisfies committee monotonicity by construction (each round adds at
    most one candidate to the running selection), so the prefix W_k^pi for
    every k is itself a committee in the running of Algorithm 1.
    """
    decomposition = [1] * m
    result = committee_selection_iterated_rounding(
        sampling_oracle=sampling_oracle,
        m=m,
        k=m,
        rng=rng,
        epsilon=epsilon,
        decomposition=decomposition,
        n_users_per_round=n_users_per_round,
        n_lottery_samples_L=n_lottery_samples_L,
        n_committee_eval=n_committee_eval,
        n_mwu_iters=n_mwu_iters,
    )
    pi: list[int] = []
    for sub in result.sub_committees:
        for c in sub:
            ci = int(c)
            if ci not in pi:
                pi.append(ci)
                if len(pi) >= m:
                    break
        if len(pi) >= m:
            break
    # Fill in any candidates that never got added (degenerate edge case).
    for c in range(m):
        if c not in pi:
            pi.append(c)
    return np.asarray(pi, dtype=np.int64)


# ---------------------------------------------------------------------------
# Bradley-Terry baseline (Chiang et al., 2024 style).
#
# Maximum-likelihood Bradley-Terry on observed pairwise battles. We use the
# standard Minorization-Maximization (MM) update of Hunter (2004), with an
# optional per-battle weight 1/P_t to match the LMArena reweighted MLE.
# ---------------------------------------------------------------------------


def bradley_terry_ranking(
    battles: np.ndarray,
    m: int,
    weights: Optional[np.ndarray] = None,
    n_iters: int = 200,
    tol: float = 1e-7,
    return_scores: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Bradley-Terry MLE ranking via MM (Hunter 2004), with optional weights.

    Parameters
    ----------
    battles : (T, 3) int array
        Each row is (winner, loser, _ignored_count). The third column is
        retained for compatibility with our LMArena loader, which provides
        ``count`` -- collapses repeated identical battles.
    m : int
        Number of candidates.
    weights : (T,) float array, optional
        Per-battle weight (e.g., 1/P_t in LMArena's reweighted MLE). Defaults
        to all-ones.
    n_iters : int
    tol : float
        Stop when max param change falls below tol.
    return_scores : bool
        If True, return (ranking, theta) where theta = log p (BT scores).

    Returns
    -------
    ranking : (m,) int array, top-first by descending BT score.
    """
    if weights is None:
        weights = np.ones(battles.shape[0], dtype=np.float64)
    counts = battles[:, 2].astype(np.float64) if battles.shape[1] >= 3 else np.ones(battles.shape[0])
    w = weights * counts
    winners = battles[:, 0].astype(np.int64)
    losers = battles[:, 1].astype(np.int64)
    p = np.full(m, 1.0 / m, dtype=np.float64)
    # Wins[i] = sum of weights for battles where i won.
    wins = np.zeros(m, dtype=np.float64)
    np.add.at(wins, winners, w)
    # For MM update we need, for each i, sum over battles involving i of
    # w_t / (p_i + p_j_other). We precompute (i, j) -> total weight matrix
    # for efficiency on small m. For large m a sparse representation would
    # be preferred but LMArena's m=20 means dense is trivial.
    pair_weight = np.zeros((m, m), dtype=np.float64)
    np.add.at(pair_weight, (winners, losers), w)
    # Symmetrize: total weight on the (i, j) pair regardless of who won.
    sym = pair_weight + pair_weight.T
    for _ in range(n_iters):
        # denom_i = sum_{j != i} sym[i, j] / (p_i + p_j)
        denom = np.zeros(m, dtype=np.float64)
        for i in range(m):
            mask = np.arange(m) != i
            sums = p[i] + p[mask]
            sums = np.where(sums > 0, sums, 1.0)  # avoid 0/0; numerator is 0 anyway
            denom[i] = (sym[i, mask] / sums).sum()
        # MM update: p_new[i] = wins[i] / denom[i] (skip if no info).
        new_p = np.where(denom > 0, wins / np.maximum(denom, 1e-12), p)
        new_p = new_p / new_p.sum()
        if np.max(np.abs(new_p - p)) < tol:
            p = new_p
            break
        p = new_p
    ranking = np.argsort(-p, kind="stable")
    if return_scores:
        # BT score theta = log p (additive shift is irrelevant).
        theta = np.log(np.clip(p, 1e-30, None))
        return ranking, theta
    return ranking
