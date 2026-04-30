"""End-to-end LMArena reproduction.

Runs the experiment in Section 5.2 of Haghtalab et al. (2026):

    1. Load the LMArena 140k dataset and filter to the top m=20 LLMs.
    2. Compute per-category Bradley-Terry rankings (Mallows central rankings).
    3. Build a semi-synthetic Mallows mixture user distribution with one
       component per category and mixture weights = relative pairwise-comparison
       counts per category.
    4. For each dispersion phi in {0.1, 0.5, 0.9}:
         a. Compute the BT ranking using the global pairwise-comparison sample
            (matches the LMArena leaderboard mechanism).
         b. Compute the Algorithm 2 (geometric checkpoints) ranking.
         c. Compute the Algorithm 3 (committee monotonicity) ranking.
         d. Compute the gamma_hat stability curve for each ranking at every k.
    5. Save numeric results and plots.

Outputs (relative to repository root):
    results/results.json
    results/run_log.md  (appended)
    results/fig_stability_vs_bt.png
    results/fig_mallows_clusters.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the package importable when run as a script from any cwd.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from pluralistic_leaderboards.algorithms import (  # noqa: E402
    bradley_terry_ranking,
    make_mallows_mixture_oracle,
    ranking_via_committee_monotonicity,
    ranking_via_geometric_checkpoints,
)
from pluralistic_leaderboards.data import (  # noqa: E402
    load_lmarena,
    per_category_central_rankings,
)
from pluralistic_leaderboards.evaluation import (  # noqa: E402
    stability_approximation_factor,
    stability_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-models", type=int, default=20)
    p.add_argument("--max-battles", type=int, default=20_000)
    p.add_argument("--phis", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--n-eval-users", type=int, default=10_000)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n-users-per-round", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"[1/5] Loading LMArena data (n_models={args.n_models}, max_battles={args.max_battles}) ...")
    sample = load_lmarena(
        n_models=args.n_models,
        max_battles=args.max_battles,
        seed=args.seed,
    )
    print(f"      loaded {sample.n_battles} battles across {sample.n_models} models")
    print(f"      categories: {list(sample.category_to_idx.keys())}")
    print(f"      mixture weights: {sample.category_mixture_weights.round(3).tolist()}")
    print(f"[2/5] Computing per-category Bradley-Terry central rankings ...")
    central_rankings_dict = per_category_central_rankings(sample)
    cat_names = list(sample.category_to_idx.keys())
    centers = [central_rankings_dict[c] for c in cat_names]
    weights = sample.category_mixture_weights
    # Print top model per category for sanity
    for c, r in zip(cat_names, centers):
        print(f"      [{c}] top: {sample.model_names[int(r[0])]}")
    print(f"[3/5] Computing global Bradley-Terry ranking on observed battles ...")
    bt_global = bradley_terry_ranking(sample.battles, m=sample.n_models)
    print(f"      BT top-3: {[sample.model_names[i] for i in bt_global[:3]]}")
    results: dict = {
        "config": {
            "n_models": sample.n_models,
            "max_battles_used": sample.n_battles,
            "phis": list(args.phis),
            "n_eval_users": args.n_eval_users,
            "epsilon": args.epsilon,
            "n_users_per_round": args.n_users_per_round,
            "seed": args.seed,
        },
        "model_names": sample.model_names,
        "categories": cat_names,
        "category_mixture_weights": weights.round(6).tolist(),
        "category_top_models": {
            c: sample.model_names[int(r[0])] for c, r in zip(cat_names, centers)
        },
        "bt_global_ranking": [sample.model_names[i] for i in bt_global.tolist()],
        "phi_runs": [],
    }
    print(f"[4/5] Running rankings + stability comparison for phi in {list(args.phis)} ...")
    per_phi_curves: dict[float, dict[str, np.ndarray]] = {}
    per_phi_rankings: dict[float, dict[str, list[str]]] = {}
    for phi in args.phis:
        print(f"  phi = {phi}")
        rng_local = np.random.default_rng(args.seed + int(phi * 1000))
        oracle = make_mallows_mixture_oracle(centers=centers, phi=phi, weights=weights)
        # Eval users for ground-truth gamma_hat estimates.
        eval_users = oracle(args.n_eval_users, np.random.default_rng(args.seed + 9999))
        # Bradley-Terry on a fresh sample of pairwise comparisons drawn from
        # the eval users (= "ranking by Borda count over all sampled users",
        # per the paper, equivalent in the limit to BT on all pairwise
        # comparisons). For pragmatism we re-use the global BT here, which is
        # the canonical LMArena baseline.
        bt = bt_global.copy()
        # Algorithm 2
        t_a2 = time.time()
        a2 = ranking_via_geometric_checkpoints(
            sampling_oracle=oracle,
            m=sample.n_models,
            rng=rng_local,
            growth_factor=2.0,
            epsilon=args.epsilon,
            n_users_per_round=args.n_users_per_round,
            n_lottery_samples_L=15,
            n_committee_eval=80,
            n_mwu_iters=12,
        )
        a2_time = time.time() - t_a2
        # Algorithm 3
        t_a3 = time.time()
        a3 = ranking_via_committee_monotonicity(
            sampling_oracle=oracle,
            m=sample.n_models,
            rng=rng_local,
            epsilon=args.epsilon,
            n_users_per_round=args.n_users_per_round,
            n_lottery_samples_L=15,
            n_committee_eval=80,
            n_mwu_iters=12,
        )
        a3_time = time.time() - t_a3
        # Stability curves
        bt_curve = stability_curve(eval_users, bt, m=sample.n_models)
        a2_curve = stability_curve(eval_users, a2, m=sample.n_models)
        a3_curve = stability_curve(eval_users, a3, m=sample.n_models)
        per_phi_curves[phi] = {"bt": bt_curve, "a2": a2_curve, "a3": a3_curve}
        per_phi_rankings[phi] = {
            "bt": [sample.model_names[i] for i in bt.tolist()],
            "a2": [sample.model_names[i] for i in a2.tolist()],
            "a3": [sample.model_names[i] for i in a3.tolist()],
        }
        # Summary metrics for the run record
        results["phi_runs"].append({
            "phi": phi,
            "rankings": per_phi_rankings[phi],
            "stability_curves": {
                "bt": bt_curve.round(4).tolist(),
                "a2": a2_curve.round(4).tolist(),
                "a3": a3_curve.round(4).tolist(),
            },
            "n_unstable_prefixes": {
                "bt": int((bt_curve > 1.0).sum()),
                "a2": int((a2_curve > 1.0).sum()),
                "a3": int((a3_curve > 1.0).sum()),
            },
            "max_gamma": {
                "bt": float(bt_curve.max()),
                "a2": float(a2_curve.max()),
                "a3": float(a3_curve.max()),
            },
            "mean_gamma": {
                "bt": float(bt_curve.mean()),
                "a2": float(a2_curve.mean()),
                "a3": float(a3_curve.mean()),
            },
            "runtime_s": {"a2": round(a2_time, 2), "a3": round(a3_time, 2)},
        })
        print(
            f"    BT max-gamma={bt_curve.max():.3f} "
            f"A2 max-gamma={a2_curve.max():.3f} "
            f"A3 max-gamma={a3_curve.max():.3f}  "
            f"(unstable prefixes: BT={int((bt_curve > 1.0).sum())} "
            f"A2={int((a2_curve > 1.0).sum())} "
            f"A3={int((a3_curve > 1.0).sum())})"
        )
    print(f"[5/5] Plotting and saving outputs to {args.out_dir} ...")
    _plot_stability_curves(per_phi_curves, args.phis, args.out_dir / "fig_stability_vs_bt.png")
    _plot_mallows_clusters(centers, weights, cat_names, sample.model_names, args.out_dir / "fig_mallows_clusters.png")
    results_path = args.out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    elapsed = time.time() - t0
    results["wallclock_s"] = round(elapsed, 1)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"      results saved to {results_path}")
    print(f"      wall time: {elapsed:.1f}s")


def _plot_stability_curves(per_phi: dict, phis: list, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(phis), figsize=(5 * len(phis), 4), sharey=True)
    if len(phis) == 1:
        axes = [axes]
    for ax, phi in zip(axes, phis):
        curves = per_phi[phi]
        k_axis = np.arange(1, len(curves["bt"]) + 1)
        ax.plot(k_axis, curves["bt"], "o-", label="Bradley-Terry", color="C0")
        ax.plot(k_axis, curves["a2"], "v-", label="Algorithm 2", color="C2")
        ax.plot(k_axis, curves["a3"], "x-", label="Algorithm 3", color="C1")
        ax.axhline(1.0, ls="--", color="gray", alpha=0.6, label="Stability threshold")
        ax.set_xlabel("Top-k committee size")
        ax.set_title(f"Ranking stability (phi = {phi})")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"Stability approximation factor $\hat{\gamma}$")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_mallows_clusters(
    centers: list[np.ndarray],
    weights: np.ndarray,
    cat_names: list[str],
    model_names: list[str],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    K = len(centers)
    m = centers[0].shape[0]
    fig, ax = plt.subplots(figsize=(min(2 + 0.4 * m, 12), max(3, 0.3 * K)))
    # Heatmap: row = category, col = model, value = position in central ranking.
    grid = np.empty((K, m))
    for k, c in enumerate(centers):
        for pos, model_idx in enumerate(c):
            grid[k, model_idx] = pos
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(K))
    yticklabels = [f"{cat_names[k]} (w={weights[k]:.2f})" for k in range(K)]
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xticks(range(m))
    ax.set_xticklabels(model_names, rotation=70, ha="right", fontsize=7)
    ax.set_title("Per-category Bradley-Terry central rankings (lower = better)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("Rank position")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
