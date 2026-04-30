"""Smoke tests for data.py.

The full LMArena dataset is large (1.6 GB) and download time would dominate
test runtime. We test the loader's data-flattening logic on synthetic minimal
inputs and gate the actual HuggingFace download behind an env flag.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pluralistic_leaderboards.data import (
    LMArenaSample,
    _flatten_category_tag,
    per_category_central_rankings,
)


def test_flatten_category_tag_priorities():
    assert _flatten_category_tag(None) == "general__low"
    assert _flatten_category_tag({}) == "general__low"
    # Math takes priority over creative_writing.
    assert _flatten_category_tag(
        {"math_v0.1": {"math": True}, "creative_writing_v0.1": {"creative_writing": True}}
    ).startswith("math__")
    # Creative writing.
    cat = _flatten_category_tag({"creative_writing_v0.1": {"creative_writing": True}})
    assert cat.startswith("creative_writing__")
    # Instruction following.
    cat = _flatten_category_tag({"if_v0.1": {"if": True, "score": 4}})
    assert cat.startswith("instruction_following__")
    # Difficulty buckets via criteria_v0.1 truthy counts.
    high = _flatten_category_tag({
        "math_v0.1": {"math": True},
        "criteria_v0.1": {k: True for k in (
            "complexity", "creativity", "domain_knowledge",
            "problem_solving", "real_world", "specificity", "technical_accuracy"
        )}
    })
    assert high == "math__high"
    low = _flatten_category_tag({
        "math_v0.1": {"math": True},
        "criteria_v0.1": {k: False for k in (
            "complexity", "creativity", "domain_knowledge",
            "problem_solving", "real_world", "specificity", "technical_accuracy"
        )}
    })
    assert low == "math__low"


def test_per_category_central_rankings_on_synthetic():
    """Build a tiny synthetic LMArenaSample and check per-category BT rankings."""
    raw = pd.DataFrame({
        "winner": [0, 0, 1, 2, 2],
        "loser":  [1, 2, 2, 0, 1],
        "category": ["math", "math", "math", "creative_writing", "creative_writing"],
    })
    grouped = raw.groupby(["winner", "loser"]).size().reset_index(name="count")
    battles = grouped[["winner", "loser", "count"]].to_numpy(dtype=np.int64)
    sample = LMArenaSample(
        battles=battles,
        raw_battles=raw,
        model_names=["A", "B", "C"],
        n_models=3,
        n_battles=raw.shape[0],
        category_to_idx={"math": 0, "creative_writing": 1},
        category_battle_counts=np.array([3, 2], dtype=np.float64),
        category_mixture_weights=np.array([0.6, 0.4], dtype=np.float64),
    )
    rankings = per_category_central_rankings(sample)
    assert "math" in rankings and "creative_writing" in rankings
    # Math: 0 wins twice, 1 wins once, 2 loses both -> 0 should be on top.
    assert rankings["math"][0] == 0
    # Creative writing: 2 wins both -> 2 should be on top.
    assert rankings["creative_writing"][0] == 2


@pytest.mark.skipif(
    os.environ.get("PLURALISTIC_LEADERBOARDS_LIVE_LMARENA") != "1",
    reason="Live LMArena fetch is slow; set PLURALISTIC_LEADERBOARDS_LIVE_LMARENA=1 to enable.",
)
def test_load_lmarena_live_smoke():
    from pluralistic_leaderboards.data import load_lmarena

    sample = load_lmarena(n_models=20, max_battles=2_000, seed=0)
    assert sample.n_models == 20
    assert sample.n_battles > 0
    assert sample.battles.shape[1] == 3
    assert all(0 <= int(w) < sample.n_models for w in sample.battles[:, 0])
    assert all(0 <= int(l) < sample.n_models for l in sample.battles[:, 1])
    assert abs(sample.category_mixture_weights.sum() - 1.0) < 1e-6
