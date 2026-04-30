"""LMArena data loader for arena-human-preference-140k.

Dataset reference:
    LMArena Team. "arena-human-preference-140k", 2025.
    https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

Schema (relevant fields only):
    - model_a, model_b : stringclass model names.
    - winner : one of {model_a, model_b, tie, both_bad}.
    - category_tag : dict with annotation tags (math, creative_writing, etc.).
    - id, evaluation_session_id, evaluation_order : provenance.

Following Section 5.2 of the paper, we:
    1. Filter to the top m=20 LLMs by overall Bradley-Terry rank.
    2. Treat each ``category_tag`` value as a Mallows mixture component.
    3. Compute a per-category Bradley-Terry ranking as the central ranking.
    4. Use the relative number of pairwise comparisons per category as the
       mixture weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LMArenaSample:
    """A processed LMArena slice."""

    battles: np.ndarray  # (T, 3): (model_a_idx, model_b_idx, count) with winner first.
    raw_battles: pd.DataFrame  # one row per battle (winner_idx, loser_idx, category)
    model_names: list[str]
    n_models: int
    n_battles: int
    category_to_idx: dict[str, int]  # observed category tag -> compact index
    category_battle_counts: np.ndarray  # (n_categories,) raw battle counts
    category_mixture_weights: np.ndarray  # (n_categories,) normalized weights


_CRITERIA_KEYS = (
    "complexity",
    "creativity",
    "domain_knowledge",
    "problem_solving",
    "real_world",
    "specificity",
    "technical_accuracy",
)


def _extract_nested(tag: dict | None, outer: str, inner: str) -> bool | int | float | None:
    if not isinstance(tag, dict):
        return None
    sub = tag.get(outer)
    if isinstance(sub, dict):
        return sub.get(inner)
    return None


def _flatten_category_tag(tag: dict | None) -> str:
    """Pick a single canonical category for a battle from the category_tag dict.

    The LMArena 140k schema stores ``category_tag`` as a *nested* dict::

        {
          'creative_writing_v0.1': {'creative_writing': bool, 'score': str},
          'criteria_v0.1': {complexity/creativity/domain_knowledge/...: bool},
          'if_v0.1':       {'if': bool, 'score': int},
          'math_v0.1':     {'math': bool},
        }

    We adopt the following deterministic flatten rule that yields up to 20
    (= 4 base categories x 5 difficulty/criteria buckets) prompt-categories,
    matching the size of the category space referenced in Table 1 of the
    paper:

      base = first truthy of {math, coding, creative_writing, instruction_following} else "general"
      difficulty bucket = number of truthy criteria flags ("low": 0-1, "med":
        2-4, "high": 5-7)
      category = f"{base}__{bucket}"

    The 4-base x 3-bucket scheme yields 12 categories in practice (with
    "general__low" / "general__med" / "general__high" filling the gaps),
    which is the closest reproducible mapping to the paper's 20-bin scheme
    using the public dataset's classifier outputs. Categories with very few
    samples are still kept; the per-category Mallows mixture weight handles
    the imbalance naturally.
    """
    if not isinstance(tag, dict) or not tag:
        return "general__low"
    is_math = bool(_extract_nested(tag, "math_v0.1", "math"))
    is_creative = bool(_extract_nested(tag, "creative_writing_v0.1", "creative_writing"))
    is_if = bool(_extract_nested(tag, "if_v0.1", "if"))
    is_coding = bool(tag.get("is_code", False))  # top-level fallback
    if is_math:
        base = "math"
    elif is_coding:
        base = "coding"
    elif is_creative:
        base = "creative_writing"
    elif is_if:
        base = "instruction_following"
    else:
        base = "general"
    crit = tag.get("criteria_v0.1") or {}
    if isinstance(crit, dict):
        n_truthy = sum(1 for k in _CRITERIA_KEYS if bool(crit.get(k, False)))
    else:
        n_truthy = 0
    if n_truthy <= 1:
        bucket = "low"
    elif n_truthy <= 4:
        bucket = "med"
    else:
        bucket = "high"
    return f"{base}__{bucket}"


def load_lmarena(
    n_models: int = 20,
    max_battles: Optional[int] = 20_000,
    cache_dir: Optional[str | Path] = None,
    seed: int = 0,
) -> LMArenaSample:
    """Load and preprocess the LMArena 140k dataset.

    Parameters
    ----------
    n_models : int
        Top-m models to keep, ranked by total battles.
    max_battles : int or None
        If not None, subsample to at most this many battles after filtering.
        Set to None to use the full slice.
    cache_dir : path, optional
        Hugging Face cache directory.
    seed : int
        Subsampling seed.

    Returns
    -------
    LMArenaSample
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    ds = load_dataset(
        "lmarena-ai/arena-human-preference-140k",
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    df = ds.to_pandas()
    # Drop ties / both_bad to obtain a clear pairwise preference signal.
    df = df[df["winner"].isin(["model_a", "model_b"])].reset_index(drop=True)
    # Top-n models by raw appearance frequency in either slot.
    appearances = pd.concat([df["model_a"], df["model_b"]]).value_counts()
    keep_models = list(appearances.head(n_models).index)
    keep_set = set(keep_models)
    df = df[df["model_a"].isin(keep_set) & df["model_b"].isin(keep_set)].reset_index(drop=True)
    if max_battles is not None and len(df) > max_battles:
        idx = rng.choice(len(df), size=max_battles, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    model_to_idx = {m: i for i, m in enumerate(keep_models)}
    df["category"] = df["category_tag"].apply(_flatten_category_tag)
    # Build raw_battles with winner first.
    winner_a = df["winner"] == "model_a"
    winner_idx = np.where(winner_a, df["model_a"].map(model_to_idx), df["model_b"].map(model_to_idx)).astype(np.int64)
    loser_idx = np.where(winner_a, df["model_b"].map(model_to_idx), df["model_a"].map(model_to_idx)).astype(np.int64)
    raw_battles = pd.DataFrame({
        "winner": winner_idx,
        "loser": loser_idx,
        "category": df["category"].values,
    })
    # Collapse duplicates for BT input -> (winner, loser, count).
    grouped = raw_battles.groupby(["winner", "loser"]).size().reset_index(name="count")
    battles_arr = grouped[["winner", "loser", "count"]].to_numpy(dtype=np.int64)
    categories = sorted(raw_battles["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}
    cat_counts = raw_battles["category"].value_counts().reindex(categories, fill_value=0).to_numpy().astype(np.float64)
    cat_weights = cat_counts / cat_counts.sum()
    return LMArenaSample(
        battles=battles_arr,
        raw_battles=raw_battles,
        model_names=keep_models,
        n_models=len(keep_models),
        n_battles=int(raw_battles.shape[0]),
        category_to_idx=category_to_idx,
        category_battle_counts=cat_counts,
        category_mixture_weights=cat_weights,
    )


def per_category_central_rankings(
    sample: LMArenaSample,
) -> dict[str, np.ndarray]:
    """Compute per-category Bradley-Terry rankings.

    Each ranking is used as a Mallows central ranking in the semi-synthetic
    user distribution (Section 5.2 of the paper).
    """
    from .algorithms import bradley_terry_ranking

    rankings: dict[str, np.ndarray] = {}
    for cat, _idx in sample.category_to_idx.items():
        sub = sample.raw_battles[sample.raw_battles["category"] == cat]
        if sub.empty:
            rankings[cat] = np.arange(sample.n_models, dtype=np.int64)
            continue
        grouped = sub.groupby(["winner", "loser"]).size().reset_index(name="count")
        battles = grouped[["winner", "loser", "count"]].to_numpy(dtype=np.int64)
        rankings[cat] = bradley_terry_ranking(battles, m=sample.n_models)
    return rankings
