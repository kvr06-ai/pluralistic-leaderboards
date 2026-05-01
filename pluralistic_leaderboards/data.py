"""LMArena data loaders.

Two HuggingFace datasets are supported:

1. ``lmarena-ai/arena-human-preference-140k`` (canonical) -- general public
   votes, nested ``category_tag`` dict.
2. ``lmarena-ai/arena-expert-5k`` (Nov 2025, sensitivity-check) -- expert-only
   votes, ``occupational_tags`` dict with 23 boolean keys (no ``category_tag``).

Schema cheatsheet (relevant fields only):

    arena-human-preference-140k:
        - model_a, model_b   : str
        - winner             : str in {model_a, model_b, tie, both_bad}
        - category_tag       : nested dict (math_v0.1, creative_writing_v0.1,
          if_v0.1, criteria_v0.1, ...).
        - id, evaluation_session_id, evaluation_order : provenance.

    arena-expert-5k:
        - id                 : str
        - model_a, model_b   : str (105 unique models in source)
        - winner             : str in {model_a, model_b, tie, both_bad}
        - conversation_a/b   : str (conversation transcripts; not used here).
        - full_conversation  : str (multi-turn context; not used here).
        - language           : str (35 unique codes, e.g. en/zh/ru).
        - occupational_tags  : flat dict[str, bool] over 23 occupational
          categories (business, engineering, mathematical, medicine, ...).
        - evaluation_order   : int (1-24).
        NB: there is no ``category_tag`` field. The flatten rule below uses
        ``occupational_tags`` (first truthy occupation) as the category and
        falls back to ``general`` when no tag fires.

Following Section 5.2 of the paper, we:
    1. Filter to the top m models by overall battle frequency.
    2. Treat each category value as a Mallows mixture component.
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


# ---------------------------------------------------------------------------
# arena-expert-5k loader (sensitivity-check dataset).
# ---------------------------------------------------------------------------

# 23 occupational categories from the arena-expert-5k schema (boolean fields
# inside ``occupational_tags``). Order is arbitrary -- when multiple tags fire
# we pick the first truthy in this canonical order to keep flatten output
# deterministic.
_OCCUPATION_KEYS = (
    "business_and_management_and_financial_operations",
    "community_and_social_service",
    "construction_and_extraction",
    "education",
    "engineering_and_architecture",
    "entertainment_and_sports_and_media",
    "farming_and_fishing_and_forestry",
    "food_preparation_and_serving",
    "legal_and_government",
    "life_and_physical_and_social_science",
    "mathematical",
    "medicine_and_healthcare",
    "office_and_administrative_support",
    "personal_care_and_service",
    "philosophy_and_religion_and_theology",
    "production_and_industrial",
    "real_estate",
    "sales_and_retail",
    "software_and_it_services",
    "technology_hardware_and_equipment",
    "travel",
    "visual_arts_and_design",
    "writing_and_literature_and_language",
)

# Compact short-name aliases for plot legibility (the raw keys are too long
# to show on a heatmap axis). The mapping is unambiguous because each raw
# key has a unique short form.
_OCCUPATION_SHORT = {
    "business_and_management_and_financial_operations": "business",
    "community_and_social_service": "community",
    "construction_and_extraction": "construction",
    "education": "education",
    "engineering_and_architecture": "engineering",
    "entertainment_and_sports_and_media": "entertainment",
    "farming_and_fishing_and_forestry": "farming",
    "food_preparation_and_serving": "food",
    "legal_and_government": "legal",
    "life_and_physical_and_social_science": "science",
    "mathematical": "math",
    "medicine_and_healthcare": "medicine",
    "office_and_administrative_support": "office",
    "personal_care_and_service": "personal_care",
    "philosophy_and_religion_and_theology": "philosophy",
    "production_and_industrial": "production",
    "real_estate": "real_estate",
    "sales_and_retail": "sales",
    "software_and_it_services": "software",
    "technology_hardware_and_equipment": "hardware",
    "travel": "travel",
    "visual_arts_and_design": "visual_arts",
    "writing_and_literature_and_language": "writing",
}


def _flatten_occupational_tag(tag: dict | None) -> str:
    """Pick a single canonical occupational category for an arena-expert-5k row.

    The arena-expert-5k schema stores ``occupational_tags`` as a *flat* dict
    of 23 boolean fields (e.g. ``mathematical: True``,
    ``software_and_it_services: True``). For the Mallows mixture, we collapse
    each row to a single category by picking the first truthy occupation in
    the canonical ``_OCCUPATION_KEYS`` order. Rows with no truthy tag (the
    most common case for short prompts) fall back to ``general``.

    Returns the *short* alias name (e.g. ``mathematical`` -> ``math``,
    ``software_and_it_services`` -> ``software``) for downstream
    readability. The mapping is one-to-one so no information is lost.
    """
    if not isinstance(tag, dict) or not tag:
        return "general"
    for key in _OCCUPATION_KEYS:
        if bool(tag.get(key, False)):
            return _OCCUPATION_SHORT[key]
    return "general"


def load_lmarena_expert5k(
    n_models: int = 20,
    max_battles: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    seed: int = 0,
) -> LMArenaSample:
    """Load and preprocess the LMArena ``arena-expert-5k`` dataset.

    Parallel to :func:`load_lmarena` but targets the expert-only 5k slice
    (``lmarena-ai/arena-expert-5k``, last updated 2025-11-05). The schema
    differs from the 140k dataset in two ways relevant here:

    - There is no ``category_tag`` field. Categories are derived from
      ``occupational_tags`` (a flat dict of 23 boolean occupations) via
      :func:`_flatten_occupational_tag`.
    - There are 105 unique models. Top-m filtering applies as before.

    Returns the same :class:`LMArenaSample` dataclass so downstream code in
    :func:`per_category_central_rankings`, the algorithms module, and the
    evaluation module is reused unchanged.

    Parameters
    ----------
    n_models : int
        Top-m models to keep, ranked by total appearance frequency (model_a
        or model_b). Defaults to 20 for parity with the 140k canonical run;
        callers should reduce this if the dataset has fewer than ``n_models``
        sufficiently-supported models.
    max_battles : int or None
        If not None, subsample to at most this many battles after filtering.
        Defaults to ``None`` because the source dataset is already small (~5k).
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
        "lmarena-ai/arena-expert-5k",
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
    df["category"] = df["occupational_tags"].apply(_flatten_occupational_tag)
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
    cat_weights = cat_counts / cat_counts.sum() if cat_counts.sum() > 0 else cat_counts
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
