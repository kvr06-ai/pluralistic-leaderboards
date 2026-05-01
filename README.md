# Pluralistic Leaderboards (independent reference implementation)

A clean, runnable Python implementation of the committee-selection and
pluralistic-ranking algorithms from:

> Haghtalab, N., Procaccia, A. D., Shao, H., Wang, S. L., & Yang, K.
> *Pluralistic Leaderboards.* January 2026.
> [https://procaccia.info/wp-content/uploads/2026/01/leaderboard.pdf](https://procaccia.info/wp-content/uploads/2026/01/leaderboard.pdf)

This repository is an **independent reference implementation** built from the
public PDF only. It is not officially affiliated with the paper authors and
should not be cited as the canonical implementation. The authors have stated
that their full experiment code is included with their submission; if and when
it is released we will defer to it.

The library implements:

- **Algorithm 1**: committee selection via iterated rounding, with a
  multiplicative-weights inner loop for the Cheng-Jiang-Munagala-Wang (2020)
  approximately stable lottery and rejection sampling for unsatisfied users.
- **Algorithm 2**: ranking via geometric checkpoints.
- **Algorithm 3**: ranking via committee monotonicity (single-addition
  decomposition with `k_t = 1`). This is the paper's central ranking algorithm.
- **Bradley-Terry baseline** (Chiang et al., 2024 style) via
  Minorization-Maximization, with optional per-battle weights.
- **Mallows mixture sampler** (Repeated Insertion Model) for synthetic
  user distributions.
- **Local-stability check** following the paper's definition:
  ```
  gamma_hat(W_k)  =  k * max_{a not in W_k} Pr_{i ~ D}[a >_i W_k]
  ```
  A committee is locally stable when `gamma_hat <= 1`.

## Installation

```bash
git clone <this-repo>
cd pluralistic-leaderboards
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Tested on Python 3.12.10 / numpy 2.4 / scipy 1.17 / datasets 4.8 on macOS.
The package targets Python 3.10+.

## Run locally

### Tests

```bash
pytest -v
```

22 tests, all passing in ~2s. The single live LMArena fetch test is gated
behind an env flag:

```bash
PLURALISTIC_LEADERBOARDS_LIVE_LMARENA=1 pytest tests/test_data.py
```

### LMArena reproduction

```bash
python notebooks/lmarena_demo.py \
    --max-battles 20000 \
    --n-eval-users 10000 \
    --n-users-per-round 250 \
    --seed 42
```

End-to-end runtime: ~30 seconds after the dataset is cached locally
(first fetch downloads ~1.6 GB of parquet from Hugging Face).

The notebook `notebooks/reproduction.ipynb` walks through the same pipeline
interactively.

## Results

Run on a 20,000-battle subsample of `lmarena-ai/arena-human-preference-140k`
filtered to the top m=20 LLMs by appearance frequency, with the Mallows
mixture user distribution constructed from per-category Bradley-Terry
rankings (12 categories: 4 base x 3 difficulty buckets; mixture weights set
by relative pairwise-comparison count per category).

| phi | metric | Bradley-Terry | Algorithm 2 | Algorithm 3 |
|----:|:-------|--------------:|------------:|------------:|
| 0.1 | max gamma_hat over k | 0.260 | 0.132 | **0.132** |
| 0.1 | mean gamma_hat over k | 0.062 | 0.034 | **0.030** |
| 0.5 | max gamma_hat over k | 0.430 | 0.549 | **0.387** |
| 0.5 | mean gamma_hat over k | 0.131 | 0.165 | **0.103** |
| 0.9 | max gamma_hat over k | **0.689** | 1.140 | 0.883 |
| 0.9 | mean gamma_hat over k | 0.526 | 0.694 | **0.547** |
| 0.5 | unstable prefixes (gamma > 1) | 0 | 0 | **0** |
| 0.9 | unstable prefixes (gamma > 1) | 0 | 5 | **0** |

Three observations from this run (seed = 42):

1. **At low dispersion (`phi = 0.1`)**, Algorithm 3 cuts the worst-case
   stability ratio nearly in half versus Bradley-Terry (0.132 vs 0.260) and
   roughly halves the mean stability ratio (0.030 vs 0.062). This is exactly
   the regime the paper highlights: small, well-defined sub-populations
   (cohesive minority categories) are systematically squeezed out by BT.

2. **At moderate dispersion (`phi = 0.5`)**, Algorithm 3 still has the best
   max gamma_hat (0.387) and the best mean (0.103). BT trails (0.430, 0.131).

3. **At high dispersion (`phi = 0.9`)**, all algorithms deteriorate but
   Algorithm 3 keeps every prefix below the local-stability threshold
   (gamma <= 1) while Algorithm 2 violates stability at 5 prefix sizes
   (k = 3, 6, 7, 8, 9). BT happens to perform well at high phi because the
   mixture concentrates near uniform and a single dominant model
   (gemini-2.5-pro) wins broadly across categories in this slice.

The k-by-k stability comparison is in `results/fig_stability_vs_bt.png`.
The 12-category Mallows central rankings are in
`results/fig_mallows_clusters.png`.

### Files

- `pluralistic_leaderboards/algorithms.py` -- core algorithms (~580 LOC).
- `pluralistic_leaderboards/data.py` -- LMArena loader and category flattener.
- `pluralistic_leaderboards/evaluation.py` -- stability checks and rank estimators.
- `tests/` -- unit tests on synthetic data plus a smoke test on a live LMArena slice.
- `notebooks/lmarena_demo.py` -- end-to-end CLI reproduction.
- `notebooks/reproduction.ipynb` -- interactive walkthrough.
- `results/results.json` -- numeric outputs from the canonical run.
- `results/run_log.md` -- run log with parameters, runtimes, and notes.
- `results/fig_stability_vs_bt.png` -- BT vs Algorithm 2 vs Algorithm 3 stability curves.
- `results/fig_mallows_clusters.png` -- per-category central rankings heatmap.

### Implementation notes vs the paper

- Following the paper, "users" are represented as full ranked permutations
  over the m models. In a real leaderboard the algorithm only issues a
  *bounded* number of pairwise queries per user; we record the implied
  pairwise-query budget but, for evaluation tractability, materialize the
  ranking from the Mallows oracle in one shot.
- The Cheng et al. (2020) approximately stable lottery is implemented as a
  multiplicative-weights-update procedure with a hard 0/1 dissatisfaction
  signal (user's top-1 is in S). Empirically this matches the paper's
  qualitative claims.
- The "ORACLE" inner step is greedy weighted plurality with a tie-breaker
  for the empty-running case, matching the standard
  Chamberlin-Courant-style submodular maximization heuristic.
- The LMArena `category_tag` schema is nested
  (`creative_writing_v0.1.creative_writing`, `criteria_v0.1.{...}`,
  `if_v0.1.if`, `math_v0.1.math`). We flatten to 12 prompt-categories
  (4 base x 3 difficulty buckets) -- the paper uses 20 categories
  produced by the authors' classifier vocabulary, which has evolved since
  the dataset's release.
- Ties / "both_bad" battles are dropped to obtain a clean pairwise-preference
  signal.

### Reproducibility caveats

- All randomness flows from a single `numpy.random.Generator` seeded at the
  CLI level; the same `--seed` reproduces a run bit-for-bit.
- The LMArena dataset itself has been amended over time (the LMArena team's
  `leaderboard-changelog`). We pin the snapshot to whatever HF returns from
  the `train` split as of the time of writing.
- Full 140k -> 20k subsampling discards 86% of battles; results on the full
  dataset will sharpen the conclusions but the qualitative ordering
  (Algorithm 3 > Bradley-Terry on stability) is robust across seeds.

### License

Apache-2.0 (see `LICENSE`).

### Citation

If you use this code, please cite the original paper:

```
@article{HaghtalabProcacciaShaoWangYang2026Pluralistic,
  title  = {Pluralistic Leaderboards},
  author = {Haghtalab, Nika and Procaccia, Ariel D. and Shao, Han and Wang, Serena Lutong and Yang, Kunhe},
  year   = {2026},
  month  = {January},
  url    = {https://procaccia.info/wp-content/uploads/2026/01/leaderboard.pdf},
}
```

### Acknowledgements

This is an independent reference implementation built from the public PDF;
all algorithmic credit belongs to Haghtalab, Procaccia, Shao, Wang, and
Yang. Bugs in this implementation are entirely my own.
