"""Is any of the WESAD tuning real, or is it all fold noise?

A single pooled balanced accuracy hides how much it moves from subject to subject. With only 15
subjects, LOSO point estimates carry a standard error large enough to swallow every "improvement"
the sweeps produced -- so this scores each config *per held-out subject*, then compares configs with
a paired test on the same subjects (paired, because the same person is easy or hard for every
config, and pairing removes that shared variance).

Reports, per config: mean per-subject balanced accuracy +/- standard error, and a paired comparison
against the currently shipped configuration.

Usage: uv run python scripts/wesad_significance.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

from wesad_experiments import (  # noqa: E402
    CACHE_PATH,
    MODEL_COLUMNS,
    Config,
    _smooth,
    build_table,
)

CANDIDATES = [
    Config("shipped: step30 / ref 300s", step_sec=30.0, reference_seconds=300.0),
    Config("step10 / ref 120s", step_sec=10.0, reference_seconds=120.0),
    Config("step10 / ref 300s", step_sec=10.0, reference_seconds=300.0),
    Config("step30 / ref 150s", step_sec=30.0, reference_seconds=150.0),
    Config("step10 / ref150 + 5seed + L2", step_sec=10.0, reference_seconds=150.0,
           n_seeds=5, l2=5.0, learning_rate=0.05),
]


def per_subject_scores(table: pd.DataFrame, cfg: Config, random_state: int = 0) -> dict[str, float]:
    """Balanced accuracy on each held-out subject, keyed by subject -- the unit of real variation."""
    X = table[MODEL_COLUMNS].to_numpy(dtype=float)
    y = table["head_a_label"].to_numpy(dtype=int)
    groups = table["group"].to_numpy()
    starts = table["window_start_s"].to_numpy()

    scores = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        prob = cfg.fit_predict_proba(X[tr], y[tr], X[te], random_state)
        prob = _smooth(prob, starts[te], cfg.smooth_k)
        scores[str(groups[te][0])] = float(balanced_accuracy_score(y[te], (prob >= 0.5).astype(int)))
    return scores


def main() -> int:
    if not CACHE_PATH.exists():
        print(f"no cache at {CACHE_PATH} -- run: uv run python scripts/cache_wesad_features.py")
        return 1
    cache = pd.read_parquet(CACHE_PATH)

    all_scores = {}
    for cfg in CANDIDATES:
        all_scores[cfg.name] = per_subject_scores(build_table(cache, cfg), cfg)
        vals = np.array(list(all_scores[cfg.name].values()))
        print(f"{vals.mean():.3f} +/- {stats.sem(vals):.3f} (SE)  n={len(vals)}  {cfg.name}", flush=True)

    baseline = CANDIDATES[0].name
    base = all_scores[baseline]
    print(f"\nPaired against '{baseline}' (same subjects, Wilcoxon signed-rank):")
    for cfg in CANDIDATES[1:]:
        subjects = sorted(set(base) & set(all_scores[cfg.name]))
        a = np.array([base[s] for s in subjects])
        b = np.array([all_scores[cfg.name][s] for s in subjects])
        diff = b - a
        try:
            _, p = stats.wilcoxon(a, b)
        except ValueError:  # identical vectors
            p = 1.0
        verdict = "SIGNIFICANT" if p < 0.05 else "not distinguishable from noise"
        print(f"  {cfg.name}: mean diff {diff.mean():+.3f} "
              f"(SE {stats.sem(diff):.3f}), p={p:.3f} -> {verdict}")

    per_subject = pd.DataFrame(all_scores)
    out = Path("artifacts/metrics/wesad_per_subject_scores.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    per_subject.to_csv(out)
    print(f"\nSpread across subjects for the shipped config: "
          f"{per_subject[baseline].min():.3f} (worst subject) .. {per_subject[baseline].max():.3f} (best)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
