"""Sweep accuracy levers for Head A on WESAD, all under grouped leave-one-subject-out.

Reads the cached dense feature table (``scripts/cache_wesad_features.py``) so no experiment pays
the cvxEDA extraction cost. Every config is scored the same way -- LOSO over WESAD's subjects,
balanced accuracy on the held-out person -- so the rows are directly comparable.

Levers swept:
  * ``step_sec``      -- window density. A denser step is simply more training data; it cannot leak,
                        because LOSO holds out *people*, not windows.
  * ``smooth_k``      -- rolling-median smoothing of the predicted probability across consecutive
                        windows of the held-out subject. Stress is temporally contiguous, so a lone
                        one-window spike is usually noise. Causal (trailing window only), and the
                        live runtime already smooths via hysteresis.
  * ``robust``        -- personal-baseline reference from median/IQR instead of mean/std.
  * boosting hyperparameters.

IMPORTANT (honesty): picking the best row of this sweep by its LOSO score is selection on the same
folds you report, which biases the winner optimistically. Use ``--nested`` to re-score the chosen
grid with nested CV (config picked inside each outer fold's training data only) -- that number is
the one to quote.

Usage:
  uv run python scripts/wesad_experiments.py                 # exploratory sweep
  uv run python scripts/wesad_experiments.py --nested        # honest estimate of the tuned model
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut  # noqa: E402

from neuroshield.features.extract import FEATURE_COLUMNS  # noqa: E402
from neuroshield.features.harmonize import harmonize_labels  # noqa: E402
from neuroshield.features.personalize import (  # noqa: E402
    MIN_REFERENCE_WINDOWS,
    MIN_STD,
    PERSONALIZE_BASE,
    PERSONALIZED_COLUMNS,
    add_personalized_features,
)

CACHE_PATH = Path("artifacts/cache/wesad_features_step10.parquet")
CACHE_STEP_SEC = 10.0

MODEL_COLUMNS = list(FEATURE_COLUMNS) + PERSONALIZED_COLUMNS


@dataclass
class Config:
    name: str
    step_sec: float = 30.0
    smooth_k: int = 1  # 1 = no smoothing
    robust: bool = False
    reference_seconds: float = 300.0  # length of the personal-baseline calibration period
    max_iter: int = 200
    learning_rate: float = 0.08
    max_depth: int | None = 4
    l2: float = 1.0
    n_seeds: int = 1  # >1 averages probabilities over seeds (variance reduction)
    class_weight: str | None = None
    tune_threshold: bool = False  # pick the decision threshold inside the training fold only
    extra: dict = field(default_factory=dict)

    def estimator(self, random_state: int = 0):
        return CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                l2_regularization=self.l2,
                class_weight=self.class_weight,
                random_state=random_state,
                **self.extra,
            ),
            method="isotonic",
            cv=3,
        )

    def fit_predict_proba(self, X_tr, y_tr, X_te, random_state: int = 0):
        """Seed-ensembled probability: one fit when n_seeds==1, else the mean over seeds."""
        probs = [
            self.estimator(random_state + s).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            for s in range(self.n_seeds)
        ]
        return np.mean(probs, axis=0)


def _robust_personalize(features: pd.DataFrame, reference_seconds: float = 300.0) -> pd.DataFrame:
    """Personal baseline from median / IQR -- a heavy-tailed feature can't drag the reference."""
    result = features.copy()
    for col in PERSONALIZED_COLUMNS:
        result[col] = np.nan
    for _subject, idx in result.groupby("subject_id", sort=False).groups.items():
        frame = result.loc[idx].sort_values("window_start_s")
        accepted = frame[frame["valid_fraction"] >= 0.5]
        if len(accepted) < MIN_REFERENCE_WINDOWS:
            accepted = frame
        start = accepted["window_start_s"].min()
        ref = accepted[accepted["window_start_s"] < start + reference_seconds]
        if ref.empty:
            continue
        center = ref[PERSONALIZE_BASE].median(skipna=True)
        # IQR/1.349 is the normal-consistent robust estimate of the standard deviation.
        iqr = ref[PERSONALIZE_BASE].quantile(0.75) - ref[PERSONALIZE_BASE].quantile(0.25)
        scale = (iqr / 1.349).clip(lower=MIN_STD)
        for col in PERSONALIZE_BASE:
            result.loc[idx, f"{col}_p"] = (result.loc[idx, col] - center[col]) / scale[col]
    return result


def build_table(cache: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Subsample the dense cache to this config's step, personalize, harmonize, keep Head A rows."""
    stride = int(round(cfg.step_sec / CACHE_STEP_SEC))
    if stride < 1:
        raise ValueError(f"step_sec={cfg.step_sec} is denser than the cache ({CACHE_STEP_SEC}s)")

    frames = []
    for _subject, frame in cache.groupby("subject_id", sort=False):
        frame = frame.sort_values("window_start_s")
        frames.append(frame.iloc[::stride])
    sub = pd.concat(frames, ignore_index=True)

    sub = (
        _robust_personalize(sub, cfg.reference_seconds)
        if cfg.robust
        else add_personalized_features(sub, reference_seconds=cfg.reference_seconds)
    )
    harmonized, _ = harmonize_labels(sub, "wesad")
    return harmonized[harmonized["eligible_head_a"]].reset_index(drop=True)


def _best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy. Only ever called on *training*-fold data."""
    grid = np.linspace(0.2, 0.8, 25)
    scores = [balanced_accuracy_score(y_true, (prob >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _smooth(prob: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    """Trailing rolling median of ``prob`` in time order (causal: never looks ahead)."""
    if k <= 1:
        return prob
    series = pd.Series(prob[np.argsort(order)])
    smoothed = series.rolling(window=k, min_periods=1).median().to_numpy()
    out = np.empty_like(smoothed)
    out[np.argsort(order)] = smoothed
    return out


def loso_score(table: pd.DataFrame, cfg: Config, random_state: int = 0) -> dict:
    X = table[MODEL_COLUMNS].to_numpy(dtype=float)
    y = table["head_a_label"].to_numpy(dtype=int)
    groups = table["group"].to_numpy()
    starts = table["window_start_s"].to_numpy()

    y_true, y_pred = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        threshold = 0.5
        if cfg.tune_threshold:
            # Held-out subject is untouched: the threshold is fit on an inner split of the
            # training subjects only, then applied blind to the held-out person.
            inner_tr, inner_val = next(
                GroupKFold(n_splits=4).split(X[tr], y[tr], groups[tr])
            )
            inner_model = cfg.estimator(random_state).fit(X[tr][inner_tr], y[tr][inner_tr])
            threshold = _best_threshold(
                y[tr][inner_val], inner_model.predict_proba(X[tr][inner_val])[:, 1]
            )

        prob = cfg.fit_predict_proba(X[tr], y[tr], X[te], random_state)
        prob = _smooth(prob, starts[te], cfg.smooth_k)
        y_pred.extend((prob >= threshold).astype(int))
        y_true.extend(y[te])

    yt, yp = np.array(y_true), np.array(y_pred)
    return {
        "config": cfg.name,
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0)),
        "n_windows": int(len(yt)),
    }


def nested_score(tables: dict[str, pd.DataFrame], grid: list[Config], random_state: int = 0) -> dict:
    """Honest estimate of the *tuning procedure*: the config is chosen inside each outer fold.

    The outer fold's held-out subject never influences which config wins, so this number carries
    none of the selection optimism that reading the best row off the sweep does.
    """
    reference = tables[grid[0].name]
    outer_groups = np.unique(reference["group"].to_numpy())

    y_true, y_pred, chosen = [], [], []
    for held_out in outer_groups:
        # Inner selection: score every config by grouped CV over the *other* subjects only.
        best_cfg, best_score = None, -np.inf
        for cfg in grid:
            table = tables[cfg.name]
            inner = table[table["group"] != held_out]
            Xi = inner[MODEL_COLUMNS].to_numpy(dtype=float)
            yi = inner["head_a_label"].to_numpy(dtype=int)
            gi = inner["group"].to_numpy()
            si = inner["window_start_s"].to_numpy()

            scores = []
            for tr, te in GroupKFold(n_splits=4).split(Xi, yi, gi):
                if len(np.unique(yi[tr])) < 2 or len(np.unique(yi[te])) < 2:
                    continue
                m = cfg.estimator(random_state).fit(Xi[tr], yi[tr])
                p = _smooth(m.predict_proba(Xi[te])[:, 1], si[te], cfg.smooth_k)
                scores.append(balanced_accuracy_score(yi[te], (p >= 0.5).astype(int)))
            score = float(np.mean(scores)) if scores else -np.inf
            if score > best_score:
                best_cfg, best_score = cfg, score

        # Outer scoring: refit the winning config on all other subjects, predict the held-out one.
        table = tables[best_cfg.name]
        tr_mask = table["group"] != held_out
        te_mask = ~tr_mask
        m = best_cfg.estimator(random_state).fit(
            table.loc[tr_mask, MODEL_COLUMNS].to_numpy(dtype=float),
            table.loc[tr_mask, "head_a_label"].to_numpy(dtype=int),
        )
        p = m.predict_proba(table.loc[te_mask, MODEL_COLUMNS].to_numpy(dtype=float))[:, 1]
        p = _smooth(p, table.loc[te_mask, "window_start_s"].to_numpy(), best_cfg.smooth_k)
        y_pred.extend((p >= 0.5).astype(int))
        y_true.extend(table.loc[te_mask, "head_a_label"].to_numpy(dtype=int))
        chosen.append(best_cfg.name)
        print(f"  outer fold {held_out}: picked {best_cfg.name} (inner {best_score:.3f})", flush=True)

    yt, yp = np.array(y_true), np.array(y_pred)
    return {
        "config": "NESTED (config chosen inside each fold)",
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0)),
        "n_windows": int(len(yt)),
        "configs_chosen": chosen,
    }


# Round 1 established: dense windows help (+0.008), smoothing hurts (it blurs stress onset), and a
# robust median/IQR personal reference is worse than mean/std. Round 2 sweeps what is still open,
# all on top of the round-1 winner (dense windows, no smoothing).
GRID_ROUND1 = [
    Config("A. current shipped (step 30s)"),
    Config("B. dense windows (step 10s)", step_sec=10.0),
    Config("C. dense + smooth k=3", step_sec=10.0, smooth_k=3),
    Config("D. dense + smooth k=5", step_sec=10.0, smooth_k=5),
    Config("E. dense + smooth k=9", step_sec=10.0, smooth_k=9),
    Config("F. robust personal ref + dense + smooth k=5", step_sec=10.0, smooth_k=5, robust=True),
    Config("G. deeper trees + dense + smooth k=5", step_sec=10.0, smooth_k=5, max_depth=6, max_iter=300),
    Config("H. stronger L2 + dense + smooth k=5", step_sec=10.0, smooth_k=5, l2=5.0, learning_rate=0.05),
]

_DENSE = {"step_sec": 10.0}
GRID_ROUND2 = [
    Config("B. dense (round-1 winner)", **_DENSE),
    Config("I. dense + 5-seed ensemble", **_DENSE, n_seeds=5),
    Config("J. dense + class_weight balanced", **_DENSE, class_weight="balanced"),
    Config("K. dense + in-fold tuned threshold", **_DENSE, tune_threshold=True),
    Config("L. dense + slow/long boosting", **_DENSE, learning_rate=0.03, max_iter=600),
    Config("M. dense + unlimited depth", **_DENSE, max_depth=None),
    Config("N. dense + smooth k=2", **_DENSE, smooth_k=2),
    Config("O. dense + 5-seed + stronger L2", **_DENSE, n_seeds=5, l2=5.0, learning_rate=0.05),
]

# The realistic candidate set, for the nested-CV run that produces the number we actually quote.
# Deliberately small: nested CV refits every candidate inside every outer fold, and a grid stuffed
# with near-duplicates just buys noise.
GRID_NESTED = [
    Config("B. dense", **_DENSE),
    Config("I. dense + 5-seed ensemble", **_DENSE, n_seeds=5),
    Config("M. dense + unlimited depth", **_DENSE, max_depth=None),
    Config("O. dense + 5-seed + stronger L2", **_DENSE, n_seeds=5, l2=5.0, learning_rate=0.05),
]

# Round 3. Fixing the window-count reference bug (it silently shortened the calibration period at a
# denser step) revealed that the *length of the calibration period* is itself a live hyperparameter --
# and the accidentally-short one was better. It is legitimately tunable: the app decides how long it
# asks a user to sit still. Sweep it honestly instead of inheriting it from a bug.
GRID_ROUND3 = [
    Config("P. step30 / ref 300s  (currently shipped)", step_sec=30.0, reference_seconds=300.0),
    Config("Q. step10 / ref 120s", **_DENSE, reference_seconds=120.0),
    Config("R. step10 / ref 150s", **_DENSE, reference_seconds=150.0),
    Config("S. step10 / ref 300s", **_DENSE, reference_seconds=300.0),
    Config("T. step10 / ref 600s", **_DENSE, reference_seconds=600.0),
    Config("U. step30 / ref 150s", step_sec=30.0, reference_seconds=150.0),
    Config("V. step10 / ref 150s + 5-seed + L2", **_DENSE, reference_seconds=150.0,
           n_seeds=5, l2=5.0, learning_rate=0.05),
    Config("W. step10 / ref 150s + unlimited depth", **_DENSE, reference_seconds=150.0, max_depth=None),
]

GRIDS = {"1": GRID_ROUND1, "2": GRID_ROUND2, "3": GRID_ROUND3, "nested": GRID_NESTED}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", action="store_true", help="run the honest nested-CV estimate too")
    parser.add_argument("--round", choices=sorted(GRIDS), default="1")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args()
    grid = GRIDS[args.round]

    if not args.cache.exists():
        print(f"no cache at {args.cache} -- run: uv run python scripts/cache_wesad_features.py")
        return 1
    cache = pd.read_parquet(args.cache)
    print(f"cache: {len(cache)} windows, {cache['subject_id'].nunique()} subjects\n", flush=True)

    tables = {cfg.name: build_table(cache, cfg) for cfg in grid}

    results = []
    for cfg in grid:
        r = loso_score(tables[cfg.name], cfg)
        results.append(r)
        print(f"{r['balanced_accuracy']:.3f} bal-acc | {r['macro_f1']:.3f} F1 | "
              f"{r['n_windows']:5d} windows | {cfg.name}", flush=True)

    if args.nested:
        print("\nnested CV (honest estimate of the tuned model)...", flush=True)
        r = nested_score(tables, grid)
        results.append(r)
        print(f"\n{r['balanced_accuracy']:.3f} bal-acc | {r['macro_f1']:.3f} F1 | {r['config']}")

    out = Path(f"artifacts/metrics/wesad_experiments_round{args.round}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series({"results": results}).to_json(out, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
