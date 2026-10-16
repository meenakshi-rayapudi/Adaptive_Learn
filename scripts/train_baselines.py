"""
Trains the baseline models on the ASSISTments benchmark data in data/benchmarks.

    python -m scripts.train_baselines

It reads data/benchmarks/assistments_snapshots.csv (one row per attempt: the features
Q, A, T, R, F, E, H computed from the student's past, plus whether they then failed),
splits it by time (earliest 80% train, latest 20% test), trains three simple models
and prints ROC-AUC, PR-AUC and Brier score on the held-out late 20%.

To rebuild that CSV from the raw ASSISTments file:
    python -m scripts.train_baselines --raw-csv path/to/skill_builder_data.csv

Features that are empty or constant in this data (ASSISTments has no dates, flashcards
or recommendations) are skipped automatically.
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.dataset import BENCHMARK_SNAPSHOTS as SNAPSHOTS
from engine.dataset import MIN_ATTEMPTS_FOR_ML, load_benchmark_split, to_model_frame
from engine.feature_spec import FEATURE_NAMES

ARTIFACT = os.path.join("engine", "artifacts", "baseline_assistments.joblib")


def make_models():
    return {
        "prior_only": Pipeline([("clf", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "hist_gbdt": Pipeline([
            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42)),
        ]),
    }


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def main():
    parser = argparse.ArgumentParser(description="Train baseline models on the ASSISTments benchmark data.")
    parser.add_argument("--raw-csv", help="Raw ASSISTments skill-builder CSV; rebuilds the snapshots file first.")
    parser.add_argument("--min-prior", type=int, default=MIN_ATTEMPTS_FOR_ML,
                        help="Only use attempts where the student already had this many earlier attempts.")
    args = parser.parse_args()

    if args.raw_csv:
        from engine.benchmarks import load_assistments_events
        from engine.dataset import build_snapshot_dataset
        data = load_assistments_events(args.raw_csv)
        build_snapshot_dataset(data.events_by_student, wallclock=False).to_csv(SNAPSHOTS, index=False)
        print(f"Rebuilt {SNAPSHOTS}")
    elif not os.path.exists(SNAPSHOTS):
        sys.exit(f"{SNAPSHOTS} not found. Put the data there, or pass --raw-csv.")

    train, test, features = load_benchmark_split(min_prior=args.min_prior)
    print(f"Split by time: {len(train):,} train / {len(test):,} test "
          f"(fail rate {train['y_fail'].mean():.1%} / {test['y_fail'].mean():.1%})")

    skipped = [f for f in FEATURE_NAMES if f not in features]
    print(f"Using features {features}; skipped (empty, constant or left out): {skipped}")

    X_train, X_test = to_model_frame(train)[features], to_model_frame(test)[features]
    results, fitted = {}, {}
    for name, model in make_models().items():
        model.fit(X_train, train["y_fail"])
        results[name] = evaluate(model, X_test, test["y_fail"])
        fitted[name] = model

    print("\nBaselines on the held-out latest 20% of attempts:\n")
    print(pd.DataFrame(results).T[["roc_auc", "pr_auc", "brier"]].round(4).to_string())

    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    joblib.dump({"models": fitted, "feature_names": features, "source": "assistments", "metrics": results}, ARTIFACT)
    print(f"\nSaved the fitted pipelines (imputer + scaler + model) to {ARTIFACT}")


if __name__ == "__main__":
    main()
