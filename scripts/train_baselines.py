"""
Builds the normalized feature table from a data source, splits it chronologically
(80% early / 20% late), trains baseline models, and reports evaluation baselines.

This is the Week 3 baseline only. The full model tournament (LightGBM, XGBoost,
MLP, Random Forest, SHAP) is Week 4.

Usage:
    python -m scripts.train_baselines --source db
    python -m scripts.train_baselines --source ednet --train-csv path/train.csv --questions-csv path/questions.csv --max-users 2000
    python -m scripts.train_baselines --source assistments --train-csv path/skill_builder_data.csv
"""

import argparse
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.dataset import (MIN_ATTEMPTS_FOR_ML, assert_no_temporal_leakage, build_snapshot_dataset,
                            temporal_split, to_model_frame)
from engine.feature_spec import FEATURE_NAMES

OUT_DIR = os.path.join("data", "benchmarks")
ARTIFACT_DIR = os.path.join("engine", "artifacts")


def load_source(args):
    if args.source == "db":
        from database.db import get_session, init_db
        from database.schema import Student
        from engine.features import load_student_events
        init_db()
        session = get_session()
        try:
            ids = [s.id for s in session.query(Student).all()]
            events = {sid: load_student_events(sid, session=session) for sid in ids}
        finally:
            session.close()
        return events, True, "global"

    from engine import benchmarks
    if not args.train_csv:
        sys.exit("--train-csv is required for this source.")
    if args.source == "ednet":
        if not args.questions_csv:
            sys.exit("--questions-csv is required for the ednet source.")
        data = benchmarks.load_ednet_events(args.train_csv, args.questions_csv, args.max_users, args.max_rows)
    else:
        data = benchmarks.load_assistments_events(args.train_csv, args.max_users, args.max_rows)
    return data.events_by_student, data.wallclock, data.split_mode


class ZeroFillAllNanColumns(BaseEstimator, TransformerMixin):
    """
    Fills columns that are 100% NaN with a constant 0.0.
    HistGradientBoostingClassifier natively handles partially missing values (NaNs),
    but crashes if a column has ZERO observed values (100% NaN) because it cannot
    compute binning thresholds.
    """
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.all_nan_cols_ = [c for c in X_df.columns if X_df[c].isna().all()]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for c in self.all_nan_cols_:
            X_df[c] = X_df[c].fillna(0.0)
        return X_df


def make_models():
    return {
        "prior_only": Pipeline([("clf", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "hist_gbdt": Pipeline([
            ("fix_all_nan", ZeroFillAllNanColumns()),
            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42)),
        ]),
    }


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def main():
    parser = argparse.ArgumentParser(description="Train AdaptiveLearn baseline models on a temporal split.")
    parser.add_argument("--source", choices=["db", "ednet", "assistments"], default="db")
    parser.add_argument("--train-csv")
    parser.add_argument("--questions-csv")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--min-prior", type=int, default=MIN_ATTEMPTS_FOR_ML,
                        help="Only train on attempts where the student already had this many prior attempts.")
    args = parser.parse_args()

    started = time.time()
    events, wallclock, split_mode = load_source(args)
    df = build_snapshot_dataset(events, wallclock=wallclock)
    if df.empty:
        sys.exit("No quiz attempts found in the chosen source.")

    os.makedirs(OUT_DIR, exist_ok=True)
    snapshot_path = os.path.join(OUT_DIR, f"{args.source}_snapshots.csv")
    df.to_csv(snapshot_path, index=False)
    print(f"Normalized {len(df):,} attempts from {df['student_id'].nunique():,} students -> {snapshot_path}")

    df = df[df["n_prior"] >= args.min_prior]
    train, test = temporal_split(df, train_frac=0.8, mode=split_mode)
    assert_no_temporal_leakage(train, test, mode=split_mode)
    print(f"Temporal split ({split_mode}): {len(train):,} train / {len(test):,} test "
          f"(fail rate {train['y_fail'].mean():.1%} / {test['y_fail'].mean():.1%})")

    if test["y_fail"].nunique() < 2 or train["y_fail"].nunique() < 2:
        sys.exit("Train or test set has only one class; cannot compute AUC. Use more data.")

    X_train, X_test = to_model_frame(train), to_model_frame(test)
    results, fitted = {}, {}
    for name, model in make_models().items():
        model.fit(X_train, train["y_fail"])
        results[name] = evaluate(model, X_test, test["y_fail"])
        fitted[name] = model

    table = pd.DataFrame(results).T[["roc_auc", "pr_auc", "brier"]].round(4)
    print(f"\nEvaluation baselines on the held-out late 20% (features: {FEATURE_NAMES}):\n")
    print(table.to_string())

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifact_path = os.path.join(ARTIFACT_DIR, f"baseline_{args.source}.joblib")
    joblib.dump({
        "models": fitted, "feature_names": FEATURE_NAMES, "source": args.source,
        "split_mode": split_mode, "metrics": results,
    }, artifact_path)
    print(f"\nSaved fitted pipelines (imputer + scaler + model) -> {artifact_path}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
