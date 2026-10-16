"""
Model tournament: five models, the same ASSISTments data, the same time-based split.

    python -m scripts.model_tournament

Every model is trained on the earliest 80% of attempts and scored on the latest 20%
(ROC-AUC, PR-AUC, Brier score, and the time one prediction takes). Hyperparameters are
fixed, sensible defaults: no tuning, so no model gets an unfair edge.

The winner is the tree model with the best ROC-AUC (TreeSHAP explanations need a tree
model). It is saved to engine/artifacts/model.pkl and the table to
docs/model_tournament_results.md.
"""

import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from engine.dataset import load_benchmark_split, to_model_frame

MODEL_PATH = os.path.join("engine", "artifacts", "model.pkl")
RESULTS_DOC = os.path.join("docs", "model_tournament_results.md")
TREE_MODELS = ["random_forest", "lightgbm", "xgboost"]


def make_models():
    # Linear and neural models need missing values filled and features scaled; the tree models do not.
    prepare = lambda: [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    return {
        "logistic_regression": Pipeline(prepare() + [("clf", LogisticRegression(max_iter=1000))]),
        "mlp": Pipeline(prepare() + [("clf", MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
                                                           max_iter=100, random_state=42))]),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=42),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=42),
    }


def p95_latency_ms(model, X, n=200):
    """95th percentile of the time one single-student prediction takes."""
    times = []
    for i in range(n):
        start = time.perf_counter()
        model.predict_proba(X.iloc[[i]])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 95))


def main():
    train, test, features = load_benchmark_split()
    print(f"Data: {len(train):,} train / {len(test):,} test attempts | features used: {features}\n")
    X_train, X_test = to_model_frame(train)[features], to_model_frame(test)[features]
    y_train, y_test = train["y_fail"], test["y_fail"]

    results, fitted = {}, {}
    for name, model in make_models().items():
        start = time.time()
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(y_test, p),
            "pr_auc": average_precision_score(y_test, p),
            "brier": brier_score_loss(y_test, p),
            "latency_ms": p95_latency_ms(model, X_test),
        }
        fitted[name] = model
        print(f"  {name:<20} trained in {time.time() - start:5.1f}s   ROC-AUC {results[name]['roc_auc']:.4f}")

    table = pd.DataFrame(results).T[["roc_auc", "pr_auc", "brier", "latency_ms"]]
    best_overall = table["roc_auc"].idxmax()
    winner = table.loc[TREE_MODELS, "roc_auc"].idxmax()

    print("\nResults on the held-out latest 20% of attempts:\n")
    print(table.round(4).to_string())
    print(f"\nBest ROC-AUC overall: {best_overall}. Packaged winner (best tree model): {winner}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": fitted[winner], "name": winner, "feature_names": features,
                 "metrics": results[winner]}, MODEL_PATH)
    print(f"Saved {winner} to {MODEL_PATH}")

    with open(RESULTS_DOC, "w", encoding="utf-8") as f:
        f.write("# Model tournament results\n\n")
        f.write(f"Data: ASSISTments benchmark, {len(train):,} training attempts (earliest 80%) and "
                f"{len(test):,} test attempts (latest 20%). Features used: {', '.join(features)}.\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | Brier (lower is better) | Latency p95 (ms) |\n|---|---|---|---|---|\n")
        for name, row in table.iterrows():
            mark = " (winner)" if name == winner else ""
            f.write(f"| {name}{mark} | {row.roc_auc:.4f} | {row.pr_auc:.4f} | {row.brier:.4f} | {row.latency_ms:.1f} |\n")
        f.write(f"\nThe winner is the tree model with the best ROC-AUC, because TreeSHAP explanations need a tree model. "
                f"Hyperparameters are fixed defaults with no tuning.\n")
    print(f"Wrote {RESULTS_DOC}")


if __name__ == "__main__":
    main()
