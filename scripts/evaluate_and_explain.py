"""
Week 5 Automated Model Evaluation & Global Interpretability.

Tasks:
1. Evaluates model performance calibration curves (Brier score, reliability curve)
   and recalibrates feature importances on new/organic student attempt data or held-out test data.
2. Generates and exports global interpretability visualizations:
   - Global SHAP summary bee-swarm plot (reports/shap_summary.png)
   - Calibration curve (reports/calibration_curve.png)
   - Feature importances chart (reports/feature_importances.png)

Usage:
    python -m scripts.evaluate_and_explain
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

from engine.dataset import load_benchmark_split, to_model_frame
from engine.predictor import MODEL_PATH, load_model

REPORTS_DIR = os.path.join("reports")


def generate_calibration_plot(model, X_test, y_test, output_path: str):
    """Computes and plots probability calibration curve compared to perfectly calibrated baseline."""
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", color="#2563EB", label=f"Model (Brier: {brier_score_loss(y_test, y_prob):.4f})")
    plt.xlabel("Mean Predicted Probability (Deficit Score)")
    plt.ylabel("Fraction of True Deficits (Failures)")
    plt.title("Model Performance Calibration Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Calibration curve saved -> {output_path}")


def generate_feature_importance_plot(model, feature_names, output_path: str):
    """Plots normalized feature importances from the tree model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(range(len(feature_names)), importances[indices], color="#3B82F6", align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.ylabel("Gini / Split Importance")
    plt.xlabel("Feature Name")
    plt.title("Winning Model Feature Importances (Tree Splits)")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Feature importances plot saved -> {output_path}")


def generate_global_shap_beeswarm(model, X_sample, output_path: str):
    """Computes global TreeSHAP values on a sample and saves the summary beeswarm plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        # binary classification: positive class is class 1 (failure/deficit)
        shap_values = shap_values[-1]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Global Model Interpretability (SHAP Summary Plot)", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] SHAP summary bee-swarm plot saved -> {output_path}")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    package = load_model()
    model = package["model"]
    feature_names = package["feature_names"]

    print(f"Loaded tournament winning model: {package['name']} with features {feature_names}")

    train, test, used_features = load_benchmark_split()
    X_test = to_model_frame(test)[feature_names]
    y_test = test["y_fail"]

    # 1. Performance calibration curve
    cal_path = os.path.join(REPORTS_DIR, "calibration_curve.png")
    generate_calibration_plot(model, X_test, y_test, cal_path)

    # 2. Feature importances chart
    fi_path = os.path.join(REPORTS_DIR, "feature_importances.png")
    generate_feature_importance_plot(model, feature_names, fi_path)

    # 3. Global SHAP summary beeswarm plot (using a representative sample for speed)
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    generate_global_shap_beeswarm(model, shap_sample, shap_path)

    # 4. Organic student attempts validation & recalibration check against SQLite
    print("\n[VALIDATION] Validating model performance on live/organic student attempts from SQLite:")
    try:
        from database.db import get_session, init_db
        from database.schema import Student
        from engine.dataset import build_snapshot_dataset
        from engine.features import load_student_events

        init_db()
        session = get_session()
        try:
            student_ids = [s.id for s in session.query(Student).all()]
            events_by_student = {sid: load_student_events(sid, session=session) for sid in student_ids}
        finally:
            session.close()

        organic_df = build_snapshot_dataset(events_by_student, wallclock=True)
        if not organic_df.empty and len(organic_df) >= 10:
            X_organic = to_model_frame(organic_df)[feature_names]
            y_organic = organic_df["y_fail"]

            if y_organic.nunique() > 1:
                p_org = model.predict_proba(X_organic)[:, 1]
                auc_org = roc_auc_score(y_organic, p_org)
                pr_org = average_precision_score(y_organic, p_org)
                brier_org = brier_score_loss(y_organic, p_org)
                print(f"  - Evaluated on {len(organic_df):,} organic attempts across {organic_df['student_id'].nunique()} students")
                print(f"  - Organic ROC-AUC: {auc_org:.4f} | PR-AUC: {pr_org:.4f} | Brier: {brier_org:.4f}")
            else:
                print(f"  - {len(organic_df)} attempts found, but only 1 target class observed.")
        else:
            print("  - Insufficient organic attempts in SQLite for separate validation (< 10 rows).")
    except Exception as e:
        print(f"  - Could not load organic attempts ({e})")

    print("\n[SUCCESS] Week 5 automated evaluation and interpretability artifacts generated successfully.")


if __name__ == "__main__":
    main()

