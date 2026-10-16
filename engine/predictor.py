"""
Packaged inference: one call turns a student's history on a topic into a deficit score
(the model's predicted chance they fail the next question on it) plus the top 3 reasons.

    from engine.predictor import predict_topic_deficit
    predict_topic_deficit("S001", "D0001_T3")

The model is the tournament winner saved by scripts/model_tournament.py.
"""

import os

import joblib
import pandas as pd

from engine.explainer import explain
from engine.feature_spec import FEATURE_NAMES
from engine.features import compute_feature_vector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl")
_package = None


def load_model():
    global _package
    if _package is None:
        _package = joblib.load(MODEL_PATH)
    return _package


def predict_topic_deficit(student_id, topic_id, as_of=None, session=None):
    """Looks up the student's history for the topic, then scores it with predict_from_vector."""
    return predict_from_vector(compute_feature_vector(student_id, topic_id, as_of=as_of, session=session))


def predict_from_vector(vector):
    """
    Scores a FeatureVector. Returns {"deficit": 0.62, "model": "lightgbm", "reasons": [...]}, where
    deficit is in [0, 1] (higher = more likely to fail) and each of the top 3 reasons looks like:
      {"feature": "A", "value": 0.33, "shap": 0.41, "direction": "raises"}
    `value` is the student's raw value for that feature (None if it is missing).
    """
    package = load_model()
    model, names = package["model"], package["feature_names"]

    model_ready = dict(zip(FEATURE_NAMES, vector.to_model_array()))
    raw = dict(zip(FEATURE_NAMES, vector.to_raw_array()))

    X = pd.DataFrame([[model_ready[f] for f in names]], columns=names)
    deficit = float(model.predict_proba(X)[0, 1])

    reasons = explain(model, X, top_k=3)
    for reason in reasons:
        reason["value"] = raw[reason["feature"]]
        reason["direction"] = "raises" if reason["shap"] > 0 else "lowers"
    return {"deficit": round(deficit, 3), "model": package["name"], "reasons": reasons}
