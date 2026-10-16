"""
TreeSHAP explanations for the packaged tree model.

For one prediction, SHAP splits the model's output into one contribution per feature.
A positive value pushed the student's failure risk up, a negative one pushed it down.
The values are exact for tree models, are in log-odds (the model's internal scale), and
together with the model's base value add up to its raw output.
"""

import numpy as np
import shap

_explainers = {}  # building a TreeExplainer takes a moment, so keep one per model


def _positive_class(values):
    """Different SHAP versions return a list per class or a 3D array; keep the 'fails' class as 2D."""
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    return values[:, :, -1] if values.ndim == 3 else values


def explain(model, X, top_k=3):
    """
    X is a one-row DataFrame in model-ready form. Returns the top_k features by size of effect,
    biggest first: [{"feature": "A", "shap": 0.41}, ...]
    """
    if id(model) not in _explainers:
        _explainers[id(model)] = shap.TreeExplainer(model)
    values = _positive_class(_explainers[id(model)].shap_values(X))[0]

    order = np.argsort(-np.abs(values))[:top_k]
    return [{"feature": X.columns[i], "shap": float(values[i])} for i in order]
