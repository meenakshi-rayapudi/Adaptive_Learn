"""
Topic recommender: ranks a document's topics by how much a student needs them.

The handoff (docs/architectural_decisions.md, Decision 3):
  * fewer than 3 quiz attempts -> the cold-start formula (engine/cold_start.py)
  * 3 or more attempts         -> the trained model (engine/predictor.py), with SHAP reasons

If the model can't be used for any reason (file missing, lightgbm or shap not installed, ...)
the cold-start formula is used instead, so the app never crashes over it.
"""

import json
from typing import Dict, List, Optional

from database.db import get_session
from database.schema import Topic, QuizAttempt, Recommendation
from engine.cold_start import cold_start_score, recommend_priority
from engine.dataset import MIN_ATTEMPTS_FOR_ML
from engine.features import compute_topic_vectors

COLD_START_VERSION = "cold_start_v1"


def cold_start_deficit(fv) -> float:
    """
    Day-1 weakness score. A missing value counts against the topic: never studied = fully stale
    (handled inside cold_start_score), no flashcards = no mastery, never quizzed = 50/50 on accuracy.
    """
    accuracy = fv.a if fv.a is not None else 0.5
    mastery = fv.f if fv.f is not None else 0.0
    return cold_start_score(accuracy, fv.r, mastery)


def _ml_prediction(fv):
    """The model's prediction for one topic, or None if the model can't be used."""
    try:
        from engine.predictor import predict_from_vector  # lightgbm and shap are only imported here
        return predict_from_vector(fv)
    except Exception as e:
        print(f"Warning: ML model unavailable, using the cold-start formula instead ({type(e).__name__}: {e})")
        return None


def recommend_topics(student_id: str, document_id: str, limit: int = 3, session=None) -> List[Dict]:
    """
    The topics the student should study, most needed first. Each item:
      {"topic_id", "name", "deficit", "priority", "accuracy", "days_since_study",
       "model_version", "reasons"}
    `reasons` holds the top 3 SHAP reasons when the model scored the topic, otherwise None.
    """
    own_session = session is None
    session = session or get_session()
    try:
        topics = session.query(Topic).filter(Topic.document_id == document_id).all()
        if not topics:
            return []

        attempts = session.query(QuizAttempt).filter(QuizAttempt.student_id == student_id).count()
        use_model = attempts >= MIN_ATTEMPTS_FOR_ML
        names = {t.id: t.name for t in topics}

        recommendations = []
        for fv in compute_topic_vectors(student_id, list(names), session=session):
            prediction = _ml_prediction(fv) if use_model else None
            if use_model and prediction is None:
                use_model = False  # the model is broken: don't retry (and warn) for every topic

            if prediction:
                deficit, version, reasons = prediction["deficit"], f"ml_{prediction['model']}", prediction["reasons"]
            else:
                deficit, version, reasons = round(cold_start_deficit(fv), 3), COLD_START_VERSION, None

            recommendations.append({
                "topic_id": fv.topic_id, "name": names[fv.topic_id], "deficit": deficit,
                "priority": recommend_priority(deficit), "accuracy": fv.a, "days_since_study": fv.r,
                "model_version": version, "reasons": reasons,
            })

        recommendations.sort(key=lambda r: r["deficit"], reverse=True)
        return recommendations[:limit]
    finally:
        if own_session:
            session.close()


def save_recommendation(student_id: Optional[str], recommendation: Dict, followed: Optional[bool] = None,
                        session=None) -> Optional[int]:
    """
    Stores one recommendation with the model version and its SHAP reasons (as JSON), so we can
    later see what was suggested, why, and whether the student followed it.
    Returns the saved id, or None. Never raises.
    """
    if not student_id:
        return None

    own_session = session is None
    try:
        session = session or get_session()
        row = Recommendation(
            student_id=student_id, topic_id=recommendation["topic_id"],
            deficit_score=recommendation["deficit"], priority=recommendation["priority"],
            model_version=recommendation["model_version"],
            shap_explanation=json.dumps(recommendation["reasons"]) if recommendation.get("reasons") else None,
            is_followed=followed,
        )
        session.add(row)
        session.commit()
        return row.id
    except Exception as e:
        print(f"Warning: could not save recommendation: {e}")
        if session is not None:
            session.rollback()
        return None
    finally:
        if own_session and session is not None:
            session.close()
