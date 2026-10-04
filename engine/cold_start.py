"""Week 2 cold-start recommendation heuristic.

The Day 1 recommender is a deterministic scoring model that estimates the
student's deficit on a topic before enough interaction data is available.
It mirrors the Week 2 planning formula in docs/weekly_team_plan_and_study_guide.md:

    Deficit = 0.5 * (1 - Accuracy) + 0.3 * RecencyPenalty + 0.2 * (1 - FlashcardMastery)

where RecencyPenalty is the scaled value of days_since_last_review in [0, 1].
"""

from __future__ import annotations

from typing import Optional


def clip_scale(value: float, cap: float = 30.0) -> float:
    """Clamp a numeric value and scale it to the [0, 1] range."""
    numeric_value = float(value)
    if cap <= 0:
        return 0.0
    return min(max(0.0, numeric_value), cap) / cap


def recency_penalty(days_since_last_study: Optional[float], cap: float = 30.0) -> float:
    """Return the normalized penalty for a stale topic review.

    A brand-new student or an unseen topic gets a recency penalty of 1.0,
    which is the maximum deficit signal until more evidence arrives.
    """
    if days_since_last_study is None:
        return 1.0
    return clip_scale(float(days_since_last_study), cap)


def cold_start_score(
    accuracy: float,
    days_since_last_review: Optional[float],
    flashcard_mastery: float,
    cap: float = 30.0,
) -> float:
    """Compute the Topic Deficit Score on Day 1 using the Week 2 heuristic."""
    acc = max(0.0, min(1.0, float(accuracy)))
    mastery = max(0.0, min(1.0, float(flashcard_mastery)))
    recency = recency_penalty(days_since_last_review, cap=cap)
    score = 0.5 * (1.0 - acc) + 0.3 * recency + 0.2 * (1.0 - mastery)
    return max(0.0, min(1.0, score))


def recommend_priority(deficit_score: float) -> str:
    """Map a cold-start score to a human-friendly recommendation band."""
    if deficit_score >= 0.7:
        return "high"
    if deficit_score >= 0.4:
        return "medium"
    return "low"
