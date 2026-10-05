"""
Feature vector specification for the Learner Brain ML model.

This module is the single source of truth for the shape and math of the
input vector described in docs/model_selection_and_benchmarks.md (Part 2)
and docs/architectural_decisions.md (Decision 2/6):

    x = [Q, A, T, R, F, E, H]

It defines *what each feature means, its raw range, and how it is
transformed into a model-ready value*. It intentionally does NOT query the
database — aggregating raw SQLite rows (quiz_attempts, flashcard_events,
recommendations, ...) into these raw values is Week 3 scope
(engine/features.py: `compute_feature_vector(student_id, topic_id)`).
Keeping the math here means that module can import and reuse it verbatim
instead of re-deriving the transforms.

--------------------------------------------------------------------------
Feature definitions (raw ranges, before model-ready transform):

  Q  in [0.0, 1.0]   Global recent quiz performance (accuracy ratio over the
                      student's last N attempts across all topics).
  A  in [0.0, 1.0]   Topic-specific historical accuracy
                      (correct / total for this (student, topic) pair).
  T  in [500, 120000] Average response latency in milliseconds for this topic
                      (cognitive struggle signal; log-normally distributed).
  R  in [0, 30+]     Days elapsed since the last study event (quiz attempt OR
                      flashcard review) on this topic (spaced-decay recency).
  F  in [0.0, 1.0]   Flashcard mastery score derived from SM-2 ease factor and
                      interval (see docs/architectural_decisions.md, Decision 7).
  E  in [0, 500]     Engagement intensity: count of remedial guides read plus
                      chat questions asked about this topic.
  H  in [0, 10+]     Historical recommendation adherence: count of prior
                      recommendations for this topic the student acted on.

Any of these may be missing (NaN) for a brand-new (student, topic) pair —
e.g. a student who never touches flashcards has F = NaN. Tree-based models
(XGBoost/LightGBM) route NaN natively, which is *why* they were chosen over
deep learning (see docs/model_selection_and_benchmarks.md, Section 3).

--------------------------------------------------------------------------
Model-ready transforms (applied per-feature before feeding the model):

  Q' = Q                                  (already a clean ratio)
  A' = A                                  (already a clean ratio)
  T' = log1p(T)                           (compresses the long right tail of
                                            latency into an approximately
                                            linear scale; log1p avoids -inf
                                            at T=0)
  R' = min(R, R_CAP) / R_CAP               (clipped-and-scaled recency; caps
                                            "hasn't studied this in forever"
                                            at a fixed penalty rather than
                                            growing unbounded)
  F' = F                                  (already a clean ratio; NaN if the
                                            student has never reviewed a
                                            flashcard on this topic)
  E' = log1p(E)                           (count data is heavily right-skewed)
  H' = min(H, H_CAP) / H_CAP               (clipped-and-scaled adherence ratio)

The cold-start heuristic (Week 2, engine/cold_start.py) only needs the raw
A, R, F values directly per its formula:
  Deficit = 0.5*(1 - A) + 0.3*RecencyPenalty(R) + 0.2*(1 - F)
so `recency_penalty()` below is shared by both the heuristic and the
ML feature transform.
"""

import math
from dataclasses import dataclass, fields
from typing import Optional, List, Dict, Any

FEATURE_NAMES: List[str] = ["Q", "A", "T", "R", "F", "E", "H"]

FEATURE_BOUNDS: Dict[str, tuple] = {
    "Q": (0.0, 1.0),
    "A": (0.0, 1.0),
    "T": (500, 120_000),   # milliseconds
    "R": (0, 30),           # days (soft cap; see R_CAP)
    "F": (0.0, 1.0),
    "E": (0, 500),           # count
    "H": (0, 10),            # count (soft cap; see H_CAP)
}

R_CAP = 30.0   # days; recency penalty saturates beyond this
H_CAP = 10.0   # prior recommendations; adherence ratio saturates beyond this


@dataclass
class FeatureVector:
    """Raw (untransformed) feature values for one (student_id, topic_id) pair."""
    student_id: str
    topic_id: str
    q: Optional[float] = None   # None if the student has no quiz history at all
    a: Optional[float] = None   # None if the student has never been quizzed on this topic
    t: Optional[float] = None   # milliseconds; None if the student has no quiz history for this topic
    r: Optional[float] = None   # days since last study event; None if never studied
    f: Optional[float] = None   # flashcard mastery ratio; None if never reviewed
    e: float = 0.0
    h: float = 0.0

    def to_raw_array(self) -> List[Optional[float]]:
        """Raw values in FEATURE_NAMES order, NaN-friendly for tree models."""
        return [self.q, self.a, self.t, self.r, self.f, self.e, self.h]

    def to_model_array(self) -> List[float]:
        """Model-ready values (transformed per the spec above), NaN for missing modalities."""
        nan = float("nan")
        return [
            self.q if self.q is not None else nan,
            self.a if self.a is not None else nan,
            log1p_transform(self.t) if self.t is not None else nan,
            recency_penalty(self.r) if self.r is not None else nan,
            self.f if self.f is not None else nan,
            log1p_transform(self.e),
            clip_scale(self.h, H_CAP),
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def log1p_transform(x: float) -> float:
    """Compresses right-skewed count/latency data. See module docstring."""
    return math.log1p(max(0.0, x))


def clip_scale(x: float, cap: float) -> float:
    """Clips x to [0, cap] and scales to [0, 1]."""
    return min(max(0.0, x), cap) / cap


def recency_penalty(days_since_last_study: float, cap: float = R_CAP) -> float:
    """
    Shared by the Week 2 cold-start heuristic and the ML feature transform.
    0.0 = studied today, 1.0 = at or beyond the cap (default 30 days).
    """
    return clip_scale(days_since_last_study, cap)


def forgetting_curve_retention(days_elapsed: float, stability: float) -> float:
    """
    Ebbinghaus forgetting curve: R(t) = e^(-t/S).
    `stability` (S) is memory stability in days, driven by SM-2 ease factor
    and interval (docs/architectural_decisions.md, Decision 7). Used to
    derive the flashcard mastery feature F from raw SM-2 state.
    """
    if stability <= 0:
        return 0.0
    return math.exp(-days_elapsed / stability)
