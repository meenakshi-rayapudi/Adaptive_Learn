"""
Feature aggregation: raw study events -> the vector x = [Q, A, T, R, F, E, H]
for any (student_id, topic_id) as of a point in time.

One implementation serves both worlds, so training and serving cannot drift:
  * `FeatureState` is an incremental accumulator. Feed it events in
    chronological order with `update()`, ask for a vector with `vector()`.
  * The live app path (`compute_feature_vector`) loads a student's rows from
    SQLite, replays them through a FeatureState, and reads the vector.
  * The training path (engine/dataset.py) replays the same events and takes a
    snapshot *before* each attempt, so every training row only sees the past.

Exact definitions (see engine/feature_spec.py for ranges and transforms):

  Q  mean correctness of the student's last QUIZ_WINDOW (20) quiz questions,
     across ALL topics. NaN if the student has never answered a question.
  A  correct / total over every prior question on THIS topic. NaN if none.
  T  mean response latency (ms) over this topic's prior questions, clipped to
     [500, 120000]. NaN if no latency was recorded.
  R  days between `as_of` and the last quiz or flashcard event on this topic.
     NaN if never studied (or if the data source has no wall-clock time).
  F  flashcard mastery. For each card (latest SM-2 state only):
         confidence = min(repetition_count, 3) / 3
         stability  = interval_days / -ln(0.9)      (so retention is 90% exactly
                                                     when the card comes due)
         retention  = exp(-days_since_review / stability)   (Ebbinghaus)
         F_card     = confidence * retention
     F = mean(F_card) over the topic's cards. A card that was just failed
     (repetition_count = 0) scores 0. NaN if the topic has no flashcard events.
  E  count of engagement events on this topic (remedial guides read, chat
     questions asked). 0 if none.
  H  count of earlier recommendations on this topic that the student followed.
     Known limitation: the schema stores no "followed at" time, so a followed
     recommendation is counted from its creation time.

Missing values stay NaN on purpose: tree models route NaN natively, which is
why they were chosen (docs/model_selection_and_benchmarks.md).
"""

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from engine.feature_spec import FeatureVector, FEATURE_BOUNDS, forgetting_curve_retention

QUIZ_WINDOW = 20
MASTERY_REPS = 3
RETENTION_AT_DUE = 0.9
SECONDS_PER_DAY = 86400.0

QUIZ, FLASHCARD, ENGAGEMENT, REC_FOLLOWED = "quiz", "flashcard", "engagement", "rec_followed"


@dataclass
class Event:
    """One thing a student did, in a source-agnostic shape (DB rows and benchmark rows both map to this)."""
    kind: str                          # quiz | flashcard | engagement | rec_followed
    ts: float                          # seconds on a monotonic axis (epoch seconds for DB data)
    topic_id: str
    correct: Optional[bool] = None     # quiz
    latency_ms: Optional[float] = None  # quiz
    group_id: Optional[str] = None     # quiz: questions of one attempt share a group_id
    card_key: Optional[str] = None     # flashcard
    interval_days: float = 0.0         # flashcard (SM-2 state after this review)
    repetition_count: int = 0          # flashcard (SM-2 state after this review)
    count: float = 1.0                 # engagement / rec_followed


class _TopicState:
    __slots__ = ("n", "correct", "lat_sum", "lat_n", "last_study_ts", "cards", "engagement", "followed")

    def __init__(self):
        self.n = 0
        self.correct = 0
        self.lat_sum = 0.0
        self.lat_n = 0
        self.last_study_ts: Optional[float] = None
        self.cards: Dict[str, Tuple[float, float, int]] = {}
        self.engagement = 0.0
        self.followed = 0.0


class FeatureState:
    """
    Incremental per-student accumulator. Events MUST be applied in chronological
    order; `vector(as_of=t)` is only valid if every applied event is at or before t.
    `wallclock=False` is for data with no real timestamps (ASSISTments order ids):
    R and F then stay NaN instead of reporting meaningless "days".
    """

    def __init__(self, wallclock: bool = True):
        self.wallclock = wallclock
        self.recent_quiz: deque = deque(maxlen=QUIZ_WINDOW)
        self.topics: Dict[str, _TopicState] = {}
        self.attempts_seen = 0

    def _topic(self, topic_id: str) -> _TopicState:
        state = self.topics.get(topic_id)
        if state is None:
            state = self.topics[topic_id] = _TopicState()
        return state

    def update(self, e: Event) -> None:
        t = self._topic(e.topic_id)
        if e.kind == QUIZ:
            if e.correct is not None:
                self.recent_quiz.append(1.0 if e.correct else 0.0)
                t.n += 1
                t.correct += 1 if e.correct else 0
            if e.latency_ms is not None and not math.isnan(e.latency_ms):
                t.lat_sum += e.latency_ms
                t.lat_n += 1
            t.last_study_ts = e.ts if t.last_study_ts is None else max(t.last_study_ts, e.ts)
        elif e.kind == FLASHCARD:
            t.cards[e.card_key or "_"] = (e.ts, e.interval_days, e.repetition_count)
            t.last_study_ts = e.ts if t.last_study_ts is None else max(t.last_study_ts, e.ts)
        elif e.kind == ENGAGEMENT:
            t.engagement += e.count
        elif e.kind == REC_FOLLOWED:
            t.followed += e.count

    def vector(self, student_id: str, topic_id: str, as_of: float) -> FeatureVector:
        t = self.topics.get(topic_id) or _TopicState()

        q = sum(self.recent_quiz) / len(self.recent_quiz) if self.recent_quiz else None
        a = t.correct / t.n if t.n else None

        lat = None
        if t.lat_n:
            lo, hi = FEATURE_BOUNDS["T"]
            lat = min(max(t.lat_sum / t.lat_n, lo), hi)

        r = f = None
        if self.wallclock:
            if t.last_study_ts is not None:
                r = max(0.0, (as_of - t.last_study_ts) / SECONDS_PER_DAY)
            f = self._flashcard_mastery(t, as_of)

        return FeatureVector(
            student_id=student_id, topic_id=topic_id,
            q=q, a=a, t=lat, r=r, f=f, e=t.engagement, h=t.followed,
        )

    @staticmethod
    def _flashcard_mastery(t: _TopicState, as_of: float) -> Optional[float]:
        if not t.cards:
            return None
        scores = []
        for reviewed_ts, interval_days, reps in t.cards.values():
            if reps <= 0:
                scores.append(0.0)
                continue
            confidence = min(reps, MASTERY_REPS) / MASTERY_REPS
            stability = max(interval_days, 1.0) / -math.log(RETENTION_AT_DUE)
            days = max(0.0, (as_of - reviewed_ts) / SECONDS_PER_DAY)
            scores.append(confidence * forgetting_curve_retention(days, stability))
        return sum(scores) / len(scores)


# --------------------------------------------------------------------------
# Database access
# --------------------------------------------------------------------------

def to_epoch(value: dt.datetime) -> float:
    """DB timestamps are naive UTC (datetime.utcnow); convert to epoch seconds."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def load_student_events(student_id: str, session=None) -> List[Event]:
    """Reads one student's raw rows from SQLite and returns them as time-ordered Events."""
    from database.db import get_session, init_db
    from database.schema import QuizAttempt, QuizQuestion, FlashcardEvent, EngagementEvent, Recommendation

    own_session = session is None
    if own_session:
        init_db()  # idempotent; adds any tables missing from an older database file
    session = session or get_session()
    try:
        events: List[Event] = []

        rows = (
            session.query(QuizQuestion, QuizAttempt)
            .join(QuizAttempt, QuizQuestion.quiz_attempt_id == QuizAttempt.id)
            .filter(QuizAttempt.student_id == student_id)
            .order_by(QuizQuestion.id)
            .all()
        )
        for question, attempt in rows:
            topic_id = question.topic_id or attempt.topic_id
            if topic_id is None or question.is_correct is None:
                continue
            events.append(Event(
                kind=QUIZ, ts=to_epoch(attempt.started_at), topic_id=topic_id,
                correct=bool(question.is_correct),
                latency_ms=float(question.latency_ms) if question.latency_ms is not None else None,
                group_id=str(attempt.id),
            ))

        for c in session.query(FlashcardEvent).filter(FlashcardEvent.student_id == student_id).order_by(FlashcardEvent.id):
            if c.topic_id is None:
                continue
            events.append(Event(
                kind=FLASHCARD, ts=to_epoch(c.reviewed_at), topic_id=c.topic_id,
                card_key=f"{c.topic_id}|{c.card_front}",
                interval_days=float(c.interval_days), repetition_count=int(c.repetition_count),
            ))

        for g in session.query(EngagementEvent).filter(EngagementEvent.student_id == student_id).order_by(EngagementEvent.id):
            if g.topic_id is None:
                continue
            events.append(Event(kind=ENGAGEMENT, ts=to_epoch(g.created_at), topic_id=g.topic_id))

        for r in session.query(Recommendation).filter(
            Recommendation.student_id == student_id, Recommendation.is_followed.is_(True)
        ).order_by(Recommendation.id):
            events.append(Event(kind=REC_FOLLOWED, ts=to_epoch(r.created_at), topic_id=r.topic_id))

        events.sort(key=lambda e: e.ts)  # stable: ties keep insertion order
        return events
    finally:
        if own_session:
            session.close()


def compute_feature_vector(student_id: str, topic_id: str,
                           as_of: Optional[dt.datetime] = None, session=None) -> FeatureVector:
    """Current (or historical, via `as_of`) feature vector for one (student, topic) pair."""
    as_of_ts = to_epoch(as_of) if as_of is not None else to_epoch(dt.datetime.utcnow())
    state = FeatureState()
    for event in load_student_events(student_id, session=session):
        if event.ts <= as_of_ts:
            state.update(event)
    return state.vector(student_id, topic_id, as_of_ts)


def compute_topic_vectors(student_id: str, topic_ids: Optional[List[str]] = None,
                          as_of: Optional[dt.datetime] = None, session=None) -> List[FeatureVector]:
    """Vectors for several topics in one pass (what the recommender needs). Defaults to every topic the student has touched."""
    as_of_ts = to_epoch(as_of) if as_of is not None else to_epoch(dt.datetime.utcnow())
    state = FeatureState()
    for event in load_student_events(student_id, session=session):
        if event.ts <= as_of_ts:
            state.update(event)
    topic_ids = topic_ids if topic_ids is not None else list(state.topics.keys())
    return [state.vector(student_id, tid, as_of_ts) for tid in topic_ids]
