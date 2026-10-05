"""
Learner service: the one place the agent and the UI go to ask about a student.

It sits between three things:
  * the database (topics, engagement log),
  * the ML engine (engine/features.py turns raw logs into x = [Q, A, T, R, F, E, H]),
  * the tutor agent (core/engine.py), which uses it to pick and quiz weak topics.

Everything here is a plain function. Pass `session` only in tests; normally each
call opens and closes its own database session.
"""

import datetime as dt
import json
from collections import Counter
from typing import Dict, List, Optional

from database.db import get_session
from database.schema import Topic, EngagementEvent, QuizAttempt, QuizQuestion
from engine.feature_spec import recency_penalty
from engine.features import compute_topic_vectors

# A new "study session" starts after this many idle minutes.
SESSION_GAP_MINUTES = 15
# Credit for time spent reading after a session's last click (we only see timestamps, not when they left).
SESSION_TAIL_MINUTES = 2


def topic_key(topic_id: str) -> str:
    """'D26a79da19c_T3' -> 'T3'. ChromaDB stores the short key, the database stores the full id."""
    return topic_id.split("_")[-1]


def get_topic(topic_id: str, document_id: Optional[str] = None, session=None) -> Optional[Dict]:
    """Looks a topic up by full id ('D1_T3') or short key ('T3', needs document_id)."""
    candidates = [topic_id]
    if document_id:
        candidates.append(f"{document_id}_{topic_key(topic_id)}")

    own_session = session is None
    session = session or get_session()
    try:
        topic = session.query(Topic).filter(Topic.id.in_(candidates)).first()
        if topic is None:
            return None
        return {"id": topic.id, "key": topic.topic_key, "name": topic.name, "description": topic.description}
    finally:
        if own_session:
            session.close()


def _deficit(fv) -> float:
    """
    Day-1 weakness score from docs/architectural_decisions.md (Decision 3):
        0.5 * (1 - accuracy) + 0.3 * recency penalty + 0.2 * (1 - flashcard mastery)
    A missing value counts against the topic: never studied = fully stale,
    no flashcards = no mastery, never quizzed = 50/50 on accuracy.
    Week 4 swaps this for the trained model once a student has 3+ attempts.
    """
    accuracy = fv.a if fv.a is not None else 0.5
    recency = recency_penalty(fv.r) if fv.r is not None else 1.0
    mastery = fv.f if fv.f is not None else 0.0
    return 0.5 * (1 - accuracy) + 0.3 * recency + 0.2 * (1 - mastery)


def get_weak_topics(student_id: str, document_id: str, limit: int = 3, session=None) -> List[Dict]:
    """The student's weakest topics in a document, weakest first."""
    own_session = session is None
    session = session or get_session()
    try:
        topics = session.query(Topic).filter(Topic.document_id == document_id).all()
        if not topics:
            return []

        vectors = compute_topic_vectors(student_id, [t.id for t in topics], session=session)
        names = {t.id: t.name for t in topics}

        ranked = [
            {"topic_id": fv.topic_id, "name": names[fv.topic_id], "deficit": round(_deficit(fv), 3),
             "accuracy": fv.a, "days_since_study": fv.r}
            for fv in vectors
        ]
        ranked.sort(key=lambda row: row["deficit"], reverse=True)
        return ranked[:limit]
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Saving quiz results
# ---------------------------------------------------------------------------

def _tag_questions(quiz: List[Dict], document_id: Optional[str], topics: Optional[List[Dict]]) -> List[Optional[str]]:
    """
    Full topic id for every question. Drill quizzes already carry one; for the rest we pick the
    closest topic by text similarity (same local matching used for document chunks, no LLM call).
    """
    topic_ids = [q.get("topic_id") for q in quiz]
    missing = [i for i, t in enumerate(topic_ids) if not t]

    if missing and topics and document_id:
        from core.topic_extractor import assign_topics_to_chunks
        texts = [f"{quiz[i].get('question', '')} {quiz[i].get('answer', '')}" for i in missing]
        for i, tagged in zip(missing, assign_topics_to_chunks(texts, topics)):
            if tagged["topic_id"]:
                topic_ids[i] = f"{document_id}_{tagged['topic_id']}"
    return topic_ids


def log_quiz_attempt(student_id: Optional[str], document_id: Optional[str], quiz: List[Dict],
                     user_answers: Dict[int, str], topics: Optional[List[Dict]] = None,
                     at: Optional[dt.datetime] = None, session=None) -> Optional[int]:
    """
    Saves a finished quiz (one attempt row plus one row per question) so it counts toward the
    student's weak topics. `user_answers` maps question number -> chosen answer.
    Returns the attempt id, or None if nothing was saved. Never raises.
    """
    if not student_id or not quiz:
        return None

    own_session = session is None
    try:
        session = session or get_session()
        topic_ids = _tag_questions(quiz, document_id, topics)
        correct = [user_answers.get(i) == q.get("answer") for i, q in enumerate(quiz)]
        now = at or dt.datetime.utcnow()

        tagged = [t for t in topic_ids if t]
        attempt = QuizAttempt(
            student_id=student_id, document_id=document_id,
            topic_id=Counter(tagged).most_common(1)[0][0] if tagged else None,
            score=sum(correct), total_questions=len(quiz), accuracy=sum(correct) / len(quiz),
            started_at=now, completed_at=now,
        )
        session.add(attempt)
        session.flush()

        for i, q in enumerate(quiz):
            session.add(QuizQuestion(
                quiz_attempt_id=attempt.id, topic_id=topic_ids[i],
                question_text=q.get("question", ""), question_type=q.get("type", "mcq"),
                options=json.dumps(q.get("options", [])), correct_answer=q.get("answer", ""),
                user_answer=user_answers.get(i), is_correct=correct[i],
            ))
        session.commit()
        return attempt.id
    except Exception as e:
        print(f"Warning: could not save quiz attempt: {e}")
        if session is not None:
            session.rollback()
        return None
    finally:
        if own_session and session is not None:
            session.close()


# ---------------------------------------------------------------------------
# Study time tracking
# ---------------------------------------------------------------------------

def log_activity(student_id: Optional[str], event_type: str, document_id: Optional[str] = None,
                 topic_id: Optional[str] = None, at: Optional[dt.datetime] = None, session=None) -> None:
    """
    Records that the student did something (event_type: 'chat_question' or 'summary_read').
    Study time is worked out from these timestamps later. Never raises: a logging
    problem should not break the tutor.
    """
    if not student_id:
        return

    own_session = session is None
    try:
        session = session or get_session()
        session.add(EngagementEvent(
            student_id=student_id, document_id=document_id, topic_id=topic_id,
            event_type=event_type, created_at=at or dt.datetime.utcnow(),
        ))
        session.commit()
    except Exception as e:
        print(f"Warning: could not log study activity: {e}")
    finally:
        if own_session and session is not None:
            session.close()


def minutes_from_timestamps(times: List[dt.datetime]) -> float:
    """
    Groups timestamps into sessions (a gap over SESSION_GAP_MINUTES starts a new one) and
    adds up each session's length plus SESSION_TAIL_MINUTES.
    """
    if not times:
        return 0.0

    times = sorted(times)
    total = 0.0
    start = previous = times[0]
    for t in times[1:]:
        if (t - previous).total_seconds() / 60 > SESSION_GAP_MINUTES:
            total += (previous - start).total_seconds() / 60 + SESSION_TAIL_MINUTES
            start = t
        previous = t
    total += (previous - start).total_seconds() / 60 + SESSION_TAIL_MINUTES
    return round(total, 1)


def get_study_minutes(student_id: str, days: int = 7, now: Optional[dt.datetime] = None, session=None) -> float:
    """Total minutes studied in the last `days` days."""
    now = now or dt.datetime.utcnow()
    since = now - dt.timedelta(days=days)

    own_session = session is None
    session = session or get_session()
    try:
        rows = (
            session.query(EngagementEvent.created_at)
            .filter(EngagementEvent.student_id == student_id,
                    EngagementEvent.created_at >= since, EngagementEvent.created_at <= now)
            .all()
        )
        return minutes_from_timestamps([r[0] for r in rows])
    finally:
        if own_session:
            session.close()
