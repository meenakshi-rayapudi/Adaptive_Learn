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
from database.schema import Topic, EngagementEvent, QuizAttempt, QuizQuestion, FlashcardEvent
from engine.recommender import recommend_topics
from engine.spaced_repetition import update_card_review

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


def get_weak_topics(student_id: str, document_id: str, limit: int = 3, session=None) -> List[Dict]:
    """
    The student's weakest topics in a document, weakest first. Ranked by the trained model once the
    student has 3+ quiz attempts, by the cold-start formula before that (see engine/recommender.py).
    """
    return recommend_topics(student_id, document_id, limit=limit, session=session)


# ---------------------------------------------------------------------------
# Saving quiz results
# ---------------------------------------------------------------------------

def _tag_items(items: List[Dict], document_id: Optional[str], topics: Optional[List[Dict]], text_of) -> List[Optional[str]]:
    """
    Full topic id for every quiz question or flashcard. Items from a drill already carry a real one.
    generate_quiz and generate_flashcards stamp a placeholder like "General" when they aren't given a
    topic, so only ids that belong to this document count as real; the rest get the closest topic by
    text similarity (same local matching used for document chunks, no LLM call).
    """
    prefix = f"{document_id}_" if document_id else None
    topic_ids = [item.get("topic_id") if prefix and str(item.get("topic_id") or "").startswith(prefix) else None
                 for item in items]
    missing = [i for i, t in enumerate(topic_ids) if not t]

    if missing and topics and document_id:
        from core.topic_extractor import assign_topics_to_chunks
        texts = [text_of(items[i]) for i in missing]
        for i, tagged in zip(missing, assign_topics_to_chunks(texts, topics)):
            tid = tagged["topic_id"]
            if tid:
                topic_ids[i] = tid if tid.startswith(prefix) else f"{prefix}{tid}"  # topics may carry short or full ids
    return topic_ids


def _tag_questions(quiz: List[Dict], document_id: Optional[str], topics: Optional[List[Dict]]) -> List[Optional[str]]:
    return _tag_items(quiz, document_id, topics, lambda q: f"{q.get('question', '')} {q.get('answer', '')}")


def log_quiz_attempt(student_id: Optional[str], document_id: Optional[str], quiz: List[Dict],
                     user_answers: Dict[int, str], topics: Optional[List[Dict]] = None,
                     seconds_spent: Optional[float] = None,
                     at: Optional[dt.datetime] = None, session=None) -> Optional[int]:
    """
    Saves a finished quiz (one attempt row plus one row per question) so it counts toward the
    student's weak topics. `user_answers` maps question number -> chosen answer.
    `seconds_spent` is the time the whole quiz took; each question is given an equal share as its
    response time (the quiz is answered on one page, so per-question times aren't measured).
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

        latency_ms = None
        if seconds_spent is not None:
            latency_ms = int(min(max(seconds_spent * 1000 / len(quiz), 500), 120_000))  # same bounds as feature T

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
                user_answer=user_answers.get(i), is_correct=correct[i], latency_ms=latency_ms,
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


def log_flashcard_review(student_id: Optional[str], document_id: Optional[str], card: Dict, passed: bool,
                         topics: Optional[List[Dict]] = None, at: Optional[dt.datetime] = None,
                         session=None) -> Optional[int]:
    """
    Saves one flashcard review and moves the card's spaced-repetition (SM-2) schedule forward.
    passed=True is a good recall (quality 4), passed=False a miss (quality 2). The schedule lives on the
    card itself (ease_factor, interval_days, repetition_count) and is updated even if saving fails.
    Returns the saved event id, or None. Never raises.
    """
    quality = 4 if passed else 2
    card["ease_factor"], card["interval_days"], card["repetition_count"] = update_card_review(
        card.get("ease_factor", 2.5), card.get("interval_days", 0), card.get("repetition_count", 0), quality)

    if not student_id:
        return None

    own_session = session is None
    try:
        session = session or get_session()
        topic_id = _tag_items([card], document_id, topics, lambda c: f"{c.get('front', '')} {c.get('back', '')}")[0]
        if topic_id:
            card["topic_id"] = topic_id  # remembered, so this card isn't matched again on its next review

        now = at or dt.datetime.utcnow()
        event = FlashcardEvent(
            student_id=student_id, document_id=document_id, topic_id=topic_id,
            card_front=card.get("front", ""), card_back=card.get("back", ""), quality=quality,
            ease_factor=card["ease_factor"], interval_days=card["interval_days"],
            repetition_count=card["repetition_count"],
            reviewed_at=now, next_review_at=now + dt.timedelta(days=card["interval_days"]),
        )
        session.add(event)
        session.commit()
        return event.id
    except Exception as e:
        print(f"Warning: could not save flashcard review: {e}")
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
