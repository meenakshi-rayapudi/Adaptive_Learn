import os
import json
from typing import Optional, Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .schema import Base, QuizAttempt, QuizQuestion, FlashcardEvent

DB_PATH = os.getenv("ADAPTIVELEARN_DB_PATH", os.path.join("data", "adaptivelearn.db"))

_engine = None
_SessionFactory = None


def get_engine():
    """Lazy singleton SQLAlchemy engine bound to the local SQLite file."""
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def init_db():
    """Creates all tables if they do not already exist. Safe to call on every app startup."""
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    """Returns a new SQLAlchemy session. Caller is responsible for closing it."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def log_quiz_attempt(
    student_id: str,
    topic_id: str,
    score: int,
    total_questions: int,
    document_id: Optional[str] = None,
    started_at=None,
    completed_at=None,
) -> Dict[str, Any]:
    """Persist one quiz attempt and return a dictionary payload for the UI/backend."""
    session = get_session()
    try:
        attempts = session.query(QuizAttempt).filter(QuizAttempt.student_id == student_id).count()
        accuracy = (float(score) / float(total_questions)) if total_questions else 0.0
        attempt = QuizAttempt(
            student_id=student_id,
            document_id=document_id,
            topic_id=topic_id,
            score=int(score),
            total_questions=int(total_questions),
            accuracy=accuracy,
            started_at=started_at,
            completed_at=completed_at,
        )
        session.add(attempt)
        session.commit()
        session.refresh(attempt)
        return {
            "id": attempt.id,
            "student_id": attempt.student_id,
            "document_id": attempt.document_id,
            "topic_id": attempt.topic_id,
            "score": attempt.score,
            "total_questions": attempt.total_questions,
            "accuracy": attempt.accuracy,
        }
    finally:
        session.close()


def log_question_event(
    quiz_attempt_id: int,
    topic_id: str,
    question_text: str,
    question_type: str,
    options: list,
    correct_answer: str,
    user_answer: Optional[str],
    is_correct: bool,
    latency_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist a single quiz question row and return its metadata."""
    session = get_session()
    try:
        question = QuizQuestion(
            quiz_attempt_id=quiz_attempt_id,
            topic_id=topic_id,
            question_text=question_text,
            question_type=question_type,
            options=json.dumps(options),
            correct_answer=str(correct_answer),
            user_answer=str(user_answer) if user_answer is not None else None,
            is_correct=bool(is_correct),
            latency_ms=latency_ms,
        )
        session.add(question)
        session.commit()
        session.refresh(question)
        return {
            "id": question.id,
            "quiz_attempt_id": question.quiz_attempt_id,
            "topic_id": question.topic_id,
            "question_text": question.question_text,
            "question_type": question.question_type,
            "correct_answer": question.correct_answer,
            "user_answer": question.user_answer,
            "is_correct": question.is_correct,
            "latency_ms": question.latency_ms,
        }
    finally:
        session.close()


def log_flashcard_review(
    student_id: str,
    topic_id: str,
    quality: int,
    ease_factor: float = 2.5,
    interval_days: int = 0,
    repetition_count: int = 0,
    document_id: Optional[str] = None,
    card_front: str = "",
    card_back: str = "",
) -> Dict[str, Any]:
    """Persist one flashcard review event using the SM-2 quality rating blur."""
    session = get_session()
    try:
        event = FlashcardEvent(
            student_id=student_id,
            document_id=document_id,
            topic_id=topic_id,
            card_front=card_front or "Review concept",
            card_back=card_back or "Flashcard explanation",
            quality=int(quality),
            ease_factor=float(ease_factor),
            interval_days=int(interval_days),
            repetition_count=int(repetition_count),
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return {
            "id": event.id,
            "student_id": event.student_id,
            "document_id": event.document_id,
            "topic_id": event.topic_id,
            "quality": event.quality,
            "ease_factor": event.ease_factor,
            "interval_days": event.interval_days,
            "repetition_count": event.repetition_count,
        }
    finally:
        session.close()
