"""
Relational schema for AdaptiveLearn's "Learner Brain".

Scope (Week 1): students, documents, topics, chunks, quiz_attempts,
quiz_questions, flashcard_events, recommendations.

Design notes:
- Student/Document/Topic use short, human-readable string primary keys
  (e.g. "S001", "D0001", "D0001_T1") because they are referenced directly
  in prompts, UI badges, and the ASSISTments/EdNet-style feature vector
  spec (engine/feature_spec.py), where readable IDs make debugging and
  interview walkthroughs far easier than opaque surrogate keys.
- Topics are scoped to a single document for v1 (see docs/architectural_decisions.md,
  Decision 11) — cross-document topic merging is a v2 concern.
- Chunk rows store only a preview + the ChromaDB id; the actual chunk text
  and embedding live in ChromaDB (core/vector_store.py). This table exists
  so SQL joins can answer "which topics does this document have and how
  many chunks tag each one" without opening the vector store.
"""

import datetime as dt
from typing import Optional, List

from sqlalchemy import ForeignKey, String, Integer, Float, Boolean, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


class Student(Base):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    archetype: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    quiz_attempts: Mapped[List["QuizAttempt"]] = relationship(back_populates="student")
    flashcard_events: Mapped[List["FlashcardEvent"]] = relationship(back_populates="student")
    recommendations: Mapped[List["Recommendation"]] = relationship(back_populates="student")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(32), default="pdf")  # pdf | youtube | synthetic
    uploaded_by: Mapped[Optional[str]] = mapped_column(ForeignKey("students.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    topics: Mapped[List["Topic"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)  # e.g. "D0001_T3"
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    topic_key: Mapped[str] = mapped_column(String(16))  # e.g. "T3" (short form used in prompts/UI)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped["Document"] = relationship(back_populates="topics")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="topic")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    topic_id: Mapped[Optional[str]] = mapped_column(ForeignKey("topics.id"), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chroma_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    text_preview: Mapped[str] = mapped_column(String(300), default="")
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # cosine sim to assigned topic
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    topic: Mapped[Optional["Topic"]] = relationship(back_populates="chunks")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    document_id: Mapped[Optional[str]] = mapped_column(ForeignKey("documents.id"), nullable=True)
    topic_id: Mapped[Optional[str]] = mapped_column(ForeignKey("topics.id"), nullable=True)
    score: Mapped[int] = mapped_column(Integer, default=0)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    student: Mapped["Student"] = relationship(back_populates="quiz_attempts")
    questions: Mapped[List["QuizQuestion"]] = relationship(back_populates="attempt", cascade="all, delete-orphan")


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quiz_attempt_id: Mapped[int] = mapped_column(ForeignKey("quiz_attempts.id"))
    topic_id: Mapped[Optional[str]] = mapped_column(ForeignKey("topics.id"), nullable=True)
    question_text: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(16), default="mcq")  # mcq | true_false
    options: Mapped[str] = mapped_column(Text, default="")  # JSON-encoded list
    correct_answer: Mapped[str] = mapped_column(String(300))
    user_answer: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # cognitive struggle signal (T)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    attempt: Mapped["QuizAttempt"] = relationship(back_populates="questions")


class FlashcardEvent(Base):
    __tablename__ = "flashcard_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    document_id: Mapped[Optional[str]] = mapped_column(ForeignKey("documents.id"), nullable=True)
    topic_id: Mapped[Optional[str]] = mapped_column(ForeignKey("topics.id"), nullable=True)
    card_front: Mapped[str] = mapped_column(Text)
    card_back: Mapped[str] = mapped_column(Text)
    quality: Mapped[int] = mapped_column(Integer)  # SM-2 recall quality, 0-5
    ease_factor: Mapped[float] = mapped_column(Float, default=2.5)
    interval_days: Mapped[int] = mapped_column(Integer, default=0)
    repetition_count: Mapped[int] = mapped_column(Integer, default=0)
    reviewed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    next_review_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    student: Mapped["Student"] = relationship(back_populates="flashcard_events")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"))
    deficit_score: Mapped[float] = mapped_column(Float)  # D in [0.0, 1.0]
    priority: Mapped[str] = mapped_column(String(16), default="medium")  # high | medium | low
    model_version: Mapped[str] = mapped_column(String(32), default="cold_start_v1")
    shap_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON-encoded top-3 SHAP features
    is_followed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    student: Mapped["Student"] = relationship(back_populates="recommendations")
