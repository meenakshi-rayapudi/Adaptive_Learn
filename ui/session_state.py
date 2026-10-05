"""
Single place that owns Streamlit session-state initialization and the
active student_id / document_id, so no other module has to guess whether
a key already exists before touching it.
"""

import datetime as dt

import streamlit as st
from core.learner_service import SESSION_GAP_MINUTES
from database.db import init_db

SESSION_DEFAULTS = {
    "session_started_at": None,
    "last_active_at": None,
    "flashcards": None,
    "quiz": None,
    "remedial_guide": None,
    "missed_questions": [],
    "session_results": [],
    "messages": [],
    "latest_audio": None,
    "study_plan": None,
    "quiz_submitted": False,
    "current_file": None,
    "student_id": None,
    "document_id": None,
    "topics": [],
    "topic_chunk_counts": {},
}


def init_session_state():
    """Initializes the SQLite schema and any missing session-state keys. Idempotent."""
    init_db()
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default() if callable(default) else default
    touch_session()


def start_new_session() -> None:
    now = dt.datetime.now()
    st.session_state.session_started_at = now
    st.session_state.last_active_at = now


def touch_session() -> None:
    """
    Call once per full page run (button click, chat message, ...). If the student has been
    idle longer than SESSION_GAP_MINUTES, this starts a fresh study session; otherwise the
    current one keeps running. The timer's own auto-refresh does not call this, so a tab
    left open does not count as studying.
    """
    last = st.session_state.get("last_active_at")
    if last is None or dt.datetime.now() - last > dt.timedelta(minutes=SESSION_GAP_MINUTES):
        st.session_state.session_started_at = dt.datetime.now()
    st.session_state.last_active_at = dt.datetime.now()


def is_session_idle() -> bool:
    last = st.session_state.get("last_active_at")
    return last is None or dt.datetime.now() - last > dt.timedelta(minutes=SESSION_GAP_MINUTES)


def session_minutes() -> int:
    started = st.session_state.get("session_started_at")
    if started is None:
        return 0
    return int((dt.datetime.now() - started).total_seconds() // 60)


def set_active_student(student_id: str) -> None:
    st.session_state.student_id = student_id
    start_new_session()
    agent = st.session_state.get("agent_executor")
    if agent is not None:
        agent.student_id = student_id  # so "weakest topic" is worked out for the right student


def get_active_student_id():
    return st.session_state.get("student_id")


def set_active_document(document_id: str, topics: list, topic_chunk_counts: dict) -> None:
    st.session_state.document_id = document_id
    st.session_state.topics = topics
    st.session_state.topic_chunk_counts = topic_chunk_counts


def get_active_document_id():
    return st.session_state.get("document_id")


def reset_document_artifacts() -> None:
    """Clears per-document study artifacts without touching the active student."""
    st.session_state.flashcards = None
    st.session_state.quiz = None
    st.session_state.remedial_guide = None
    st.session_state.missed_questions = []
