"""
Single place that owns Streamlit session-state initialization and the
active student_id / document_id, so no other module has to guess whether
a key already exists before touching it.
"""

import streamlit as st
from database.db import init_db

SESSION_DEFAULTS = {
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


def set_active_student(student_id: str) -> None:
    st.session_state.student_id = student_id


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
