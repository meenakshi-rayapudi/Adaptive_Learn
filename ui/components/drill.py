import streamlit as st
from core.learner_service import get_weak_topics, get_topic


def _label(weak_topic: dict) -> str:
    if weak_topic["accuracy"] is None:
        return f"{weak_topic['name']}  ·  not quizzed yet"
    return f"{weak_topic['name']}  ·  {weak_topic['accuracy']:.0%} accuracy"


def show_weak_topic_drill():
    """'Review Weak Topic': lists the document's topics, weakest first, and starts a focused 5-question quiz on the one picked."""
    st.markdown("#### 🎯 Targeted Remedial Drill")

    student_id = st.session_state.get("student_id")
    document_id = st.session_state.get("document_id")
    agent = st.session_state.get("agent_executor")

    if not (student_id and document_id and agent):
        st.caption("Pick a student and upload a document to unlock drills on your weakest topics.")
        return

    weak_topics = get_weak_topics(student_id, document_id, limit=50)
    if not weak_topics:
        st.caption("No topics were found for this document yet.")
        return

    by_id = {w["topic_id"]: w for w in weak_topics}
    topic_id = st.selectbox(
        "Topics (weakest first, untouched topics count as weak)",
        list(by_id),
        format_func=lambda t: _label(by_id[t]),
        key="drill_topic_id",
    )

    if st.button("🔍 Review Weak Topic", type="primary", use_container_width=True):
        agent.artifacts.quiz = None  # so a failed attempt can't show an older quiz
        with st.spinner(f"Building a 5-question drill on {by_id[topic_id]['name']}..."):
            message = agent.make_targeted_quiz(topic_id, "medium")

        if agent.artifacts.quiz:
            st.session_state.quiz = agent.artifacts.quiz
            st.session_state.remedial_guide = None
            st.session_state.missed_questions = []
            st.session_state.quiz_submitted = False
            if "user_answers" in st.session_state:
                del st.session_state.user_answers
            st.rerun()
        else:
            st.error(message)


def show_drill_banner(quiz) -> None:
    """Small label above a quiz that came from the drill, so the student knows what it covers."""
    topic_id = quiz[0].get("topic_id") if quiz else None
    topic = get_topic(topic_id) if topic_id else None
    if topic:
        st.info(f"🎯 Targeted drill on: **{topic['name']}**")
