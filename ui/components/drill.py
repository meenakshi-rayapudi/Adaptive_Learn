import streamlit as st
from core.learner_service import get_weak_topics, get_topic
from engine.recommender import save_recommendation
from ui.components.recommendation_card import show_recommendation_card


def _label(weak_topic: dict) -> str:
    if weak_topic["accuracy"] is None:
        return f"{weak_topic['name']}  ·  not quizzed yet"
    return f"{weak_topic['name']}  ·  {weak_topic['accuracy']:.0%} accuracy"


def _take_targeted_quiz(agent, rec: dict) -> None:
    agent.artifacts.quiz = None  # so a failed attempt can't show an older quiz
    with st.spinner(f"Building a 5-question quiz on {rec['name']}..."):
        message = agent.make_targeted_quiz(rec["topic_id"], "medium")

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


def _launch_flashcard_drill(agent, rec: dict) -> None:
    agent.artifacts.flashcards = None
    with st.spinner(f"Building flashcards on {rec['name']}..."):
        message = agent.make_targeted_flashcards(rec["topic_id"])

    if agent.artifacts.flashcards:
        st.session_state.flashcards = agent.artifacts.flashcards
        st.session_state.card_study_idx = 0
        st.session_state.drill_message = (
            f"🗂️ {len(agent.artifacts.flashcards)} flashcards on {rec['name']} are ready. "
            f"Open the **Knowledge Flashcards** tab to start."
        )
        st.rerun()
    else:
        st.error(message)


def show_weak_topic_drill():
    """Lists the document's topics weakest first and shows a Recommendation Breakdown card for the one picked."""
    st.markdown("#### 🎯 Targeted Remedial Drill")

    message = st.session_state.pop("drill_message", None)
    if message:
        st.success(message)

    student_id = st.session_state.get("student_id")
    document_id = st.session_state.get("document_id")
    agent = st.session_state.get("agent_executor")

    if not (student_id and document_id and agent):
        st.caption("Pick a student and upload a document to unlock drills on your weakest topics.")
        return

    recommendations = get_weak_topics(student_id, document_id, limit=50)
    if not recommendations:
        st.caption("No topics were found for this document yet.")
        return

    by_id = {r["topic_id"]: r for r in recommendations}
    topic_id = st.selectbox(
        "Topics (weakest first, untouched topics count as weak)",
        list(by_id),
        format_func=lambda t: _label(by_id[t]),
        key="drill_topic_id",
    )
    rec = by_id[topic_id]

    with st.container(border=True):
        show_recommendation_card(rec)
        quiz_col, cards_col = st.columns(2)
        take_quiz = quiz_col.button("📝 Take Targeted Quiz", type="primary", use_container_width=True)
        launch_cards = cards_col.button("🗂️ Launch Flashcard Drill", use_container_width=True)

    if take_quiz or launch_cards:
        save_recommendation(student_id, rec, followed=True)  # the student acted on it
        if take_quiz:
            _take_targeted_quiz(agent, rec)
        else:
            _launch_flashcard_drill(agent, rec)


def show_drill_banner(quiz) -> None:
    """Small label above a quiz that came from the drill, so the student knows what it covers."""
    topic_id = quiz[0].get("topic_id") if quiz else None
    topic = get_topic(topic_id) if topic_id else None
    if topic:
        st.info(f"🎯 Targeted drill on: **{topic['name']}**")
