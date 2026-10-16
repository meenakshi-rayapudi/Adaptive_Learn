import streamlit as st
from core.learner_service import log_flashcard_review
from ui.flashcard_ui import display_flashcards


def render_flashcards_tab(send_agent_goal) -> None:
    """Tab 2: Active Recall Flashcard Decks. `send_agent_goal` is the shared
    agent-dispatch callback owned by ui/app.py (keeps chat, flashcards, and
    quiz tabs all funneling through one artifact-sync code path)."""
    st.subheader("Active Recall Flashcard Decks")
    st.caption("Spaced repetition flashcards generated autonomously by the agent from your study materials.")

    review_cards = [c for c in (st.session_state.flashcards or []) if c.get("status") == "review"]

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🤖 Ask Agent to Generate Flashcards Deck", use_container_width=True):
            send_agent_goal(
                "Please analyze the uploaded study material, identify the 10 most critical high-yield concepts, and create an active-recall flashcard deck.",
                spinner_text="AI Tutor is analyzing key concepts and creating flashcards..."
            )
            st.rerun()

    with col2:
        if st.session_state.flashcards:
            if st.button("🔊 Ask Agent to Narrate Flashcards to Audio", use_container_width=True):
                deck_summary = "Here is an audio review of your flashcards: " + " ".join(
                    [f"Concept: {c.get('front')}. Definition: {c.get('back')}." for c in st.session_state.flashcards[:5]]
                )
                send_agent_goal(
                    f"Please convert this flashcard review into spoken audio narration: {deck_summary}",
                    spinner_text="AI Tutor is generating neural audio for your flashcards..."
                )
                st.rerun()

    st.divider()

    def on_quiz_drill(cards):
        review_terms = ", ".join([f"'{c.get('front')}'" for c in cards])
        send_agent_goal(
            f"The student is struggling with the following concepts from their flashcards: {review_terms}. Please generate a focused 5-question practice quiz specifically testing these concepts to reinforce active recall.",
            spinner_text="AI Tutor is crafting a targeted review quiz..."
        )
        st.session_state.remedial_guide = None
        st.session_state.missed_questions = []
        st.session_state.quiz_submitted = False
        if "user_answers" in st.session_state:
            del st.session_state.user_answers
        st.rerun()

    def on_review(card, passed):
        log_flashcard_review(st.session_state.get("student_id"), st.session_state.get("document_id"),
                             card, passed, topics=st.session_state.get("topics"))

    if st.session_state.flashcards:
        display_flashcards(st.session_state.flashcards, on_quiz_drill=on_quiz_drill, on_review=on_review)
    else:
        st.info("No flashcard deck active. Click the button above or ask your AI Tutor in chat to build a flashcard deck!")
