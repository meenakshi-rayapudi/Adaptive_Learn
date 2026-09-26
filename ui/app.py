import streamlit as st
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from ui.components.header import show_header
from ui.components.sidebar import show_sidebar
from ui.components.chat import chat_interface
from ui.components.flashcards_tab import render_flashcards_tab
from ui.components.quiz_tab import render_quiz_tab
from ui.styles.css import load_css
from ui.session_state import init_session_state
from utils.analytics import show_analytics
from core.engine import run_agent_query

st.set_page_config(
    page_title="Veras - AI Academic Tutor",
    page_icon="🎓",
    layout="wide"
)

# Load custom visual styling
st.markdown(load_css(), unsafe_allow_html=True)
show_header()

init_session_state()
document_source = show_sidebar()


def send_agent_goal(prompt: str, spinner_text: str = "Agent is reasoning..."):
    """
    Sends an autonomous directive to the Agent and captures all tool artifacts.
    Shared by the chat, flashcards, and quiz tabs so every entry point syncs
    artifacts (flashcards/quiz/study_plan/remedial_guide/audio) the same way.
    """
    if "agent_executor" not in st.session_state:
        st.warning("Please upload a study document or YouTube link first.")
        return None

    with st.spinner(spinner_text):
        history = []
        for m in st.session_state.messages:
            role = "human" if m["role"] == "user" else "ai"
            history.append((role, m["content"]))

        result = run_agent_query(st.session_state.agent_executor, prompt, history)

        if isinstance(result, dict):
            response_text = result.get("response", "")
            artifacts = result.get("artifacts", {})

            if artifacts.get("flashcards"):
                st.session_state.flashcards = artifacts["flashcards"]
            if artifacts.get("quiz"):
                st.session_state.quiz = artifacts["quiz"]
            if artifacts.get("study_plan"):
                st.session_state.study_plan = artifacts["study_plan"]
            if artifacts.get("remedial_guide"):
                st.session_state.remedial_guide = artifacts["remedial_guide"]
            if artifacts.get("audio_file"):
                st.session_state.latest_audio = artifacts["audio_file"]

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            return result
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": str(result)})
            return {"response": str(result), "artifacts": {}}


# ================= MAIN NAVIGATION TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["💬 Study Companion (Chat)", "🗂️ Knowledge Flashcards", "📝 Practice Quiz & Remediation", "📊 Learning Analytics"])


# ---------- TAB 1: STUDY COMPANION (CHAT) ----------
with tab1:
    st.subheader("Autonomous Academic Companion")
    st.caption("Chat with your AI Tutor. Ask for summaries, deep conceptual explanations, practice problems, or audio narration.")

    if not st.session_state.get("current_file"):
        st.info("👋 Upload a PDF textbook or paste a YouTube lecture link in the sidebar to get started!")

    # Render interactive chat
    chat_interface()

    # Agent Audio Player widget if audio was synthesized
    if st.session_state.get("latest_audio") and os.path.exists(st.session_state.latest_audio):
        st.divider()
        st.subheader("🎧 AI Synthesized Audio Narration")
        with open(st.session_state.latest_audio, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            label="⬇ Download Audio Lesson (.mp3)",
            data=audio_bytes,
            file_name="study_narration.mp3",
            mime="audio/mp3"
        )


# ---------- TAB 2: FLASHCARDS ----------
with tab2:
    render_flashcards_tab(send_agent_goal)


# ---------- TAB 3: PRACTICE QUIZ & ADAPTIVE REMEDIATION ----------
with tab3:
    render_quiz_tab(send_agent_goal)


# ---------- TAB 4: LEARNING ANALYTICS DASHBOARD ----------
with tab4:
    show_analytics()
