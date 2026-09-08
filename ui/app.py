import streamlit as st
import sys
import os
import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from ui.components.header import show_header
from ui.components.sidebar import show_sidebar
from ui.components.chat import chat_interface
from ui.styles.css import load_css
from ui.flashcard_ui import display_flashcards
from utils.analytics import show_analytics, save_quiz_result
from core.engine import run_agent_query

st.set_page_config(
    page_title="Veras - AI Academic Tutor",
    page_icon="🎓",
    layout="wide"
)

# Load custom visual styling
st.markdown(load_css(), unsafe_allow_html=True)
show_header()
document_source = show_sidebar()

# ================= SESSION STATE INITIALIZATION =================
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "remedial_guide" not in st.session_state:
    st.session_state.remedial_guide = None
if "missed_questions" not in st.session_state:
    st.session_state.missed_questions = []
if "session_results" not in st.session_state:
    st.session_state.session_results = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_audio" not in st.session_state:
    st.session_state.latest_audio = None


def send_agent_goal(prompt: str, spinner_text: str = "Agent is reasoning..."):
    """
    Sends an autonomous directive to the Agent and captures all tool artifacts.
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

    def on_quiz_drill(review_cards):
        review_terms = ", ".join([f"'{c.get('front')}'" for c in review_cards])
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

    if st.session_state.flashcards:
        display_flashcards(st.session_state.flashcards, on_quiz_drill=on_quiz_drill)
    else:
        st.info("No flashcard deck active. Click the button above or ask your AI Tutor in chat to build a flashcard deck!")


# ---------- TAB 3: PRACTICE QUIZ & ADAPTIVE REMEDIATION ----------
with tab3:
    st.subheader("Practice Assessment & Cognitive Remediation")
    st.caption("AI-generated assessments with automated error diagnosis and personalized remedial guides.")

    top_col1, top_col2 = st.columns([1, 1])
    with top_col1:
        if st.button("🤖 Ask Agent to Design a New Practice Assessment", use_container_width=True):
            send_agent_goal(
                "Please generate a rigorous 5-question practice quiz with a mix of MCQ and True/False questions based on the core principles in this document.",
                spinner_text="AI Tutor is designing your assessment..."
            )
            st.session_state.remedial_guide = None
            st.session_state.missed_questions = []
            st.session_state.quiz_submitted = False
            if "user_answers" in st.session_state:
                del st.session_state.user_answers
            st.rerun()

    with top_col2:
        if st.session_state.get("quiz_submitted"):
            if st.button("🔄 Retake Current Quiz", use_container_width=True):
                st.session_state.quiz_submitted = False
                st.session_state.remedial_guide = None
                st.session_state.missed_questions = []
                if "user_answers" in st.session_state:
                    del st.session_state.user_answers
                st.rerun()

    st.divider()

    if st.session_state.quiz:
        if "user_answers" not in st.session_state:
            st.session_state.user_answers = {}

        # ── STATE A: QUIZ HAS BEEN SUBMITTED (SHOW SCORE, ANSWERS & REMEDIAL GUIDE) ──
        if st.session_state.get("quiz_submitted"):
            score = st.session_state.get("last_score", 0)
            total_q = len(st.session_state.quiz)
            accuracy = score / total_q if total_q > 0 else 0

            # Score Banner
            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric("Final Score", f"{score} / {total_q}")
            score_col2.metric("Accuracy", f"{int(accuracy * 100)}%")
            score_col3.metric("Result", "🌟 Perfect" if score == total_q else f"⚠️ {len(st.session_state.missed_questions)} Gaps Identified")

            if score == total_q:
                st.balloons()
                st.success("🎉 Excellent! You scored 100% on this assessment. You have mastered these concepts!")
            else:
                st.warning(f"You missed {len(st.session_state.missed_questions)} question(s). Review the detailed answers and study guide below.")

            # Detailed Answer Explanations
            st.subheader("📝 Question Breakdown & Explanations")
            for i, q in enumerate(st.session_state.quiz):
                user_ans = st.session_state.user_answers.get(i, "No answer selected")
                correct_ans = q.get("answer")
                is_correct = (user_ans == correct_ans)

                with st.expander(
                    f"{'✅' if is_correct else '❌'} Q{i+1}: {q.get('question')}",
                    expanded=not is_correct
                ):
                    st.markdown(f"**Question:** {q.get('question')}")
                    if is_correct:
                        st.success(f"**Your Answer:** `{user_ans}` (Correct!)")
                    else:
                        st.error(f"**Your Answer:** `{user_ans}` | **Correct Answer:** `{correct_ans}`")
                    st.info(f"💡 **Explanation:** {q.get('explanation', 'Ground truth concept from source material.')}")

            # Personalized Remedial Study Guide
            if st.session_state.get("remedial_guide"):
                st.divider()
                st.subheader("📘 AI Remedial Study Guide")
                st.markdown(st.session_state.remedial_guide)
                st.download_button(
                    label="📥 Download Remedial Study Guide (.md)",
                    data=st.session_state.remedial_guide,
                    file_name="remedial_study_guide.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # ── STATE B: QUIZ IS ACTIVE (SHOW INTERACTIVE FORM) ──
        else:
            st.info("Select your answers and click **Submit Assessment** at the bottom.")
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"#### Q{i+1}: {q.get('question')}")

                if q.get("type") == "mcq":
                    st.session_state.user_answers[i] = st.radio(
                        f"Select your answer for Q{i+1}:",
                        q.get("options", []),
                        key=f"quiz_radio_{i}"
                    )
                elif q.get("type") == "true_false":
                    st.session_state.user_answers[i] = st.radio(
                        f"Select True or False for Q{i+1}:",
                        ["True", "False"],
                        key=f"quiz_radio_{i}"
                    )
                st.divider()

            if st.button("📊 Submit Assessment & Receive Feedback", use_container_width=True):
                score = 0
                st.session_state.missed_questions = []

                for i, q in enumerate(st.session_state.quiz):
                    user_ans = st.session_state.user_answers.get(i)
                    correct_ans = q.get("answer")

                    if user_ans == correct_ans:
                        score += 1
                    else:
                        st.session_state.missed_questions.append({
                            "question": q.get("question"),
                            "answer": correct_ans,
                            "user_answer": user_ans,
                            "explanation": q.get("explanation", "")
                        })

                total_q = len(st.session_state.quiz)
                accuracy = score / total_q if total_q > 0 else 0
                topic = st.session_state.get("current_file", "General Study Session")

                st.session_state.last_score = score
                st.session_state.last_total = total_q
                st.session_state.last_accuracy = accuracy
                st.session_state.quiz_submitted = True

                # Persist score to CSV analytics
                save_quiz_result(topic, score, total_q)
                st.session_state.session_results.append({
                    "topic": topic,
                    "correct": accuracy,
                    "type": "quiz",
                    "date": datetime.date.today().strftime("%Y-%m-%d")
                })

                # If there are missed questions, have agent generate remedial guide
                if st.session_state.missed_questions:
                    missed_summary = "\n".join([
                        f"- Question: {m['question']}\n  Correct: {m['answer']}\n  Student Answer: {m['user_answer']}\n  Note: {m.get('explanation', '')}"
                        for m in st.session_state.missed_questions
                    ])
                    send_agent_goal(
                        f"The student completed a practice quiz on '{topic}' and missed the following questions:\n{missed_summary}\n\nPlease analyze their errors, diagnose the underlying conceptual gaps, and generate a Personalized Remedial Mastery Guide with memory mnemonics.",
                        spinner_text="AI Tutor is performing cognitive diagnosis and crafting your remedial guide..."
                    )

                st.rerun()
    else:
        st.info("No practice quiz active. Click the button above to ask the AI Tutor to design an assessment from your material.")


# ---------- TAB 4: LEARNING ANALYTICS DASHBOARD ----------
with tab4:
    show_analytics()
