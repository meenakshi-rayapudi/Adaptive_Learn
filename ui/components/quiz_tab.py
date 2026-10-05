import datetime
import streamlit as st
from core.learner_service import log_quiz_attempt
from ui.components.drill import show_weak_topic_drill, show_drill_banner
from utils.analytics import save_quiz_result


def render_quiz_tab(send_agent_goal) -> None:
    """Tab 3: Practice Quiz & Adaptive Remediation. `send_agent_goal` is the
    shared agent-dispatch callback owned by ui/app.py."""
    st.subheader("Practice Assessment & Cognitive Remediation")
    st.caption("AI-generated assessments with automated error diagnosis and personalized remedial guides.")

    show_weak_topic_drill()
    st.divider()

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

    if not st.session_state.quiz:
        st.info("No practice quiz active. Click the button above to ask the AI Tutor to design an assessment from your material.")
        return

    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    show_drill_banner(st.session_state.quiz)

    if st.session_state.get("quiz_submitted"):
        _render_results(send_agent_goal)
    else:
        _render_active_quiz(send_agent_goal)


def _render_active_quiz(send_agent_goal) -> None:
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

        # Save to the database so this quiz counts toward the student's weak topics
        log_quiz_attempt(
            st.session_state.get("student_id"), st.session_state.get("document_id"),
            st.session_state.quiz, st.session_state.user_answers, topics=st.session_state.get("topics"),
        )

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


def _render_results(send_agent_goal) -> None:
    score = st.session_state.get("last_score", 0)
    total_q = len(st.session_state.quiz)
    accuracy = score / total_q if total_q > 0 else 0

    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric("Final Score", f"{score} / {total_q}")
    score_col2.metric("Accuracy", f"{int(accuracy * 100)}%")
    score_col3.metric("Result", "🌟 Perfect" if score == total_q else f"⚠️ {len(st.session_state.missed_questions)} Gaps Identified")

    if score == total_q:
        st.balloons()
        st.success("🎉 Excellent! You scored 100% on this assessment. You have mastered these concepts!")
    else:
        st.warning(f"You missed {len(st.session_state.missed_questions)} question(s). Review the detailed answers and study guide below.")

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
