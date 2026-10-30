import streamlit as st
from core.learner_service import get_weak_topics
from ui.components.recommendation_card import PRIORITY_LABELS, reasons_in_words


def show_recommendation_banner():
    """
    'Recommended Next Topic' strip at the top of the chat and quiz tabs: the student's single most
    needed topic, how urgent it is, and the first reasons why. Shows nothing until a student and a
    document are loaded.
    """
    student_id = st.session_state.get("student_id")
    document_id = st.session_state.get("document_id")
    if not (student_id and document_id):
        return

    top = get_weak_topics(student_id, document_id, limit=1)
    if not top:
        return

    rec = top[0]
    text = (f"🎯 **Recommended Next Topic: {rec['name']}** · "
            f"{PRIORITY_LABELS.get(rec['priority'], rec['priority'])} · {rec['deficit']:.0%} risk")
    why = " · ".join(reasons_in_words(rec)[:2])
    if why:
        text += f"  \n{why}"

    (st.warning if rec["priority"] == "high" else st.info)(text)
