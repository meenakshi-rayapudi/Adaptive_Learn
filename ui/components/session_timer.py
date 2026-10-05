import streamlit as st
from core.learner_service import get_study_minutes
from ui.session_state import session_minutes, is_session_idle


@st.fragment(run_every="30s")
def show_session_timer(student_id: str):
    """Sidebar clock for the current study session. Refreshes itself every 30 seconds."""
    if is_session_idle():
        st.caption("⏸️ Session paused. Click anything to start a new one.")
    else:
        st.metric("⏱️ This study session", f"{session_minutes()} min")
    st.caption(f"📅 Logged in the last 7 days: {get_study_minutes(student_id):.0f} min")
