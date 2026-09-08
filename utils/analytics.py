import streamlit as st
import pandas as pd
import os
import datetime

def save_quiz_result(topic, score, total):
    """
    Saves the quiz performance to a CSV file.
    """
    file_path = "data/audio/performance.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    accuracy = score / total if total > 0 else 0
    date = datetime.date.today().strftime("%Y-%m-%d")
    
    new_data = pd.DataFrame([{
        "topic": topic,
        "correct": accuracy,
        "type": "quiz",
        "date": date
    }])
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df = pd.concat([df, new_data], ignore_index=True)
        except Exception:
            df = new_data
    else:
        df = new_data
        
    df.to_csv(file_path, index=False)

def show_analytics():

    st.header("Learning Analytics Dashboard")

    file_path = "data/audio/performance.csv"
    
    # Load data from CSV if it exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading analytics data: {e}")
            return
    else:
        # Fallback to current session results if CSV doesn't exist yet
        if not st.session_state.get("session_results"):
            st.warning("You haven't taken a quiz yet. Please complete at least one quiz to unlock your learning analytics!")
            return
        df = pd.DataFrame(st.session_state.session_results)

    if df.empty:
        st.warning("No study data found yet. Start learning to see your progress!")
        return

    # Convert date column to datetime for better sorting/plotting
    df['date'] = pd.to_datetime(df['date'])

    # Dashboard Layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy Over Time
        st.subheader("Progress Over Time")
        accuracy = df.groupby("date")["correct"].mean()
        st.line_chart(accuracy)

    with col2:
        # Daily Study Activity
        st.subheader("Study Activity")
        activity = df.groupby("date").size()
        st.bar_chart(activity)

    # Topic Mastery
    st.subheader("Topic Mastery")
    topic_accuracy = df.groupby("topic")["correct"].mean().sort_values(ascending=False)
    st.bar_chart(topic_accuracy)

    # Overall Mastery
    st.divider()
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        mastery = df["correct"].mean() * 100
        st.metric("Overall Mastery Score", f"{mastery:.2f}%")
        
    with m_col2:
        total_quizzes = len(df)
        st.metric("Total Quizzes Taken", total_quizzes)