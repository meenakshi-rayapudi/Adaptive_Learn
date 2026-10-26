"""
Week 5: Executive Learning Analytics Dashboard (Tab 4).

Features 4 rich Plotly charts powered directly by SQLite relational queries:
  Visual 1: Topic Mastery (Horizontal Bar / Radar)
  Visual 2: Spaced Repetition Retention Decay Curve (Ebbinghaus forgetting curve vs SM-2 cards)
  Visual 3: Chronological Accuracy Trend Line (interactive Plotly line with moving average)
  Visual 4: Study Time & Activity Heatmap / Daily Engagement Bar
"""

import math
import os
import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import learner_service
from database.db import get_session
from database.schema import QuizAttempt, QuizQuestion, Topic, FlashcardEvent


def render_learning_analytics_dashboard():
    st.header("📊 Advanced Learning Analytics Dashboard")
    st.caption("Real-time cognitive progress, retention curves, and topic mastery powered by relational SQLite logs.")

    student_id = st.session_state.get("student_id")
    document_id = st.session_state.get("document_id")

    if not student_id:
        st.info("👋 Select or create a student in the sidebar to view their learning analytics.")
        return

    session = get_session()
    try:
        # Load core metrics
        attempts_count = session.query(QuizAttempt).filter(QuizAttempt.student_id == student_id).count()
        total_questions = (
            session.query(QuizQuestion)
            .join(QuizAttempt, QuizQuestion.quiz_attempt_id == QuizAttempt.id)
            .filter(QuizAttempt.student_id == student_id)
            .count()
        )
        correct_questions = (
            session.query(QuizQuestion)
            .join(QuizAttempt, QuizQuestion.quiz_attempt_id == QuizAttempt.id)
            .filter(QuizAttempt.student_id == student_id, QuizQuestion.is_correct == True)
            .count()
        )
        study_data = learner_service.get_study_time_distribution(student_id, days=14, session=session)
        cards_data = learner_service.get_flashcard_retention_breakdown(student_id, document_id=document_id, session=session)
        accuracy_trends = learner_service.get_accuracy_trends_by_topic(student_id, document_id=document_id, session=session)
    finally:
        session.close()

    overall_acc = (correct_questions / total_questions) if total_questions > 0 else 0.0

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Accuracy", f"{overall_acc:.1%}", f"{correct_questions}/{total_questions} questions")
    k2.metric("Total Quizzes", attempts_count)
    k3.metric("Study Time (14d)", f"{study_data['total_minutes']:.0f} mins")
    k4.metric("Active Flashcards", cards_data["total_cards"], f"{cards_data['counts']['mastered']} mastered")

    st.divider()

    # ---------------------------------------------------------
    # Visual 1 & Visual 3: Accuracy & Topic Mastery
    # ---------------------------------------------------------
    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.subheader("🎯 Topic Mastery")
        if accuracy_trends:
            df_acc = pd.DataFrame(accuracy_trends)
            topic_summary = (
                df_acc.groupby("topic_name")["is_correct"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"mean": "accuracy", "count": "attempts"})
                .sort_values("accuracy", ascending=True)
            )

            fig_mastery = px.bar(
                topic_summary,
                x="accuracy",
                y="topic_name",
                orientation="h",
                color="accuracy",
                color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                range_color=[0, 1],
                text=topic_summary["accuracy"].apply(lambda v: f"{v:.0%}"),
                labels={"accuracy": "Mastery (Accuracy)", "topic_name": "Topic"},
            )
            fig_mastery.update_layout(
                margin=dict(l=10, r=20, t=20, b=20),
                height=340,
                xaxis=dict(tickformat=".0%", range=[0, 1.05]),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_mastery, use_container_width=True)
        else:
            st.info("Take a quiz to see topic mastery breakdown.")

    with row1_right:
        st.subheader("📈 Chronological Accuracy Trend")
        if accuracy_trends:
            df_acc = pd.DataFrame(accuracy_trends)
            df_acc["date"] = pd.to_datetime(df_acc["date"])
            daily_acc = df_acc.groupby("date")["is_correct"].mean().reset_index()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily_acc["date"],
                y=daily_acc["is_correct"],
                mode="lines+markers",
                name="Daily Accuracy",
                line=dict(color="#3B82F6", width=3),
                marker=dict(size=8, color="#1D4ED8"),
            ))

            if len(daily_acc) > 3:
                daily_acc["rolling"] = daily_acc["is_correct"].rolling(window=3, min_periods=1).mean()
                fig_trend.add_trace(go.Scatter(
                    x=daily_acc["date"],
                    y=daily_acc["rolling"],
                    mode="lines",
                    name="3-Day Trend",
                    line=dict(color="#93C5FD", width=2, dash="dash"),
                ))

            fig_trend.update_layout(
                margin=dict(l=10, r=20, t=20, b=20),
                height=340,
                yaxis=dict(tickformat=".0%", range=[0, 1.05], title="Accuracy"),
                xaxis=dict(title="Date"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No quiz submissions logged yet.")

    st.divider()

    # ---------------------------------------------------------
    # Visual 2 & Visual 4: Retention Decay Curve & Activity Heatmap
    # ---------------------------------------------------------
    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("🧠 Spaced Repetition Retention Decay Curve")
        st.caption("Ebbinghaus memory decay: $R = e^{-t / S}$ modeled on student's actual SM-2 review history.")

        card_list = cards_data.get("cards", [])
        t_days = np.linspace(0, 30, 60)
        fig_decay = go.Figure()

        if card_list:
            # Group cards by category and compute student's empirical mean stability S
            # For each card: stability S is proportional to (interval_days * ease_factor / 2.5)
            s_by_cat = {"new": [], "review": [], "mastered": []}
            for c in card_list:
                cat = c["category"]
                # S = max(1.0, interval * (ease / 2.5))
                s_val = max(1.0, (c["interval_days"] or 1.0) * ((c["ease_factor"] or 2.5) / 2.5))
                s_by_cat[cat].append(s_val)

            colors = {"new": "#EF4444", "review": "#F59E0B", "mastered": "#10B981"}
            styles = {"new": "dot", "review": "solid", "mastered": "solid"}
            widths = {"new": 1.5, "review": 2, "mastered": 3}

            for cat in ["new", "review", "mastered"]:
                s_values = s_by_cat[cat]
                count = len(s_values)
                if count > 0:
                    mean_s = float(np.mean(s_values))
                    r_curve = np.exp(-t_days / mean_s)
                    fig_decay.add_trace(go.Scatter(
                        x=t_days,
                        y=r_curve,
                        mode="lines",
                        name=f"{cat.title()} Cards ({count}, S={mean_s:.1f}d)",
                        line=dict(color=colors[cat], dash=styles[cat], width=widths[cat]),
                    ))
        else:
            # Standard theoretical baseline if no cards reviewed yet
            r_new = np.exp(-t_days / 1.5)
            r_review = np.exp(-t_days / 6.0)
            r_mastered = np.exp(-t_days / 21.0)
            fig_decay.add_trace(go.Scatter(x=t_days, y=r_new, mode="lines", name="New (Baseline)", line=dict(color="#EF4444", dash="dot")))
            fig_decay.add_trace(go.Scatter(x=t_days, y=r_review, mode="lines", name="Review (Baseline)", line=dict(color="#F59E0B")))
            fig_decay.add_trace(go.Scatter(x=t_days, y=r_mastered, mode="lines", name="Mastered (Baseline)", line=dict(color="#10B981", width=3)))

        # 80% Retention threshold line
        fig_decay.add_hline(y=0.8, line_dash="dash", line_color="gray", annotation_text="Target Retention (80%)")

        fig_decay.update_layout(
            margin=dict(l=10, r=20, t=20, b=20),
            height=340,
            xaxis=dict(title="Days Elapsed Since Review (t)"),
            yaxis=dict(tickformat=".0%", range=[0, 1.05], title="Estimated Recall Probability (R)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_decay, use_container_width=True)

    with row2_right:
        st.subheader("⏱️ Study Activity & Weekly Distribution")
        view_mode = st.radio("View", ["7x24 Activity Heatmap", "Daily Study Time", "Weekly Distribution"], horizontal=True, label_visibility="collapsed")

        if view_mode == "7x24 Activity Heatmap":
            # Visual 4: Study Time & Activity Heatmap
            heatmap_data = study_data.get("heatmap_matrix") or [[0]*24 for _ in range(7)]
            days_labels = study_data.get("days_of_week") or ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            hours_labels = [f"{h:02d}:00" for h in range(24)]

            fig_hm = px.imshow(
                heatmap_data,
                labels=dict(x="Hour of Day", y="Day of Week", color="Events"),
                x=hours_labels,
                y=days_labels,
                color_continuous_scale="Viridis",
                aspect="auto",
            )
            fig_hm.update_layout(
                margin=dict(l=10, r=20, t=10, b=20),
                height=300,
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        elif view_mode == "Daily Study Time":
            daily_mins = study_data["daily_minutes"]
            if daily_mins:
                df_study = pd.DataFrame(list(daily_mins.items()), columns=["Date", "Minutes"]).sort_values("Date")
                fig_study = px.bar(
                    df_study,
                    x="Date",
                    y="Minutes",
                    color="Minutes",
                    color_continuous_scale="Blues",
                    text=df_study["Minutes"].apply(lambda m: f"{m:.0f}m" if m > 0 else ""),
                )
                fig_study.update_layout(
                    margin=dict(l=10, r=20, t=10, b=20),
                    height=300,
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Minutes"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_study, use_container_width=True)
            else:
                st.info("No study events logged in the selected window.")

        else:
            # Weekly distribution breakdown
            weekly_dist = study_data.get("weekly_distribution") or {}
            df_weekly = pd.DataFrame(list(weekly_dist.items()), columns=["Day", "Minutes"])
            fig_week = px.bar(
                df_weekly,
                x="Day",
                y="Minutes",
                color="Minutes",
                color_continuous_scale="Teal",
                text=df_weekly["Minutes"].apply(lambda m: f"{m:.0f}m" if m > 0 else ""),
            )
            fig_week.update_layout(
                margin=dict(l=10, r=20, t=10, b=20),
                height=300,
                xaxis=dict(title="Day of Week"),
                yaxis=dict(title="Total Study Minutes"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_week, use_container_width=True)

    # ---------------------------------------------------------
    # Model Explainability & Diagnostics Section
    # ---------------------------------------------------------
    with st.expander("🔍 Model Diagnostics & Global Interpretability (SHAP & Calibration)"):
        st.write("Artifacts exported by automated model evaluation on held-out test attempts:")
        cal_img = os.path.join("reports", "calibration_curve.png")
        shap_img = os.path.join("reports", "shap_summary.png")
        fi_img = os.path.join("reports", "feature_importances.png")

        c1, c2, c3 = st.columns(3)
        if os.path.exists(cal_img):
            c1.image(cal_img, caption="Reliability Calibration Curve")
        if os.path.exists(fi_img):
            c2.image(fi_img, caption="Feature Importances (Tree Splits)")
        if os.path.exists(shap_img):
            c3.image(shap_img, caption="Global SHAP Bee-Swarm Plot")

