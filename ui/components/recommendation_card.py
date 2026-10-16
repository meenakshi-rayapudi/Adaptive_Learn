import html

import streamlit as st

PRIORITY_COLORS = {"high": "#DC2626", "medium": "#D97706", "low": "#059669"}
PRIORITY_LABELS = {"high": "High priority", "medium": "Medium priority", "low": "Low priority"}


def _sentence(reason: dict):
    """One SHAP reason (see engine/predictor.py) as a plain-English bullet, or None if there is nothing to say."""
    feature, value, raises = reason["feature"], reason["value"], reason["direction"] == "raises"

    if feature == "A":
        if value is None:
            text = "you haven't been quizzed on this topic yet"
        else:
            text = f"your accuracy on this topic is {'only ' if raises else ''}{value:.0%}"
    elif feature == "Q" and value is not None:
        text = f"your recent quiz results overall are {'below' if raises else 'above'} average ({value:.0%} correct)"
    elif feature == "T" and value is not None:
        text = f"you take {'longer' if raises else 'less time'} than average per question here (about {value / 1000:.0f} seconds)"
    elif feature == "R" and value is not None:
        text = f"it has been {value:.0f} days since you last studied it" if raises else f"you studied it recently ({value:.0f} days ago)"
    elif feature == "F" and value is not None:
        text = f"your flashcard mastery of this topic is {'low' if raises else 'high'} ({value:.0%})"
    else:
        return None
    return f"{'🔺' if raises else '🟢'} {text[0].upper()}{text[1:]}"


def reasons_in_words(rec: dict) -> list:
    """
    Plain-English bullets for why a topic was recommended. With a model score they come from its SHAP
    reasons; with the starter (cold-start) formula they come from the formula's own inputs.
    """
    if rec.get("reasons"):
        return [b for b in (_sentence(r) for r in rec["reasons"]) if b]

    bullets = []
    accuracy, days = rec.get("accuracy"), rec.get("days_since_study")
    if accuracy is None:
        bullets.append("🔺 You haven't been quizzed on this topic yet")
    else:
        bullets.append(f"{'🔺' if accuracy < 0.6 else '🟢'} Your accuracy on this topic is {accuracy:.0%}")

    if days is None:
        bullets.append("🔺 You haven't studied this topic yet")
    elif days >= 1:
        bullets.append(f"{'🔺' if days >= 3 else '🟢'} It has been {days:.0f} days since you last studied it")
    else:
        bullets.append("🟢 You studied this topic today")
    return bullets


def show_recommendation_card(rec: dict) -> None:
    """Topic title, priority badge, deficit-risk gauge and the 'why' bullets for one recommendation."""
    color = PRIORITY_COLORS.get(rec["priority"], "#6B7280")
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap;'>"
        f"<span style='font-size:1.35em; font-weight:700;'>{html.escape(rec['name'])}</span>"
        f"<span style='background:{color}; color:white; padding:3px 12px; border-radius:14px; "
        f"font-size:0.8em; font-weight:600;'>{PRIORITY_LABELS.get(rec['priority'], rec['priority'])}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.progress(min(max(rec["deficit"], 0.0), 1.0), text=f"Deficit risk: {rec['deficit']:.0%}")

    st.markdown("**Why this is recommended**")
    for bullet in reasons_in_words(rec):
        st.markdown(f"- {bullet}")

    if rec["model_version"].startswith("ml_"):
        st.caption("Predicted by the learning model from your study history.")
    else:
        st.caption("Starter estimate. The learning model takes over after your 3rd quiz.")
