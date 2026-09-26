import streamlit as st

# Fixed palette cycled by topic index so colors stay stable across reruns
# (topic order doesn't change once extracted).
BADGE_COLORS = [
    "#2563EB",  # blue
    "#059669",  # green
    "#D97706",  # amber
    "#DB2777",  # pink
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DC2626",  # red
    "#4D7C0F",  # olive
]


def render_topic_badges(topics: list, topic_chunk_counts: dict) -> None:
    """Renders color-coded pills for each extracted topic with its chunk count."""
    if not topics:
        return

    st.markdown("**🗂️ Detected Topics**")

    badges_html = "<div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;'>"
    for i, topic in enumerate(topics):
        color = BADGE_COLORS[i % len(BADGE_COLORS)]
        count = topic_chunk_counts.get(topic["topic_id"], 0)
        title = topic.get("description", "").replace('"', "'")
        badges_html += (
            f"<span title=\"{title}\" style='"
            f"background:{color}; color:white; padding:4px 10px; border-radius:14px; "
            f"font-size:12px; font-weight:600; white-space:nowrap;'>"
            f"{topic['name']} · {count}</span>"
        )
    badges_html += "</div>"

    st.markdown(badges_html, unsafe_allow_html=True)
