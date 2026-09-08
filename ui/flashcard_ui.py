import streamlit as st
import streamlit.components.v1 as components
import os
from datetime import datetime, timedelta
from utils.audio_utils import generate_audio


# ─────────────────────────────────────────────
#  SPACED REPETITION HELPERS
# ─────────────────────────────────────────────

def _now_str() -> str:
    return datetime.utcnow().isoformat()


def _mark_seen(card: dict) -> None:
    """Stamp the card with the current time."""
    card["last_seen"] = _now_str()


def _sr_sort_key(card: dict):
    """
    Spaced Repetition sorting priority:
      0 – review cards (needs practice)      ← ALWAYS AT TOP
      1 – new unreviewed cards                ← SECOND
      2 – mastered cards                      ← SINK TO BOTTOM
    """
    status = card.get("status", "new")
    review_count = card.get("review_count", 0)

    if status == "review":
        return (0, -review_count)   # Most reviewed cards first among review cards
    if status == "new":
        return (1, 0)
    if status == "mastered":
        return (2, 0)
    return (3, 0)


def apply_spaced_repetition(cards: list) -> list:
    """
    Return a re-ordered copy of cards according to Spaced Repetition priority.
    'review' cards bubble to the top; 'mastered' cards sink to the bottom.
    """
    return sorted(cards, key=_sr_sort_key)


# ─────────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────────

def play_audio(text, key):
    temp_file = f"temp_card_{key}.mp3"
    try:
        audio_path = generate_audio(text, temp_file)
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/mp3")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        st.error(f"Audio Error: {e}")


# ─────────────────────────────────────────────
#  FLASHCARD HTML
# ─────────────────────────────────────────────

def get_flashcard_html(front, back, status, key, height=220):
    colors = {
        "new":      "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)",
        "mastered": "linear-gradient(135deg, #10B981 0%, #047857 100%)",
        "review":   "linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)",
    }
    bg_color = colors.get(status, colors["new"])

    badge_labels = {"new": "🆕 New Concept", "mastered": "✅ Mastered", "review": "🔁 Needs Review"}
    badge = badge_labels.get(status, "")

    html = f"""
    <style>
    .flashcard-{key} {{
        background-color: transparent;
        width: 100%;
        height: {height}px;
        perspective: 1000px;
        cursor: pointer;
        margin-bottom: 10px;
    }}
    .flashcard-inner-{key} {{
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
    }}
    .flashcard-{key}:hover .flashcard-inner-{key} {{
        transform: rotateY(180deg);
    }}
    .flashcard-front-{key}, .flashcard-back-{key} {{
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 14px;
        padding: 24px;
        backface-visibility: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-size: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        color: white;
        box-sizing: border-box;
    }}
    .flashcard-front-{key} {{
        background: {bg_color};
    }}
    .flashcard-back-{key} {{
        background: linear-gradient(135deg, #475569 0%, #1E293B 100%);
        transform: rotateY(180deg);
        font-size: 14px;
        font-weight: 400;
        line-height: 1.5;
        overflow-y: auto;
    }}
    .badge-{key} {{
        font-size: 12px;
        font-weight: 700;
        opacity: 0.9;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
        background: rgba(255,255,255,0.2);
        padding: 4px 10px;
        border-radius: 20px;
    }}
    .hint-{key} {{
        font-size: 11px;
        opacity: 0.75;
        margin-top: 10px;
        font-style: italic;
    }}
    </style>
    <div class="flashcard-{key}">
        <div class="flashcard-inner-{key}">
            <div class="flashcard-front-{key}">
                <span class="badge-{key}">{badge}</span>
                <div>{front}</div>
                <div class="hint-{key}">Hover / Tap to flip</div>
            </div>
            <div class="flashcard-back-{key}">
                <div style="font-weight: 700; margin-bottom: 8px; font-size: 12px; color: #94A3B8; text-transform: uppercase;">Explanation</div>
                <div>{back}</div>
            </div>
        </div>
    </div>
    """
    return html


# ─────────────────────────────────────────────
#  MAIN DISPLAY FUNCTION
# ─────────────────────────────────────────────

def display_flashcards(cards, on_quiz_drill=None):
    if not cards:
        st.info("No flashcards generated.")
        return

    # ── Initialise card fields ──────────────────
    for card in cards:
        card.setdefault("status",    "new")
        card.setdefault("last_seen", None)
        card.setdefault("review_count", 0)

    # ── Mastery Stats Banner ───────────────────
    total    = len(cards)
    mastered = sum(1 for c in cards if c.get("status") == "mastered")
    review   = sum(1 for c in cards if c.get("status") == "review")
    new      = sum(1 for c in cards if c.get("status") == "new")

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Total Cards", total)
    p_col2.metric("New 🆕", new)
    p_col3.metric("Mastered ✅", mastered)
    p_col4.metric("Need Review 🔁", review)

    if total > 0:
        progress = mastered / total
        st.progress(progress, text=f"Deck Mastery: {int(progress * 100)}% ({mastered}/{total} cards mastered)")

    # ── Targeted AI Drill Banner ────────────────
    if review > 0 and on_quiz_drill is not None:
        st.write("")
        st.warning(f"🔁 **{review} concept(s) queued for review!** Spaced Repetition has bubbled them to the top of your deck.")
        if st.button(f"🎯 AI Practice Drill: Quiz Me on {review} Review Concept(s)", type="primary", use_container_width=True, key="drill_review_cards_btn"):
            review_cards = [c for c in cards if c.get("status") == "review"]
            on_quiz_drill(review_cards)

    st.divider()

    # ── Pedagogical Explanation & Mode Switcher ─
    mode_col1, mode_col2 = st.columns([1, 1])
    with mode_col1:
        view_mode = st.radio(
            "Study Mode:",
            ["🎴 Focused Study Mode (1-by-1 Active Recall)", "🗂️ Grid View (Browse All Cards)"],
            horizontal=True,
            index=0
        )
    with mode_col2:
        filter_status = st.selectbox(
            "Filter Deck:",
            ["All Cards", "Need Review Only (Difficult)", "New Only", "Mastered Only"],
            index=0
        )

    # ── Filter Cards ───────────────────────────
    status_map = {
        "Need Review Only (Difficult)": "review",
        "New Only": "new",
        "Mastered Only": "mastered"
    }

    for i, card in enumerate(cards):
        card["original_index"] = i

    if filter_status == "All Cards":
        active_deck = list(cards)
    else:
        target = status_map[filter_status]
        active_deck = [c for c in cards if c.get("status") == target]

    if not active_deck:
        st.info(f"No cards found matching filter: **{filter_status}**")
        return

    # Apply Spaced Repetition priority (review cards surfaced first)
    active_deck = apply_spaced_repetition(active_deck)

    # ─────────────────────────────────────────────
    #  VIEW MODE 1: FOCUSED STUDY MODE (1-by-1)
    # ─────────────────────────────────────────────
    if "🎴" in view_mode:
        if "card_study_idx" not in st.session_state:
            st.session_state.card_study_idx = 0

        # Boundary check
        if st.session_state.card_study_idx >= len(active_deck):
            st.session_state.card_study_idx = 0

        curr_idx = st.session_state.card_study_idx
        current_card = active_deck[curr_idx]
        orig_i = current_card["original_index"]

        _mark_seen(current_card)

        st.caption(f"Card {curr_idx + 1} of {len(active_deck)} | Original #{orig_i + 1}")

        # Render Large Focus Card
        card_html = get_flashcard_html(
            current_card.get("front", "Concept"),
            current_card.get("back", "Definition"),
            current_card["status"],
            f"focus_{orig_i}",
            height=260
        )
        components.html(card_html, height=280)

        # Action Buttons
        act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1, 1])
        
        with act_col1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.card_study_idx = (curr_idx - 1) % len(active_deck)
                st.rerun()

        with act_col2:
            if st.button("🔁 Need Review", type="secondary", use_container_width=True):
                current_card["status"] = "review"
                current_card["review_count"] = current_card.get("review_count", 0) + 1
                _mark_seen(current_card)
                st.session_state.card_study_idx = (curr_idx + 1) % len(active_deck)
                st.rerun()

        with act_col3:
            if st.button("✅ Mastered", type="primary", use_container_width=True):
                current_card["status"] = "mastered"
                current_card["review_count"] = 0
                _mark_seen(current_card)
                st.session_state.card_study_idx = (curr_idx + 1) % len(active_deck)
                st.rerun()

        with act_col4:
            if st.button("➡️ Next", use_container_width=True):
                st.session_state.card_study_idx = (curr_idx + 1) % len(active_deck)
                st.rerun()

        # Audio button
        st.write("")
        if st.button("🔊 Narrate This Flashcard", key=f"focus_audio_{orig_i}"):
            play_audio(
                f"{current_card.get('front', '')}. Explanation: {current_card.get('back', '')}",
                f"focus_{orig_i}"
            )

    # ─────────────────────────────────────────────
    #  VIEW MODE 2: GRID VIEW (BROWSE ALL)
    # ─────────────────────────────────────────────
    else:
        cols = st.columns(3)
        for idx, card in enumerate(active_deck):
            col_idx = idx % 3
            orig_i  = card["original_index"]

            with cols[col_idx]:
                _mark_seen(card)

                card_html = get_flashcard_html(
                    card.get("front", "Concept"),
                    card.get("back",  "Definition"),
                    card["status"],
                    orig_i,
                )
                components.html(card_html, height=240)

                # Status buttons
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Mastered", key=f"master_{orig_i}"):
                        card["status"]       = "mastered"
                        card["review_count"] = 0
                        _mark_seen(card)
                        st.rerun()
                with btn_col2:
                    if st.button("🔁 Review", key=f"review_{orig_i}"):
                        card["status"]        = "review"
                        card["review_count"]  = card.get("review_count", 0) + 1
                        _mark_seen(card)
                        st.rerun()

                # Audio
                if st.button("🔊 Audio", key=f"audio_{orig_i}"):
                    play_audio(
                        f"{card.get('front', '')}. {card.get('back', '')}",
                        orig_i,
                    )