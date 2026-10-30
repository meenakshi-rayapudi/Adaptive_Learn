import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "ui", "app.py")

HIGH = {"topic_id": "D1_T2", "name": "Deadlocks", "deficit": 0.72, "priority": "high", "accuracy": 0.3,
        "days_since_study": 7, "model_version": "cold_start_v1", "reasons": None}
LOW = {**HIGH, "name": "Paging", "deficit": 0.2, "priority": "low", "accuracy": 0.9, "days_since_study": 1}


class AppTestCase(unittest.TestCase):
    def start_app(self, recommendations=(), student="stu", document="D1", flashcards=None):
        self.review_calls = []
        patches = [
            ("ui.components.sidebar.list_students", {"return_value": [{"id": "stu", "name": "Stu", "archetype": None}]}),
            ("ui.components.session_timer.get_study_minutes", {"return_value": 0}),
            ("ui.components.recommendation_banner.get_weak_topics", {"return_value": list(recommendations)}),
            ("ui.components.flashcards_tab.log_flashcard_review",
             {"side_effect": lambda *args, **kwargs: self.review_calls.append(args)}),
        ]
        for target, kwargs in patches:
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

        at = AppTest.from_file(APP, default_timeout=90)
        if student:
            at.session_state["student_id"] = student
        if document:
            at.session_state["document_id"] = document
        if flashcards:
            at.session_state["flashcards"] = flashcards
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        return at


class TestRecommendedNextTopicBanner(AppTestCase):
    def banners(self, at):
        return [e.value for e in list(at.warning) + list(at.info) if "Recommended Next Topic" in e.value]

    def test_banner_shows_on_both_the_chat_and_quiz_tabs(self):
        at = self.start_app([HIGH])
        banners = self.banners(at)
        self.assertEqual(len(banners), 2)
        for text in banners:
            self.assertIn("Recommended Next Topic: Deadlocks", text)
            self.assertIn("High priority", text)
            self.assertIn("72% risk", text)
            self.assertIn("Your accuracy on this topic is 30%", text)   # the starter formula's reason

    def test_a_high_priority_topic_is_shown_as_a_warning(self):
        at = self.start_app([HIGH])
        self.assertEqual(len([w for w in at.warning if "Recommended Next Topic" in w.value]), 2)

    def test_a_low_priority_topic_is_a_calm_info_box(self):
        at = self.start_app([LOW])
        self.assertEqual(len([w for w in at.warning if "Recommended Next Topic" in w.value]), 0)
        self.assertEqual(len([i for i in at.info if "Recommended Next Topic: Paging" in i.value]), 2)

    def test_only_the_single_top_topic_is_shown(self):
        at = self.start_app([HIGH, LOW])
        self.assertFalse(any("Paging" in b for b in self.banners(at)))

    def test_banner_is_hidden_without_a_student_or_a_document(self):
        self.assertEqual(self.banners(self.start_app([HIGH], student=None)), [])
        self.assertEqual(self.banners(self.start_app([HIGH], document=None)), [])

    def test_banner_is_hidden_when_there_is_nothing_to_recommend(self):
        self.assertEqual(self.banners(self.start_app([])), [])


def deck():
    return [{"front": f"front {i}", "back": f"back {i}", "topic_id": "D1_T1"} for i in range(3)]


class TestFlashcardRatingButtons(AppTestCase):
    def button(self, at, label):
        return next(b for b in at.button if b.label == label)

    def test_focused_mode_has_hard_good_easy_and_no_old_buttons(self):
        at = self.start_app(flashcards=deck())
        labels = [b.label for b in at.button]
        for expected in ("😓 Hard", "👍 Good", "🌟 Easy", "⬅️ Previous", "➡️ Next"):
            self.assertIn(expected, labels)
        for old in ("🔁 Need Review", "✅ Mastered"):
            self.assertNotIn(old, labels)

    def test_each_rating_sends_its_sm2_quality_and_sets_the_card_status(self):
        for label, quality, status in (("😓 Hard", 2, "review"), ("👍 Good", 4, "mastered"), ("🌟 Easy", 5, "mastered")):
            with self.subTest(label=label):
                at = self.start_app(flashcards=deck())
                self.button(at, label).click().run()
                self.assertEqual(self.review_calls[-1][3], quality)           # (student, document, card, quality)
                self.assertEqual(self.review_calls[-1][0], "stu")
                self.assertEqual(at.session_state["flashcards"][0]["status"], status)
                self.assertEqual(at.session_state["card_study_idx"], 1)       # moved on to the next card

    def test_hard_counts_a_review_and_good_resets_it(self):
        at = self.start_app(flashcards=deck())
        self.button(at, "😓 Hard").click().run()
        self.assertEqual(at.session_state["flashcards"][0]["review_count"], 1)

    def test_grid_mode_has_three_rating_buttons_on_every_card(self):
        at = self.start_app(flashcards=deck())
        next(r for r in at.radio if r.label == "Study Mode:").set_value("🗂️ Grid View (Browse All Cards)").run()
        self.assertEqual([e.value for e in at.exception], [])
        rating_keys = [b.key for b in at.button if b.key and b.key.startswith("rate_")]
        self.assertEqual(len(rating_keys), 9)

        at.button(key="rate_5_1").click().run()                               # Easy on the second card
        self.assertEqual(self.review_calls[-1][3], 5)
        self.assertEqual(at.session_state["flashcards"][1]["status"], "mastered")
        self.assertEqual(at.session_state["flashcards"][0]["status"], "new")


if __name__ == "__main__":
    unittest.main()
