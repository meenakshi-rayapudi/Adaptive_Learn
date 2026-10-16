import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

from core.engine import TutorAgent
from ui.components.recommendation_card import reasons_in_words


def reason(feature, value, direction, shap=0.5):
    return {"feature": feature, "value": value, "direction": direction, "shap": shap}


class TestPlainEnglishReasons(unittest.TestCase):
    def words(self, *reasons):
        return reasons_in_words({"reasons": list(reasons)})

    def test_low_accuracy_is_described_with_the_real_number(self):
        self.assertEqual(self.words(reason("A", 0.3, "raises")), ["🔺 Your accuracy on this topic is only 30%"])

    def test_good_accuracy_is_described_as_a_strength(self):
        self.assertEqual(self.words(reason("A", 0.9, "lowers")), ["🟢 Your accuracy on this topic is 90%"])

    def test_recent_results_and_response_time(self):
        bullets = self.words(reason("Q", 0.25, "raises"), reason("T", 14500, "raises"), reason("T", 3000, "lowers"))
        self.assertEqual(bullets, [
            "🔺 Your recent quiz results overall are below average (25% correct)",
            "🔺 You take longer than average per question here (about 14 seconds)",
            "🟢 You take less time than average per question here (about 3 seconds)",
        ])

    def test_never_quizzed_is_said_plainly(self):
        self.assertEqual(self.words(reason("A", None, "raises")), ["🔺 You haven't been quizzed on this topic yet"])

    def test_missing_values_and_unknown_features_are_skipped(self):
        self.assertEqual(self.words(reason("Q", None, "raises"), reason("T", None, "lowers"), reason("X", 1, "raises")), [])

    def test_recency_and_flashcards_are_understood_too(self):
        bullets = self.words(reason("R", 7, "raises"), reason("F", 0.1, "raises"))
        self.assertEqual(bullets, ["🔺 It has been 7 days since you last studied it",
                                   "🔺 Your flashcard mastery of this topic is low (10%)"])

    def test_starter_formula_explains_itself_from_its_own_inputs(self):
        rec = {"reasons": None, "accuracy": 0.3, "days_since_study": 7}
        self.assertEqual(reasons_in_words(rec), ["🔺 Your accuracy on this topic is 30%",
                                                 "🔺 It has been 7 days since you last studied it"])

    def test_starter_formula_with_no_history(self):
        rec = {"reasons": None, "accuracy": None, "days_since_study": None}
        self.assertEqual(reasons_in_words(rec), ["🔺 You haven't been quizzed on this topic yet",
                                                 "🔺 You haven't studied this topic yet"])

    def test_starter_formula_with_a_fresh_strong_topic(self):
        rec = {"reasons": None, "accuracy": 0.9, "days_since_study": 0.2}
        self.assertEqual(reasons_in_words(rec), ["🟢 Your accuracy on this topic is 90%", "🟢 You studied this topic today"])


class TestFlashcardDrillMethod(unittest.TestCase):
    def make_agent(self):
        agent = object.__new__(TutorAgent)   # skip __init__: no LLM needed for these tests
        agent.vector_db, agent.full_text, agent.student_id, agent.document_id = None, "doc text", "stu", "D1"
        agent.artifacts = types.SimpleNamespace(flashcards=None, quiz=None)
        return agent

    def run_method(self, topic=None, cards=None, **kwargs):
        topic = topic if topic is not None else {"id": "D1_T3", "key": "T3", "name": "Paging", "description": "page tables"}
        agent = self.make_agent()
        with mock.patch("core.engine.learner_service.get_topic", return_value=topic), \
             mock.patch("core.engine.search_topic", return_value="topic text"), \
             mock.patch("core.engine.generate_flashcards", return_value=cards) as generate:
            message = agent.make_targeted_flashcards("D1_T3", **kwargs)
        return agent, message, generate

    def test_builds_a_deck_tagged_with_the_topic(self):
        agent, message, generate = self.run_method(cards=[{"front": "f", "back": "b"}] * 2)
        self.assertEqual(len(agent.artifacts.flashcards), 2)
        self.assertIn("Successfully created 2 flashcards", message)
        generate.assert_called_once_with("topic text", topic="Paging", num_cards=8, topic_id="D1_T3")

    def test_an_empty_generation_leaves_the_deck_alone(self):
        agent, message, _ = self.run_method(cards=[])
        self.assertIsNone(agent.artifacts.flashcards)
        self.assertIn("Could not generate", message)

    def test_an_unknown_topic_is_reported(self):
        agent = self.make_agent()
        with mock.patch("core.engine.learner_service.get_topic", return_value=None):
            message = agent.make_targeted_flashcards("nope")
        self.assertIn("not found", message)
        self.assertIsNone(agent.artifacts.flashcards)

    def test_a_blank_topic_uses_the_weakest_one(self):
        agent = self.make_agent()
        topic = {"id": "D1_T2", "key": "T2", "name": "Deadlocks", "description": ""}
        with mock.patch("core.engine.learner_service.get_weak_topics", return_value=[{"topic_id": "D1_T2"}]) as weak, \
             mock.patch("core.engine.learner_service.get_topic", return_value=topic) as get_topic, \
             mock.patch("core.engine.search_topic", return_value="text"), \
             mock.patch("core.engine.generate_flashcards", return_value=[{"front": "f", "back": "b"}]):
            agent.make_targeted_flashcards("")
        weak.assert_called_once()
        get_topic.assert_called_once_with("D1_T2", "D1")


class FakeAgent:
    """Stands in for the tutor so the buttons can be tested without an LLM."""
    student_id = "stu"

    def __init__(self, works=True):
        self.works = works
        self.artifacts = types.SimpleNamespace(quiz=None, flashcards=None)

    def make_targeted_quiz(self, topic_id, difficulty):
        if self.works:
            self.artifacts.quiz = [{"question": "q", "type": "mcq", "options": ["a"], "answer": "a", "topic_id": topic_id}]
            return "ok"
        return "Could not generate a quiz"

    def make_targeted_flashcards(self, topic_id, num_cards=8):
        if self.works:
            self.artifacts.flashcards = [{"front": "f", "back": "b", "status": "new"}] * 3
            return "ok"
        return "Could not generate flashcards"


RECOMMENDATIONS = [
    {"topic_id": "D1_T2", "name": "Deadlocks", "deficit": 0.72, "priority": "high", "accuracy": 0.3,
     "days_since_study": 7, "model_version": "ml_lightgbm",
     "reasons": [reason("A", 0.3, "raises"), reason("Q", 0.4, "raises"), reason("T", 4000, "lowers")]},
    {"topic_id": "D1_T1", "name": "Paging", "deficit": 0.2, "priority": "low", "accuracy": 0.9,
     "days_since_study": 1, "model_version": "cold_start_v1", "reasons": None},
]


class TestCardAndButtons(unittest.TestCase):
    def start_app(self, works=True):
        saved = []
        for target, kwargs in [("ui.components.sidebar.list_students",
                                {"return_value": [{"id": "stu", "name": "Stu", "archetype": None}]}),
                               ("ui.components.session_timer.get_study_minutes", {"return_value": 0}),
                               ("ui.components.drill.get_weak_topics", {"return_value": RECOMMENDATIONS}),
                               ("ui.components.drill.save_recommendation",
                                {"side_effect": lambda *args, **kw: saved.append((args, kw))})]:
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.agent = FakeAgent(works)
        at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "ui", "app.py"), default_timeout=90)
        at.session_state["student_id"] = "stu"
        at.session_state["document_id"] = "D1"
        at.session_state["agent_executor"] = self.agent
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        return at, saved

    def button(self, at, label):
        return next(b for b in at.button if b.label == label)

    def test_the_card_shows_title_badge_risk_and_why(self):
        at, _ = self.start_app()
        text = " ".join(m.value for m in list(at.markdown) + list(at.caption))
        for expected in ("Deadlocks", "High priority", "Why this is recommended",
                         "Your accuracy on this topic is only 30%", "Predicted by the learning model"):
            self.assertIn(expected, text)
        self.assertIn("Deficit risk: 72%", str([p.proto.text for p in at.get("progress")]))

    def test_both_action_buttons_are_there(self):
        at, _ = self.start_app()
        labels = [b.label for b in at.button]
        self.assertIn("📝 Take Targeted Quiz", labels)
        self.assertIn("🗂️ Launch Flashcard Drill", labels)

    def test_take_targeted_quiz_loads_the_quiz_and_saves_the_recommendation(self):
        at, saved = self.start_app()
        self.button(at, "📝 Take Targeted Quiz").click().run()
        self.assertEqual(at.session_state["quiz"][0]["topic_id"], "D1_T2")
        self.assertFalse(at.session_state["quiz_submitted"])
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0][1]["topic_id"], "D1_T2")
        self.assertTrue(saved[0][1]["followed"])

    def test_launch_flashcard_drill_loads_the_deck_and_points_to_the_tab(self):
        at, saved = self.start_app()
        self.button(at, "🗂️ Launch Flashcard Drill").click().run()
        self.assertEqual(len(at.session_state["flashcards"]), 3)
        self.assertEqual(at.session_state["card_study_idx"], 0)
        self.assertIn("Knowledge Flashcards", " ".join(s.value for s in at.success))
        self.assertTrue(saved[0][1]["followed"])

    def test_a_failed_drill_shows_the_reason_instead_of_crashing(self):
        at, _ = self.start_app(works=False)
        self.button(at, "🗂️ Launch Flashcard Drill").click().run()
        self.assertIn("Could not generate flashcards", " ".join(e.value for e in at.error))
        self.button(at, "📝 Take Targeted Quiz").click().run()
        self.assertIn("Could not generate a quiz", " ".join(e.value for e in at.error))
        self.assertEqual([e.value for e in at.exception], [])


if __name__ == "__main__":
    unittest.main()
