import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db import init_db, log_quiz_attempt, log_question_event, log_flashcard_review
from engine.cold_start import cold_start_score, recency_penalty
from engine.spaced_repetition import update_card_review
from models.schemas import FlashcardItem, QuizQuestionItem


class TestWeek2ColdStartAndSM2(unittest.TestCase):
    def test_cold_start_formula(self):
        score = cold_start_score(accuracy=0.4, days_since_last_review=10.0, flashcard_mastery=0.2)
        expected = 0.5 * (1 - 0.4) + 0.3 * recency_penalty(10.0) + 0.2 * (1 - 0.2)
        self.assertAlmostEqual(score, expected)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_spaced_repetition_update(self):
        ef, interval, reps = update_card_review(2.5, 0, 0, 5)
        self.assertGreater(ef, 2.0)
        self.assertGreaterEqual(interval, 1)
        self.assertGreaterEqual(reps, 1)

    def test_topic_tagging_is_supported_in_schemas(self):
        item = FlashcardItem(
            front="What is a queue?",
            back="A FIFO data structure.",
            topic_id="D0001_T1",
        )
        self.assertEqual(item.topic_id, "D0001_T1")

        question = QuizQuestionItem(
            question="A queue is FIFO.",
            type="true_false",
            options=["True", "False"],
            answer="True",
            explanation="A queue processes in order received.",
            topic_id="D0001_T1",
        )
        self.assertEqual(question.topic_id, "D0001_T1")

        legacy_item = FlashcardItem(front="What is a stack?", back="LIFO data structure.")
        self.assertIsNone(legacy_item.topic_id)

        legacy_question = QuizQuestionItem(
            question="A stack is FIFO.",
            type="true_false",
            options=["True", "False"],
            answer="False",
        )
        self.assertIsNone(legacy_question.topic_id)


class TestWeek2DatabaseLogging(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_quiz_and_flashcard_logging(self):
        student_id = "S_WEEK2_001"
        topic_id = "D0001_T1"

        attempt = log_quiz_attempt(student_id, topic_id, score=3, total_questions=5)
        self.assertEqual(attempt["student_id"], student_id)
        self.assertEqual(attempt["topic_id"], topic_id)
        self.assertAlmostEqual(attempt["accuracy"], 0.6)

        question = log_question_event(
            attempt["id"],
            topic_id,
            "What is FIFO?",
            "mcq",
            ["First In First Out", "Last In First Out", "First In Last Out", "Fast Input Fast Output"],
            "First In First Out",
            "First In First Out",
            True,
            1800,
        )
        self.assertEqual(question["topic_id"], topic_id)
        self.assertTrue(question["is_correct"])

        review = log_flashcard_review(
            student_id,
            topic_id,
            quality=4,
            ease_factor=2.7,
            interval_days=6,
            repetition_count=2,
        )
        self.assertEqual(review["student_id"], student_id)
        self.assertEqual(review["topic_id"], topic_id)
        self.assertEqual(review["quality"], 4)


if __name__ == "__main__":
    unittest.main()
