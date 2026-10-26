import unittest
import datetime as dt
from database.db import get_session, init_db
from database.schema import Student, Document, Topic, QuizAttempt, QuizQuestion, FlashcardEvent, EngagementEvent
from core import learner_service


class TestWeek5Analytics(unittest.TestCase):

    def setUp(self):
        init_db()
        self.session = get_session()
        self.student_id = "TEST_STUDENT_W5"
        self.document_id = "TEST_DOC_W5"
        self.topic_id = "TEST_DOC_W5_T1"

        # Cleanup test entities
        self._cleanup()

        # Create basic student and topic
        student = Student(id=self.student_id, name="Test W5 Learner")
        doc = Document(id=self.document_id, filename="test.pdf", source_type="pdf")
        topic = Topic(id=self.topic_id, document_id=self.document_id, topic_key="T1", name="Thermodynamics", description="Heat and energy")
        self.session.add_all([student, doc, topic])
        self.session.commit()

    def tearDown(self):
        self._cleanup()
        self.session.close()

    def _cleanup(self):
        self.session.query(QuizQuestion).filter(QuizQuestion.topic_id == self.topic_id).delete()
        self.session.query(QuizAttempt).filter(QuizAttempt.student_id == self.student_id).delete()
        self.session.query(FlashcardEvent).filter(FlashcardEvent.student_id == self.student_id).delete()
        self.session.query(EngagementEvent).filter(EngagementEvent.student_id == self.student_id).delete()
        self.session.query(Topic).filter(Topic.id == self.topic_id).delete()
        self.session.query(Document).filter(Document.id == self.document_id).delete()
        self.session.query(Student).filter(Student.id == self.student_id).delete()
        self.session.commit()

    def test_accuracy_trends_by_topic(self):
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        attempt = QuizAttempt(student_id=self.student_id, document_id=self.document_id, score=2, total_questions=3, accuracy=0.67, started_at=now)
        self.session.add(attempt)
        self.session.commit()

        q1 = QuizQuestion(quiz_attempt_id=attempt.id, topic_id=self.topic_id, question_text="Q1", correct_answer="A", user_answer="A", is_correct=True, created_at=now)
        q2 = QuizQuestion(quiz_attempt_id=attempt.id, topic_id=self.topic_id, question_text="Q2", correct_answer="B", user_answer="C", is_correct=False, created_at=now)
        self.session.add_all([q1, q2])
        self.session.commit()

        trends = learner_service.get_accuracy_trends_by_topic(self.student_id, document_id=self.document_id, session=self.session)
        self.assertEqual(len(trends), 2)
        self.assertEqual(trends[0]["topic_id"], self.topic_id)
        self.assertEqual(trends[0]["topic_name"], "Thermodynamics")
        self.assertTrue(trends[0]["is_correct"])
        self.assertFalse(trends[1]["is_correct"])

    def test_flashcard_retention_breakdown(self):
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        c1 = FlashcardEvent(student_id=self.student_id, document_id=self.document_id, topic_id=self.topic_id, card_front="Card 1", card_back="Ans 1", quality=4, repetition_count=0, interval_days=0, reviewed_at=now)
        c2 = FlashcardEvent(student_id=self.student_id, document_id=self.document_id, topic_id=self.topic_id, card_front="Card 2", card_back="Ans 2", quality=4, repetition_count=1, interval_days=3, reviewed_at=now)
        c3 = FlashcardEvent(student_id=self.student_id, document_id=self.document_id, topic_id=self.topic_id, card_front="Card 3", card_back="Ans 3", quality=5, repetition_count=4, interval_days=15, reviewed_at=now)
        self.session.add_all([c1, c2, c3])
        self.session.commit()

        breakdown = learner_service.get_flashcard_retention_breakdown(self.student_id, document_id=self.document_id, session=self.session)
        self.assertEqual(breakdown["total_cards"], 3)
        self.assertEqual(breakdown["counts"]["new"], 1)
        self.assertEqual(breakdown["counts"]["review"], 1)
        self.assertEqual(breakdown["counts"]["mastered"], 1)

    def test_study_time_distribution(self):
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        e1 = EngagementEvent(student_id=self.student_id, document_id=self.document_id, topic_id=self.topic_id, event_type="chat_question", created_at=now - dt.timedelta(minutes=5))
        e2 = EngagementEvent(student_id=self.student_id, document_id=self.document_id, topic_id=self.topic_id, event_type="chat_question", created_at=now)
        self.session.add_all([e1, e2])
        self.session.commit()

        dist = learner_service.get_study_time_distribution(self.student_id, days=7, now=now, session=self.session)
        self.assertGreater(dist["total_minutes"], 0)
        self.assertIn(now.strftime("%Y-%m-%d"), dist["daily_minutes"])


if __name__ == "__main__":
    unittest.main()
