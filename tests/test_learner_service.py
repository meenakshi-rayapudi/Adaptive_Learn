import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import learner_service as svc
from database.schema import Base, Student, Document, Topic, QuizAttempt, QuizQuestion
from tools import quiz_tool
from tools.retrieval_tool import search_topic

T0 = dt.datetime(2026, 3, 1, 10, 0, 0)


def minutes(n):
    return T0 + dt.timedelta(minutes=n)


class TestStudyTime(unittest.TestCase):
    def test_no_activity_is_zero(self):
        self.assertEqual(svc.minutes_from_timestamps([]), 0.0)

    def test_single_click_gets_the_tail_credit(self):
        self.assertEqual(svc.minutes_from_timestamps([minutes(0)]), svc.SESSION_TAIL_MINUTES)

    def test_close_events_form_one_session(self):
        # 0 -> 10 min is one session: 10 minutes + tail
        self.assertEqual(svc.minutes_from_timestamps([minutes(0), minutes(4), minutes(10)]), 10 + svc.SESSION_TAIL_MINUTES)

    def test_a_long_gap_starts_a_new_session(self):
        total = svc.minutes_from_timestamps([minutes(0), minutes(5), minutes(120)])
        self.assertEqual(total, (5 + svc.SESSION_TAIL_MINUTES) + svc.SESSION_TAIL_MINUTES)

    def test_order_does_not_matter(self):
        self.assertEqual(svc.minutes_from_timestamps([minutes(10), minutes(0)]),
                         svc.minutes_from_timestamps([minutes(0), minutes(10)]))


class TestWithDatabase(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        s = self.session
        s.add_all([
            Student(id="stu", name="Stu"),
            Document(id="D1", filename="d.pdf"),
            Topic(id="D1_T1", document_id="D1", topic_key="T1", name="Paging", description="page tables"),
            Topic(id="D1_T2", document_id="D1", topic_key="T2", name="Deadlocks", description="banker's algorithm"),
            Topic(id="D1_T3", document_id="D1", topic_key="T3", name="Scheduling", description="round robin"),
        ])
        s.commit()
        # Recently aced Paging, recently failed Deadlocks, never touched Scheduling.
        self.add_attempt("D1_T1", [True, True, True, True], days_ago=1)
        self.add_attempt("D1_T2", [False, False, False, True], days_ago=1)

    def tearDown(self):
        self.session.close()

    def add_attempt(self, topic_id, outcomes, days_ago):
        when = dt.datetime.utcnow() - dt.timedelta(days=days_ago)
        attempt = QuizAttempt(student_id="stu", document_id="D1", topic_id=topic_id, started_at=when)
        self.session.add(attempt)
        self.session.flush()
        for ok in outcomes:
            self.session.add(QuizQuestion(quiz_attempt_id=attempt.id, topic_id=topic_id, question_text="q",
                                          correct_answer="a", is_correct=ok, latency_ms=5000))
        self.session.commit()

    def test_weak_topics_are_ranked_weakest_first(self):
        ranked = svc.get_weak_topics("stu", "D1", limit=3, session=self.session)
        self.assertEqual([r["name"] for r in ranked], ["Scheduling", "Deadlocks", "Paging"])
        self.assertGreater(ranked[0]["deficit"], ranked[1]["deficit"])
        self.assertGreater(ranked[1]["deficit"], ranked[2]["deficit"])

    def test_limit_is_respected(self):
        self.assertEqual(len(svc.get_weak_topics("stu", "D1", limit=1, session=self.session)), 1)

    def test_unknown_document_has_no_weak_topics(self):
        self.assertEqual(svc.get_weak_topics("stu", "nope", session=self.session), [])

    def test_new_student_gets_every_topic_back(self):
        self.assertEqual(len(svc.get_weak_topics("brand_new", "D1", session=self.session)), 3)

    def test_get_topic_by_full_id_or_short_key(self):
        self.assertEqual(svc.get_topic("D1_T2", session=self.session)["name"], "Deadlocks")
        self.assertEqual(svc.get_topic("T2", document_id="D1", session=self.session)["name"], "Deadlocks")
        self.assertIsNone(svc.get_topic("T2", session=self.session))
        self.assertEqual(svc.get_topic("T2", "D1", session=self.session)["key"], "T2")

    def test_logged_activity_turns_into_study_minutes(self):
        now = minutes(300)
        svc.log_activity("stu", "chat_question", "D1", at=minutes(290), session=self.session)
        svc.log_activity("stu", "summary_read", "D1", at=minutes(296), session=self.session)
        svc.log_activity("stu", "chat_question", "D1", at=minutes(100), session=self.session)
        # Two sessions inside the window: (290 -> 296) and the lone click at 100.
        expected = (6 + svc.SESSION_TAIL_MINUTES) + svc.SESSION_TAIL_MINUTES
        self.assertEqual(svc.get_study_minutes("stu", days=1, now=now, session=self.session), expected)

    def test_old_activity_is_outside_the_window(self):
        svc.log_activity("stu", "chat_question", at=T0, session=self.session)
        self.assertEqual(svc.get_study_minutes("stu", days=1, now=T0 + dt.timedelta(days=5), session=self.session), 0.0)

    def test_logging_without_a_student_does_nothing(self):
        svc.log_activity(None, "chat_question", session=self.session)  # must not raise

    def drill_quiz(self):
        return [
            {"question": "q1", "type": "mcq", "options": ["a", "b"], "answer": "a", "topic_id": "D1_T3"},
            {"question": "q2", "type": "true_false", "options": ["True", "False"], "answer": "True", "topic_id": "D1_T3"},
            {"question": "q3", "type": "mcq", "options": ["a", "b"], "answer": "b", "topic_id": "D1_T3"},
        ]

    def test_saved_quiz_has_the_right_score_and_rows(self):
        attempt_id = svc.log_quiz_attempt("stu", "D1", self.drill_quiz(), {0: "a", 1: "False", 2: "b"}, session=self.session)
        attempt = self.session.get(QuizAttempt, attempt_id)
        self.assertEqual((attempt.score, attempt.total_questions), (2, 3))
        self.assertAlmostEqual(attempt.accuracy, 2 / 3)
        self.assertEqual(attempt.topic_id, "D1_T3")
        rows = self.session.query(QuizQuestion).filter_by(quiz_attempt_id=attempt_id).order_by(QuizQuestion.id).all()
        self.assertEqual([r.is_correct for r in rows], [True, False, True])
        self.assertEqual(rows[1].user_answer, "False")

    def test_a_saved_quiz_shows_up_in_weak_topics(self):
        before = {r["name"]: r for r in svc.get_weak_topics("stu", "D1", session=self.session)}
        self.assertIsNone(before["Scheduling"]["accuracy"])

        svc.log_quiz_attempt("stu", "D1", self.drill_quiz(), {0: "b", 1: "False", 2: "a"}, session=self.session)

        after = {r["name"]: r for r in svc.get_weak_topics("stu", "D1", session=self.session)}
        self.assertEqual(after["Scheduling"]["accuracy"], 0.0)

    def test_untagged_questions_get_the_closest_topic(self):
        import core.topic_extractor as te
        original = te.assign_topics_to_chunks
        te.assign_topics_to_chunks = lambda texts, topics: [{"chunk": t, "topic_id": "T2", "similarity": 0.9} for t in texts]
        try:
            quiz = [{"question": "What is the Banker's algorithm?", "type": "mcq", "options": ["a"], "answer": "a"}]
            attempt_id = svc.log_quiz_attempt("stu", "D1", quiz, {0: "a"},
                                              topics=[{"topic_id": "T2", "name": "Deadlocks", "description": ""}],
                                              session=self.session)
        finally:
            te.assign_topics_to_chunks = original
        row = self.session.query(QuizQuestion).filter_by(quiz_attempt_id=attempt_id).one()
        self.assertEqual(row.topic_id, "D1_T2")

    def test_nothing_is_saved_without_a_student_or_questions(self):
        self.assertIsNone(svc.log_quiz_attempt(None, "D1", self.drill_quiz(), {}, session=self.session))
        self.assertIsNone(svc.log_quiz_attempt("stu", "D1", [], {}, session=self.session))


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt

        class Reply:
            content = "[]"
        return Reply()


class TestTargetedQuizPieces(unittest.TestCase):
    def run_quiz(self, difficulty):
        fake = FakeLLM()
        original = quiz_tool.get_llm
        quiz_tool.get_llm = lambda: fake
        try:
            quiz_tool.generate_quiz("some study text", topic="Deadlocks", difficulty=difficulty)
        finally:
            quiz_tool.get_llm = original
        return fake.prompt

    def test_difficulty_reaches_the_prompt(self):
        self.assertIn("Difficulty: hard", self.run_quiz("hard"))
        self.assertIn("Difficulty: easy", self.run_quiz("EASY "))

    def test_unknown_difficulty_falls_back_to_medium(self):
        self.assertIn("Difficulty: medium", self.run_quiz("impossible"))

    def test_topic_search_filters_on_the_topic_key(self):
        class FakeStore:
            def similarity_search(self, query, k, filter):
                self.seen = filter

                class Doc:
                    page_content = "chunk text"
                return [Doc()]

        store = FakeStore()
        self.assertEqual(search_topic("T2", "deadlocks", store), "chunk text")
        self.assertEqual(store.seen, {"topic_id": "T2"})

    def test_topic_search_without_a_store_is_empty(self):
        self.assertEqual(search_topic("T2", "x", None), "")


if __name__ == "__main__":
    unittest.main()
