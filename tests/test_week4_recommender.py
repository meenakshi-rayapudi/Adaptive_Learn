import datetime as dt
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import Base, Student, Document, Topic, QuizAttempt, QuizQuestion, Recommendation
from engine.features import compute_feature_vector
from engine.predictor import MODEL_PATH
from engine.recommender import COLD_START_VERSION, recommend_topics, save_recommendation


class RecommenderTestCase(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        s = self.session
        s.add_all([Student(id=n, name=n) for n in ("newbie", "two", "three")])
        s.add(Document(id="D", filename="d.pdf"))
        s.add_all([Topic(id=f"D_T{i}", document_id="D", topic_key=f"T{i}", name=f"Topic {i}") for i in (1, 2, 3)])
        s.commit()
        self.add_attempts("two", 2)
        self.add_attempts("three", 3)

    def tearDown(self):
        self.session.close()

    def add_attempts(self, student, count):
        now = dt.datetime.utcnow()
        for i in range(count):
            attempt = QuizAttempt(student_id=student, document_id="D", topic_id="D_T1",
                                  started_at=now - dt.timedelta(days=count - i))
            self.session.add(attempt)
            self.session.flush()
            self.session.add(QuizQuestion(quiz_attempt_id=attempt.id, topic_id="D_T1", question_text="q",
                                          correct_answer="a", is_correct=(i % 2 == 0), latency_ms=5000))
        self.session.commit()

    def recommend(self, student, **kwargs):
        return recommend_topics(student, "D", session=self.session, **kwargs)


class TestHandoff(RecommenderTestCase):
    def test_fewer_than_three_attempts_use_the_cold_start_formula(self):
        for student in ("newbie", "two"):
            recs = self.recommend(student)
            self.assertEqual({r["model_version"] for r in recs}, {COLD_START_VERSION})
            self.assertTrue(all(r["reasons"] is None for r in recs))

    @unittest.skipUnless(os.path.exists(MODEL_PATH), "run scripts/model_tournament.py first")
    def test_three_attempts_hand_over_to_the_model(self):
        recs = self.recommend("three")
        self.assertTrue(all(r["model_version"].startswith("ml_") for r in recs))
        self.assertTrue(all(len(r["reasons"]) == 3 for r in recs))
        self.assertTrue(all(0 <= r["deficit"] <= 1 for r in recs))

    def test_results_are_ranked_and_labelled_by_priority(self):
        recs = self.recommend("two")
        self.assertEqual([r["deficit"] for r in recs], sorted((r["deficit"] for r in recs), reverse=True))
        for r in recs:
            expected = "high" if r["deficit"] >= 0.7 else "medium" if r["deficit"] >= 0.4 else "low"
            self.assertEqual(r["priority"], expected)

    def test_limit_and_unknown_document(self):
        self.assertEqual(len(self.recommend("two", limit=1)), 1)
        self.assertEqual(recommend_topics("two", "nope", session=self.session), [])


class TestFallback(RecommenderTestCase):
    def test_a_failing_model_falls_back_to_cold_start_instead_of_crashing(self):
        with mock.patch("engine.predictor.predict_from_vector", side_effect=RuntimeError("model broke")):
            recs = self.recommend("three")
        self.assertEqual(len(recs), 3)
        self.assertEqual({r["model_version"] for r in recs}, {COLD_START_VERSION})

    def test_missing_ml_libraries_fall_back_to_cold_start(self):
        with mock.patch.dict(sys.modules, {"engine.predictor": None}):   # makes "import engine.predictor" fail
            recs = self.recommend("three")
        self.assertEqual({r["model_version"] for r in recs}, {COLD_START_VERSION})

    def test_a_missing_model_file_falls_back_to_cold_start(self):
        with mock.patch("engine.predictor.MODEL_PATH", os.path.join("no", "such", "model.pkl")), \
             mock.patch("engine.predictor._package", None):
            recs = self.recommend("three")
        self.assertEqual({r["model_version"] for r in recs}, {COLD_START_VERSION})

    def test_a_broken_model_is_tried_once_not_once_per_topic(self):
        with mock.patch("engine.predictor.predict_from_vector", side_effect=RuntimeError("boom")) as broken:
            self.recommend("three")
        self.assertEqual(broken.call_count, 1)


class TestSavingRecommendations(RecommenderTestCase):
    def sample(self, reasons):
        return {"topic_id": "D_T2", "deficit": 0.62, "priority": "medium", "model_version": "ml_lightgbm",
                "reasons": reasons}

    def test_saves_model_version_and_shap_reasons_as_json(self):
        reasons = [{"feature": "A", "value": 0.33, "shap": 0.41, "direction": "raises"}]
        rec_id = save_recommendation("three", self.sample(reasons), followed=True, session=self.session)
        row = self.session.get(Recommendation, rec_id)
        self.assertEqual((row.topic_id, row.model_version, row.priority, row.is_followed),
                         ("D_T2", "ml_lightgbm", "medium", True))
        self.assertAlmostEqual(row.deficit_score, 0.62)
        self.assertEqual(json.loads(row.shap_explanation), reasons)

    def test_cold_start_recommendations_have_no_shap_column(self):
        rec_id = save_recommendation("two", self.sample(None), session=self.session)
        row = self.session.get(Recommendation, rec_id)
        self.assertIsNone(row.shap_explanation)
        self.assertIsNone(row.is_followed)

    def test_only_followed_recommendations_count_toward_feature_h(self):
        save_recommendation("two", self.sample(None), followed=True, session=self.session)
        save_recommendation("two", self.sample(None), followed=None, session=self.session)
        self.assertEqual(compute_feature_vector("two", "D_T2", session=self.session).h, 1)

    def test_saving_never_raises(self):
        self.assertIsNone(save_recommendation(None, self.sample(None), session=self.session))
        self.assertIsNone(save_recommendation("two", {"topic_id": "D_T2"}, session=self.session))  # missing fields


if __name__ == "__main__":
    unittest.main()
