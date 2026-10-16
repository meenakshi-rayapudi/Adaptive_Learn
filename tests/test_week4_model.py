import datetime as dt
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import Base, Student, Document, Topic, QuizAttempt, QuizQuestion
from engine.dataset import SNAPSHOT_COLUMNS, load_benchmark_split
from engine.explainer import explain, _positive_class
from engine.predictor import MODEL_PATH, predict_topic_deficit


class TestExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"Q": rng.random(3000), "A": rng.random(3000), "T": rng.random(3000)})
        fails = (X["A"] + rng.normal(0, 0.1, 3000)) < 0.5      # failing is driven by low topic accuracy
        cls.model = LGBMClassifier(n_estimators=60, verbose=-1, random_state=0).fit(X, fails.astype(int))

    def row(self, q, a, t):
        return pd.DataFrame([{"Q": q, "A": a, "T": t}])

    def test_returns_the_top_three_biggest_effects_first(self):
        reasons = explain(self.model, self.row(0.5, 0.05, 0.5), top_k=3)
        self.assertEqual(len(reasons), 3)
        sizes = [abs(r["shap"]) for r in reasons]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual(reasons[0]["feature"], "A")

    def test_low_accuracy_raises_risk_and_high_accuracy_lowers_it(self):
        low = explain(self.model, self.row(0.5, 0.05, 0.5))[0]
        high = explain(self.model, self.row(0.5, 0.95, 0.5))[0]
        self.assertGreater(low["shap"], 0)
        self.assertLess(high["shap"], 0)

    def test_contributions_add_up_exactly_to_the_models_output(self):
        X = self.row(0.3, 0.2, 0.9)
        explainer = shap.TreeExplainer(self.model)
        values = _positive_class(explainer.shap_values(X))[0]
        base = np.ravel(explainer.expected_value)[-1]
        margin = self.model.predict(X, raw_score=True)[0]
        self.assertAlmostEqual(base + values.sum(), margin, places=4)

    def test_missing_values_are_explained_too(self):
        reasons = explain(self.model, self.row(np.nan, np.nan, np.nan))
        self.assertEqual(len(reasons), 3)


@unittest.skipUnless(os.path.exists(MODEL_PATH), "run scripts/model_tournament.py first to create the model")
class TestPredictTopicDeficit(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        s = self.session
        s.add_all([Student(id="weak", name="w"), Student(id="strong", name="s"), Student(id="new", name="n"),
                   Document(id="D", filename="d.pdf"),
                   Topic(id="D_T1", document_id="D", topic_key="T1", name="Deadlocks")])
        s.commit()
        now = dt.datetime.utcnow()
        for student, outcomes, latency in [("weak", [False, False, True, False, False], 30000),
                                           ("strong", [True, True, True, True, False], 4000)]:
            for day, ok in enumerate(outcomes):
                attempt = QuizAttempt(student_id=student, document_id="D", topic_id="D_T1",
                                      started_at=now - dt.timedelta(days=10 - day))
                s.add(attempt)
                s.flush()
                s.add(QuizQuestion(quiz_attempt_id=attempt.id, topic_id="D_T1", question_text="q",
                                   correct_answer="a", is_correct=ok, latency_ms=latency))
        s.commit()

    def tearDown(self):
        self.session.close()

    def test_a_struggling_student_scores_higher_risk_than_a_strong_one(self):
        weak = predict_topic_deficit("weak", "D_T1", session=self.session)
        strong = predict_topic_deficit("strong", "D_T1", session=self.session)
        self.assertGreater(weak["deficit"], strong["deficit"])

    def test_result_has_a_score_and_three_reasons_with_raw_values(self):
        result = predict_topic_deficit("weak", "D_T1", session=self.session)
        self.assertTrue(0 <= result["deficit"] <= 1)
        self.assertEqual(len(result["reasons"]), 3)
        reason = next(r for r in result["reasons"] if r["feature"] == "A")
        self.assertAlmostEqual(reason["value"], 0.2)           # raw accuracy, not a transformed number
        self.assertIn(reason["direction"], ("raises", "lowers"))

    def test_a_student_with_no_history_still_gets_a_valid_score(self):
        result = predict_topic_deficit("new", "D_T1", session=self.session)
        self.assertTrue(0 <= result["deficit"] <= 1)
        self.assertIsNone(next(r for r in result["reasons"] if r["feature"] == "A")["value"])


class TestBenchmarkLoader(unittest.TestCase):
    def test_features_without_information_or_left_out_are_dropped(self):
        n = 40
        df = pd.DataFrame({
            "student_id": ["s"] * n, "topic_id": ["t"] * n, "ts": np.arange(n, dtype=float),
            "n_prior": np.arange(n), "n_questions": 1,
            "Q": np.linspace(0, 1, n), "A": np.linspace(1, 0, n), "T": np.linspace(1000, 9000, n),
            "R": np.nan, "F": np.nan, "E": np.arange(n, dtype=float), "H": 0.0,
            "y_deficit": [0.0, 1.0] * (n // 2), "y_fail": [0, 1] * (n // 2),
        })[SNAPSHOT_COLUMNS]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "snapshots.csv")
            df.to_csv(path, index=False)
            train, test, features = load_benchmark_split(min_prior=3, path=path)
        self.assertEqual(features, ["Q", "A", "T"])   # R, F empty; H constant; E left out
        self.assertLess(train["ts"].max(), test["ts"].min())


if __name__ == "__main__":
    unittest.main()
