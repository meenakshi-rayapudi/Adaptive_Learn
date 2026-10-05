import datetime as dt
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import (Base, Student, Document, Topic, QuizAttempt, QuizQuestion,
                             FlashcardEvent, EngagementEvent, Recommendation)
from engine.dataset import (assert_no_temporal_leakage, build_snapshot_dataset,
                            temporal_split, to_model_frame)
from engine.features import (QUIZ_WINDOW, ENGAGEMENT, FLASHCARD, QUIZ, REC_FOLLOWED, Event,
                             FeatureState, compute_feature_vector, load_student_events, to_epoch)

DAY = 86400.0


def quiz(ts, topic, correct, latency=None, group=None):
    return Event(kind=QUIZ, ts=ts, topic_id=topic, correct=correct, latency_ms=latency, group_id=group)


class TestFeatureMath(unittest.TestCase):
    def test_brand_new_student_has_missing_modalities(self):
        fv = FeatureState().vector("s", "t1", as_of=0)
        self.assertIsNone(fv.q)
        self.assertIsNone(fv.a)
        self.assertIsNone(fv.t)
        self.assertIsNone(fv.r)
        self.assertIsNone(fv.f)
        self.assertEqual(fv.e, 0)
        self.assertEqual(fv.h, 0)

    def test_topic_accuracy_latency_and_recency(self):
        s = FeatureState()
        for ts, ok, lat in [(0, True, 1000), (0, False, 2000), (0, True, 3000)]:
            s.update(quiz(ts, "x", ok, lat))
        fv = s.vector("s", "x", as_of=2 * DAY)
        self.assertAlmostEqual(fv.a, 2 / 3)
        self.assertAlmostEqual(fv.t, 2000)
        self.assertAlmostEqual(fv.r, 2.0)

    def test_latency_is_clipped_to_spec_bounds(self):
        s = FeatureState()
        s.update(quiz(0, "x", True, latency=10))
        self.assertEqual(s.vector("s", "x", 0).t, 500)

    def test_q_is_global_but_a_is_per_topic(self):
        s = FeatureState()
        s.update(quiz(0, "easy", True))
        s.update(quiz(1, "easy", True))
        s.update(quiz(2, "hard", False))
        s.update(quiz(3, "hard", False))
        hard = s.vector("s", "hard", as_of=10)
        self.assertAlmostEqual(hard.q, 0.5)
        self.assertEqual(hard.a, 0.0)

    def test_q_only_looks_at_the_recent_window(self):
        s = FeatureState()
        for i in range(QUIZ_WINDOW):
            s.update(quiz(i, "x", False))
        for i in range(QUIZ_WINDOW):
            s.update(quiz(100 + i, "x", True))
        self.assertEqual(s.vector("s", "x", as_of=1000).q, 1.0)

    def test_flashcard_mastery_matches_forgetting_curve(self):
        s = FeatureState()
        # Mastered card, reviewed exactly one interval ago -> retention is 90% by construction.
        s.update(Event(kind=FLASHCARD, ts=0, topic_id="x", card_key="a", interval_days=6, repetition_count=3))
        self.assertAlmostEqual(s.vector("s", "x", as_of=6 * DAY).f, 0.9)
        # A just-failed card (repetition_count = 0) contributes zero mastery.
        s.update(Event(kind=FLASHCARD, ts=0, topic_id="x", card_key="b", interval_days=1, repetition_count=0))
        self.assertAlmostEqual(s.vector("s", "x", as_of=6 * DAY).f, 0.45)

    def test_only_latest_review_of_a_card_counts(self):
        s = FeatureState()
        s.update(Event(kind=FLASHCARD, ts=0, topic_id="x", card_key="a", interval_days=1, repetition_count=0))
        s.update(Event(kind=FLASHCARD, ts=DAY, topic_id="x", card_key="a", interval_days=6, repetition_count=3))
        self.assertAlmostEqual(s.vector("s", "x", as_of=DAY + 6 * DAY).f, 0.9)

    def test_engagement_and_followed_recommendations(self):
        s = FeatureState()
        s.update(Event(kind=ENGAGEMENT, ts=0, topic_id="x"))
        s.update(Event(kind=ENGAGEMENT, ts=1, topic_id="x", count=2))
        s.update(Event(kind=REC_FOLLOWED, ts=2, topic_id="x"))
        fv = s.vector("s", "x", as_of=3)
        self.assertEqual(fv.e, 3)
        self.assertEqual(fv.h, 1)
        self.assertEqual(s.vector("s", "other", as_of=3).e, 0)

    def test_without_wallclock_recency_and_flashcards_stay_missing(self):
        s = FeatureState(wallclock=False)
        s.update(quiz(5, "x", True))
        fv = s.vector("s", "x", as_of=99)
        self.assertIsNone(fv.r)
        self.assertIsNone(fv.f)
        self.assertEqual(fv.a, 1.0)

    def test_model_frame_keeps_nan(self):
        df = pd.DataFrame([{"Q": None, "A": None, "T": 8000, "R": 6, "F": None, "E": 3, "H": 1}])
        out = to_model_frame(df)
        self.assertTrue(math.isnan(out.loc[0, "Q"]))
        self.assertAlmostEqual(out.loc[0, "R"], 0.2)
        self.assertAlmostEqual(out.loc[0, "H"], 0.1)
        self.assertAlmostEqual(out.loc[0, "E"], math.log1p(3))


class TestSnapshotDataset(unittest.TestCase):
    def setUp(self):
        self.events = {"s": [
            *[quiz(0, "x", ok, group="g1") for ok in (True, True, True, False)],       # 75% accuracy
            *[quiz(DAY, "x", ok, group="g2") for ok in (False, False, True, False)],   # 25% accuracy
            *[quiz(2 * DAY, "x", ok, group="g3") for ok in (True, True)],
        ]}

    def test_one_row_per_attempt_with_continuous_and_binary_labels(self):
        df = build_snapshot_dataset(self.events)
        self.assertEqual(len(df), 3)
        first, second = df.iloc[0], df.iloc[1]
        self.assertAlmostEqual(first["y_deficit"], 0.25)
        self.assertEqual(first["y_fail"], 0)
        self.assertAlmostEqual(second["y_deficit"], 0.75)
        self.assertEqual(second["y_fail"], 1)
        self.assertEqual(first["n_questions"], 4)

    def test_snapshots_only_see_the_past(self):
        df = build_snapshot_dataset(self.events)
        self.assertTrue(math.isnan(df.iloc[0]["A"]))             # nothing known before attempt 1
        self.assertAlmostEqual(df.iloc[1]["A"], 0.75)            # exactly attempt 1's accuracy
        self.assertAlmostEqual(df.iloc[2]["A"], 4 / 8)           # attempts 1 and 2 combined
        self.assertEqual(list(df["n_prior"]), [0, 1, 2])

    def test_changing_a_later_outcome_never_changes_earlier_features(self):
        before = build_snapshot_dataset(self.events)
        altered = {"s": [Event(**{**e.__dict__, "correct": not e.correct}) if e.group_id == "g3" else e
                         for e in self.events["s"]]}
        after = build_snapshot_dataset(altered)
        feature_cols = ["Q", "A", "T", "R", "F", "E", "H"]
        pd.testing.assert_frame_equal(before.iloc[:3][feature_cols], after.iloc[:3][feature_cols])

    def test_ungrouped_events_become_single_question_rows(self):
        df = build_snapshot_dataset({"s": [quiz(0, "x", True), quiz(1, "x", False)]})
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["y_fail"]), [0, 1])


class TestTemporalSplit(unittest.TestCase):
    def frame(self):
        rows = [{"student_id": f"s{i % 5}", "ts": float(i), "y_fail": i % 2} for i in range(100)]
        return pd.DataFrame(rows)

    def test_global_split_is_chronological_80_20(self):
        train, test = temporal_split(self.frame(), 0.8, mode="global")
        self.assertEqual((len(train), len(test)), (80, 20))
        assert_no_temporal_leakage(train, test, mode="global")
        self.assertLess(train["ts"].max(), test["ts"].min())

    def test_global_split_never_straddles_tied_timestamps(self):
        df = self.frame()
        df.loc[75:85, "ts"] = 80.0
        train, test = temporal_split(df, 0.8, mode="global")
        assert_no_temporal_leakage(train, test, mode="global")

    def test_per_student_split_holds_out_each_students_latest_rows(self):
        train, test = temporal_split(self.frame(), 0.8, mode="per_student")
        self.assertEqual((len(train), len(test)), (80, 20))
        assert_no_temporal_leakage(train, test, mode="per_student")

    def test_leakage_check_catches_a_shuffled_split(self):
        df = self.frame()
        with self.assertRaises(AssertionError):
            assert_no_temporal_leakage(df.iloc[50:], df.iloc[:50], mode="global")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            temporal_split(self.frame(), mode="random")


class TestDatabasePath(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        s = self.session
        s.add_all([Student(id="stu", name="Stu"), Document(id="D", filename="d.pdf"), Topic(id="D_T1", document_id="D", topic_key="T1", name="Recursion")])
        s.commit()

        self.t0 = dt.datetime(2026, 1, 1, 12, 0, 0)
        for day, outcomes in [(0, [True, False, True, True]), (2, [False, False, True, False])]:
            when = self.t0 + dt.timedelta(days=day)
            attempt = QuizAttempt(student_id="stu", document_id="D", topic_id="D_T1", started_at=when)
            s.add(attempt)
            s.flush()
            for ok in outcomes:
                s.add(QuizQuestion(quiz_attempt_id=attempt.id, topic_id="D_T1", question_text="q",
                                   correct_answer="a", is_correct=ok, latency_ms=4000))
        s.add(FlashcardEvent(student_id="stu", topic_id="D_T1", card_front="f", card_back="b", quality=4,
                             interval_days=6, repetition_count=3, reviewed_at=self.t0 + dt.timedelta(days=2)))
        s.add(EngagementEvent(student_id="stu", topic_id="D_T1", event_type="chat_question",
                              created_at=self.t0 + dt.timedelta(days=1)))
        s.add(Recommendation(student_id="stu", topic_id="D_T1", deficit_score=0.7, is_followed=True,
                             created_at=self.t0 + dt.timedelta(days=1)))
        s.add(Recommendation(student_id="stu", topic_id="D_T1", deficit_score=0.6, is_followed=False,
                             created_at=self.t0 + dt.timedelta(days=1)))
        s.commit()

    def tearDown(self):
        self.session.close()

    def test_vector_aggregated_from_database_rows(self):
        as_of = self.t0 + dt.timedelta(days=8)
        fv = compute_feature_vector("stu", "D_T1", as_of=as_of, session=self.session)
        self.assertAlmostEqual(fv.a, 4 / 8)
        self.assertAlmostEqual(fv.q, 4 / 8)
        self.assertAlmostEqual(fv.t, 4000)
        self.assertAlmostEqual(fv.r, 6.0)        # last study event was the day-2 quiz/flashcard
        self.assertAlmostEqual(fv.f, 0.9)        # reviewed one full interval (6 days) before as_of
        self.assertEqual(fv.e, 1)
        self.assertEqual(fv.h, 1)                # only the followed recommendation counts

    def test_as_of_in_the_past_hides_later_rows(self):
        fv = compute_feature_vector("stu", "D_T1", as_of=self.t0 + dt.timedelta(days=1), session=self.session)
        self.assertAlmostEqual(fv.a, 3 / 4)      # second attempt (day 2) is invisible
        self.assertIsNone(fv.f)

    def test_unknown_student_yields_all_missing(self):
        fv = compute_feature_vector("nobody", "D_T1", session=self.session)
        self.assertIsNone(fv.a)

    def test_database_events_feed_the_same_snapshot_builder(self):
        events = {"stu": load_student_events("stu", session=self.session)}
        df = build_snapshot_dataset(events)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["ts"], to_epoch(self.t0))
        self.assertAlmostEqual(df.iloc[1]["A"], 3 / 4)
        self.assertAlmostEqual(df.iloc[1]["y_deficit"], 0.75)


if __name__ == "__main__":
    unittest.main()
