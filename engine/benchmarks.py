"""
Adapters that normalize public benchmark logs into AdaptiveLearn's Event stream,
so they flow through the exact same feature code as live app data
(engine/features.py -> engine/dataset.py).

Supported inputs (download these yourself; nothing here fetches data):

  EdNet / Riiid  ("riiid-test-answer-prediction" on Kaggle)
      train.csv      timestamp, user_id, content_id, content_type_id,
                     answered_correctly, prior_question_elapsed_time,
                     prior_question_had_explanation
      questions.csv  question_id, tags, part
  ASSISTments    skill_builder_data*.csv (2009-2010 "skill builder")
      order_id, user_id, correct, skill_id, ms_first_response, hint_count, original

Mapping onto x = [Q, A, T, R, F, E, H]:
  Q, A   from correctness                              (topic = first EdNet tag / ASSISTments skill)
  T      EdNet: elapsed time of the following bundle   (prior_question_elapsed_time is
                reported on the NEXT row, so it is shifted back one row per user)
         ASSISTments: ms_first_response
  R      EdNet: real elapsed days (timestamps are relative to each user's first event,
                which is fine because only differences are used).
         ASSISTments 2009 has no dates, so R is NaN.
  E      EdNet: 1 per "explanation was shown" flag.  ASSISTments: hint_count.
  F, H   not recorded by either dataset -> NaN (a genuinely missing modality, which
         tree models handle natively).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from engine.features import Event, QUIZ, ENGAGEMENT


@dataclass
class BenchmarkData:
    name: str
    events_by_student: Dict[str, List[Event]]
    wallclock: bool       # False -> R and F are NaN (no real dates in the source)
    split_mode: str       # "global" or "per_student" (see engine/dataset.temporal_split)


def load_ednet_events(train_csv: str, questions_csv: str,
                      max_users: Optional[int] = None, max_rows: Optional[int] = None) -> BenchmarkData:
    cols = ["timestamp", "user_id", "content_id", "content_type_id", "answered_correctly",
            "prior_question_elapsed_time", "prior_question_had_explanation"]
    df = pd.read_csv(train_csv, usecols=cols, nrows=max_rows)
    df = df[(df["content_type_id"] == 0) & (df["answered_correctly"].isin([0, 1]))]

    questions = pd.read_csv(questions_csv)
    first_tag = questions["tags"].astype(str).str.split().str[0]
    questions["topic"] = first_tag.where(questions["tags"].notna(), "part" + questions["part"].astype(str))
    df = df.merge(questions[["question_id", "topic"]], left_on="content_id", right_on="question_id", how="inner")

    if max_users:
        keep = df["user_id"].drop_duplicates().head(max_users)
        df = df[df["user_id"].isin(keep)]

    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    by_user = df.groupby("user_id")
    df["latency_ms"] = by_user["prior_question_elapsed_time"].shift(-1)
    explained = df["prior_question_had_explanation"].astype(str).str.lower().eq("true")
    df["had_explanation"] = explained.groupby(df["user_id"]).shift(-1, fill_value=False)

    events: Dict[str, List[Event]] = {}
    for row in df.itertuples(index=False):
        ts = row.timestamp / 1000.0
        topic = str(row.topic)
        user_events = events.setdefault(str(row.user_id), [])
        user_events.append(Event(
            kind=QUIZ, ts=ts, topic_id=topic, correct=bool(row.answered_correctly),
            latency_ms=None if pd.isna(row.latency_ms) else float(row.latency_ms),
        ))
        if row.had_explanation:
            user_events.append(Event(kind=ENGAGEMENT, ts=ts, topic_id=topic))

    return BenchmarkData("ednet", events, wallclock=True, split_mode="per_student")


def load_assistments_events(csv_path: str, max_users: Optional[int] = None,
                            max_rows: Optional[int] = None) -> BenchmarkData:
    wanted = {"order_id", "user_id", "correct", "skill_id", "ms_first_response", "hint_count", "original"}
    df = pd.read_csv(csv_path, encoding="latin-1", usecols=lambda c: c in wanted, nrows=max_rows)

    df = df.dropna(subset=["skill_id"])
    df = df[df["correct"].isin([0, 1])]
    if "original" in df.columns:
        df = df[df["original"] == 1]   # main problems only; scaffolding rows would double-count

    if max_users:
        keep = df["user_id"].drop_duplicates().head(max_users)
        df = df[df["user_id"].isin(keep)]

    df = df.sort_values("order_id", kind="stable")

    events: Dict[str, List[Event]] = {}
    for row in df.itertuples(index=False):
        ts = float(row.order_id)
        topic = f"skill_{int(row.skill_id)}"
        latency = row.ms_first_response if pd.notna(row.ms_first_response) and row.ms_first_response > 0 else None
        user_events = events.setdefault(str(row.user_id), [])
        user_events.append(Event(
            kind=QUIZ, ts=ts, topic_id=topic, correct=bool(row.correct),
            latency_ms=None if latency is None else float(latency),
        ))
        hints = getattr(row, "hint_count", 0)
        if pd.notna(hints) and hints > 0:
            user_events.append(Event(kind=ENGAGEMENT, ts=ts, topic_id=topic, count=float(hints)))

    return BenchmarkData("assistments", events, wallclock=False, split_mode="global")
