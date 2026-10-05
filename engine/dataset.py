"""
Training-table construction and the chronological train/test split.

Row definition (one row per quiz attempt):
    features  x = [Q, A, T, R, F, E, H] computed from events strictly BEFORE
              the attempt started (so the row only sees the past);
    label     y_deficit = 1 - accuracy on that attempt, in [0, 1]
              y_fail    = 1 if accuracy < FAIL_ACCURACY_THRESHOLD else 0
              (this is "failure on the next encounter" from the ADR; the
              model's predicted probability becomes the deficit score D).

The ADR also mentions a DecayPenalty term in Y. It is deliberately left out:
decay is already an input (R), and folding it into the label would make Y a
function of a feature and inflate every metric.

Questions that belong to the same attempt share one snapshot (the state before
the attempt), exactly how the recommender is used in production: it scores
topics between attempts, not mid-quiz.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from engine.feature_spec import FEATURE_NAMES, R_CAP, H_CAP
from engine.features import Event, FeatureState, QUIZ

FAIL_ACCURACY_THRESHOLD = 0.6
MIN_ATTEMPTS_FOR_ML = 3   # student_attempts >= 3 hands over from the cold-start heuristic to the ML model

SNAPSHOT_COLUMNS = ["student_id", "topic_id", "ts", "n_prior", "n_questions"] + FEATURE_NAMES + ["y_deficit", "y_fail"]


def build_snapshot_dataset(events_by_student: Dict[str, List[Event]], wallclock: bool = True) -> pd.DataFrame:
    rows = []
    for student_id, events in events_by_student.items():
        events = sorted(events, key=lambda e: e.ts)

        attempts: Dict[Tuple[str, str], List[Event]] = {}
        for e in events:
            if e.kind == QUIZ and e.group_id is not None:
                attempts.setdefault((e.group_id, e.topic_id), []).append(e)

        state = FeatureState(wallclock=wallclock)
        applied = set()

        for e in events:
            if e.kind != QUIZ:
                state.update(e)
                continue

            key = (e.group_id, e.topic_id) if e.group_id is not None else None
            if key is not None:
                if key in applied:
                    continue
                applied.add(key)
            members = attempts[key] if key is not None else [e]

            outcomes = [m.correct for m in members if m.correct is not None]
            if outcomes:
                fv = state.vector(student_id, e.topic_id, as_of=e.ts)
                accuracy = sum(outcomes) / len(outcomes)
                rows.append({
                    "student_id": student_id, "topic_id": e.topic_id, "ts": e.ts,
                    "n_prior": state.attempts_seen, "n_questions": len(outcomes),
                    "Q": fv.q, "A": fv.a, "T": fv.t, "R": fv.r, "F": fv.f, "E": fv.e, "H": fv.h,
                    "y_deficit": 1.0 - accuracy,
                    "y_fail": int(accuracy < FAIL_ACCURACY_THRESHOLD),
                })

            for m in members:
                state.update(m)
            state.attempts_seen += 1

    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the model-ready transforms from engine/feature_spec.py column-wise. NaN is preserved."""
    out = df[FEATURE_NAMES].astype(float).copy()
    out["T"] = np.log1p(out["T"].clip(lower=0))
    out["R"] = out["R"].clip(lower=0, upper=R_CAP) / R_CAP
    out["E"] = np.log1p(out["E"].clip(lower=0))
    out["H"] = out["H"].clip(lower=0, upper=H_CAP) / H_CAP
    return out


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8, time_col: str = "ts",
                   mode: str = "global") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: the earliest `train_frac` of attempts train, the latest test.
    Never shuffles. Two modes:
      global       one cutoff time for everyone (train T0 -> T_split, test T_split -> now).
                   Use when `ts` is a shared clock (the app database, ASSISTments order ids).
      per_student  each student's own earliest 80% / latest 20%. Use when `ts` is only
                   meaningful within one student's timeline (EdNet/Riiid timestamps).
    """
    if mode == "global":
        ordered = np.sort(df[time_col].to_numpy())
        if len(ordered) < 2:
            raise ValueError("Need at least 2 rows to split.")
        cutoff = ordered[min(int(len(ordered) * train_frac), len(ordered) - 1)]
        train, test = df[df[time_col] < cutoff], df[df[time_col] >= cutoff]
        if train.empty or test.empty:
            raise ValueError("Timestamps are too tied to form a train/test split; use mode='per_student'.")
        return train.copy(), test.copy()

    if mode == "per_student":
        train_parts, test_parts = [], []
        for _, group in df.groupby("student_id", sort=False):
            group = group.sort_values(time_col, kind="stable")
            n = len(group)
            if n < 2:
                train_parts.append(group)
                continue
            k = max(1, min(n - 1, int(round(n * train_frac))))
            train_parts.append(group.iloc[:k])
            test_parts.append(group.iloc[k:])
        if not test_parts:
            raise ValueError("No student has enough rows to produce a test set.")
        return pd.concat(train_parts), pd.concat(test_parts)

    raise ValueError(f"Unknown split mode: {mode!r}")


def assert_no_temporal_leakage(train: pd.DataFrame, test: pd.DataFrame, time_col: str = "ts",
                               mode: str = "global") -> None:
    """Raises AssertionError if any test row is older than a training row."""
    if mode == "global":
        assert train[time_col].max() < test[time_col].min(), "train/test overlap in time"
        return
    train_max = train.groupby("student_id")[time_col].max()
    test_min = test.groupby("student_id")[time_col].min()
    joined = pd.concat([train_max.rename("train_max"), test_min.rename("test_min")], axis=1).dropna()
    assert (joined["train_max"] <= joined["test_min"]).all(), "a student's test rows precede their training rows"
