"""
Synthetic student data generator (Week 1, Person 1 / ML deliverable).

Produces 50 realistic synthetic student profiles across 5 archetypes
(Fast Learner, Struggling Student, Crammer, Consistent Reviewer,
Disengaged), simulating 30 days of quiz attempts and flashcard reviews
against a fixed "synthetic benchmark" document + topic set.

This gives Person 1 a dataset to validate the feature vector spec
(engine/feature_spec.py) and gives everyone else realistic data to build
against before real users exist (see docs/architectural_decisions.md,
Part 1, Strategy #3: "Synthetic Student Calibration").

Usage:
    python -m scripts.seed_student_data
    python -m scripts.seed_student_data --num-students 50 --days 30 --seed 42
    python -m scripts.seed_student_data --reset   # wipe prior synthetic data first
"""

import argparse
import datetime as dt
import os
import random
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from database.db import init_db, get_session
from database.crud import get_or_create_student, create_document, replace_topics
from database.schema import QuizAttempt, QuizQuestion, FlashcardEvent, Student

SYNTHETIC_DOCUMENT_ID = "SYN_BENCHMARK"
SYNTHETIC_DOCUMENT_NAME = "Synthetic CS Fundamentals Benchmark Set"

SYNTHETIC_TOPICS = [
    {"topic_id": "T1", "name": "Variables & Data Types", "description": "Primitive types, type coercion, and variable scope."},
    {"topic_id": "T2", "name": "Control Flow", "description": "Conditionals, loops, and branching logic."},
    {"topic_id": "T3", "name": "Functions & Scope", "description": "Function definitions, parameters, closures, and lexical scope."},
    {"topic_id": "T4", "name": "Recursion", "description": "Base cases, call stacks, and recursive problem decomposition."},
    {"topic_id": "T5", "name": "Arrays & Lists", "description": "Indexing, mutation, and common array algorithms."},
    {"topic_id": "T6", "name": "Trees & Graphs", "description": "Traversal strategies, binary trees, and graph representations."},
    {"topic_id": "T7", "name": "Sorting Algorithms", "description": "Comparison sorts, complexity trade-offs, and stability."},
    {"topic_id": "T8", "name": "Deadlock Handling", "description": "Banker's algorithm, prevention, detection, and recovery."},
]

# Per-archetype behavioral parameters used to simulate 30 days of activity.
ARCHETYPES = {
    "Fast Learner": dict(
        base_accuracy=0.55, accuracy_growth=0.014, activity_prob=0.55,
        latency_ms=(3000, 9000), flashcard_prob=0.60, quality_bias=1.0,
        burst_days=None,
    ),
    "Struggling Student": dict(
        base_accuracy=0.30, accuracy_growth=0.003, activity_prob=0.50,
        latency_ms=(9000, 25000), flashcard_prob=0.35, quality_bias=-0.8,
        burst_days=None,
    ),
    "Crammer": dict(
        base_accuracy=0.40, accuracy_growth=0.0, activity_prob=0.12,
        latency_ms=(6000, 15000), flashcard_prob=0.20, quality_bias=0.0,
        burst_days=range(24, 30),   # last ~6 days: cram hard
        burst_activity_prob=0.85,
    ),
    "Consistent Reviewer": dict(
        base_accuracy=0.65, accuracy_growth=0.006, activity_prob=0.80,
        latency_ms=(4000, 10000), flashcard_prob=0.85, quality_bias=0.6,
        burst_days=None,
    ),
    "Disengaged": dict(
        base_accuracy=0.45, accuracy_growth=0.0, activity_prob=0.08,
        latency_ms=(5000, 12000), flashcard_prob=0.10, quality_bias=-0.3,
        burst_days=None,
    ),
}

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _sm2_update(ease_factor: float, interval: int, reps: int, quality: int):
    """
    Standard SuperMemo SM-2 update. Duplicated here (rather than imported)
    because engine/spaced_repetition.py is Week 2 scope; this local copy
    lets the seed generator produce realistic ease/interval progressions
    without taking a dependency on not-yet-built code.
    """
    if quality < 3:
        reps = 0
        interval = 1
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = round(interval * ease_factor)

    new_ef = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = _clip(new_ef, 1.3, 2.8)
    return new_ef, interval, reps


def _activity_probability(archetype_cfg: dict, day: int) -> float:
    burst_days = archetype_cfg.get("burst_days")
    if burst_days and day in burst_days:
        return archetype_cfg.get("burst_activity_prob", archetype_cfg["activity_prob"])
    return archetype_cfg["activity_prob"]


def generate_students(num_students: int, seed: int):
    """Builds the (student_id, name, archetype) roster, evenly split across archetypes."""
    rng = random.Random(seed)
    archetype_names = list(ARCHETYPES.keys())
    students = []
    for i in range(num_students):
        archetype = archetype_names[i % len(archetype_names)]
        student_id = f"SYN{i + 1:03d}"
        name = f"{archetype.replace(' ', '')}_{i + 1:03d}"
        students.append({"id": student_id, "name": name, "archetype": archetype})
    rng.shuffle(students)
    return students


def simulate_student(student: dict, num_days: int, rng: random.Random, session):
    archetype_cfg = ARCHETYPES[student["archetype"]]
    start_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=num_days)

    # Per-(topic) SM-2 state, evolved across the simulated month.
    card_state = {t["topic_id"]: {"ef": 2.5, "interval": 0, "reps": 0} for t in SYNTHETIC_TOPICS}

    for day in range(num_days):
        current_date = start_date + dt.timedelta(days=day)
        activity_prob = _activity_probability(archetype_cfg, day)
        if rng.random() >= activity_prob:
            continue

        num_topics_today = rng.choice([1, 1, 2])
        todays_topics = rng.sample(SYNTHETIC_TOPICS, k=min(num_topics_today, len(SYNTHETIC_TOPICS)))

        for topic in todays_topics:
            topic_full_id = f"{SYNTHETIC_DOCUMENT_ID}_{topic['topic_id']}"

            # ---- Quiz attempt ----
            accuracy = _clip(
                archetype_cfg["base_accuracy"] + archetype_cfg["accuracy_growth"] * day + rng.uniform(-0.12, 0.12),
                0.05, 0.98,
            )
            num_questions = rng.randint(3, 6)
            score = round(accuracy * num_questions)

            attempt_start = current_date.replace(
                hour=rng.randint(7, 22), minute=rng.randint(0, 59), second=0
            )
            attempt = QuizAttempt(
                student_id=student["id"],
                document_id=SYNTHETIC_DOCUMENT_ID,
                topic_id=topic_full_id,
                score=score,
                total_questions=num_questions,
                accuracy=score / num_questions,
                started_at=attempt_start,
                completed_at=attempt_start + dt.timedelta(minutes=rng.randint(2, 12)),
            )
            session.add(attempt)
            session.flush()  # need attempt.id for the questions below

            lo, hi = archetype_cfg["latency_ms"]
            for q_idx in range(num_questions):
                is_correct = rng.random() < accuracy
                latency_ms = int(_clip(rng.uniform(lo, hi) * (1.4 if not is_correct else 1.0), 500, 120_000))
                session.add(QuizQuestion(
                    quiz_attempt_id=attempt.id,
                    topic_id=topic_full_id,
                    question_text=f"Practice question {q_idx + 1} on {topic['name']}",
                    question_type="mcq",
                    options="[]",
                    correct_answer="Option A",
                    user_answer="Option A" if is_correct else "Option B",
                    is_correct=is_correct,
                    latency_ms=latency_ms,
                    created_at=attempt_start,
                ))

            # ---- Flashcard review(s) ----
            if rng.random() < archetype_cfg["flashcard_prob"]:
                state = card_state[topic["topic_id"]]
                num_reviews = rng.randint(1, 3)
                for _ in range(num_reviews):
                    quality = round(_clip(3 + archetype_cfg["quality_bias"] + rng.uniform(-1.2, 1.2), 0, 5))
                    new_ef, new_interval, new_reps = _sm2_update(state["ef"], state["interval"], state["reps"], quality)
                    state["ef"], state["interval"], state["reps"] = new_ef, new_interval, new_reps

                    reviewed_at = attempt_start + dt.timedelta(minutes=rng.randint(1, 30))
                    session.add(FlashcardEvent(
                        student_id=student["id"],
                        document_id=SYNTHETIC_DOCUMENT_ID,
                        topic_id=topic_full_id,
                        card_front=f"Key concept in {topic['name']}",
                        card_back=topic["description"],
                        quality=quality,
                        ease_factor=new_ef,
                        interval_days=new_interval,
                        repetition_count=new_reps,
                        reviewed_at=reviewed_at,
                        next_review_at=reviewed_at + dt.timedelta(days=new_interval),
                    ))

    session.commit()


def reset_synthetic_data(session):
    """Deletes all rows previously generated by this script (students prefixed 'SYN')."""
    synthetic_students = session.query(Student).filter(Student.is_synthetic.is_(True)).all()
    ids = [s.id for s in synthetic_students]
    if ids:
        session.query(QuizQuestion).filter(
            QuizQuestion.quiz_attempt_id.in_(
                session.query(QuizAttempt.id).filter(QuizAttempt.student_id.in_(ids))
            )
        ).delete(synchronize_session=False)
        session.query(QuizAttempt).filter(QuizAttempt.student_id.in_(ids)).delete(synchronize_session=False)
        session.query(FlashcardEvent).filter(FlashcardEvent.student_id.in_(ids)).delete(synchronize_session=False)
        session.query(Student).filter(Student.id.in_(ids)).delete(synchronize_session=False)
        session.commit()
    print(f"Reset: removed {len(ids)} prior synthetic students and their events.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AdaptiveLearn student data.")
    parser.add_argument("--num-students", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true", help="Delete previously generated synthetic data first.")
    args = parser.parse_args()

    init_db()
    session = get_session()

    try:
        if args.reset:
            reset_synthetic_data(session)

        create_document(SYNTHETIC_DOCUMENT_ID, SYNTHETIC_DOCUMENT_NAME, source_type="synthetic")
        replace_topics(SYNTHETIC_DOCUMENT_ID, SYNTHETIC_TOPICS)

        roster = generate_students(args.num_students, args.seed)
        rng = random.Random(args.seed)

        for student in roster:
            get_or_create_student(student["id"], name=student["name"], archetype=student["archetype"], is_synthetic=True)
            simulate_student(student, args.days, rng, session)
            print(f"  Simulated {student['id']} ({student['archetype']}) over {args.days} days")

        counts = {}
        for s in roster:
            counts[s["archetype"]] = counts.get(s["archetype"], 0) + 1
        print(f"\nDone. Generated {len(roster)} synthetic students across {args.days} days.")
        print(f"Archetype breakdown: {counts}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
