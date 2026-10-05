"""Core SuperMemo SM-2 update helpers for Week 2.

The implementation matches the deterministic logic already used in
scripts/seed_student_data.py so that simulated students and live learners share
one consistent memory model.
"""

from __future__ import annotations


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def update_card_review(ease_factor: float, interval: int, repetitions: int, quality: int):
    """Apply the SuperMemo SM-2 review update.

    Returns a tuple of (new_ease_factor, new_interval, new_repetitions).
    """
    quality_int = int(quality)
    if quality_int < 3:
        new_repetitions = 0
        new_interval = 1
    else:
        new_repetitions = int(repetitions) + 1
        if new_repetitions == 1:
            new_interval = 1
        elif new_repetitions == 2:
            new_interval = 6
        else:
            new_interval = max(1, round(int(interval) * float(ease_factor)))

    new_ef = float(ease_factor) + (0.1 - (5 - quality_int) * (0.08 + (5 - quality_int) * 0.02))
    new_ef = _clip(new_ef, 1.3, 2.8)
    return new_ef, new_interval, new_repetitions
