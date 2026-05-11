"""Synthetic learning-record scenario for exercise recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Exercise:
    exercise_id: int
    concept: str
    difficulty: float


@dataclass
class ResponseRecord:
    student_id: int
    exercise_id: int
    concept: str
    correct: int


@dataclass
class LearningScenario:
    exercises: list[Exercise]
    records: list[ResponseRecord]
    true_mastery: np.ndarray
    concept_names: list[str]
    student_count: int


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def make_demo_scenario(seed: int = 79) -> LearningScenario:
    """Create deterministic student-exercise response data."""
    rng = np.random.default_rng(seed)
    concept_names = ["linear_algebra", "calculus", "probability", "neural_network", "optimization"]
    student_count = 72
    exercises_per_concept = 10
    exercises: list[Exercise] = []
    exercise_id = 1
    for concept in concept_names:
        for local_id in range(exercises_per_concept):
            difficulty = 0.18 + 0.07 * local_id + rng.normal(0.0, 0.025)
            exercises.append(Exercise(exercise_id=exercise_id, concept=concept, difficulty=float(np.clip(difficulty, 0.12, 0.92))))
            exercise_id += 1

    aptitude = rng.normal(0.0, 0.85, size=(student_count, 1))
    concept_bias = rng.normal(0.0, 0.55, size=(student_count, len(concept_names)))
    true_mastery = _sigmoid(aptitude + concept_bias)
    concept_index = {name: idx for idx, name in enumerate(concept_names)}

    records: list[ResponseRecord] = []
    for student_id in range(student_count):
        attempted = rng.choice(len(exercises), size=26, replace=False)
        for exercise_pos in attempted:
            exercise = exercises[int(exercise_pos)]
            mastery = true_mastery[student_id, concept_index[exercise.concept]]
            logit = 4.2 * (mastery - exercise.difficulty) + rng.normal(0.0, 0.35)
            correct = int(rng.random() < float(_sigmoid(np.array([logit]))[0]))
            records.append(
                ResponseRecord(
                    student_id=student_id,
                    exercise_id=exercise.exercise_id - 1,
                    concept=exercise.concept,
                    correct=correct,
                )
            )

    return LearningScenario(
        exercises=exercises,
        records=records,
        true_mastery=true_mastery,
        concept_names=concept_names,
        student_count=student_count,
    )


def response_matrix(scenario: LearningScenario) -> np.ndarray:
    """Return a student-exercise matrix with NaN for unobserved responses."""
    matrix = np.full((scenario.student_count, len(scenario.exercises)), np.nan)
    for record in scenario.records:
        matrix[record.student_id, record.exercise_id] = record.correct
    return matrix
