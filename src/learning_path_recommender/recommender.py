"""Matrix-factorization learning-path recommender."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import LearningScenario, response_matrix


@dataclass
class EvaluationResult:
    strategy_name: str
    predicted_matrix: np.ndarray
    test_entries: list[tuple[int, int, int]]
    rmse: float
    accuracy: float
    auc_like_score: float
    concept_gap_mae: float
    recommendations: dict[int, list[int]]
    recommendation_difficulty_match: float


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def train_test_mask(matrix: np.ndarray, seed: int = 79) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    rng = np.random.default_rng(seed)
    train = matrix.copy()
    observed = np.argwhere(~np.isnan(matrix))
    test_size = int(len(observed) * 0.20)
    selected = observed[rng.choice(len(observed), size=test_size, replace=False)]
    test_entries: list[tuple[int, int, int]] = []
    for student, exercise in selected:
        value = int(matrix[student, exercise])
        train[student, exercise] = np.nan
        test_entries.append((int(student), int(exercise), value))
    return train, test_entries


def baseline_concept_average(scenario: LearningScenario) -> EvaluationResult:
    matrix = response_matrix(scenario)
    train, test_entries = train_test_mask(matrix)
    concept_to_exercises: dict[str, list[int]] = {}
    for exercise in scenario.exercises:
        concept_to_exercises.setdefault(exercise.concept, []).append(exercise.exercise_id - 1)

    predicted = np.full_like(matrix, 0.5, dtype=float)
    global_mean = float(np.nanmean(train))
    for student in range(matrix.shape[0]):
        student_mean = np.nanmean(train[student])
        if np.isnan(student_mean):
            student_mean = global_mean
        for concept, exercise_ids in concept_to_exercises.items():
            concept_values = train[student, exercise_ids]
            if np.all(np.isnan(concept_values)):
                concept_mean = student_mean
            else:
                concept_mean = np.nanmean(concept_values)
            predicted[student, exercise_ids] = 0.65 * concept_mean + 0.35 * global_mean

    return _evaluate("concept_average", scenario, predicted, test_entries)


def matrix_factorization(scenario: LearningScenario, factors: int = 8, epochs: int = 180, lr: float = 0.035, reg: float = 0.025) -> EvaluationResult:
    matrix = response_matrix(scenario)
    train, test_entries = train_test_mask(matrix)
    rng = np.random.default_rng(17)
    student_count, exercise_count = train.shape
    observed = np.argwhere(~np.isnan(train))

    global_logit = np.log(np.nanmean(train) / max(1e-6, 1.0 - np.nanmean(train)))
    student_vec = rng.normal(0.0, 0.12, size=(student_count, factors))
    exercise_vec = rng.normal(0.0, 0.12, size=(exercise_count, factors))
    student_bias = np.zeros(student_count)
    exercise_bias = np.zeros(exercise_count)

    for _ in range(epochs):
        rng.shuffle(observed)
        for student, exercise in observed:
            target = train[student, exercise]
            logit = global_logit + student_bias[student] + exercise_bias[exercise] + student_vec[student] @ exercise_vec[exercise]
            pred = float(_sigmoid(np.array([logit]))[0])
            error = pred - target
            grad = error * pred * (1.0 - pred)
            old_student = student_vec[student].copy()
            student_vec[student] -= lr * (grad * exercise_vec[exercise] + reg * student_vec[student])
            exercise_vec[exercise] -= lr * (grad * old_student + reg * exercise_vec[exercise])
            student_bias[student] -= lr * (grad + reg * student_bias[student])
            exercise_bias[exercise] -= lr * (grad + reg * exercise_bias[exercise])

    logits = global_logit + student_bias[:, None] + exercise_bias[None, :] + student_vec @ exercise_vec.T
    predicted = _sigmoid(logits)
    return _evaluate("matrix_factorization", scenario, predicted, test_entries)


def _concept_gap_mae(scenario: LearningScenario, predicted: np.ndarray) -> float:
    concept_index = {name: idx for idx, name in enumerate(scenario.concept_names)}
    gaps = []
    for exercise in scenario.exercises:
        concept_id = concept_index[exercise.concept]
        predicted_mastery = predicted[:, exercise.exercise_id - 1]
        true_mastery = scenario.true_mastery[:, concept_id]
        gaps.extend(np.abs(predicted_mastery - true_mastery))
    return float(np.mean(gaps))


def _recommend(scenario: LearningScenario, predicted: np.ndarray, student_ids: list[int]) -> dict[int, list[int]]:
    matrix = response_matrix(scenario)
    recommendations: dict[int, list[int]] = {}
    for student in student_ids:
        unseen = np.where(np.isnan(matrix[student]))[0]
        scores = []
        for exercise_id in unseen:
            exercise = scenario.exercises[int(exercise_id)]
            success_prob = predicted[student, exercise_id]
            challenge = 1.0 - abs(success_prob - 0.68)
            difficulty_bonus = 0.12 * exercise.difficulty
            scores.append((challenge + difficulty_bonus, int(exercise_id)))
        recommendations[student] = [item[1] for item in sorted(scores, reverse=True)[:5]]
    return recommendations


def _difficulty_match(scenario: LearningScenario, predicted: np.ndarray, recommendations: dict[int, list[int]]) -> float:
    diffs = []
    for student, exercise_ids in recommendations.items():
        for exercise_id in exercise_ids:
            diffs.append(abs(predicted[student, exercise_id] - 0.68))
    return float(np.mean(diffs)) if diffs else 0.0


def _evaluate(strategy_name: str, scenario: LearningScenario, predicted: np.ndarray, test_entries: list[tuple[int, int, int]]) -> EvaluationResult:
    actual = np.array([value for _, _, value in test_entries], dtype=float)
    scores = np.array([predicted[student, exercise] for student, exercise, _ in test_entries], dtype=float)
    rmse = float(np.sqrt(np.mean((scores - actual) ** 2)))
    accuracy = float(np.mean((scores >= 0.5) == actual))
    positive = scores[actual == 1]
    negative = scores[actual == 0]
    if len(positive) and len(negative):
        auc_like = float(np.mean(positive[:, None] > negative[None, :]))
    else:
        auc_like = 0.5
    recs = _recommend(scenario, predicted, list(range(8)))
    return EvaluationResult(
        strategy_name=strategy_name,
        predicted_matrix=predicted,
        test_entries=test_entries,
        rmse=rmse,
        accuracy=accuracy,
        auc_like_score=auc_like,
        concept_gap_mae=_concept_gap_mae(scenario, predicted),
        recommendations=recs,
        recommendation_difficulty_match=_difficulty_match(scenario, predicted, recs),
    )


def compare_recommenders(scenario: LearningScenario) -> tuple[EvaluationResult, EvaluationResult]:
    baseline = baseline_concept_average(scenario)
    optimized = matrix_factorization(scenario)
    return baseline, optimized
