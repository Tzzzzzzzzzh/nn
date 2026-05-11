"""Visualization for learning path recommendation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recommender import EvaluationResult
from scenario import LearningScenario, response_matrix


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_response_matrix(scenario: LearningScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "response_matrix.png"
    matrix = response_matrix(scenario)
    plt.figure(figsize=(10.0, 6.0))
    display = np.nan_to_num(matrix, nan=-0.25)
    plt.imshow(display, aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=1.0)
    plt.colorbar(label="response: -0.25 missing, 0 wrong, 1 correct")
    plt.title("Observed student-exercise response matrix")
    plt.xlabel("exercise id")
    plt.ylabel("student id")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_mastery_heatmap(scenario: LearningScenario, optimized: EvaluationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "predicted_mastery_heatmap.png"
    concept_means = []
    for concept in scenario.concept_names:
        exercise_ids = [exercise.exercise_id - 1 for exercise in scenario.exercises if exercise.concept == concept]
        concept_means.append(np.mean(optimized.predicted_matrix[:, exercise_ids], axis=1))
    mastery = np.column_stack(concept_means)
    plt.figure(figsize=(9.0, 6.2))
    plt.imshow(mastery, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.colorbar(label="predicted mastery")
    plt.xticks(range(len(scenario.concept_names)), scenario.concept_names, rotation=25, ha="right")
    plt.ylabel("student id")
    plt.title("Matrix-factorization predicted concept mastery")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_recommendation_scores(scenario: LearningScenario, optimized: EvaluationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "recommendation_scores.png"
    students = sorted(optimized.recommendations.keys())
    values = []
    labels = []
    for student in students:
        recs = optimized.recommendations[student]
        values.append([optimized.predicted_matrix[student, exercise_id] for exercise_id in recs])
        labels.append(f"S{student}")
    plt.figure(figsize=(9.6, 5.6))
    plt.imshow(np.array(values), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="predicted success probability")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(5), ["rec1", "rec2", "rec3", "rec4", "rec5"])
    plt.title("Recommended exercise success probabilities")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: EvaluationResult, optimized: EvaluationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["RMSE", "accuracy", "AUC-like", "mastery gap"]
    baseline_values = [baseline.rmse, baseline.accuracy, baseline.auc_like_score, baseline.concept_gap_mae]
    optimized_values = [optimized.rmse, optimized.accuracy, optimized.auc_like_score, optimized.concept_gap_mae]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="concept average")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="matrix factorization")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Learning recommender metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_recommendations_csv(scenario: LearningScenario, optimized: EvaluationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "recommendations.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("student_id,rank,exercise_id,concept,difficulty,predicted_success\n")
        for student, exercise_ids in optimized.recommendations.items():
            for rank, exercise_id in enumerate(exercise_ids, start=1):
                exercise = scenario.exercises[exercise_id]
                handle.write(
                    f"{student},{rank},{exercise_id},{exercise.concept},{exercise.difficulty:.3f},"
                    f"{optimized.predicted_matrix[student, exercise_id]:.4f}\n"
                )
    return path


def create_visualizations(
    scenario: LearningScenario,
    baseline: EvaluationResult,
    optimized: EvaluationResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_response_matrix(scenario, output_dir),
        plot_mastery_heatmap(scenario, optimized, output_dir),
        plot_recommendation_scores(scenario, optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_recommendations_csv(scenario, optimized, output_dir),
    ]
