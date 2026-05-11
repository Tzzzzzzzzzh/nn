from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import build_metrics
from recommender import compare_recommenders
from scenario import make_demo_scenario


def test_matrix_factorization_improves_prediction() -> None:
    scenario = make_demo_scenario(seed=79)
    baseline, optimized = compare_recommenders(scenario)

    assert optimized.rmse < baseline.rmse
    assert optimized.accuracy >= baseline.accuracy
    assert optimized.auc_like_score >= baseline.auc_like_score


def test_recommendations_are_unseen_exercises() -> None:
    scenario = make_demo_scenario(seed=79)
    _, optimized = compare_recommenders(scenario)
    seen = {(record.student_id, record.exercise_id) for record in scenario.records}

    for student, exercise_ids in optimized.recommendations.items():
        assert len(exercise_ids) == 5
        assert all((student, exercise_id) not in seen for exercise_id in exercise_ids)


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=79)

    assert metrics["improvement"]["rmse_percent"] > 3
    assert (tmp_path / "response_matrix.png").exists()
    assert (tmp_path / "predicted_mastery_heatmap.png").exists()
    assert (tmp_path / "recommendation_scores.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
