from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from features import graph_features
from main import build_metrics
from model import compare_models
from scenario import make_demo_scenario


def test_graph_features_improve_fraud_ranking() -> None:
    scenario = make_demo_scenario(seed=89)
    baseline, optimized = compare_models(scenario)

    assert optimized.f1_score >= baseline.f1_score
    assert optimized.auc_like_score >= baseline.auc_like_score
    assert optimized.top_k_recall >= baseline.top_k_recall


def test_graph_feature_matrix_has_expected_rows() -> None:
    scenario = make_demo_scenario(seed=89)
    features = graph_features(scenario)

    assert features.shape[0] == len(scenario.accounts)
    assert features.shape[1] > 10


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=89)

    assert metrics["improvement"]["f1_delta"] >= 0
    assert (tmp_path / "transaction_graph_projection.png").exists()
    assert (tmp_path / "fraud_score_distribution.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "adjacency_heatmap.png").exists()
    assert (tmp_path / "metrics.json").exists()
