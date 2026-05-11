from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from forecast import compare_forecasters
from main import build_metrics
from scenario import make_demo_scenario


def test_attention_forecast_improves_error() -> None:
    scenario = make_demo_scenario(seed=67)
    baseline, optimized = compare_forecasters(scenario)

    assert optimized.mae < baseline.mae
    assert optimized.rmse < baseline.rmse
    assert optimized.mape_percent < baseline.mape_percent


def test_attention_weights_are_valid_probabilities() -> None:
    scenario = make_demo_scenario(seed=67)
    _, optimized = compare_forecasters(scenario)

    assert optimized.attention_weights is not None
    row_sums = optimized.attention_weights.sum(axis=1)
    assert abs(row_sums.min() - 1.0) < 1e-6
    assert abs(row_sums.max() - 1.0) < 1e-6


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=67)

    assert metrics["improvement"]["mae_percent"] > 5
    assert (tmp_path / "forecast_curve.png").exists()
    assert (tmp_path / "error_distribution.png").exists()
    assert (tmp_path / "attention_weights.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
