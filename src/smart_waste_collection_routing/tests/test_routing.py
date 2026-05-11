from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import build_metrics
from routing import compare_routes
from scenario import make_demo_scenario


def test_urgency_route_reduces_overflow_risk() -> None:
    scenario = make_demo_scenario(seed=53)
    baseline, optimized = compare_routes(scenario)

    assert len(optimized.stops) >= len(baseline.stops) - 1
    assert optimized.average_overflow_risk < baseline.average_overflow_risk
    assert optimized.overflow_count <= baseline.overflow_count


def test_route_respects_capacity_limit() -> None:
    scenario = make_demo_scenario(seed=53)
    _, optimized = compare_routes(scenario)

    assert optimized.collected_fill_percent <= scenario.truck_capacity_percent
    assert optimized.total_service_time_min <= scenario.shift_duration_min
    assert len(set(optimized.route_bin_ids)) == len(optimized.route_bin_ids)


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=53)

    assert metrics["improvement"]["average_overflow_risk_percent"] > 5
    assert (tmp_path / "route_map_comparison.png").exists()
    assert (tmp_path / "overflow_risk_timeline.png").exists()
    assert (tmp_path / "district_service_counts.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
