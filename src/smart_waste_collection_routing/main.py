"""Run the smart waste collection routing demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing import compare_routes
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 53) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_routes(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)

    metrics = {
        "project": "smart_waste_collection_routing",
        "seed": seed,
        "bin_count": len(scenario.bins),
        "truck_capacity_percent": scenario.truck_capacity_percent,
        "shift_duration_min": scenario.shift_duration_min,
        "baseline_nearest_neighbor": {
            "served_bins": len(baseline.stops),
            "total_distance_km": round(baseline.total_distance_km, 4),
            "collected_fill_percent": round(baseline.collected_fill_percent, 4),
            "overflow_count": baseline.overflow_count,
            "average_overflow_risk": round(baseline.average_overflow_risk, 4),
            "high_priority_served": baseline.high_priority_served,
            "route_efficiency_score": round(baseline.route_efficiency_score, 4),
        },
        "optimized_urgency_aware": {
            "served_bins": len(optimized.stops),
            "total_distance_km": round(optimized.total_distance_km, 4),
            "collected_fill_percent": round(optimized.collected_fill_percent, 4),
            "overflow_count": optimized.overflow_count,
            "average_overflow_risk": round(optimized.average_overflow_risk, 4),
            "high_priority_served": optimized.high_priority_served,
            "route_efficiency_score": round(optimized.route_efficiency_score, 4),
        },
        "improvement": {
            "overflow_count_percent": round(_improvement(float(baseline.overflow_count), float(optimized.overflow_count)), 2),
            "average_overflow_risk_percent": round(
                _improvement(baseline.average_overflow_risk, optimized.average_overflow_risk), 2
            ),
            "high_priority_served_delta": optimized.high_priority_served - baseline.high_priority_served,
            "efficiency_score_delta": round(optimized.route_efficiency_score - baseline.route_efficiency_score, 2),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart waste collection routing demo")
    parser.add_argument("--seed", type=int, default=53, help="random seed for deterministic waste-bin scenario")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
