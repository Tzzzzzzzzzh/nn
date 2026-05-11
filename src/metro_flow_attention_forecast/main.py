"""Run the metro passenger-flow attention forecasting demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forecast import compare_forecasters
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 67) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_forecasters(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)
    metrics = {
        "project": "metro_flow_attention_forecast",
        "seed": seed,
        "station_name": scenario.station_name,
        "sample_count": len(scenario.flow),
        "interval_minutes": scenario.interval_minutes,
        "baseline_seasonal": {
            "mae": round(baseline.mae, 4),
            "rmse": round(baseline.rmse, 4),
            "mape_percent": round(baseline.mape_percent, 4),
            "overload_recall": round(baseline.overload_recall, 4),
            "overload_precision": round(baseline.overload_precision, 4),
        },
        "optimized_attention": {
            "mae": round(optimized.mae, 4),
            "rmse": round(optimized.rmse, 4),
            "mape_percent": round(optimized.mape_percent, 4),
            "overload_recall": round(optimized.overload_recall, 4),
            "overload_precision": round(optimized.overload_precision, 4),
        },
        "improvement": {
            "mae_percent": round(_improvement(baseline.mae, optimized.mae), 2),
            "rmse_percent": round(_improvement(baseline.rmse, optimized.rmse), 2),
            "mape_percent": round(_improvement(baseline.mape_percent, optimized.mape_percent), 2),
            "overload_recall_delta": round(optimized.overload_recall - baseline.overload_recall, 4),
            "overload_precision_delta": round(optimized.overload_precision - baseline.overload_precision, 4),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metro passenger-flow attention forecasting demo")
    parser.add_argument("--seed", type=int, default=67, help="random seed for deterministic flow generation")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
