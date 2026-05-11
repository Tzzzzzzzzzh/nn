"""Run the learning path recommender demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender import compare_recommenders
from scenario import make_demo_scenario
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 79) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_recommenders(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)
    metrics = {
        "project": "learning_path_recommender",
        "seed": seed,
        "student_count": scenario.student_count,
        "exercise_count": len(scenario.exercises),
        "record_count": len(scenario.records),
        "baseline_concept_average": {
            "rmse": round(baseline.rmse, 4),
            "accuracy": round(baseline.accuracy, 4),
            "auc_like_score": round(baseline.auc_like_score, 4),
            "concept_gap_mae": round(baseline.concept_gap_mae, 4),
            "recommendation_difficulty_match": round(baseline.recommendation_difficulty_match, 4),
        },
        "optimized_matrix_factorization": {
            "rmse": round(optimized.rmse, 4),
            "accuracy": round(optimized.accuracy, 4),
            "auc_like_score": round(optimized.auc_like_score, 4),
            "concept_gap_mae": round(optimized.concept_gap_mae, 4),
            "recommendation_difficulty_match": round(optimized.recommendation_difficulty_match, 4),
        },
        "improvement": {
            "rmse_percent": round(_improvement(baseline.rmse, optimized.rmse), 2),
            "accuracy_delta": round(optimized.accuracy - baseline.accuracy, 4),
            "auc_like_delta": round(optimized.auc_like_score - baseline.auc_like_score, 4),
            "concept_gap_percent": round(_improvement(baseline.concept_gap_mae, optimized.concept_gap_mae), 2),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning path recommendation with matrix factorization")
    parser.add_argument("--seed", type=int, default=79, help="random seed for deterministic learning data")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
