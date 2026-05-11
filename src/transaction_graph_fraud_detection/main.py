"""Run transaction graph fraud detection demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import compare_models
from scenario import make_demo_scenario
from visualization import create_visualizations


def _delta(after: float, before: float) -> float:
    return after - before


def build_metrics(output_dir: Path, seed: int = 89) -> dict[str, object]:
    scenario = make_demo_scenario(seed=seed)
    baseline, optimized = compare_models(scenario)
    outputs = create_visualizations(scenario, baseline, optimized, output_dir)
    metrics = {
        "project": "transaction_graph_fraud_detection",
        "seed": seed,
        "account_count": len(scenario.accounts),
        "transaction_count": len(scenario.transactions),
        "fraud_account_count": sum(account.risk_label for account in scenario.accounts),
        "baseline_statistical_features": {
            "accuracy": round(baseline.accuracy, 4),
            "precision": round(baseline.precision, 4),
            "recall": round(baseline.recall, 4),
            "f1_score": round(baseline.f1_score, 4),
            "auc_like_score": round(baseline.auc_like_score, 4),
            "top_k_recall": round(baseline.top_k_recall, 4),
        },
        "optimized_graph_features": {
            "accuracy": round(optimized.accuracy, 4),
            "precision": round(optimized.precision, 4),
            "recall": round(optimized.recall, 4),
            "f1_score": round(optimized.f1_score, 4),
            "auc_like_score": round(optimized.auc_like_score, 4),
            "top_k_recall": round(optimized.top_k_recall, 4),
        },
        "improvement": {
            "precision_delta": round(_delta(optimized.precision, baseline.precision), 4),
            "recall_delta": round(_delta(optimized.recall, baseline.recall), 4),
            "f1_delta": round(_delta(optimized.f1_score, baseline.f1_score), 4),
            "auc_like_delta": round(_delta(optimized.auc_like_score, baseline.auc_like_score), 4),
            "top_k_recall_delta": round(_delta(optimized.top_k_recall, baseline.top_k_recall), 4),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transaction graph fraud detection demo")
    parser.add_argument("--seed", type=int, default=89, help="random seed for deterministic graph generation")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
