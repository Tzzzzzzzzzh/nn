"""Run synthetic news topic classification demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier import compare_classifiers
from dataset import make_demo_dataset
from visualization import create_visualizations


def build_metrics(output_dir: Path, seed: int = 113) -> dict[str, object]:
    dataset = make_demo_dataset(seed=seed)
    baseline, optimized = compare_classifiers(dataset)
    outputs = create_visualizations(dataset, baseline, optimized, output_dir)
    metrics = {
        "project": "news_topic_text_classifier",
        "seed": seed,
        "sample_count": len(dataset.texts),
        "class_count": len(dataset.label_names),
        "baseline_keyword": {
            "accuracy": round(baseline.accuracy, 4),
            "macro_f1": round(baseline.macro_f1, 4),
        },
        "optimized_tfidf_softmax": {
            "accuracy": round(optimized.accuracy, 4),
            "macro_f1": round(optimized.macro_f1, 4),
            "final_loss": round((optimized.loss_curve or [0.0])[-1], 6),
        },
        "improvement": {
            "accuracy_delta": round(optimized.accuracy - baseline.accuracy, 4),
            "macro_f1_delta": round(optimized.macro_f1 - baseline.macro_f1, 4),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic news topic TF-IDF softmax classifier")
    parser.add_argument("--seed", type=int, default=113, help="random seed for deterministic text generation")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
