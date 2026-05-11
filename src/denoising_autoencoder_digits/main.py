"""Run synthetic digit denoising with a Numpy autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import make_demo_dataset
from model import compare_denoisers
from visualization import create_visualizations


def _improvement(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-9) * 100.0


def build_metrics(output_dir: Path, seed: int = 101) -> dict[str, object]:
    dataset = make_demo_dataset(seed=seed)
    baseline, optimized = compare_denoisers(dataset)
    outputs = create_visualizations(dataset, baseline, optimized, output_dir)
    metrics = {
        "project": "denoising_autoencoder_digits",
        "seed": seed,
        "sample_count": int(len(dataset.labels)),
        "image_size": dataset.image_size,
        "noise_std": dataset.noise_std,
        "baseline_mean_filter": {
            "mse": round(baseline.mse, 6),
            "psnr_db": round(baseline.psnr_db, 4),
            "edge_contrast": round(baseline.edge_contrast, 4),
            "template_accuracy": round(baseline.template_accuracy, 4),
        },
        "optimized_autoencoder": {
            "mse": round(optimized.mse, 6),
            "psnr_db": round(optimized.psnr_db, 4),
            "edge_contrast": round(optimized.edge_contrast, 4),
            "template_accuracy": round(optimized.template_accuracy, 4),
            "final_training_loss": round((optimized.training_loss or [0.0])[-1], 6),
        },
        "improvement": {
            "mse_percent": round(_improvement(baseline.mse, optimized.mse), 2),
            "psnr_delta_db": round(optimized.psnr_db - baseline.psnr_db, 4),
            "template_accuracy_delta": round(optimized.template_accuracy - baseline.template_accuracy, 4),
        },
        "generated_files": [path.name for path in outputs],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic digit denoising autoencoder demo")
    parser.add_argument("--seed", type=int, default=101, help="random seed for deterministic dataset generation")
    parser.add_argument("--output", type=Path, default=Path("assets"), help="directory for generated assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.output, seed=args.seed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
