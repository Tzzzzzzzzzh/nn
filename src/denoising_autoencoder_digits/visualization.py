"""Visualization helpers for digit denoising."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import DigitDataset
from model import DenoiseResult


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_reconstruction_grid(dataset: DigitDataset, baseline: DenoiseResult, optimized: DenoiseResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "reconstruction_grid.png"
    sample_count = 10
    indices = optimized.test_indices[:sample_count]
    fig, axes = plt.subplots(4, sample_count, figsize=(12.0, 5.0))
    rows = [
        ("clean", dataset.clean[indices]),
        ("noisy", dataset.noisy[indices]),
        ("mean", baseline.reconstructed[:sample_count]),
        ("autoencoder", optimized.reconstructed[:sample_count]),
    ]
    for row_idx, (label, images) in enumerate(rows):
        for col in range(sample_count):
            axes[row_idx, col].imshow(images[col].reshape(dataset.image_size, dataset.image_size), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col].axis("off")
            if col == 0:
                axes[row_idx, col].set_ylabel(label)
    plt.suptitle("Digit denoising reconstruction examples")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_training_loss(optimized: DenoiseResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "training_loss.png"
    losses = optimized.training_loss or []
    plt.figure(figsize=(8.8, 5.2))
    plt.plot(losses, color="#2f80ed", linewidth=2.0)
    plt.title("Denoising autoencoder training loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_error_distribution(dataset: DigitDataset, baseline: DenoiseResult, optimized: DenoiseResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "pixel_error_distribution.png"
    clean = dataset.clean[optimized.test_indices]
    base_error = np.abs(baseline.reconstructed - clean).reshape(-1)
    opt_error = np.abs(optimized.reconstructed - clean).reshape(-1)
    bins = np.linspace(0, 1, 30)
    plt.figure(figsize=(9.4, 5.6))
    plt.hist(base_error, bins=bins, color="#eb5757", alpha=0.58, label="mean filter")
    plt.hist(opt_error, bins=bins, color="#27ae60", alpha=0.58, label="autoencoder")
    plt.title("Pixel absolute error distribution")
    plt.xlabel("absolute pixel error")
    plt.ylabel("pixel count")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: DenoiseResult, optimized: DenoiseResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["MSE", "PSNR/10", "edge contrast", "template acc"]
    baseline_values = [baseline.mse, baseline.psnr_db / 10.0, baseline.edge_contrast, baseline.template_accuracy]
    optimized_values = [optimized.mse, optimized.psnr_db / 10.0, optimized.edge_contrast, optimized.template_accuracy]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9.8, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="mean filter")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="autoencoder")
    plt.xticks(x, labels)
    plt.title("Denoising metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_reconstruction_csv(dataset: DigitDataset, baseline: DenoiseResult, optimized: DenoiseResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "denoising_metrics_by_sample.csv"
    clean = dataset.clean[optimized.test_indices]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("sample_index,label,mean_filter_mse,autoencoder_mse\n")
        for local_id, sample_index in enumerate(optimized.test_indices):
            base_mse = np.mean((baseline.reconstructed[local_id] - clean[local_id]) ** 2)
            auto_mse = np.mean((optimized.reconstructed[local_id] - clean[local_id]) ** 2)
            handle.write(f"{sample_index},{dataset.labels[sample_index]},{base_mse:.6f},{auto_mse:.6f}\n")
    return path


def create_visualizations(dataset: DigitDataset, baseline: DenoiseResult, optimized: DenoiseResult, output_dir: Path) -> list[Path]:
    return [
        plot_reconstruction_grid(dataset, baseline, optimized, output_dir),
        plot_training_loss(optimized, output_dir),
        plot_error_distribution(dataset, baseline, optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_reconstruction_csv(dataset, baseline, optimized, output_dir),
    ]
