"""Visualization helpers for news topic classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classifier import ClassificationResult
from dataset import TextDataset


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(result: ClassificationResult, dataset: TextDataset, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(6.8, 5.8))
    plt.imshow(result.confusion_matrix, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks(range(len(dataset.label_names)), dataset.label_names, rotation=25, ha="right")
    plt.yticks(range(len(dataset.label_names)), dataset.label_names)
    for y in range(result.confusion_matrix.shape[0]):
        for x in range(result.confusion_matrix.shape[1]):
            plt.text(x, y, str(result.confusion_matrix[y, x]), ha="center", va="center", color="#222222")
    plt.title("TF-IDF softmax confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_loss_curve(result: ClassificationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "training_loss.png"
    plt.figure(figsize=(8.8, 5.2))
    plt.plot(result.loss_curve or [], color="#2f80ed", linewidth=2.0)
    plt.title("Softmax classifier training loss")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: ClassificationResult, optimized: ClassificationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["accuracy", "macro F1"]
    baseline_values = [baseline.accuracy, baseline.macro_f1]
    optimized_values = [optimized.accuracy, optimized.macro_f1]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(7.6, 5.4))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="keyword")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="TF-IDF softmax")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Topic classification metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_top_words(result: ClassificationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "top_words.png"
    labels = []
    counts = []
    for topic, words in result.top_words.items():
        labels.extend([f"{topic}\n{word}" for word in words[:5]])
        counts.extend(list(range(5, 0, -1)))
    plt.figure(figsize=(11.0, 5.8))
    plt.bar(range(len(labels)), counts, color="#27ae60")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("relative rank")
    plt.title("Top weighted words by topic")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_predictions_csv(dataset: TextDataset, result: ClassificationResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "classification_predictions.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("row,true_label,predicted_label,confidence\n")
        for row, (truth, pred, probs) in enumerate(zip(result.y_true, result.y_pred, result.probabilities)):
            handle.write(
                f"{row},{dataset.label_names[int(truth)]},{dataset.label_names[int(pred)]},{float(np.max(probs)):.5f}\n"
            )
    return path


def create_visualizations(
    dataset: TextDataset,
    baseline: ClassificationResult,
    optimized: ClassificationResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_confusion_matrix(optimized, dataset, output_dir),
        plot_loss_curve(optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        plot_top_words(optimized, output_dir),
        write_predictions_csv(dataset, optimized, output_dir),
    ]
