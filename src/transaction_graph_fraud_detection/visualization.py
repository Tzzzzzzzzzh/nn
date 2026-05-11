"""Visualization for transaction graph fraud detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from features import adjacency_matrix, labels
from model import FraudResult
from scenario import FraudScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_graph_projection(scenario: FraudScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "transaction_graph_projection.png"
    y = labels(scenario)
    rng = np.random.default_rng(5)
    segment_angle = {segment: i * 2 * np.pi / len(scenario.segments) for i, segment in enumerate(scenario.segments)}
    coords = []
    for account in scenario.accounts:
        angle = segment_angle[account.segment] + rng.normal(0.0, 0.22)
        radius = 1.0 + rng.normal(0.0, 0.13) + 0.25 * account.risk_label
        coords.append((radius * np.cos(angle), radius * np.sin(angle)))
    coords = np.array(coords)
    adjacency = adjacency_matrix(scenario)
    plt.figure(figsize=(8.0, 7.2))
    edge_pairs = np.argwhere(np.triu(adjacency > 0, k=1))
    for source, target in edge_pairs[:420]:
        plt.plot([coords[source, 0], coords[target, 0]], [coords[source, 1], coords[target, 1]], color="#d0d0d0", linewidth=0.35, alpha=0.28)
    plt.scatter(coords[y == 0, 0], coords[y == 0, 1], s=28, color="#2f80ed", alpha=0.78, label="normal")
    plt.scatter(coords[y == 1, 0], coords[y == 1, 1], s=52, color="#eb5757", alpha=0.90, label="fraud")
    plt.title("Transaction graph projection")
    plt.axis("off")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_score_distribution(baseline: FraudResult, optimized: FraudResult, scenario: FraudScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "fraud_score_distribution.png"
    y_test = labels(scenario)[optimized.test_indices]
    plt.figure(figsize=(9.5, 5.6))
    plt.hist(optimized.probabilities[y_test == 0], bins=18, color="#2f80ed", alpha=0.62, label="normal score")
    plt.hist(optimized.probabilities[y_test == 1], bins=18, color="#eb5757", alpha=0.72, label="fraud score")
    plt.title("Graph-enhanced fraud probability distribution")
    plt.xlabel("predicted fraud probability")
    plt.ylabel("account count")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: FraudResult, optimized: FraudResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels_text = ["precision", "recall", "F1", "top-k recall"]
    baseline_values = [baseline.precision, baseline.recall, baseline.f1_score, baseline.top_k_recall]
    optimized_values = [optimized.precision, optimized.recall, optimized.f1_score, optimized.top_k_recall]
    x = np.arange(len(labels_text))
    width = 0.36
    plt.figure(figsize=(9.6, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="statistical")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="graph-enhanced")
    plt.xticks(x, labels_text)
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Fraud detection metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_adjacency_heatmap(scenario: FraudScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "adjacency_heatmap.png"
    adjacency = adjacency_matrix(scenario)
    order = np.argsort(labels(scenario))
    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(adjacency[np.ix_(order, order)], aspect="auto", cmap="magma")
    plt.title("Transaction adjacency heatmap")
    plt.xlabel("account index")
    plt.ylabel("account index")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_scores_csv(baseline: FraudResult, optimized: FraudResult, scenario: FraudScenario, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "fraud_scores.csv"
    y = labels(scenario)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("account_id,true_label,statistical_score,graph_enhanced_score\n")
        base_scores = {idx: score for idx, score in zip(baseline.test_indices, baseline.probabilities)}
        opt_scores = {idx: score for idx, score in zip(optimized.test_indices, optimized.probabilities)}
        for idx in optimized.test_indices:
            handle.write(f"{idx},{y[idx]},{base_scores[idx]:.5f},{opt_scores[idx]:.5f}\n")
    return path


def create_visualizations(scenario: FraudScenario, baseline: FraudResult, optimized: FraudResult, output_dir: Path) -> list[Path]:
    return [
        plot_graph_projection(scenario, output_dir),
        plot_score_distribution(baseline, optimized, scenario, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        plot_adjacency_heatmap(scenario, output_dir),
        write_scores_csv(baseline, optimized, scenario, output_dir),
    ]
