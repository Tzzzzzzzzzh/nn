from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import make_demo_dataset
from main import build_metrics
from model import compare_denoisers


def test_autoencoder_reduces_reconstruction_error() -> None:
    dataset = make_demo_dataset(seed=101)
    baseline, optimized = compare_denoisers(dataset)

    assert optimized.mse < baseline.mse
    assert optimized.psnr_db > baseline.psnr_db


def test_training_loss_decreases() -> None:
    dataset = make_demo_dataset(seed=101)
    _, optimized = compare_denoisers(dataset)

    assert optimized.training_loss is not None
    assert optimized.training_loss[-1] < optimized.training_loss[0]


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=101)

    assert metrics["improvement"]["mse_percent"] > 10
    assert (tmp_path / "reconstruction_grid.png").exists()
    assert (tmp_path / "training_loss.png").exists()
    assert (tmp_path / "pixel_error_distribution.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "metrics.json").exists()
