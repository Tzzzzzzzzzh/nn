from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from classifier import compare_classifiers
from main import build_metrics
from dataset import make_demo_dataset


def test_tfidf_softmax_matches_or_beats_keyword_baseline() -> None:
    dataset = make_demo_dataset(seed=113)
    baseline, optimized = compare_classifiers(dataset)

    assert optimized.accuracy >= baseline.accuracy
    assert optimized.macro_f1 >= baseline.macro_f1
    assert optimized.loss_curve is not None
    assert optimized.loss_curve[-1] < optimized.loss_curve[0]


def test_classifier_outputs_all_classes() -> None:
    dataset = make_demo_dataset(seed=113)
    _, optimized = compare_classifiers(dataset)

    assert optimized.confusion_matrix.shape == (len(dataset.label_names), len(dataset.label_names))
    assert set(optimized.y_pred).issubset(set(range(len(dataset.label_names))))


def test_main_exports_metrics_and_visualizations(tmp_path: Path) -> None:
    metrics = build_metrics(tmp_path, seed=113)

    assert metrics["optimized_tfidf_softmax"]["accuracy"] >= metrics["baseline_keyword"]["accuracy"]
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "training_loss.png").exists()
    assert (tmp_path / "metric_comparison.png").exists()
    assert (tmp_path / "top_words.png").exists()
    assert (tmp_path / "metrics.json").exists()
