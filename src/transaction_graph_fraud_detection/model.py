"""Logistic fraud classifiers with and without graph features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from features import base_features, graph_features, labels
from scenario import FraudScenario


@dataclass
class FraudResult:
    strategy_name: str
    probabilities: np.ndarray
    test_indices: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_like_score: float
    top_k_recall: float


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    return (train_x - mean) / std, (test_x - mean) / std


def _split_indices(y: np.ndarray, seed: int = 89) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = []
    test = []
    for label in [0, 1]:
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        split = int(len(indices) * 0.70)
        train.extend(indices[:split].tolist())
        test.extend(indices[split:].tolist())
    return np.array(train, dtype=int), np.array(test, dtype=int)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _train_logistic(train_x: np.ndarray, train_y: np.ndarray, epochs: int = 900, lr: float = 0.08, reg: float = 0.015) -> tuple[np.ndarray, float]:
    weights = np.zeros(train_x.shape[1])
    bias = 0.0
    pos_weight = float(np.sum(train_y == 0) / max(np.sum(train_y == 1), 1))
    for _ in range(epochs):
        logits = train_x @ weights + bias
        pred = _sigmoid(logits)
        sample_weight = np.where(train_y == 1, pos_weight, 1.0)
        error = (pred - train_y) * sample_weight
        weights -= lr * ((train_x.T @ error) / len(train_y) + reg * weights)
        bias -= lr * float(np.mean(error))
    return weights, bias


def _metrics(probabilities: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float, float, float, float]:
    pred = probabilities >= 0.5
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    tn = float(np.sum((pred == 0) & (y_true == 0)))
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    pos = probabilities[y_true == 1]
    neg = probabilities[y_true == 0]
    auc_like = float(np.mean(pos[:, None] > neg[None, :])) if len(pos) and len(neg) else 0.5
    k = max(1, int(np.sum(y_true == 1)))
    top_k = np.argsort(probabilities)[-k:]
    top_k_recall = float(np.sum(y_true[top_k] == 1) / max(np.sum(y_true == 1), 1))
    return float(accuracy), float(precision), float(recall), float(f1), auc_like, top_k_recall


def train_and_evaluate(scenario: FraudScenario, strategy_name: str) -> FraudResult:
    y = labels(scenario)
    x = base_features(scenario) if strategy_name == "statistical_features" else graph_features(scenario)
    train_idx, test_idx = _split_indices(y)
    train_x, test_x = _standardize(x[train_idx], x[test_idx])
    weights, bias = _train_logistic(train_x, y[train_idx])
    probabilities = _sigmoid(test_x @ weights + bias)
    accuracy, precision, recall, f1, auc_like, top_k = _metrics(probabilities, y[test_idx])
    return FraudResult(
        strategy_name=strategy_name,
        probabilities=probabilities,
        test_indices=test_idx,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_like_score=auc_like,
        top_k_recall=top_k,
    )


def compare_models(scenario: FraudScenario) -> tuple[FraudResult, FraudResult]:
    baseline = train_and_evaluate(scenario, "statistical_features")
    optimized = train_and_evaluate(scenario, "graph_enhanced_features")
    return baseline, optimized
