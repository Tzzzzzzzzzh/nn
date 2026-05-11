"""Keyword baseline and TF-IDF softmax classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset import TOPIC_WORDS, TextDataset, train_test_split
from vectorizer import build_vocabulary, tfidf_matrix, tokenize


@dataclass
class ClassificationResult:
    strategy_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    top_words: dict[str, list[str]]
    loss_curve: list[float] | None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> tuple[float, float, np.ndarray]:
    confusion = np.zeros((class_count, class_count), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth, pred] += 1
    f1_scores = []
    for label in range(class_count):
        tp = confusion[label, label]
        fp = confusion[:, label].sum() - tp
        fn = confusion[label, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_scores.append(2.0 * precision * recall / max(precision + recall, 1e-9))
    return float(np.mean(y_true == y_pred)), float(np.mean(f1_scores)), confusion


def keyword_baseline(dataset: TextDataset) -> ClassificationResult:
    _, test_idx = train_test_split(dataset)
    label_names = dataset.label_names
    predictions = []
    probabilities = []
    for text in [dataset.texts[idx] for idx in test_idx]:
        tokens = tokenize(text)
        scores = np.zeros(len(label_names), dtype=float)
        for label_id, label_name in enumerate(label_names):
            topic_set = set(TOPIC_WORDS[label_name])
            scores[label_id] = sum(token in topic_set for token in tokens)
        if scores.sum() == 0:
            probs = np.ones(len(label_names)) / len(label_names)
        else:
            probs = (scores + 0.05) / np.sum(scores + 0.05)
        probabilities.append(probs)
        predictions.append(int(np.argmax(probs)))
    y_true = dataset.labels[test_idx]
    y_pred = np.array(predictions, dtype=int)
    probs = np.array(probabilities)
    accuracy, macro_f1, confusion = _metrics(y_true, y_pred, len(label_names))
    return ClassificationResult(
        strategy_name="keyword_baseline",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probs,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=confusion,
        top_words={name: TOPIC_WORDS[name][:6] for name in label_names},
        loss_curve=None,
    )


def train_softmax_classifier(x_train: np.ndarray, y_train: np.ndarray, class_count: int, epochs: int = 650, lr: float = 0.75, reg: float = 0.015) -> tuple[np.ndarray, np.ndarray, list[float]]:
    weights = np.zeros((x_train.shape[1], class_count), dtype=float)
    bias = np.zeros(class_count, dtype=float)
    y_onehot = np.eye(class_count)[y_train]
    losses: list[float] = []
    for _ in range(epochs):
        probs = _softmax(x_train @ weights + bias)
        loss = -float(np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))) + reg * float(np.sum(weights * weights))
        losses.append(loss)
        grad = (probs - y_onehot) / len(y_train)
        weights -= lr * (x_train.T @ grad + reg * weights)
        bias -= lr * grad.sum(axis=0)
    return weights, bias, losses


def tfidf_softmax_classifier(dataset: TextDataset) -> ClassificationResult:
    train_idx, test_idx = train_test_split(dataset)
    train_texts = [dataset.texts[idx] for idx in train_idx]
    test_texts = [dataset.texts[idx] for idx in test_idx]
    vocabulary = build_vocabulary(train_texts)
    x_train, x_test, _ = tfidf_matrix(train_texts, test_texts, vocabulary)
    y_train = dataset.labels[train_idx]
    y_true = dataset.labels[test_idx]
    weights, bias, losses = train_softmax_classifier(x_train, y_train, len(dataset.label_names))
    probabilities = _softmax(x_test @ weights + bias)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy, macro_f1, confusion = _metrics(y_true, y_pred, len(dataset.label_names))
    inverse_vocab = {idx: word for word, idx in vocabulary.items()}
    top_words: dict[str, list[str]] = {}
    for label_id, label_name in enumerate(dataset.label_names):
        top_indices = np.argsort(weights[:, label_id])[-8:][::-1]
        top_words[label_name] = [inverse_vocab[idx] for idx in top_indices]
    return ClassificationResult(
        strategy_name="tfidf_softmax",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=confusion,
        top_words=top_words,
        loss_curve=losses,
    )


def compare_classifiers(dataset: TextDataset) -> tuple[ClassificationResult, ClassificationResult]:
    baseline = keyword_baseline(dataset)
    optimized = tfidf_softmax_classifier(dataset)
    return baseline, optimized
