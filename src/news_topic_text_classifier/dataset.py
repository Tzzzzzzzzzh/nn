"""Synthetic news-topic text dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextDataset:
    texts: list[str]
    labels: np.ndarray
    label_names: list[str]


TOPIC_WORDS = {
    "technology": [
        "algorithm",
        "neural",
        "cloud",
        "chip",
        "model",
        "robot",
        "data",
        "software",
        "sensor",
        "platform",
    ],
    "finance": [
        "market",
        "stock",
        "bank",
        "credit",
        "fund",
        "trade",
        "risk",
        "profit",
        "loan",
        "inflation",
    ],
    "sports": [
        "match",
        "team",
        "score",
        "coach",
        "player",
        "league",
        "training",
        "final",
        "season",
        "stadium",
    ],
    "health": [
        "doctor",
        "patient",
        "vaccine",
        "hospital",
        "diet",
        "sleep",
        "clinic",
        "therapy",
        "exercise",
        "medicine",
    ],
}

COMMON_WORDS = [
    "today",
    "report",
    "city",
    "public",
    "plan",
    "update",
    "service",
    "analysis",
    "global",
    "new",
    "expert",
    "policy",
    "future",
    "community",
]


def make_demo_dataset(seed: int = 113, samples_per_topic: int = 95) -> TextDataset:
    """Create deterministic short news snippets with overlapping vocabulary."""
    rng = np.random.default_rng(seed)
    label_names = list(TOPIC_WORDS.keys())
    texts: list[str] = []
    labels: list[int] = []
    for label_id, topic in enumerate(label_names):
        topic_words = TOPIC_WORDS[topic]
        rival_topics = [name for name in label_names if name != topic]
        for _ in range(samples_per_topic):
            main_count = int(rng.integers(4, 7))
            common_count = int(rng.integers(4, 7))
            rival_count = int(rng.integers(2, 5))
            words = rng.choice(topic_words, size=main_count, replace=True).tolist()
            words += rng.choice(COMMON_WORDS, size=common_count, replace=True).tolist()
            rival_topic = str(rng.choice(rival_topics))
            words += rng.choice(TOPIC_WORDS[rival_topic], size=rival_count, replace=True).tolist()
            if rng.random() < 0.22:
                words.append("breaking")
            if rng.random() < 0.18:
                words.append("research")
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(label_id)
    return TextDataset(texts=texts, labels=np.array(labels, dtype=int), label_names=label_names)


def train_test_split(dataset: TextDataset, test_ratio: float = 0.25, seed: int = 113) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for label in np.unique(dataset.labels):
        indices = np.where(dataset.labels == label)[0]
        rng.shuffle(indices)
        split = int(len(indices) * (1.0 - test_ratio))
        train_indices.extend(indices[:split].tolist())
        test_indices.extend(indices[split:].tolist())
    return np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)
