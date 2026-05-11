"""Bag-of-words and TF-IDF vectorization."""

from __future__ import annotations

from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def build_vocabulary(texts: list[str], min_count: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    words = sorted([word for word, count in counter.items() if count >= min_count])
    return {word: index for index, word in enumerate(words)}


def count_matrix(texts: list[str], vocabulary: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=float)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            if token in vocabulary:
                matrix[row, vocabulary[token]] += 1.0
    return matrix


def tfidf_matrix(train_texts: list[str], test_texts: list[str], vocabulary: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_counts = count_matrix(train_texts, vocabulary)
    test_counts = count_matrix(test_texts, vocabulary)
    doc_freq = np.sum(train_counts > 0, axis=0)
    idf = np.log((1.0 + len(train_texts)) / (1.0 + doc_freq)) + 1.0
    train_tf = train_counts / np.maximum(train_counts.sum(axis=1, keepdims=True), 1.0)
    test_tf = test_counts / np.maximum(test_counts.sum(axis=1, keepdims=True), 1.0)
    train_tfidf = train_tf * idf
    test_tfidf = test_tf * idf
    return train_tfidf, test_tfidf, idf
