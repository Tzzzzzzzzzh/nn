"""Synthetic 8x8 digit dataset for denoising autoencoder demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DigitDataset:
    clean: np.ndarray
    noisy: np.ndarray
    labels: np.ndarray
    image_size: int
    noise_std: float


_DIGIT_TEMPLATES = np.array(
    [
        [
            "01111110",
            "11000011",
            "11000111",
            "11001111",
            "11110011",
            "11100011",
            "11000011",
            "01111110",
        ],
        [
            "00011000",
            "00111000",
            "01111000",
            "00011000",
            "00011000",
            "00011000",
            "00011000",
            "01111110",
        ],
        [
            "01111110",
            "11000011",
            "00000011",
            "00001110",
            "00111000",
            "01100000",
            "11000000",
            "11111111",
        ],
        [
            "11111110",
            "00000011",
            "00000110",
            "00111100",
            "00000110",
            "00000011",
            "11000011",
            "01111110",
        ],
        [
            "00001110",
            "00011110",
            "00110110",
            "01100110",
            "11111111",
            "00000110",
            "00000110",
            "00000110",
        ],
        [
            "11111111",
            "11000000",
            "11000000",
            "11111110",
            "00000011",
            "00000011",
            "11000011",
            "01111110",
        ],
        [
            "00111110",
            "01100000",
            "11000000",
            "11111110",
            "11000011",
            "11000011",
            "11000011",
            "01111110",
        ],
        [
            "11111111",
            "00000011",
            "00000110",
            "00001100",
            "00011000",
            "00110000",
            "00110000",
            "00110000",
        ],
        [
            "01111110",
            "11000011",
            "11000011",
            "01111110",
            "11000011",
            "11000011",
            "11000011",
            "01111110",
        ],
        [
            "01111110",
            "11000011",
            "11000011",
            "11000011",
            "01111111",
            "00000011",
            "00000110",
            "01111100",
        ],
    ],
    dtype=object,
)


def _template_to_image(template: np.ndarray) -> np.ndarray:
    rows = [[float(char) for char in line] for line in template]
    return np.array(rows, dtype=float)


def _shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    y_src_start = max(0, -dy)
    y_src_end = image.shape[0] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end = image.shape[1] - max(0, dx)
    y_dst_start = max(0, dy)
    x_dst_start = max(0, dx)
    shifted[y_dst_start : y_dst_start + (y_src_end - y_src_start), x_dst_start : x_dst_start + (x_src_end - x_src_start)] = image[
        y_src_start:y_src_end, x_src_start:x_src_end
    ]
    return shifted


def make_demo_dataset(seed: int = 101, samples_per_digit: int = 90, noise_std: float = 0.36) -> DigitDataset:
    """Create clean/noisy pairs from simple digit templates."""
    rng = np.random.default_rng(seed)
    clean_images = []
    noisy_images = []
    labels = []
    for label, template in enumerate(_DIGIT_TEMPLATES):
        base = _template_to_image(template)
        for _ in range(samples_per_digit):
            dy = int(rng.integers(-1, 2))
            dx = int(rng.integers(-1, 2))
            image = _shift_image(base, dy, dx)
            image = np.clip(image + rng.normal(0.0, 0.04, size=image.shape), 0.0, 1.0)
            noisy = np.clip(image + rng.normal(0.0, noise_std, size=image.shape), 0.0, 1.0)
            clean_images.append(image.reshape(-1))
            noisy_images.append(noisy.reshape(-1))
            labels.append(label)

    return DigitDataset(
        clean=np.array(clean_images, dtype=float),
        noisy=np.array(noisy_images, dtype=float),
        labels=np.array(labels, dtype=int),
        image_size=8,
        noise_std=noise_std,
    )


def train_test_split(dataset: DigitDataset, test_ratio: float = 0.25, seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
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
