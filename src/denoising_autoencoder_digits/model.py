"""Numpy denoising autoencoder and baseline filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset import DigitDataset, train_test_split


@dataclass
class DenoiseResult:
    strategy_name: str
    reconstructed: np.ndarray
    test_indices: np.ndarray
    mse: float
    psnr_db: float
    edge_contrast: float
    template_accuracy: float
    training_loss: list[float] | None


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def mean_filter(noisy: np.ndarray, image_size: int = 8) -> np.ndarray:
    images = noisy.reshape(-1, image_size, image_size)
    output = np.zeros_like(images)
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1)), mode="edge")
    for y in range(image_size):
        for x in range(image_size):
            patch = padded[:, y : y + 3, x : x + 3]
            output[:, y, x] = np.mean(patch, axis=(1, 2))
    return output.reshape(noisy.shape)


def train_autoencoder(
    train_noisy: np.ndarray,
    train_clean: np.ndarray,
    hidden_dim: int = 20,
    epochs: int = 420,
    lr: float = 0.22,
    seed: int = 13,
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    input_dim = train_noisy.shape[1]
    w1 = rng.normal(0.0, 0.18, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = rng.normal(0.0, 0.18, size=(hidden_dim, input_dim))
    b2 = np.zeros(input_dim)
    losses: list[float] = []
    n = train_noisy.shape[0]
    for _ in range(epochs):
        hidden = _sigmoid(train_noisy @ w1 + b1)
        output = _sigmoid(hidden @ w2 + b2)
        error = output - train_clean
        loss = float(np.mean(error * error))
        losses.append(loss)
        grad_out = 2.0 * error * output * (1.0 - output) / n
        grad_w2 = hidden.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_hidden = (grad_out @ w2.T) * hidden * (1.0 - hidden)
        grad_w1 = train_noisy.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}, losses


def reconstruct_autoencoder(noisy: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    hidden = _sigmoid(noisy @ params["w1"] + params["b1"])
    return _sigmoid(hidden @ params["w2"] + params["b2"])


def _edge_contrast(images: np.ndarray, image_size: int) -> float:
    grid = images.reshape(-1, image_size, image_size)
    horizontal = np.abs(np.diff(grid, axis=2)).mean()
    vertical = np.abs(np.diff(grid, axis=1)).mean()
    return float(horizontal + vertical)


def _template_accuracy(reconstructed: np.ndarray, labels: np.ndarray, clean_templates: np.ndarray) -> float:
    distances = ((reconstructed[:, None, :] - clean_templates[None, :, :]) ** 2).mean(axis=2)
    predicted = np.argmin(distances, axis=1)
    return float(np.mean(predicted == labels))


def _evaluate(strategy_name: str, clean: np.ndarray, reconstructed: np.ndarray, labels: np.ndarray, templates: np.ndarray, test_indices: np.ndarray, image_size: int, losses: list[float] | None) -> DenoiseResult:
    mse = float(np.mean((clean - reconstructed) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-9)))
    return DenoiseResult(
        strategy_name=strategy_name,
        reconstructed=reconstructed,
        test_indices=test_indices,
        mse=mse,
        psnr_db=psnr,
        edge_contrast=_edge_contrast(reconstructed, image_size),
        template_accuracy=_template_accuracy(reconstructed, labels, templates),
        training_loss=losses,
    )


def compare_denoisers(dataset: DigitDataset) -> tuple[DenoiseResult, DenoiseResult]:
    train_idx, test_idx = train_test_split(dataset)
    train_noisy = dataset.noisy[train_idx]
    train_clean = dataset.clean[train_idx]
    test_noisy = dataset.noisy[test_idx]
    test_clean = dataset.clean[test_idx]
    test_labels = dataset.labels[test_idx]
    templates = np.array([dataset.clean[np.where(dataset.labels == label)[0][0]] for label in range(10)])

    baseline_recon = mean_filter(test_noisy, dataset.image_size)
    baseline = _evaluate("mean_filter", test_clean, baseline_recon, test_labels, templates, test_idx, dataset.image_size, None)
    params, losses = train_autoencoder(train_noisy, train_clean)
    auto_recon = reconstruct_autoencoder(test_noisy, params)
    optimized = _evaluate("denoising_autoencoder", test_clean, auto_recon, test_labels, templates, test_idx, dataset.image_size, losses)
    return baseline, optimized
