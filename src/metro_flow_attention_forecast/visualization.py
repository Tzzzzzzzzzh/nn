"""Visualization for metro passenger-flow forecasting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forecast import ForecastResult
from scenario import MetroFlowScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_forecast_curve(scenario: MetroFlowScenario, baseline: ForecastResult, optimized: ForecastResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "forecast_curve.png"
    hours = baseline.test_index * scenario.interval_minutes / 60.0

    plt.figure(figsize=(10.4, 5.8))
    plt.plot(hours, baseline.actual, color="#222222", linewidth=2.1, label="actual")
    plt.plot(hours, baseline.predicted, color="#eb5757", linewidth=1.7, alpha=0.78, label="seasonal baseline")
    plt.plot(hours, optimized.predicted, color="#2f80ed", linewidth=1.9, label="attention forecast")
    plt.axhline(scenario.train_capacity * 0.88, color="#27ae60", linestyle="--", linewidth=1.4, label="overload threshold")
    plt.title("Metro passenger-flow forecast")
    plt.xlabel("time / h")
    plt.ylabel("passengers per 15 min")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_error_distribution(baseline: ForecastResult, optimized: ForecastResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "error_distribution.png"
    base_error = baseline.predicted - baseline.actual
    opt_error = optimized.predicted - optimized.actual
    bins = np.linspace(min(base_error.min(), opt_error.min()), max(base_error.max(), opt_error.max()), 28)

    plt.figure(figsize=(9.4, 5.6))
    plt.hist(base_error, bins=bins, color="#eb5757", alpha=0.62, label="seasonal baseline")
    plt.hist(opt_error, bins=bins, color="#27ae60", alpha=0.62, label="attention forecast")
    plt.title("Forecast error distribution")
    plt.xlabel("prediction error")
    plt.ylabel("count")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_attention_weights(optimized: ForecastResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "attention_weights.png"
    if optimized.attention_weights is None:
        raise ValueError("attention weights are required")
    sample = optimized.attention_weights[:96]
    plt.figure(figsize=(10.0, 5.4))
    plt.imshow(sample.T, aspect="auto", origin="lower", cmap="YlGnBu")
    plt.colorbar(label="attention weight")
    plt.yticks(range(6), ["lag1", "lag2", "lag4", "day", "2day", "week"])
    plt.title("Temporal attention weights over one test day")
    plt.xlabel("test time slot")
    plt.ylabel("lag feature")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: ForecastResult, optimized: ForecastResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["MAE", "RMSE", "MAPE", "overload precision"]
    baseline_values = [baseline.mae, baseline.rmse, baseline.mape_percent, baseline.overload_precision * 100.0]
    optimized_values = [optimized.mae, optimized.rmse, optimized.mape_percent, optimized.overload_precision * 100.0]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="seasonal baseline")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="attention forecast")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Metro flow forecasting metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_forecast_csv(scenario: MetroFlowScenario, baseline: ForecastResult, optimized: ForecastResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "forecast_result.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("time_index,hour,actual,seasonal_baseline,attention_forecast,event_factor,weather_factor\n")
        for idx, actual, base_pred, opt_pred in zip(
            baseline.test_index,
            baseline.actual,
            baseline.predicted,
            optimized.predicted,
        ):
            hour = idx * scenario.interval_minutes / 60.0
            handle.write(
                f"{idx},{hour:.3f},{actual:.3f},{base_pred:.3f},{opt_pred:.3f},"
                f"{scenario.event_factor[idx]:.3f},{scenario.weather_factor[idx]:.3f}\n"
            )
    return path


def create_visualizations(
    scenario: MetroFlowScenario,
    baseline: ForecastResult,
    optimized: ForecastResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_forecast_curve(scenario, baseline, optimized, output_dir),
        plot_error_distribution(baseline, optimized, output_dir),
        plot_attention_weights(optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_forecast_csv(scenario, baseline, optimized, output_dir),
    ]
