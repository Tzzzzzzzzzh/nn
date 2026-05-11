"""Metro flow forecasting baselines and attention-weighted predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import MetroFlowScenario, train_test_split


@dataclass
class ForecastResult:
    strategy_name: str
    test_index: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray
    mae: float
    rmse: float
    mape_percent: float
    overload_recall: float
    overload_precision: float
    attention_weights: np.ndarray | None


def _metrics(actual: np.ndarray, predicted: np.ndarray, capacity: int) -> tuple[float, float, float, float, float]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    mape = float(np.mean(np.abs(error) / np.maximum(actual, 1.0)) * 100.0)
    actual_overload = actual > capacity * 0.88
    predicted_overload = predicted > capacity * 0.88
    true_positive = float(np.sum(actual_overload & predicted_overload))
    recall = true_positive / max(float(np.sum(actual_overload)), 1.0)
    precision = true_positive / max(float(np.sum(predicted_overload)), 1.0)
    return mae, rmse, mape, float(recall), float(precision)


def seasonal_baseline(scenario: MetroFlowScenario) -> ForecastResult:
    """Predict by using the same time slot from the previous week."""
    split = train_test_split(scenario)
    points_per_day = 24 * 60 // scenario.interval_minutes
    weekly_lag = 7 * points_per_day
    test_index = np.arange(split, len(scenario.flow))
    predicted = scenario.flow[test_index - weekly_lag]
    actual = scenario.flow[test_index]
    mae, rmse, mape, recall, precision = _metrics(actual, predicted, scenario.train_capacity)
    return ForecastResult(
        strategy_name="seasonal_baseline",
        test_index=test_index,
        actual=actual,
        predicted=predicted,
        mae=mae,
        rmse=rmse,
        mape_percent=mape,
        overload_recall=recall,
        overload_precision=precision,
        attention_weights=None,
    )


def _build_features(scenario: MetroFlowScenario, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flow = scenario.flow
    points_per_day = 24 * 60 // scenario.interval_minutes
    lags = np.array([1, 2, 4, points_per_day, 2 * points_per_day, 7 * points_per_day])
    lag_values = np.column_stack([flow[indices - lag] for lag in lags])
    minute = (indices % points_per_day) / points_per_day
    sin_time = np.sin(2.0 * np.pi * minute)
    cos_time = np.cos(2.0 * np.pi * minute)
    context = np.column_stack(
        [
            sin_time,
            cos_time,
            scenario.weather_factor[indices],
            scenario.event_factor[indices],
            np.ones(len(indices)),
        ]
    )
    return lag_values, context


def attention_forecast(scenario: MetroFlowScenario) -> ForecastResult:
    """Forecast with a simple attention-weighted temporal model and ridge readout."""
    split = train_test_split(scenario)
    points_per_day = 24 * 60 // scenario.interval_minutes
    min_index = 7 * points_per_day
    train_index = np.arange(min_index, split)
    test_index = np.arange(split, len(scenario.flow))

    train_lags, train_context = _build_features(scenario, train_index)
    test_lags, test_context = _build_features(scenario, test_index)
    targets = scenario.flow[train_index]

    recent_mean = np.mean(train_lags[:, :3], axis=1)
    daily_ref = train_lags[:, 3]
    weekly_ref = train_lags[:, -1]
    volatility = np.abs(recent_mean - daily_ref)
    event_boost = train_context[:, 3] - 1.0
    train_scores = np.column_stack(
        [
            -0.012 * np.abs(train_lags[:, 0] - recent_mean),
            -0.010 * np.abs(train_lags[:, 1] - recent_mean),
            -0.008 * np.abs(train_lags[:, 2] - recent_mean),
            -0.010 * np.abs(daily_ref - weekly_ref) + 0.12 * event_boost,
            -0.007 * volatility,
            -0.006 * np.abs(weekly_ref - recent_mean) + 0.18 * event_boost,
        ]
    )
    score_shift = train_scores - np.max(train_scores, axis=1, keepdims=True)
    train_attention = np.exp(score_shift) / np.sum(np.exp(score_shift), axis=1, keepdims=True)
    attended_train = np.sum(train_attention * train_lags, axis=1)

    test_recent_mean = np.mean(test_lags[:, :3], axis=1)
    test_daily_ref = test_lags[:, 3]
    test_weekly_ref = test_lags[:, -1]
    test_volatility = np.abs(test_recent_mean - test_daily_ref)
    test_event_boost = test_context[:, 3] - 1.0
    test_scores = np.column_stack(
        [
            -0.012 * np.abs(test_lags[:, 0] - test_recent_mean),
            -0.010 * np.abs(test_lags[:, 1] - test_recent_mean),
            -0.008 * np.abs(test_lags[:, 2] - test_recent_mean),
            -0.010 * np.abs(test_daily_ref - test_weekly_ref) + 0.12 * test_event_boost,
            -0.007 * test_volatility,
            -0.006 * np.abs(test_weekly_ref - test_recent_mean) + 0.18 * test_event_boost,
        ]
    )
    test_shift = test_scores - np.max(test_scores, axis=1, keepdims=True)
    test_attention = np.exp(test_shift) / np.sum(np.exp(test_shift), axis=1, keepdims=True)
    attended_test = np.sum(test_attention * test_lags, axis=1)

    x_train = np.column_stack([attended_train, train_lags, train_context])
    x_test = np.column_stack([attended_test, test_lags, test_context])
    ridge = 0.35
    weights = np.linalg.solve(x_train.T @ x_train + ridge * np.eye(x_train.shape[1]), x_train.T @ targets)
    predicted = np.maximum(0.0, x_test @ weights)
    actual = scenario.flow[test_index]
    mae, rmse, mape, recall, precision = _metrics(actual, predicted, scenario.train_capacity)
    return ForecastResult(
        strategy_name="attention_temporal_forecast",
        test_index=test_index,
        actual=actual,
        predicted=predicted,
        mae=mae,
        rmse=rmse,
        mape_percent=mape,
        overload_recall=recall,
        overload_precision=precision,
        attention_weights=test_attention,
    )


def compare_forecasters(scenario: MetroFlowScenario) -> tuple[ForecastResult, ForecastResult]:
    baseline = seasonal_baseline(scenario)
    optimized = attention_forecast(scenario)
    return baseline, optimized
