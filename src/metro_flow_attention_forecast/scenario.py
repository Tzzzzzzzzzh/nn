"""Synthetic metro passenger-flow scenario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetroFlowScenario:
    time_index: np.ndarray
    flow: np.ndarray
    weather_factor: np.ndarray
    event_factor: np.ndarray
    train_capacity: int
    interval_minutes: int
    station_name: str


def make_demo_scenario(seed: int = 67) -> MetroFlowScenario:
    """Create deterministic 15-minute passenger-flow data for one metro station."""
    rng = np.random.default_rng(seed)
    days = 14
    interval_minutes = 15
    points_per_day = 24 * 60 // interval_minutes
    total_points = days * points_per_day
    time_index = np.arange(total_points)
    minute_of_day = (time_index % points_per_day) * interval_minutes
    day_index = time_index // points_per_day
    weekday = day_index % 7

    morning_peak = 440.0 * np.exp(-0.5 * ((minute_of_day - 8.2 * 60) / 75.0) ** 2)
    evening_peak = 390.0 * np.exp(-0.5 * ((minute_of_day - 18.1 * 60) / 95.0) ** 2)
    noon = 120.0 * np.exp(-0.5 * ((minute_of_day - 12.7 * 60) / 170.0) ** 2)
    weekend_scale = np.where(weekday >= 5, 0.72, 1.0)
    weather_factor = 1.0 + 0.10 * np.sin(day_index / 2.0) + rng.normal(0.0, 0.025, total_points)
    event_factor = np.ones(total_points)
    event_mask = ((day_index == 5) & (minute_of_day >= 17 * 60) & (minute_of_day <= 21 * 60)) | (
        (day_index == 11) & (minute_of_day >= 16 * 60) & (minute_of_day <= 20 * 60)
    )
    event_factor[event_mask] = 1.38
    base = 42.0 + morning_peak + evening_peak + noon
    noise = rng.normal(0.0, 22.0, total_points)
    flow = np.maximum(12.0, base * weekend_scale * weather_factor * event_factor + noise)

    return MetroFlowScenario(
        time_index=time_index,
        flow=flow,
        weather_factor=weather_factor,
        event_factor=event_factor,
        train_capacity=500,
        interval_minutes=interval_minutes,
        station_name="Central Square",
    )


def train_test_split(scenario: MetroFlowScenario, test_days: int = 3) -> int:
    """Return the first index of the test segment."""
    points_per_day = 24 * 60 // scenario.interval_minutes
    return len(scenario.flow) - test_days * points_per_day
