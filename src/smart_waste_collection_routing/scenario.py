"""Scenario generation for smart waste collection routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WasteBin:
    bin_id: int
    district: str
    x_km: float
    y_km: float
    fill_percent: float
    fill_rate_percent_per_hour: float
    service_time_min: float
    priority: int


@dataclass
class WasteCollectionScenario:
    bins: list[WasteBin]
    depot_xy: tuple[float, float]
    truck_capacity_percent: float
    shift_duration_min: float
    average_speed_kmph: float
    overflow_threshold_percent: float


def make_demo_scenario(seed: int = 53) -> WasteCollectionScenario:
    """Create a deterministic urban waste-bin collection scenario."""
    rng = np.random.default_rng(seed)
    districts = {
        "residential": ((1.2, 1.0), 10, 1.05),
        "school": ((4.2, 1.8), 6, 1.35),
        "business": ((2.8, 4.5), 9, 1.55),
        "park": ((6.0, 4.0), 7, 0.85),
    }
    bins: list[WasteBin] = []
    bin_id = 1
    for district, (center, count, rate_scale) in districts.items():
        for _ in range(count):
            x = float(rng.normal(center[0], 0.55))
            y = float(rng.normal(center[1], 0.45))
            fill = float(np.clip(rng.normal(68.0 + 8.0 * rate_scale, 12.0), 25.0, 96.0))
            fill_rate = float(rng.uniform(2.0, 5.6) * rate_scale)
            service_time = float(rng.uniform(3.0, 6.5))
            priority = 3 if fill > 86 or rate_scale > 1.4 else 2 if fill > 70 else 1
            bins.append(
                WasteBin(
                    bin_id=bin_id,
                    district=district,
                    x_km=round(x, 3),
                    y_km=round(y, 3),
                    fill_percent=round(fill, 2),
                    fill_rate_percent_per_hour=round(fill_rate, 2),
                    service_time_min=round(service_time, 2),
                    priority=priority,
                )
            )
            bin_id += 1

    return WasteCollectionScenario(
        bins=bins,
        depot_xy=(0.0, 0.0),
        truck_capacity_percent=1250.0,
        shift_duration_min=360.0,
        average_speed_kmph=28.0,
        overflow_threshold_percent=92.0,
    )


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean road-distance approximation with a mild detour factor."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) * 1.18)
