"""Baseline and urgency-aware waste collection route planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import WasteBin, WasteCollectionScenario, distance_km


@dataclass
class StopRecord:
    bin_id: int
    district: str
    arrival_min: float
    fill_at_arrival_percent: float
    travel_km: float
    overflow_risk: float
    priority: int


@dataclass
class RouteResult:
    strategy_name: str
    stops: list[StopRecord]
    route_bin_ids: list[int]
    total_distance_km: float
    total_service_time_min: float
    collected_fill_percent: float
    overflow_count: int
    average_overflow_risk: float
    high_priority_served: int
    route_efficiency_score: float


def _point_for_bin(bin_item: WasteBin) -> tuple[float, float]:
    return (bin_item.x_km, bin_item.y_km)


def _fill_at_time(bin_item: WasteBin, minute: float) -> float:
    return bin_item.fill_percent + bin_item.fill_rate_percent_per_hour * minute / 60.0


def _overflow_risk(bin_item: WasteBin, arrival_min: float, scenario: WasteCollectionScenario) -> float:
    projected = _fill_at_time(bin_item, arrival_min)
    return max(0.0, projected - scenario.overflow_threshold_percent)


def _can_visit(
    current_xy: tuple[float, float],
    bin_item: WasteBin,
    current_min: float,
    load_percent: float,
    scenario: WasteCollectionScenario,
) -> bool:
    travel = distance_km(current_xy, _point_for_bin(bin_item))
    travel_min = travel / scenario.average_speed_kmph * 60.0
    return (
        current_min + travel_min + bin_item.service_time_min <= scenario.shift_duration_min
        and load_percent + bin_item.fill_percent <= scenario.truck_capacity_percent
    )


def _select_nearest(
    current_xy: tuple[float, float],
    candidates: list[WasteBin],
) -> WasteBin:
    return min(candidates, key=lambda item: distance_km(current_xy, _point_for_bin(item)))


def _select_urgency_aware(
    current_xy: tuple[float, float],
    candidates: list[WasteBin],
    current_min: float,
    scenario: WasteCollectionScenario,
) -> WasteBin:
    def score(item: WasteBin) -> float:
        travel = distance_km(current_xy, _point_for_bin(item))
        travel_min = travel / scenario.average_speed_kmph * 60.0
        arrival = current_min + travel_min
        projected_fill = _fill_at_time(item, arrival)
        urgency = max(0.0, projected_fill - 82.0)
        overflow = _overflow_risk(item, arrival, scenario)
        priority_bonus = item.priority * 5.5
        return travel * 2.2 - urgency * 1.15 - overflow * 2.8 - priority_bonus

    return min(candidates, key=score)


def plan_route(scenario: WasteCollectionScenario, strategy_name: str) -> RouteResult:
    """Plan one truck route using a baseline or urgency-aware heuristic."""
    if strategy_name not in {"nearest_neighbor", "urgency_aware"}:
        raise ValueError(f"unknown routing strategy: {strategy_name}")

    remaining = scenario.bins.copy()
    current_xy = scenario.depot_xy
    current_min = 0.0
    load_percent = 0.0
    distance_total = 0.0
    stops: list[StopRecord] = []

    while remaining:
        feasible = [
            bin_item
            for bin_item in remaining
            if _can_visit(current_xy, bin_item, current_min, load_percent, scenario)
        ]
        if not feasible:
            break

        if strategy_name == "nearest_neighbor":
            selected = _select_nearest(current_xy, feasible)
        else:
            selected = _select_urgency_aware(current_xy, feasible, current_min, scenario)

        travel = distance_km(current_xy, _point_for_bin(selected))
        travel_min = travel / scenario.average_speed_kmph * 60.0
        arrival = current_min + travel_min
        fill_at_arrival = _fill_at_time(selected, arrival)
        risk = _overflow_risk(selected, arrival, scenario)
        stops.append(
            StopRecord(
                bin_id=selected.bin_id,
                district=selected.district,
                arrival_min=arrival,
                fill_at_arrival_percent=fill_at_arrival,
                travel_km=travel,
                overflow_risk=risk,
                priority=selected.priority,
            )
        )
        distance_total += travel
        current_min = arrival + selected.service_time_min
        load_percent += selected.fill_percent
        current_xy = _point_for_bin(selected)
        remaining = [item for item in remaining if item.bin_id != selected.bin_id]

    distance_total += distance_km(current_xy, scenario.depot_xy)
    total_service = sum(item.service_time_min for item in scenario.bins if item.bin_id in {stop.bin_id for stop in stops})
    served_ids = {stop.bin_id for stop in stops}
    unserved_risks = [
        max(0.0, _fill_at_time(bin_item, scenario.shift_duration_min) - scenario.overflow_threshold_percent)
        for bin_item in scenario.bins
        if bin_item.bin_id not in served_ids
    ]
    overflow_count = sum(risk > 0.0 for risk in unserved_risks)
    average_risk = float(np.mean(unserved_risks)) if unserved_risks else 0.0
    high_priority = sum(stop.priority >= 3 for stop in stops)
    efficiency = len(stops) * 12.0 + high_priority * 5.0 - distance_total * 1.4 - overflow_count * 12.0 - average_risk * 2.5

    return RouteResult(
        strategy_name=strategy_name,
        stops=stops,
        route_bin_ids=[stop.bin_id for stop in stops],
        total_distance_km=float(distance_total),
        total_service_time_min=float(total_service),
        collected_fill_percent=float(load_percent),
        overflow_count=int(overflow_count),
        average_overflow_risk=float(average_risk),
        high_priority_served=int(high_priority),
        route_efficiency_score=float(efficiency),
    )


def compare_routes(scenario: WasteCollectionScenario) -> tuple[RouteResult, RouteResult]:
    baseline = plan_route(scenario, "nearest_neighbor")
    optimized = plan_route(scenario, "urgency_aware")
    return baseline, optimized
