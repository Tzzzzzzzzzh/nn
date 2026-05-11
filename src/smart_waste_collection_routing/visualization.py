"""Visualization helpers for smart waste collection routing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from routing import RouteResult
from scenario import WasteCollectionScenario


def _prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_route_map(scenario: WasteCollectionScenario, baseline: RouteResult, optimized: RouteResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "route_map_comparison.png"
    bin_by_id = {item.bin_id: item for item in scenario.bins}

    plt.figure(figsize=(9.8, 7.0))
    for bin_item in scenario.bins:
        size = 35 + bin_item.fill_percent * 0.9
        plt.scatter(bin_item.x_km, bin_item.y_km, s=size, color="#bdbdbd", edgecolor="#333333", linewidth=0.4)
        plt.text(bin_item.x_km + 0.04, bin_item.y_km + 0.04, str(bin_item.bin_id), fontsize=7)
    plt.scatter([scenario.depot_xy[0]], [scenario.depot_xy[1]], color="#222222", marker="s", s=130, label="depot")

    def draw_route(route: RouteResult, color: str, label: str) -> None:
        points = [scenario.depot_xy] + [(bin_by_id[bin_id].x_km, bin_by_id[bin_id].y_km) for bin_id in route.route_bin_ids] + [scenario.depot_xy]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, color=color, linewidth=2.0, alpha=0.72, label=label)

    draw_route(baseline, "#eb5757", "nearest")
    draw_route(optimized, "#2f80ed", "urgency-aware")
    plt.title("Waste collection route comparison")
    plt.xlabel("x / km")
    plt.ylabel("y / km")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_fill_risk(result: RouteResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "overflow_risk_timeline.png"
    stop_ids = np.arange(1, len(result.stops) + 1)
    fill = [stop.fill_at_arrival_percent for stop in result.stops]
    risk = [stop.overflow_risk for stop in result.stops]

    plt.figure(figsize=(10.0, 5.8))
    plt.bar(stop_ids, fill, color="#27ae60", alpha=0.72, label="fill at arrival")
    plt.plot(stop_ids, risk, color="#eb5757", marker="o", linewidth=2.0, label="overflow risk")
    plt.axhline(100, color="#222222", linestyle="--", linewidth=1.4, label="overflow threshold")
    plt.title("Urgency-aware route fill and overflow risk")
    plt.xlabel("service order")
    plt.ylabel("percent")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_district_service_counts(baseline: RouteResult, optimized: RouteResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "district_service_counts.png"
    districts = sorted({stop.district for stop in baseline.stops + optimized.stops})
    baseline_counts = [sum(stop.district == district for stop in baseline.stops) for district in districts]
    optimized_counts = [sum(stop.district == district for stop in optimized.stops) for district in districts]
    x = np.arange(len(districts))
    width = 0.36

    plt.figure(figsize=(9.6, 5.6))
    plt.bar(x - width / 2, baseline_counts, width=width, color="#eb5757", label="nearest")
    plt.bar(x + width / 2, optimized_counts, width=width, color="#2f80ed", label="urgency-aware")
    plt.xticks(x, districts)
    plt.ylabel("served bins")
    plt.title("Served bins by district")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metric_comparison(baseline: RouteResult, optimized: RouteResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "metric_comparison.png"
    labels = ["distance", "overflow", "risk", "efficiency"]
    baseline_values = [
        baseline.total_distance_km,
        baseline.overflow_count,
        baseline.average_overflow_risk,
        baseline.route_efficiency_score / 10.0,
    ]
    optimized_values = [
        optimized.total_distance_km,
        optimized.overflow_count,
        optimized.average_overflow_risk,
        optimized.route_efficiency_score / 10.0,
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.0, 5.8))
    plt.bar(x - width / 2, baseline_values, width=width, color="#eb5757", label="nearest")
    plt.bar(x + width / 2, optimized_values, width=width, color="#2f80ed", label="urgency-aware")
    plt.xticks(x, labels)
    plt.ylabel("metric value")
    plt.title("Waste collection routing metric comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_route_csv(baseline: RouteResult, optimized: RouteResult, output_dir: Path) -> Path:
    _prepare(output_dir)
    path = output_dir / "route_schedule.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("strategy,order,bin_id,district,arrival_min,fill_at_arrival_percent,travel_km,overflow_risk,priority\n")
        for route in [baseline, optimized]:
            for order, stop in enumerate(route.stops, start=1):
                handle.write(
                    f"{route.strategy_name},{order},{stop.bin_id},{stop.district},"
                    f"{stop.arrival_min:.3f},{stop.fill_at_arrival_percent:.3f},"
                    f"{stop.travel_km:.3f},{stop.overflow_risk:.3f},{stop.priority}\n"
                )
    return path


def create_visualizations(
    scenario: WasteCollectionScenario,
    baseline: RouteResult,
    optimized: RouteResult,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_route_map(scenario, baseline, optimized, output_dir),
        plot_fill_risk(optimized, output_dir),
        plot_district_service_counts(baseline, optimized, output_dir),
        plot_metric_comparison(baseline, optimized, output_dir),
        write_route_csv(baseline, optimized, output_dir),
    ]
