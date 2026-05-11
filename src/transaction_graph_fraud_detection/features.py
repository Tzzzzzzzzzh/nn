"""Feature extraction for transaction graph fraud detection."""

from __future__ import annotations

import numpy as np

from scenario import FraudScenario


def labels(scenario: FraudScenario) -> np.ndarray:
    return np.array([account.risk_label for account in scenario.accounts], dtype=int)


def base_features(scenario: FraudScenario) -> np.ndarray:
    account_count = len(scenario.accounts)
    out_count = np.zeros(account_count)
    in_count = np.zeros(account_count)
    amount_sum = np.zeros(account_count)
    amount_max = np.zeros(account_count)
    night_count = np.zeros(account_count)
    cross_count = np.zeros(account_count)

    for tx in scenario.transactions:
        out_count[tx.source] += 1
        in_count[tx.target] += 1
        amount_sum[tx.source] += tx.amount
        amount_max[tx.source] = max(amount_max[tx.source], tx.amount)
        if tx.hour <= 5 or tx.hour >= 22:
            night_count[tx.source] += 1
        cross_count[tx.source] += tx.is_cross_segment

    age = np.array([account.age_days for account in scenario.accounts], dtype=float)
    return np.column_stack(
        [
            np.log1p(out_count),
            np.log1p(in_count),
            np.log1p(amount_sum),
            np.log1p(amount_max),
            night_count / np.maximum(out_count, 1.0),
            cross_count / np.maximum(out_count, 1.0),
            np.log1p(age),
        ]
    )


def adjacency_matrix(scenario: FraudScenario) -> np.ndarray:
    n = len(scenario.accounts)
    adjacency = np.zeros((n, n), dtype=float)
    for tx in scenario.transactions:
        adjacency[tx.source, tx.target] += 1.0
        adjacency[tx.target, tx.source] += 1.0
    return adjacency


def graph_features(scenario: FraudScenario) -> np.ndarray:
    base = base_features(scenario)
    adjacency = adjacency_matrix(scenario)
    degree = adjacency.sum(axis=1)
    neighbor_mean = np.zeros_like(base)
    for node in range(adjacency.shape[0]):
        if degree[node] > 0:
            weights = adjacency[node] / degree[node]
            neighbor_mean[node] = weights @ base
    two_hop = adjacency @ adjacency
    two_hop_count = np.count_nonzero(two_hop, axis=1) - 1
    clustering_proxy = np.zeros(adjacency.shape[0])
    binary = (adjacency > 0).astype(float)
    for node in range(adjacency.shape[0]):
        neighbors = np.where(binary[node] > 0)[0]
        if len(neighbors) >= 2:
            subgraph = binary[np.ix_(neighbors, neighbors)]
            clustering_proxy[node] = subgraph.sum() / (len(neighbors) * (len(neighbors) - 1))
    return np.column_stack([base, neighbor_mean, np.log1p(degree), np.log1p(two_hop_count), clustering_proxy])
