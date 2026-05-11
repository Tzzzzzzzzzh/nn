"""Synthetic transaction graph scenario for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Account:
    account_id: int
    segment: str
    risk_label: int
    age_days: int


@dataclass
class Transaction:
    source: int
    target: int
    amount: float
    hour: int
    is_cross_segment: int


@dataclass
class FraudScenario:
    accounts: list[Account]
    transactions: list[Transaction]
    segments: list[str]


def make_demo_scenario(seed: int = 89) -> FraudScenario:
    """Create deterministic accounts and transaction graph."""
    rng = np.random.default_rng(seed)
    account_count = 180
    fraud_count = 28
    segments = ["retail", "merchant", "student", "freelancer"]
    accounts: list[Account] = []
    fraud_ids = set(rng.choice(account_count, size=fraud_count, replace=False).tolist())

    for account_id in range(account_count):
        segment = str(rng.choice(segments, p=[0.42, 0.24, 0.20, 0.14]))
        if account_id in fraud_ids:
            age_days = int(rng.integers(5, 140))
        else:
            age_days = int(rng.integers(80, 2200))
        accounts.append(Account(account_id=account_id, segment=segment, risk_label=int(account_id in fraud_ids), age_days=age_days))

    transactions: list[Transaction] = []
    for account in accounts:
        base_degree = int(rng.poisson(7 if account.risk_label == 0 else 9))
        for _ in range(max(2, base_degree)):
            if account.risk_label and rng.random() < 0.34:
                candidates = list(fraud_ids - {account.account_id})
                target = int(rng.choice(candidates)) if candidates else int(rng.integers(account_count))
            else:
                target = int(rng.integers(account_count))
                while target == account.account_id:
                    target = int(rng.integers(account_count))
            amount_scale = 70.0 if account.risk_label == 0 else 115.0
            amount = float(rng.lognormal(mean=np.log(amount_scale), sigma=0.88))
            hour = int(rng.choice(np.arange(24), p=_hour_probabilities(account.risk_label)))
            is_cross = int(accounts[target].segment != account.segment)
            transactions.append(Transaction(source=account.account_id, target=target, amount=round(amount, 2), hour=hour, is_cross_segment=is_cross))

    return FraudScenario(accounts=accounts, transactions=transactions, segments=segments)


def _hour_probabilities(is_fraud: int) -> np.ndarray:
    hours = np.arange(24)
    if is_fraud:
        weights = 0.8 + 2.3 * np.exp(-0.5 * ((hours - 2) / 2.2) ** 2) + 1.5 * np.exp(-0.5 * ((hours - 23) / 2.5) ** 2)
    else:
        weights = 0.7 + 1.8 * np.exp(-0.5 * ((hours - 12) / 5.0) ** 2) + 0.8 * np.exp(-0.5 * ((hours - 19) / 3.0) ** 2)
    return weights / weights.sum()
