"""Generate a non-IID synthetic UPI fraud dataset.

The previous generator used a single deterministic fraud rule. This version
creates different bank profiles, noisy probabilistic labels, and several fraud
signals so model performance is not inflated by one obvious condition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
N = 12000

BANKS = {
    "Bank A": {
        "share": 0.34,
        "amount_scale": 1800,
        "night_bias": 0.45,
        "velocity_shift": 0.00,
        "fraud_bias": -0.15,
    },
    "Bank B": {
        "share": 0.31,
        "amount_scale": 2600,
        "night_bias": 0.34,
        "velocity_shift": 0.08,
        "fraud_bias": 0.20,
    },
    "Bank C": {
        "share": 0.35,
        "amount_scale": 2200,
        "night_bias": 0.52,
        "velocity_shift": -0.04,
        "fraud_bias": 0.05,
    },
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def make_bank_frame(bank_name: str, profile: dict, n_rows: int, start_id: int) -> pd.DataFrame:
    timestamps = pd.to_datetime("2025-10-01") + pd.to_timedelta(
        RNG.integers(0, 180 * 24 * 60, n_rows), unit="m"
    )
    hour = timestamps.hour
    is_night = ((hour < 6) | (hour >= 22)).astype(int)

    base_amount = RNG.exponential(scale=profile["amount_scale"], size=n_rows)
    festival_spike = RNG.binomial(1, 0.08, n_rows) * RNG.exponential(3000, n_rows)
    amount = np.round(base_amount + festival_spike + 50, 2)

    velocity = np.clip(
        RNG.beta(1.5, 4.5, n_rows) + profile["velocity_shift"],
        0,
        1,
    )
    account_age_days = RNG.gamma(shape=2.8, scale=180, size=n_rows).astype(int) + 1
    receiver_age_days = RNG.gamma(shape=2.2, scale=120, size=n_rows).astype(int) + 1
    is_new_receiver = RNG.binomial(1, np.clip(0.20 + velocity * 0.25, 0, 0.65), n_rows)
    failed_attempts_24h = RNG.poisson(np.clip(velocity * 2.2, 0.05, 2.8), n_rows)
    is_weekend = timestamps.dayofweek.isin([5, 6]).astype(int)

    device_type = RNG.choice(["mobile", "web"], n_rows, p=[0.86, 0.14])
    upi_app = RNG.choice(
        ["GPay", "PhonePe", "Paytm", "BHIM"],
        n_rows,
        p=[0.34, 0.31, 0.25, 0.10],
    )
    location = RNG.choice(
        ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune"],
        n_rows,
        p=[0.22, 0.21, 0.18, 0.16, 0.13, 0.10],
    )

    log_amount = np.log1p(amount)
    risk_logit = (
        -4.95
        + 0.62 * (log_amount - np.log1p(profile["amount_scale"]))
        + 2.05 * velocity
        + 0.82 * is_night
        + 1.18 * is_new_receiver
        + 0.48 * np.clip(failed_attempts_24h, 0, 5)
        + 0.68 * (account_age_days < 45)
        + 0.45 * (receiver_age_days < 30)
        + 0.32 * is_weekend
        + 0.48 * (device_type == "web")
        + 0.24 * np.isin(location, ["Delhi", "Mumbai"])
        + profile["fraud_bias"]
    )

    # Two ring-like attack patterns create non-linear pockets of fraud.
    mule_ring = (
        (amount > np.quantile(amount, 0.88))
        & (is_new_receiver == 1)
        & (velocity > 0.55)
    )
    takeover_pattern = (
        (failed_attempts_24h >= 3)
        & (account_age_days < 90)
        & (device_type == "web")
    )
    risk_logit += 1.55 * mule_ring + 1.25 * takeover_pattern

    fraud_probability = sigmoid(risk_logit)
    is_fraud = RNG.binomial(1, fraud_probability)

    return pd.DataFrame(
        {
            "transaction_id": np.arange(start_id, start_id + n_rows),
            "utr_number": [
                "UPI" + str(RNG.integers(10000000, 99999999)) for _ in range(n_rows)
            ],
            "bank_id": bank_name,
            "amount": amount,
            "timestamp": timestamps,
            "sender_id": RNG.integers(10000, 25000, n_rows),
            "receiver_id": RNG.integers(25000, 42000, n_rows),
            "device_type": device_type,
            "upi_app": upi_app,
            "location": location,
            "hour": hour,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "transaction_velocity": np.round(velocity, 3),
            "account_age_days": account_age_days,
            "receiver_age_days": receiver_age_days,
            "is_new_receiver": is_new_receiver,
            "failed_attempts_24h": failed_attempts_24h,
            "is_fraud": is_fraud.astype(int),
        }
    )


def main() -> None:
    frames = []
    next_id = 1
    assigned = 0
    bank_items = list(BANKS.items())

    for index, (bank_name, profile) in enumerate(bank_items):
        if index == len(bank_items) - 1:
            n_rows = N - assigned
        else:
            n_rows = int(N * profile["share"])
            assigned += n_rows
        frame = make_bank_frame(bank_name, profile, n_rows, next_id)
        next_id += n_rows
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset.to_csv(DATA_DIR / "upi_synthetic_dataset.csv", index=False)

    for bank_name, frame in dataset.groupby("bank_id"):
        filename = bank_name.lower().replace(" ", "_") + ".csv"
        frame.to_csv(DATA_DIR / filename, index=False)

    print("Synthetic non-IID UPI dataset generated")
    print(dataset["is_fraud"].value_counts().sort_index())
    print(dataset.groupby("bank_id")["is_fraud"].agg(["count", "sum", "mean"]))


if __name__ == "__main__":
    main()
