"""Create per-bank CSV files from the synthetic UPI dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE = DATA_DIR / "upi_synthetic_dataset.csv"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("Missing data/upi_synthetic_dataset.csv. Run gen.py first.")

    df = pd.read_csv(SOURCE)
    if "bank_id" in df.columns:
        for bank_name, frame in df.groupby("bank_id"):
            output_name = bank_name.lower().replace(" ", "_") + ".csv"
            frame.to_csv(DATA_DIR / output_name, index=False)
            print(f"{bank_name}: {frame.shape}")
        return

    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    splits = np.array_split(shuffled, 3)
    for bank_name, frame in zip(["bank_A", "bank_B", "bank_C"], splits):
        frame.to_csv(DATA_DIR / f"{bank_name}.csv", index=False)
        print(f"{bank_name}: {frame.shape}")


if __name__ == "__main__":
    main()
