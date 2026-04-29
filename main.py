"""Research-grade federated UPI fraud detection pipeline.

This script trains and evaluates:
- centralized logistic regression baseline
- local-only bank baselines
- FedAvg
- differentially private FedAvg
- robust DP-FedAvg under a Byzantine client attack

The implementation keeps raw bank data local during federated training and only
shares clipped model updates.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BANK_FILES = {
    "Bank A": DATA_DIR / "bank_A.csv",
    "Bank B": DATA_DIR / "bank_B.csv",
    "Bank C": DATA_DIR / "bank_C.csv",
}

CONFIG = {
    "random_seed": 42,
    "test_size": 0.20,
    "validation_size": 0.20,
    "federated_rounds": 18,
    "local_epochs": 4,
    "batch_size": 128,
    "learning_rate": 0.08,
    "server_learning_rate": 1.0,
    "l2": 1e-4,
    "clip_norm": 1.5,
    "noise_multiplier": 0.30,
    "delta": 1e-5,
    "attack_client": "Bank C",
    "attack_round_start": 7,
    "attack_round_end": 12,
    "attack_scale": 4.0,
}

DROP_COLUMNS = {
    "transaction_id",
    "timestamp",
    "utr_number",
    "bank_id",
}
LABEL = "is_fraud"


@dataclass
class BankDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_count: int
    fraud_train_count: int


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def predict_probability(weights: np.ndarray, bias: float, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ weights + bias)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, y_prob >= threshold, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = y_prob >= threshold
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def stratified_or_random_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    counts = df[LABEL].value_counts()
    if len(counts) == 2 and counts.min() >= 2:
        stratify = df[LABEL]
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def load_bank_frames() -> Dict[str, pd.DataFrame]:
    missing = [str(path) for path in BANK_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing bank files. Run gen.py or federated_split.py first: "
            + ", ".join(missing)
        )

    frames = {}
    for bank_name, path in BANK_FILES.items():
        df = pd.read_csv(path)
        if LABEL not in df.columns:
            raise ValueError(f"{path} does not contain required label column {LABEL!r}")
        frames[bank_name] = df
    return frames


def split_bank_frames(
    bank_frames: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for offset, (bank_name, frame) in enumerate(bank_frames.items()):
        train_val, test = stratified_or_random_split(
            frame,
            test_size=CONFIG["test_size"],
            seed=CONFIG["random_seed"] + offset,
        )
        val_fraction_of_train_val = CONFIG["validation_size"] / (1 - CONFIG["test_size"])
        train, val = stratified_or_random_split(
            train_val,
            test_size=val_fraction_of_train_val,
            seed=CONFIG["random_seed"] + 100 + offset,
        )
        splits[bank_name] = {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }
    return splits


def encode_features(df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
    cleaned = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    encoded = pd.get_dummies(cleaned, drop_first=True)
    if LABEL not in encoded.columns:
        raise ValueError(f"Encoded frame is missing label column {LABEL!r}")
    X = encoded.drop(columns=[LABEL])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def prepare_datasets() -> Tuple[
    List[BankDataset],
    StandardScaler,
    List[str],
    Dict[str, object],
]:
    bank_frames = load_bank_frames()
    splits = split_bank_frames(bank_frames)

    train_raw = pd.concat([parts["train"] for parts in splits.values()], ignore_index=True)
    train_X_raw = encode_features(train_raw)
    feature_columns = list(train_X_raw.columns)

    scaler = StandardScaler()
    scaler.fit(train_X_raw.values)

    banks: List[BankDataset] = []
    for bank_name, parts in splits.items():
        X_train = scaler.transform(encode_features(parts["train"], feature_columns).values)
        X_val = scaler.transform(encode_features(parts["val"], feature_columns).values)
        X_test = scaler.transform(encode_features(parts["test"], feature_columns).values)
        y_train = parts["train"][LABEL].astype(int).to_numpy()
        y_val = parts["val"][LABEL].astype(int).to_numpy()
        y_test = parts["test"][LABEL].astype(int).to_numpy()
        banks.append(
            BankDataset(
                name=bank_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                train_count=len(y_train),
                fraud_train_count=int(y_train.sum()),
            )
        )

    raw_all = pd.concat(bank_frames.values(), ignore_index=True)
    bank_summary = {}
    for bank_name, frame in bank_frames.items():
        bank_summary[bank_name] = {
            "samples": int(len(frame)),
            "fraud_cases": int(frame[LABEL].sum()),
            "fraud_rate": float(frame[LABEL].mean()),
            "mean_amount": float(frame["amount"].mean()) if "amount" in frame else None,
        }

    metadata = {
        "total_samples": int(len(raw_all)),
        "fraud_cases": int(raw_all[LABEL].sum()),
        "fraud_rate": float(raw_all[LABEL].mean()),
        "bank_summary": bank_summary,
    }
    return banks, scaler, feature_columns, metadata


def combine_split(
    banks: Iterable[BankDataset],
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    for bank in banks:
        X_parts.append(getattr(bank, f"X_{split_name}"))
        y_parts.append(getattr(bank, f"y_{split_name}"))
    return np.vstack(X_parts), np.concatenate(y_parts)


def positive_class_weight(y: np.ndarray) -> float:
    positives = max(int(y.sum()), 1)
    negatives = max(len(y) - positives, 1)
    return float(min(negatives / positives, 40.0))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    rng: np.random.Generator,
    initial: Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    if initial is None:
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0
    else:
        weights = initial[0].copy()
        bias = float(initial[1])

    pos_weight = positive_class_weight(y)
    batch_size = min(CONFIG["batch_size"], n_samples)

    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = order[start : start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            y_prob = predict_probability(weights, bias, X_batch)
            sample_weight = np.where(y_batch == 1, pos_weight, 1.0)
            error = (y_prob - y_batch) * sample_weight
            denom = max(float(sample_weight.sum()), 1.0)
            grad_w = X_batch.T @ error / denom + CONFIG["l2"] * weights
            grad_b = float(error.sum() / denom)
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
    return weights, bias


def clip_update(
    delta_w: np.ndarray,
    delta_b: float,
    clip_norm: float,
) -> Tuple[np.ndarray, float, float]:
    flat = np.concatenate([delta_w.ravel(), np.array([delta_b])])
    norm = float(np.linalg.norm(flat))
    if norm > clip_norm:
        scale = clip_norm / max(norm, 1e-12)
        return delta_w * scale, float(delta_b * scale), norm
    return delta_w, float(delta_b), norm


def aggregate_updates(
    updates: List[Tuple[np.ndarray, float, int]],
    method: str,
) -> Tuple[np.ndarray, float]:
    delta_w = np.vstack([u[0] for u in updates])
    delta_b = np.array([u[1] for u in updates], dtype=float)
    sample_counts = np.array([u[2] for u in updates], dtype=float)

    if method == "mean":
        weights = sample_counts / sample_counts.sum()
        return weights @ delta_w, float(weights @ delta_b)

    if method == "median":
        return np.median(delta_w, axis=0), float(np.median(delta_b))

    if method == "trimmed_mean":
        if len(updates) <= 2:
            return np.mean(delta_w, axis=0), float(np.mean(delta_b))
        sorted_w = np.sort(delta_w, axis=0)
        sorted_b = np.sort(delta_b)
        return np.mean(sorted_w[1:-1], axis=0), float(np.mean(sorted_b[1:-1]))

    raise ValueError(f"Unknown aggregation method: {method}")


def approximate_epsilon_per_round(noise_multiplier: float, delta: float) -> float:
    if noise_multiplier <= 0:
        return float("inf")
    return float(np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier)


def train_federated(
    banks: List[BankDataset],
    aggregation: str,
    use_dp: bool,
    rng: np.random.Generator,
    attack_client: Optional[str] = None,
) -> Tuple[np.ndarray, float, List[Dict[str, float]]]:
    n_features = banks[0].X_train.shape[1]
    global_w = np.zeros(n_features, dtype=float)
    global_b = 0.0
    history: List[Dict[str, float]] = []

    X_val, y_val = combine_split(banks, "val")
    attack_rounds = set(
        range(CONFIG["attack_round_start"], CONFIG["attack_round_end"] + 1)
    )

    for round_num in range(1, CONFIG["federated_rounds"] + 1):
        updates: List[Tuple[np.ndarray, float, int]] = []
        norms = []
        attacked_this_round = False

        for bank in banks:
            local_w, local_b = fit_logistic(
                bank.X_train,
                bank.y_train,
                epochs=CONFIG["local_epochs"],
                learning_rate=CONFIG["learning_rate"],
                rng=rng,
                initial=(global_w, global_b),
            )
            delta_w = local_w - global_w
            delta_b = local_b - global_b
            delta_w, delta_b, raw_norm = clip_update(
                delta_w,
                delta_b,
                CONFIG["clip_norm"],
            )
            norms.append(raw_norm)

            if attack_client == bank.name and round_num in attack_rounds:
                delta_w = -CONFIG["attack_scale"] * delta_w
                delta_b = -CONFIG["attack_scale"] * delta_b
                attacked_this_round = True

            updates.append((delta_w, delta_b, bank.train_count))

        agg_w, agg_b = aggregate_updates(updates, aggregation)

        if use_dp:
            noise_std = CONFIG["noise_multiplier"] * CONFIG["clip_norm"] / len(banks)
            agg_w = agg_w + rng.normal(0, noise_std, size=agg_w.shape)
            agg_b = float(agg_b + rng.normal(0, noise_std))

        global_w += CONFIG["server_learning_rate"] * agg_w
        global_b += CONFIG["server_learning_rate"] * agg_b

        val_prob = predict_probability(global_w, global_b, X_val)
        threshold = tune_threshold(y_val, val_prob)
        val_metrics = evaluate_predictions(y_val, val_prob, threshold)
        history.append(
            {
                "round": round_num,
                "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "mean_update_norm": float(np.mean(norms)),
                "attacked": bool(attacked_this_round),
            }
        )

    return global_w, global_b, history


def evaluate_model(
    weights: np.ndarray,
    bias: float,
    banks: List[BankDataset],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    X_val, y_val = combine_split(banks, "val")
    X_test, y_test = combine_split(banks, "test")

    val_prob = predict_probability(weights, bias, X_val)
    threshold = tune_threshold(y_val, val_prob)

    test_prob = predict_probability(weights, bias, X_test)
    overall = evaluate_predictions(y_test, test_prob, threshold)

    per_bank = {}
    for bank in banks:
        bank_prob = predict_probability(weights, bias, bank.X_test)
        per_bank[bank.name] = evaluate_predictions(bank.y_test, bank_prob, threshold)
    return overall, per_bank


def evaluate_local_only(
    banks: List[BankDataset],
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    per_bank = {}
    metric_rows = []

    for bank in banks:
        weights, bias = fit_logistic(
            bank.X_train,
            bank.y_train,
            epochs=80,
            learning_rate=CONFIG["learning_rate"],
            rng=rng,
        )
        val_prob = predict_probability(weights, bias, bank.X_val)
        threshold = tune_threshold(bank.y_val, val_prob)
        test_prob = predict_probability(weights, bias, bank.X_test)
        metrics = evaluate_predictions(bank.y_test, test_prob, threshold)
        per_bank[bank.name] = metrics
        metric_rows.append(metrics)

    overall = {
        metric: float(np.nanmean([row[metric] for row in metric_rows]))
        for metric in [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "threshold",
        ]
    }
    return overall, per_bank


def oversample_minority(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    target_positive_ratio: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)
    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return X, y

    target_positive_count = int(
        target_positive_ratio * len(negative_idx) / max(1 - target_positive_ratio, 1e-6)
    )
    extra_count = max(0, target_positive_count - len(positive_idx))
    sampled_extra = rng.choice(positive_idx, size=extra_count, replace=True)
    balanced_idx = np.concatenate([negative_idx, positive_idx, sampled_extra])
    rng.shuffle(balanced_idx)
    return X[balanced_idx], y[balanced_idx]


def evaluate_probability_model(
    model,
    banks: List[BankDataset],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    X_val, y_val = combine_split(banks, "val")
    X_test, y_test = combine_split(banks, "test")

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]
    overall = evaluate_predictions(y_test, test_prob, threshold)

    per_bank = {}
    for bank in banks:
        bank_prob = model.predict_proba(bank.X_test)[:, 1]
        per_bank[bank.name] = evaluate_predictions(bank.y_test, bank_prob, threshold)
    return overall, per_bank


def train_traditional_ml_baselines(
    banks: List[BankDataset],
    rng: np.random.Generator,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, object],
]:
    X_train, y_train = combine_split(banks, "train")
    X_balanced, y_balanced = oversample_minority(X_train, y_train, rng)

    models = {
        "ml_lr": SklearnLogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=CONFIG["random_seed"],
        ),
        "ml_rf": RandomForestClassifier(
            n_estimators=240,
            max_depth=9,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=CONFIG["random_seed"],
            n_jobs=-1,
        ),
        "ml_mlp": MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            alpha=1e-4,
            max_iter=450,
            early_stopping=True,
            random_state=CONFIG["random_seed"],
        ),
    }

    metrics = {}
    per_bank_results = {}
    fitted_models = {}
    for name, model in models.items():
        if name == "ml_mlp":
            model.fit(X_balanced, y_balanced)
        else:
            model.fit(X_train, y_train)
        overall, per_bank = evaluate_probability_model(model, banks)
        metrics[name] = make_model_summary(overall)
        per_bank_results[name] = per_bank
        fitted_models[name] = model

    return metrics, per_bank_results, fitted_models


def make_model_summary(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "auc": metrics["roc_auc"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "threshold": metrics["threshold"],
    }


def plot_model_comparison(model_comparison: Dict[str, Dict[str, float]]) -> None:
    names = list(model_comparison.keys())
    pr_auc = [model_comparison[name]["pr_auc"] for name in names]
    f1 = [model_comparison[name]["f1"] for name in names]

    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, pr_auc, width, label="PR-AUC", color="#2F80ED")
    plt.bar(x + width / 2, f1, width, label="F1", color="#27AE60")
    plt.xticks(x, [name.replace("_", "\n") for name in names], fontsize=9)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Federated Fraud Detection: Imbalanced-Data Model Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "model_comparison.png", dpi=160)
    plt.close()


def plot_privacy_budget() -> None:
    rounds = np.arange(1, CONFIG["federated_rounds"] + 1)
    eps_per_round = approximate_epsilon_per_round(
        CONFIG["noise_multiplier"], CONFIG["delta"]
    )
    epsilon = eps_per_round * rounds
    plt.figure(figsize=(9, 4.8))
    plt.plot(rounds, epsilon, marker="o", color="#9B51E0", linewidth=2)
    plt.xlabel("Federated Round")
    plt.ylabel("Approximate cumulative epsilon")
    plt.title("Privacy Budget Tracking")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "privacy_budget.png", dpi=160)
    plt.close()


def plot_per_bank_f1(per_bank_metrics: Dict[str, Dict[str, float]]) -> None:
    names = list(per_bank_metrics.keys())
    f1_scores = [per_bank_metrics[name]["f1"] for name in names]
    recalls = [per_bank_metrics[name]["recall"] for name in names]

    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, f1_scores, width, label="F1", color="#EB5757")
    plt.bar(x + width / 2, recalls, width, label="Recall", color="#F2C94C")
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Bank Test Performance")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "per_bank_f1.png", dpi=160)
    plt.close()


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def main() -> None:
    rng = np.random.default_rng(CONFIG["random_seed"])
    banks, scaler, feature_columns, dataset_metadata = prepare_datasets()
    X_train, y_train = combine_split(banks, "train")

    print("=== Federated UPI Fraud Detection ===")
    print(f"Samples: {dataset_metadata['total_samples']}")
    print(f"Fraud rate: {dataset_metadata['fraud_rate']:.3%}")
    print(f"Features: {len(feature_columns)}")

    trained_models: Dict[str, object] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}
    per_bank_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    training_history: Dict[str, List[Dict[str, float]]] = {}

    print("\nTraining centralized baseline...")
    central_w, central_b = fit_logistic(
        X_train,
        y_train,
        epochs=90,
        learning_rate=CONFIG["learning_rate"],
        rng=rng,
    )
    central_metrics, central_per_bank = evaluate_model(central_w, central_b, banks)
    trained_models["centralized"] = (central_w, central_b)
    model_metrics["centralized"] = make_model_summary(central_metrics)
    per_bank_results["centralized"] = central_per_bank

    print("Training local-only baselines...")
    local_metrics, local_per_bank = evaluate_local_only(banks, rng)
    model_metrics["local_only_avg"] = make_model_summary(local_metrics)
    per_bank_results["local_only_avg"] = local_per_bank

    print("Training traditional ML baselines (LR/RF/MLP)...")
    ml_metrics, ml_per_bank, ml_models = train_traditional_ml_baselines(banks, rng)
    model_metrics.update(ml_metrics)
    per_bank_results.update(ml_per_bank)
    trained_models.update(ml_models)

    experiments = [
        ("fedavg_mean", "mean", False, None),
        ("dp_fedavg_mean", "mean", True, None),
        ("robust_dp_median", "median", True, None),
        ("dp_mean_attack", "mean", True, CONFIG["attack_client"]),
        ("robust_dp_median_attack", "median", True, CONFIG["attack_client"]),
    ]

    for name, aggregation, use_dp, attack_client in experiments:
        attack_note = f", attack={attack_client}" if attack_client else ""
        print(f"Training {name} (aggregation={aggregation}, dp={use_dp}{attack_note})...")
        weights, bias, history = train_federated(
            banks,
            aggregation=aggregation,
            use_dp=use_dp,
            rng=rng,
            attack_client=attack_client,
        )
        overall, per_bank = evaluate_model(weights, bias, banks)
        trained_models[name] = (weights, bias)
        model_metrics[name] = make_model_summary(overall)
        per_bank_results[name] = per_bank
        training_history[name] = history

    best_overall_model = max(
        model_metrics.keys(),
        key=lambda key: model_metrics[key]["auc"],
    )
    classical_candidates = [name for name in ["ml_lr", "ml_rf", "ml_mlp"] if name in model_metrics]
    best_classical_model = max(
        classical_candidates,
        key=lambda key: model_metrics[key]["auc"],
    )
    federated_candidates = ["fedavg_mean", "dp_fedavg_mean", "robust_dp_median"]
    best_federated_model = max(
        federated_candidates,
        key=lambda key: model_metrics[key]["auc"],
    )
    private_candidates = ["dp_fedavg_mean", "robust_dp_median"]
    recommended_private_model = max(
        private_candidates,
        key=lambda key: (model_metrics[key]["f1"], model_metrics[key]["pr_auc"]),
    )
    print(f"\nBest overall by AUC: {best_overall_model}")
    print(f"Recommended privacy-preserving model: {recommended_private_model}")
    print(
        "Test PR-AUC={:.4f}, F1={:.4f}, Recall={:.4f}".format(
            model_metrics[recommended_private_model]["pr_auc"],
            model_metrics[recommended_private_model]["f1"],
            model_metrics[recommended_private_model]["recall"],
        )
    )

    eps_per_round = approximate_epsilon_per_round(
        CONFIG["noise_multiplier"], CONFIG["delta"]
    )
    privacy_budget = {
        "accountant": "basic Gaussian composition approximation",
        "delta": CONFIG["delta"],
        "noise_multiplier": CONFIG["noise_multiplier"],
        "clip_norm": CONFIG["clip_norm"],
        "epsilon_per_round": eps_per_round,
        "total_epsilon_spent": eps_per_round * CONFIG["federated_rounds"],
        "max_rounds": CONFIG["federated_rounds"],
    }

    output = {
        "configuration": CONFIG,
        "dataset": dataset_metadata,
        "model_comparison": model_metrics,
        "best_model": recommended_private_model,
        "active_prediction_model": best_overall_model,
        "best_overall_model": best_overall_model,
        "best_classical_model": best_classical_model,
        "best_federated_model": best_federated_model,
        "recommended_private_model": recommended_private_model,
        "best_model_note": "best_model is the recommended privacy-preserving DP federated model, selected by F1 then PR-AUC among non-attack DP models. It is not necessarily the highest-AUC model overall.",
        "privacy_budget": privacy_budget,
        "per_bank_evaluation": per_bank_results[recommended_private_model],
        "robustness": {
            "attack_client": CONFIG["attack_client"],
            "attack_rounds": [
                CONFIG["attack_round_start"],
                CONFIG["attack_round_end"],
            ],
            "attack_scale": CONFIG["attack_scale"],
            "mean_under_attack": model_metrics["dp_mean_attack"],
            "median_under_attack": model_metrics["robust_dp_median_attack"],
        },
        "training_history": training_history,
        "unique_features": [
            "Traditional ML baselines: LR, RF, and MLP",
            "Non-IID bank simulation",
            "FedAvg with held-out validation thresholding",
            "Differentially private clipped update aggregation",
            "Byzantine attack simulation",
            "Coordinate-wise median robust aggregation",
            "PR-AUC reporting for imbalanced fraud detection",
        ],
    }

    plot_model_comparison(model_metrics)
    plot_privacy_budget()
    plot_per_bank_f1(per_bank_results[best_overall_model])

    active_model = trained_models[best_overall_model]
    if isinstance(active_model, tuple):
        active_w, active_b = active_model
        model_artifact = {
            "model_type": "federated_logistic_regression",
            "active_model": best_overall_model,
            "recommended_private_model": recommended_private_model,
            "weights": active_w,
            "bias": float(active_b),
            "feature_columns": feature_columns,
            "threshold": model_metrics[best_overall_model]["threshold"],
            "metrics": model_metrics[best_overall_model],
        }
    else:
        model_artifact = {
            "model_type": "sklearn_probability_model",
            "active_model": best_overall_model,
            "recommended_private_model": recommended_private_model,
            "estimator": active_model,
            "feature_columns": feature_columns,
            "threshold": model_metrics[best_overall_model]["threshold"],
            "metrics": model_metrics[best_overall_model],
        }

    with open(BASE_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_artifact, f)
    with open(BASE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(BASE_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(output), f, indent=2)

    print("\nSaved:")
    print("  model.pkl")
    print("  scaler.pkl")
    print("  results.json")
    print("  model_comparison.png")
    print("  privacy_budget.png")
    print("  per_bank_f1.png")


if __name__ == "__main__":
    main()
