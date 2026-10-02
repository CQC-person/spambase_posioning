"""Experiment execution and batch runner for multi-seed data poisoning pilot study."""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.data import load_spambase, split_data
from src.attacks import random_label_flip, targeted_spam_to_ham
from src.models import get_models
from src.metrics import evaluate_predictions


def run_single_condition(
    df: pd.DataFrame,
    seed: int,
    attack: str,
    poison_rate: float
) -> List[Dict[str, any]]:
    """Execute a single experimental condition across all three classifiers.

    Ensures that for a given (seed, attack, poison_rate) condition, all three models:
    1. Receive the exact same train/test split.
    2. Receive the exact same poisoned training labels.
    3. Are evaluated on the strictly clean, isolated test set.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Spambase dataset.
    seed : int
        Experiment random seed (used for split, attack, and model initialization).
    attack : str
        Attack type: 'clean', 'random', or 'targeted'.
    poison_rate : float
        Fraction of training samples to poison (0.0 for clean baseline).

    Returns
    -------
    List[Dict[str, any]]
        Three result dictionaries (one per model) containing performance metrics.
    """
    # 1. Stratified 80/20 train-test split seeded with the experiment seed
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=seed)

    # 2. Apply poisoning attack to training labels
    if poison_rate == 0.0 or attack == "clean":
        attack_name = "clean"
        y_train_poisoned = y_train.copy()
        n_poisoned = 0
    elif attack == "random":
        attack_name = "random"
        y_train_poisoned, flipped_mask = random_label_flip(
            y_train, poison_rate=poison_rate, random_state=seed
        )
        n_poisoned = int(flipped_mask.sum())
    elif attack == "targeted":
        attack_name = "targeted"
        y_train_poisoned, flipped_mask = targeted_spam_to_ham(
            y_train, poison_rate=poison_rate, random_state=seed
        )
        n_poisoned = int(flipped_mask.sum())
    else:
        raise ValueError(f"Unknown attack type: '{attack}'. Supported: 'clean', 'random', 'targeted'.")

    # 3. Instantiate the three models using the exact same experiment seed
    models = get_models(random_state=seed)
    condition_results = []

    # 4. Train each model on the identical poisoned training set and evaluate on clean test set
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train_poisoned)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test.to_numpy(), y_pred, model_name=model_name)

            row = {
                "seed": seed,
                "attack": attack_name,
                "poison_rate": poison_rate,
                "model": model_name,
                "n_poisoned": n_poisoned,
                "accuracy": metrics["accuracy"],
                "spam_precision": metrics["spam_precision"],
                "spam_recall": metrics["spam_recall"],
                "spam_f1": metrics["spam_f1"],
                "status": "success",
                "error_message": ""
            }
        except Exception as e:
            # Explicitly log failures instead of silently deleting them
            row = {
                "seed": seed,
                "attack": attack_name,
                "poison_rate": poison_rate,
                "model": model_name,
                "n_poisoned": n_poisoned,
                "accuracy": np.nan,
                "spam_precision": np.nan,
                "spam_recall": np.nan,
                "spam_f1": np.nan,
                "status": "failed",
                "error_message": str(e)
            }
        condition_results.append(row)

    return condition_results


def run_pilot_study(
    data_path: str = "data/spambase.data",
    names_path: str = "data/spambase.names",
    seeds: Optional[List[int]] = None,
    poison_rates: Optional[List[float]] = None,
    attacks: Optional[List[str]] = None,
    output_csv: Optional[str] = "results/stage3_pilot_results.csv"
) -> pd.DataFrame:
    """Run the complete three-seed pilot study generating exactly 81 result rows.

    Structure:
    - 9 clean-baseline rows: 3 seeds x 1 rate (0.0) x 3 models
    - 72 poisoned-condition rows: 3 seeds x 2 attacks x 4 non-zero rates x 3 models

    Returns
    -------
    pd.DataFrame
        Complete results DataFrame containing all 81 experiment executions.
    """
    if seeds is None:
        seeds = [0, 1, 2]
    if poison_rates is None:
        poison_rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    if attacks is None:
        attacks = ["random", "targeted"]

    df = load_spambase(data_path=data_path, names_path=names_path)
    all_rows = []

    print(f"Starting Stage 3 Pilot Study across seeds={seeds}, attacks={attacks}, rates={poison_rates}...")

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        # 1. Clean Baseline (0% poisoning) -> 3 model rows
        clean_rows = run_single_condition(df, seed=seed, attack="clean", poison_rate=0.0)
        all_rows.extend(clean_rows)
        print(f"  [Seed {seed}] Clean Baseline (0%): 3 models recorded.")

        # 2. Poisoned Conditions (1%, 5%, 10%, 20%) -> 2 attacks x 4 rates x 3 models = 24 rows per seed
        non_zero_rates = [r for r in poison_rates if r > 0.0]
        for attack in attacks:
            for rate in non_zero_rates:
                rows = run_single_condition(df, seed=seed, attack=attack, poison_rate=rate)
                all_rows.extend(rows)
                print(f"  [Seed {seed}] Attack '{attack}' @ {rate*100:.0f}%: 3 models recorded.")

    results_df = pd.DataFrame(all_rows)

    # Verification of Invariants
    n_clean = len(results_df[results_df["attack"] == "clean"])
    n_poisoned = len(results_df[results_df["attack"] != "clean"])
    total_rows = len(results_df)

    print("\n=== Pilot Execution Verification ===")
    print(f"Clean-baseline rows:    {n_clean} (expected: 9)")
    print(f"Poisoned-condition rows: {n_poisoned} (expected: 72)")
    print(f"Total results rows:     {total_rows} (expected: 81)")

    assert total_rows == 81, f"Expected 81 rows, got {total_rows}"
    assert n_clean == 9, f"Expected 9 clean rows, got {n_clean}"
    assert n_poisoned == 72, f"Expected 72 poisoned rows, got {n_poisoned}"
    assert results_df["accuracy"].notnull().all(), "Found failed condition or NaN in metrics!"

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        # Select required columns for deliverables
        export_cols = [
            "seed", "attack", "poison_rate", "model",
            "n_poisoned", "accuracy", "spam_precision", "spam_recall", "spam_f1"
        ]
        results_df[export_cols].to_csv(output_csv, index=False)
        print(f"\nSaved verified pilot results to: {output_csv}")

    return results_df
