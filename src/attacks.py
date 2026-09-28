"""Data poisoning attack implementations for classical machine learning classifiers.

This module provides two core data poisoning attacks against binary classification models:
1. Random Label Flipping: An indiscriminate poisoning attack where labels of randomly selected
   training instances are inverted (0 -> 1 and 1 -> 0).
2. Targeted Spam-to-Ham Label Flipping: An evasion-oriented poisoning attack where randomly
   selected spam instances (label 1) are relabeled as legitimate/non-spam (label 0).

Both functions adhere strictly to the project specification:
- Poison rate is defined as: (number of poisoned samples) / (total number of training samples).
- Sample selection is performed without replacement using NumPy's RandomState.
- The functions return the poisoned labels along with a boolean mask identifying changed samples.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def random_label_flip(
    y: Union[pd.Series, np.ndarray],
    poison_rate: float,
    random_state: int = 42
) -> Tuple[Union[pd.Series, np.ndarray], np.ndarray]:
    """Perform random (indiscriminate) label flipping attack on training labels.

    Randomly selects a subset of training labels without replacement and inverts them:
    samples with label 0 become 1, and samples with label 1 become 0.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Clean binary training labels (0: non-spam, 1: spam).
    poison_rate : float
        Fraction of total training samples to poison (0.0 to 1.0).
        Defined as: number of poisoned samples / total number of training samples.
    random_state : int, default=42
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    y_poisoned : pd.Series or np.ndarray
        The modified training labels with flipped values for selected instances.
        Preserves the original type and index if input was a pandas Series.
    flipped_mask : np.ndarray
        Boolean array of length len(y) where True indicates the label was flipped.

    Raises
    ------
    ValueError
        If poison_rate is not within [0.0, 1.0] or if y contains values other than 0 and 1.
    """
    if not (0.0 <= poison_rate <= 1.0):
        raise ValueError(f"poison_rate must be between 0.0 and 1.0, got {poison_rate}")

    is_series = isinstance(y, pd.Series)
    y_arr = y.to_numpy() if is_series else np.asarray(y)

    unique_vals = np.unique(y_arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Labels must be binary (0 or 1), found values: {unique_vals}")

    n_total = len(y_arr)
    n_poison = int(np.round(n_total * poison_rate))

    flipped_mask = np.zeros(n_total, dtype=bool)
    y_poisoned_arr = y_arr.copy()

    if n_poison > 0:
        rng = np.random.RandomState(random_state)
        # Select sample indices uniformly without replacement across the entire training set
        flip_indices = rng.choice(n_total, size=n_poison, replace=False)
        flipped_mask[flip_indices] = True
        # Invert binary labels: 0 -> 1 and 1 -> 0
        y_poisoned_arr[flip_indices] = 1 - y_poisoned_arr[flip_indices]

    if is_series:
        y_poisoned = pd.Series(y_poisoned_arr, index=y.index, name=y.name)
    else:
        y_poisoned = y_poisoned_arr

    return y_poisoned, flipped_mask


def targeted_spam_to_ham(
    y: Union[pd.Series, np.ndarray],
    poison_rate: float,
    random_state: int = 42
) -> Tuple[Union[pd.Series, np.ndarray], np.ndarray]:
    """Perform targeted spam-to-ham label flipping attack on training labels.

    Randomly selects a subset of spam instances (label 1) without replacement and
    relabels them as legitimate non-spam (label 0). Non-spam instances (label 0)
    are strictly preserved and never modified.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Clean binary training labels (0: non-spam, 1: spam).
    poison_rate : float
        Fraction of total training samples to poison (0.0 to 1.0).
        Defined as: number of poisoned samples / total number of training samples.
    random_state : int, default=42
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    y_poisoned : pd.Series or np.ndarray
        The modified training labels where selected spam samples are relabeled as 0.
        Preserves the original type and index if input was a pandas Series.
    flipped_mask : np.ndarray
        Boolean array of length len(y) where True indicates the label was flipped (1 -> 0).

    Raises
    ------
    ValueError
        If poison_rate is not within [0.0, 1.0], if labels are not binary (0 and 1),
        or if the requested number of poisoned samples exceeds available spam samples.
    """
    if not (0.0 <= poison_rate <= 1.0):
        raise ValueError(f"poison_rate must be between 0.0 and 1.0, got {poison_rate}")

    is_series = isinstance(y, pd.Series)
    y_arr = y.to_numpy() if is_series else np.asarray(y)

    unique_vals = np.unique(y_arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Labels must be binary (0 or 1), found values: {unique_vals}")

    n_total = len(y_arr)
    n_poison = int(np.round(n_total * poison_rate))

    # Identify all available spam candidates (label 1)
    spam_indices = np.where(y_arr == 1)[0]
    n_spam_available = len(spam_indices)

    if n_poison > n_spam_available:
        raise ValueError(
            f"Requested {n_poison} poisoned samples (poison_rate={poison_rate:.1%}), "
            f"but only {n_spam_available} spam samples exist in the training set."
        )

    flipped_mask = np.zeros(n_total, dtype=bool)
    y_poisoned_arr = y_arr.copy()

    if n_poison > 0:
        rng = np.random.RandomState(random_state)
        # Select spam indices without replacement
        selected_spam_indices = rng.choice(spam_indices, size=n_poison, replace=False)
        flipped_mask[selected_spam_indices] = True
        # Relabel selected spam samples as non-spam (1 -> 0)
        y_poisoned_arr[selected_spam_indices] = 0

    if is_series:
        y_poisoned = pd.Series(y_poisoned_arr, index=y.index, name=y.name)
    else:
        y_poisoned = y_poisoned_arr

    return y_poisoned, flipped_mask
