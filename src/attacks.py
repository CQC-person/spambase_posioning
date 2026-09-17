"""Data poisoning attack implementations for classical machine learning classifiers."""

from typing import Tuple
import numpy as np
import pandas as pd


def random_label_flip(
    y: pd.Series,
    poison_rate: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.Series, np.ndarray]:
    """Perform random label flipping attack on training labels.
    
    Parameters
    ----------
    y : pd.Series
        Clean binary training labels (0: non-spam, 1: spam).
    poison_rate : float
        Fraction of total training samples to poison (e.g., 0.05 for 5%).
        Defined as: number of poisoned samples / total number of training samples.
    random_state : int
        RNG seed for reproducibility.
        
    Returns
    -------
    y_poisoned : pd.Series
        The modified training labels with selected indices flipped (0->1, 1->0).
    flipped_mask : np.ndarray
        Boolean array of length len(y) where True indicates the label was flipped.
    """
    if not (0.0 <= poison_rate <= 1.0):
        raise ValueError(f"poison_rate must be between 0.0 and 1.0, got {poison_rate}")
    
    n_total = len(y)
    n_poison = int(np.round(n_total * poison_rate))
    
    rng = np.random.RandomState(random_state)
    
    # Select indices to flip without replacement from the training set
    flip_indices = rng.choice(n_total, size=n_poison, replace=False)
    
    flipped_mask = np.zeros(n_total, dtype=bool)
    flipped_mask[flip_indices] = True
    
    y_poisoned = y.copy().to_numpy()
    # Invert binary labels: 0 -> 1, 1 -> 0 for selected indices
    y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
    
    y_poisoned_series = pd.Series(y_poisoned, index=y.index, name=y.name)
    return y_poisoned_series, flipped_mask
