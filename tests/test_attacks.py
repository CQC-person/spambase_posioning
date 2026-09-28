"""Unit tests for data poisoning attack implementations in src/attacks.py."""

import pytest
import numpy as np
import pandas as pd
from src.attacks import random_label_flip, targeted_spam_to_ham
from src.data import load_spambase, split_data


@pytest.fixture
def spambase_split():
    """Load spambase dataset and return stratified 80/20 train/test split."""
    df = load_spambase()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def synthetic_labels():
    """Synthetic binary labels: 60 ham (0) and 40 spam (1)."""
    labels = np.array([0] * 60 + [1] * 40)
    return pd.Series(labels, name="target")


# ==============================================================================
# 1. Exact Poison Counts
# ==============================================================================

@pytest.mark.parametrize("poison_rate", [0.0, 0.01, 0.05, 0.10, 0.20])
def test_random_flip_exact_counts(spambase_split, poison_rate):
    """Test that random_label_flip flips exactly int(round(N * poison_rate)) labels."""
    _, _, y_train, _ = spambase_split
    n_total = len(y_train)
    expected_count = int(np.round(n_total * poison_rate))

    y_poisoned, mask = random_label_flip(y_train, poison_rate=poison_rate, random_state=42)

    assert mask.sum() == expected_count, f"Mask sum {mask.sum()} != expected {expected_count}"
    assert (y_poisoned != y_train).sum() == expected_count, "Changed labels != expected count"
    assert len(y_poisoned) == n_total, "Length of poisoned labels altered"


@pytest.mark.parametrize("poison_rate", [0.0, 0.01, 0.05, 0.10, 0.20])
def test_targeted_flip_exact_counts(spambase_split, poison_rate):
    """Test that targeted_spam_to_ham flips exactly int(round(N * poison_rate)) labels."""
    _, _, y_train, _ = spambase_split
    n_total = len(y_train)
    expected_count = int(np.round(n_total * poison_rate))

    y_poisoned, mask = targeted_spam_to_ham(y_train, poison_rate=poison_rate, random_state=42)

    assert mask.sum() == expected_count, f"Mask sum {mask.sum()} != expected {expected_count}"
    assert (y_poisoned != y_train).sum() == expected_count, "Changed labels != expected count"
    assert len(y_poisoned) == n_total, "Length of poisoned labels altered"


# ==============================================================================
# 2. Correct Label Direction
# ==============================================================================

def test_random_flip_bidirectional(spambase_split):
    """Verify that random flipping modifies labels in both directions (0->1 and 1->0)."""
    _, _, y_train, _ = spambase_split
    y_poisoned, mask = random_label_flip(y_train, poison_rate=0.05, random_state=42)

    ham_to_spam = ((y_train == 0) & (y_poisoned == 1)).sum()
    spam_to_ham = ((y_train == 1) & (y_poisoned == 0)).sum()

    assert ham_to_spam > 0, "Random flip should change some ham (0) to spam (1)"
    assert spam_to_ham > 0, "Random flip should change some spam (1) to ham (0)"
    assert ham_to_spam + spam_to_ham == mask.sum(), "Sum of flips must equal total flipped mask"


def test_targeted_flip_strictly_spam_to_ham(spambase_split):
    """Verify that targeted attack flips ONLY spam (1 -> 0) and NEVER ham (0 -> 1)."""
    _, _, y_train, _ = spambase_split
    y_poisoned, mask = targeted_spam_to_ham(y_train, poison_rate=0.10, random_state=42)

    # 1. Non-spam samples (0) must never be flipped
    ham_mask = (y_train == 0)
    assert mask[ham_mask].sum() == 0, "Targeted attack modified legitimate non-spam samples!"
    assert np.all(y_poisoned[ham_mask] == 0), "Some ham samples changed value in targeted attack"

    # 2. All flipped samples must have been originally spam (1) and are now ham (0)
    flipped_original = y_train[mask]
    flipped_poisoned = y_poisoned[mask]
    assert np.all(flipped_original == 1), "All flipped samples must have original label 1"
    assert np.all(flipped_poisoned == 0), "All flipped samples must have poisoned label 0"


# ==============================================================================
# 3. Reproducibility
# ==============================================================================

def test_attack_reproducibility(synthetic_labels):
    """Verify identical results with identical seed, and different results with different seeds."""
    # Test Random Attack
    y_p1, m1 = random_label_flip(synthetic_labels, poison_rate=0.10, random_state=123)
    y_p2, m2 = random_label_flip(synthetic_labels, poison_rate=0.10, random_state=123)
    y_p3, m3 = random_label_flip(synthetic_labels, poison_rate=0.10, random_state=999)

    assert np.array_equal(y_p1, y_p2), "Identical seed must produce identical poisoned labels"
    assert np.array_equal(m1, m2), "Identical seed must produce identical flipped mask"
    assert not np.array_equal(m1, m3), "Different seeds should produce different selections"

    # Test Targeted Attack
    y_t1, mt1 = targeted_spam_to_ham(synthetic_labels, poison_rate=0.10, random_state=123)
    y_t2, mt2 = targeted_spam_to_ham(synthetic_labels, poison_rate=0.10, random_state=123)
    y_t3, mt3 = targeted_spam_to_ham(synthetic_labels, poison_rate=0.10, random_state=999)

    assert np.array_equal(y_t1, y_t2), "Identical seed must produce identical targeted labels"
    assert np.array_equal(mt1, mt2), "Identical seed must produce identical targeted mask"
    assert not np.array_equal(mt1, mt3), "Different seeds should produce different targeted selections"


# ==============================================================================
# 4. Test-Set Isolation & Immutability
# ==============================================================================

def test_test_set_isolation(spambase_split):
    """Verify that poisoning operations do not modify test set or original training labels."""
    X_train, X_test, y_train, y_test = spambase_split

    test_labels_hash_before = pd.util.hash_pandas_object(y_test).sum()
    train_labels_hash_before = pd.util.hash_pandas_object(y_train).sum()

    # Execute both attacks
    _ = random_label_flip(y_train, poison_rate=0.20, random_state=42)
    _ = targeted_spam_to_ham(y_train, poison_rate=0.20, random_state=42)

    test_labels_hash_after = pd.util.hash_pandas_object(y_test).sum()
    train_labels_hash_after = pd.util.hash_pandas_object(y_train).sum()

    assert test_labels_hash_before == test_labels_hash_after, "Test set labels were corrupted!"
    assert train_labels_hash_before == train_labels_hash_after, "Input y_train was modified in-place!"


# ==============================================================================
# 5. Edge Cases & Exception Handling
# ==============================================================================

def test_invalid_poison_rates(synthetic_labels):
    """Verify ValueError is raised for invalid poison rates outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="poison_rate must be between 0.0 and 1.0"):
        random_label_flip(synthetic_labels, poison_rate=-0.05)

    with pytest.raises(ValueError, match="poison_rate must be between 0.0 and 1.0"):
        targeted_spam_to_ham(synthetic_labels, poison_rate=1.05)


def test_targeted_exceeds_available_spam(synthetic_labels):
    """Verify ValueError when targeted poison count exceeds available spam samples."""
    # synthetic_labels has 40 spam out of 100 total (40%). Requesting 50% = 50 samples > 40 spam.
    with pytest.raises(ValueError, match="Requested 50 poisoned samples.*only 40 spam samples exist"):
        targeted_spam_to_ham(synthetic_labels, poison_rate=0.50)


def test_numpy_array_input():
    """Verify that both functions correctly support numpy arrays as input."""
    y_np = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])

    y_rand, mask_rand = random_label_flip(y_np, poison_rate=0.2, random_state=42)
    assert isinstance(y_rand, np.ndarray), "Output should be np.ndarray when input is np.ndarray"
    assert mask_rand.sum() == 2

    y_targ, mask_targ = targeted_spam_to_ham(y_np, poison_rate=0.2, random_state=42)
    assert isinstance(y_targ, np.ndarray), "Output should be np.ndarray when input is np.ndarray"
    assert mask_targ.sum() == 2
