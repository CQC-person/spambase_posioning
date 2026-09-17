"""Evaluation metrics and reporting utilities for spam classification."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = ""
) -> Dict[str, Any]:
    """Calculate accuracy, spam precision, spam recall, spam F1, and confusion matrix.
    
    Positive class is 1 (Spam), negative class is 0 (Non-spam / Ham).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        "model": model_name,
        "accuracy": acc,
        "spam_precision": prec,
        "spam_recall": rec,
        "spam_f1": f1,
        "confusion_matrix": cm,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def evaluate_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Evaluate multiple trained models on the test set.
    
    Returns
    -------
    results_df : pd.DataFrame
        Table containing accuracy, spam precision, spam recall, spam F1.
    cms : Dict[str, np.ndarray]
        Mapping of model names to 2x2 confusion matrices.
    """
    rows = []
    cms = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test.to_numpy(), y_pred, model_name=name)
        cms[name] = metrics["confusion_matrix"]
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Spam Precision": metrics["spam_precision"],
            "Spam Recall": metrics["spam_recall"],
            "Spam F1": metrics["spam_f1"],
            "True Negatives (TN)": metrics["tn"],
            "False Positives (FP)": metrics["fp"],
            "False Negatives (FN)": metrics["fn"],
            "True Positives (TP)": metrics["tp"],
        })
        
    results_df = pd.DataFrame(rows)
    return results_df, cms
