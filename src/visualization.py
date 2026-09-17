"""Visualization utilities for exploratory data analysis and model performance comparisons."""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_class_distribution(y: pd.Series, title: str = "Spambase Class Distribution", save_path: Optional[str] = None):
    """Plot bar chart showing class counts and percentages."""
    counts = y.value_counts().sort_index()
    labels = ["Non-Spam (0)", "Spam (1)"]
    percentages = (counts / len(y)) * 100

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts.values, color=["#2b5c8f", "#d95f02"], edgecolor="black", width=0.5)
    
    for bar, pct in zip(bars, percentages):
        height = int(bar.get_height())
        label_text = f"{height:,}\n({pct:.1f}%)"
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, ax


def plot_feature_comparison_box(df: pd.DataFrame, feature_name: str, title: Optional[str] = None, save_path: Optional[str] = None):
    """Plot boxplot comparing a numerical feature between Spam and Non-Spam."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(
        x="is_spam",
        y=feature_name,
        data=df,
        palette=["#2b5c8f", "#d95f02"],
        ax=ax,
        showfliers=False
    )
    ax.set_xticklabels(["Non-Spam (0)", "Spam (1)"], fontsize=10)
    ax.set_xlabel("Email Class", fontsize=11)
    ax.set_ylabel(f"{feature_name} (%)", fontsize=11)
    ax.set_title(title or f"Comparison of '{feature_name}' (Outliers Excluded)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, ax


def plot_feature_comparison_hist(df: pd.DataFrame, feature_name: str, log_scale: bool = True, title: Optional[str] = None, save_path: Optional[str] = None):
    """Plot log-scale distribution or histogram of a feature comparing Spam and Non-Spam."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    spam_data = df[df["is_spam"] == 1][feature_name]
    ham_data = df[df["is_spam"] == 0][feature_name]

    if log_scale:
        spam_data = np.log1p(spam_data)
        ham_data = np.log1p(ham_data)
        xlabel = f"log(1 + {feature_name})"
    else:
        xlabel = feature_name

    sns.kdeplot(ham_data, label="Non-Spam (0)", color="#2b5c8f", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(spam_data, label="Spam (1)", color="#d95f02", fill=True, alpha=0.3, ax=ax)

    ax.set_title(title or f"Density Distribution of '{feature_name}'", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, ax


def plot_confusion_matrices(cms: Dict[str, np.ndarray], title_prefix: str = "Clean Baseline", save_path: Optional[str] = None):
    """Plot 1x3 confusion matrices for Logistic Regression, Decision Tree, and Random Forest."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    model_names = list(cms.keys())

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = cms[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Ham (0)", "Pred Spam (1)"],
            yticklabels=["True Ham (0)", "True Spam (1)"],
            annot_kws={"size": 12, "weight": "bold"}
        )
        title_text = f"{name}\n({title_prefix})"
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, axes


def plot_clean_vs_poisoned_metrics(clean_df: pd.DataFrame, poisoned_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot 4 figures comparing Accuracy, Spam Precision, Spam Recall, and Spam F1 between Clean and 5% Poisoned."""
    metrics = [
        ("Accuracy", "Overall Accuracy"),
        ("Spam Precision", "Spam Precision (Class 1)"),
        ("Spam Recall", "Spam Recall (Class 1)"),
        ("Spam F1", "Spam F1-Score (Class 1)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    models = clean_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    for idx, (col_name, display_title) in enumerate(metrics):
        ax = axes[idx]
        clean_vals = clean_df[col_name].values
        poison_vals = poisoned_df[col_name].values
        deltas = poison_vals - clean_vals

        bars1 = ax.bar(x - width/2, clean_vals, width, label="Clean Baseline (0%)", color="#2b5c8f", edgecolor="black")
        bars2 = ax.bar(x + width/2, poison_vals, width, label="Poisoned (5% Random Flip)", color="#d95f02", edgecolor="black")

        for i, (c_bar, p_bar, delta) in enumerate(zip(bars1, bars2, deltas)):
            sign = "+" if delta >= 0 else ""
            delta_str = f"Δ: {sign}{delta:.3f}"
            ax.annotate(delta_str,
                        xy=(x[i], max(c_bar.get_height(), p_bar.get_height())),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                        color="#a50f15" if delta < 0 else "#006d2c")

        ax.set_title(display_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.5)
        ax.set_ylim(0.70, 1.05)
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.suptitle("Clean Baseline vs. 5% Random Label Flipping Attack Performance Comparison", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, axes
