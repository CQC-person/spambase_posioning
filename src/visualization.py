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
    plot_df = df.copy()
    plot_df["Email Class"] = plot_df["is_spam"].map({0: "Non-Spam (0)", 1: "Spam (1)"})
    sns.boxplot(
        x="Email Class",
        y=feature_name,
        hue="Email Class",
        data=plot_df,
        palette={"Non-Spam (0)": "#2b5c8f", "Spam (1)": "#d95f02"},
        ax=ax,
        showfliers=False,
        legend=False
    )
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


def plot_attack_class_distributions(
    random_stats_df: pd.DataFrame,
    targeted_stats_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot class distribution shifts across poisoning rates for Random vs Targeted attacks.

    Parameters
    ----------
    random_stats_df : pd.DataFrame
        DataFrame with columns ['Poison Rate', 'Spam Count', 'Ham Count', 'Spam Pct', 'Ham Pct'].
    targeted_stats_df : pd.DataFrame
        DataFrame with columns ['Poison Rate', 'Spam Count', 'Ham Count', 'Spam Pct', 'Ham Pct'].
    save_path : Optional[str]
        Path to save the generated figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rates = random_stats_df["Poison Rate"].tolist()
    x = np.arange(len(rates))
    width = 0.35

    # Subplot 1: Apparent Spam Percentage
    ax1.plot(
        rates, random_stats_df["Spam Pct"],
        marker="o", linewidth=2.5, markersize=8, color="#2b5c8f",
        label="Random Flipping (Indiscriminate / Availability)"
    )
    ax1.plot(
        rates, targeted_stats_df["Spam Pct"],
        marker="s", linewidth=2.5, markersize=8, color="#d95f02",
        label="Targeted Spam->Ham (Evasion / Integrity)"
    )

    for i, rate in enumerate(rates):
        r_pct = random_stats_df.loc[i, "Spam Pct"]
        t_pct = targeted_stats_df.loc[i, "Spam Pct"]
        ax1.annotate(f"{r_pct:.1f}%", (rate, r_pct), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color="#2b5c8f")
        ax1.annotate(f"{t_pct:.1f}%", (rate, t_pct), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#d95f02")

    ax1.set_title("Training Set Apparent Spam Proportion vs. Poison Rate", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Poisoning Rate", fontsize=11)
    ax1.set_ylabel("Apparent Spam Percentage (%)", fontsize=11)
    ax1.set_ylim(10, 50)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper right")

    # Subplot 2: Stacked Counts Comparison for Targeted Attack
    ax2.bar(x - width/2, targeted_stats_df["Ham Count"], width, label="Ham (Non-Spam)", color="#2b5c8f", edgecolor="black")
    ax2.bar(x + width/2, targeted_stats_df["Spam Count"], width, label="Spam (Remaining)", color="#d95f02", edgecolor="black")

    for i in range(len(rates)):
        h_cnt = targeted_stats_df.loc[i, "Ham Count"]
        s_cnt = targeted_stats_df.loc[i, "Spam Count"]
        ax2.annotate(f"{h_cnt}", (x[i] - width/2, h_cnt), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5)
        ax2.annotate(f"{s_cnt}", (x[i] + width/2, s_cnt), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold", color="#d95f02")

    ax2.set_title("Targeted Attack: Class Counts Breakdown Across Poison Rates", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(rates, fontsize=10)
    ax2.set_xlabel("Poisoning Rate", fontsize=11)
    ax2.set_ylabel("Number of Samples in Training Set", fontsize=11)
    ax2.set_ylim(0, 3500)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, (ax1, ax2)


def plot_pilot_learning_curves(
    results_df: pd.DataFrame,
    metric: str = "spam_recall",
    metric_display_name: str = "Spam Recall",
    save_path: Optional[str] = None
):
    """Plot multi-seed mean and standard deviation curves comparing Random vs Targeted attacks.

    Parameters
    ----------
    results_df : pd.DataFrame
        81-row pilot results DataFrame.
    metric : str
        Column name to plot ('spam_recall', 'spam_f1', 'spam_precision', 'accuracy').
    metric_display_name : str
        Display title for the metric.
    save_path : Optional[str]
        File path to save the generated plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    clean_sub = results_df[results_df["attack"] == "clean"].copy()

    random_sub = results_df[results_df["attack"] == "random"].copy()
    clean_for_random = clean_sub.copy()
    clean_for_random["attack"] = "random"
    random_full = pd.concat([clean_for_random, random_sub], ignore_index=True)

    targeted_sub = results_df[results_df["attack"] == "targeted"].copy()
    clean_for_targeted = clean_sub.copy()
    clean_for_targeted["attack"] = "targeted"
    targeted_full = pd.concat([clean_for_targeted, targeted_sub], ignore_index=True)

    model_styles = {
        "Logistic Regression": {"color": "#2b5c8f", "marker": "o", "ls": "-"},
        "Decision Tree": {"color": "#7570b3", "marker": "^", "ls": "--"},
        "Random Forest": {"color": "#1b9e77", "marker": "s", "ls": "-."}
    }

    rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    rate_labels = ["0%", "1%", "5%", "10%", "20%"]

    for ax, data_subset, attack_title in [
        (ax1, random_full, "Random Label Flipping (Availability Threat)"),
        (ax2, targeted_full, "Targeted Spam->Ham Flipping (Integrity Threat)")
    ]:
        for model_name, style in model_styles.items():
            m_data = data_subset[data_subset["model"] == model_name]
            stats = m_data.groupby("poison_rate")[metric].agg(["mean", "std"]).reindex(rates)

            means = stats["mean"].values
            stds = stats["std"].fillna(0.0).values

            ax.plot(
                rates, means,
                label=model_name,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                linewidth=2.2,
                markersize=7
            )
            ax.fill_between(
                rates,
                np.maximum(0, means - stds),
                np.minimum(1.0, means + stds),
                color=style["color"],
                alpha=0.15
            )

        ax.set_title(attack_title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(rates)
        ax.set_xticklabels(rate_labels, fontsize=10)
        ax.set_xlabel("Poisoning Rate", fontsize=11)
        ax.set_ylabel(metric_display_name, fontsize=11)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, loc="lower left" if metric == "spam_recall" else "best")

    plt.suptitle(f"Pilot Study (Seeds 0, 1, 2): {metric_display_name} Mean ± 1 Std across Poisoning Rates", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, (ax1, ax2)


def plot_pilot_multi_metric_grid(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot 2x2 grid of performance metrics (Spam Recall, Spam F1, Spam Precision, Accuracy)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    metrics_info = [
        ("spam_recall", "Spam Recall (Primary Metric)", axes[0, 0]),
        ("spam_f1", "Spam F1-Score (Primary Metric)", axes[0, 1]),
        ("spam_precision", "Spam Precision (Supporting Metric)", axes[1, 0]),
        ("accuracy", "Overall Accuracy (Supporting Metric)", axes[1, 1])
    ]

    clean_sub = results_df[results_df["attack"] == "clean"].copy()
    targeted_sub = results_df[results_df["attack"] == "targeted"].copy()
    clean_for_targeted = clean_sub.copy()
    clean_for_targeted["attack"] = "targeted"
    targeted_full = pd.concat([clean_for_targeted, targeted_sub], ignore_index=True)

    random_sub = results_df[results_df["attack"] == "random"].copy()
    clean_for_random = clean_sub.copy()
    clean_for_random["attack"] = "random"
    random_full = pd.concat([clean_for_random, random_sub], ignore_index=True)

    rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    rate_labels = ["0%", "1%", "5%", "10%", "20%"]

    model_colors = {
        "Logistic Regression": "#2b5c8f",
        "Decision Tree": "#7570b3",
        "Random Forest": "#1b9e77"
    }

    for metric_col, title, ax in metrics_info:
        for model_name, color in model_colors.items():
            # Solid line for Targeted, Dashed line for Random
            t_data = targeted_full[targeted_full["model"] == model_name]
            t_stats = t_data.groupby("poison_rate")[metric_col].agg(["mean"]).reindex(rates)
            ax.plot(
                rates, t_stats["mean"].values,
                color=color, linestyle="-", marker="o", linewidth=2.0, markersize=6,
                label=f"{model_name} (Targeted)"
            )

            r_data = random_full[random_full["model"] == model_name]
            r_stats = r_data.groupby("poison_rate")[metric_col].agg(["mean"]).reindex(rates)
            ax.plot(
                rates, r_stats["mean"].values,
                color=color, linestyle=":", marker="x", linewidth=1.5, markersize=5, alpha=0.75,
                label=f"{model_name} (Random)"
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(rates)
        ax.set_xticklabels(rate_labels, fontsize=10)
        ax.set_xlabel("Poisoning Rate", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8, loc="lower left" if "recall" in metric_col else "best")

    plt.suptitle("Multi-Metric Comparison: Targeted (Solid) vs Random (Dotted) Across Models (Seeds 0, 1, 2 Mean)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, axes
