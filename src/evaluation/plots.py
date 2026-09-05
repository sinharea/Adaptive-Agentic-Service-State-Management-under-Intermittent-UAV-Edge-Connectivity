"""
Publication-quality visualization for experimental results.

Generates all 10 required plots in both PNG and PDF format.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# Consistent style
STRATEGY_COLORS = {
    "local_execution": "#636EFA",
    "threshold_based": "#EF553B",
    "mobility_aware": "#00CC96",
    "rl_based": "#AB63FA",
    "agentic_manager": "#FFA15A",
}

STRATEGY_LABELS = {
    "local_execution": "Local Execution",
    "threshold_based": "Threshold-Based",
    "mobility_aware": "Mobility-Aware",
    "rl_based": "RL-Based",
    "agentic_manager": "Agentic (Proposed)",
}

STRATEGY_MARKERS = {
    "local_execution": "o",
    "threshold_based": "s",
    "mobility_aware": "^",
    "rl_based": "D",
    "agentic_manager": "*",
}


def _setup_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
    })


def _save_fig(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"),
                bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(agg_df: pd.DataFrame, metric: str,
                        title: str, ylabel: str,
                        output_dir: str, fig_name: str) -> None:
    """Bar chart comparing strategies on a single metric."""
    _setup_style()
    fig, ax = plt.subplots()

    strategies = agg_df["strategy"].values
    means = agg_df[f"{metric}_mean"].values
    stds = agg_df[f"{metric}_std"].values
    colors = [STRATEGY_COLORS.get(s, "#999") for s in strategies]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    bars = ax.bar(range(len(strategies)), means, yerr=stds,
                  color=colors, capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _save_fig(fig, output_dir, fig_name)


def plot_metric_vs_variable(results_df: pd.DataFrame,
                            variable: str, metric: str,
                            title: str, xlabel: str, ylabel: str,
                            output_dir: str, fig_name: str) -> None:
    """Line chart: metric vs experimental variable, one line per strategy."""
    _setup_style()
    fig, ax = plt.subplots()

    for strategy in results_df["strategy"].unique():
        sdf = results_df[results_df["strategy"] == strategy]
        grouped = sdf.groupby(variable)[metric].agg(["mean", "std"]).reset_index()

        color = STRATEGY_COLORS.get(strategy, "#999")
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = STRATEGY_MARKERS.get(strategy, "o")

        ax.errorbar(grouped[variable], grouped["mean"], yerr=grouped["std"],
                    color=color, label=label, marker=marker,
                    capsize=3, linewidth=1.5, markersize=6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    _save_fig(fig, output_dir, fig_name)


def generate_all_plots(results_df: pd.DataFrame,
                       agg_df: pd.DataFrame,
                       output_dir: str = "results/plots") -> None:
    """Generate all 10 required publication-quality plots.

    Args:
        results_df: DataFrame with all runs (strategy, seed, variable, metrics).
        agg_df: Aggregated DataFrame by strategy.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1-2: Connectivity reliability plots (if variable data available)
    if "failure_probability" in results_df.columns:
        plot_metric_vs_variable(
            results_df, "failure_probability", "total_interruption_time",
            "Fig 1: Service Interruption vs Connectivity Reliability",
            "Link Failure Probability", "Total Interruption Time (steps)",
            output_dir, "fig01_interruption_vs_connectivity")

        plot_metric_vs_variable(
            results_df, "failure_probability", "total_state_loss",
            "Fig 2: State Loss vs Connectivity Reliability",
            "Link Failure Probability", "Total State Loss (versions)",
            output_dir, "fig02_state_loss_vs_connectivity")

    # 3-4: State size plots
    if "state_size" in results_df.columns:
        plot_metric_vs_variable(
            results_df, "state_size", "total_data_transferred",
            "Fig 3: Migration Overhead vs State Size",
            "State Size (MB)", "Total Data Transferred (MB)",
            output_dir, "fig03_migration_overhead_vs_state_size")

        plot_metric_vs_variable(
            results_df, "state_size", "total_recovery_time",
            "Fig 4: Recovery Time vs State Size",
            "State Size (MB)", "Total Recovery Time (steps)",
            output_dir, "fig04_recovery_time_vs_state_size")

    # 5: UAV velocity
    if "uav_velocity" in results_df.columns:
        plot_metric_vs_variable(
            results_df, "uav_velocity", "service_availability",
            "Fig 5: Service Availability vs UAV Velocity",
            "UAV Velocity (m/s)", "Service Availability",
            output_dir, "fig05_availability_vs_velocity")

    # 6-10: Strategy comparison bar charts
    plot_comparison_bar(
        agg_df, "total_replication_bw",
        "Fig 6: Bandwidth Overhead by Strategy",
        "Total Bandwidth Overhead (MB)",
        output_dir, "fig06_bandwidth_overhead")

    plot_comparison_bar(
        agg_df, "migrations_attempted",
        "Fig 7: Number of Migrations by Strategy",
        "Number of Migrations",
        output_dir, "fig07_num_migrations")

    plot_comparison_bar(
        agg_df, "sla_violations",
        "Fig 8: SLA Violations by Strategy",
        "Number of SLA Violations",
        output_dir, "fig08_sla_violations")

    plot_comparison_bar(
        agg_df, "objective_cost",
        "Fig 9: Overall Cost by Strategy",
        "Weighted Objective Cost",
        output_dir, "fig09_overall_cost")

    # 10: Multi-metric comparison
    _plot_performance_comparison(agg_df, output_dir)

    print(f"All plots saved to {output_dir}/")


def _plot_performance_comparison(agg_df: pd.DataFrame,
                                 output_dir: str) -> None:
    """Radar/grouped bar chart comparing all strategies across key metrics."""
    _setup_style()

    metrics = [
        ("service_availability", "Availability"),
        ("migration_success_rate", "Migration Success"),
        ("sla_violations", "SLA Violations"),
        ("total_state_loss", "State Loss"),
        ("objective_cost", "Overall Cost"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        strategies = agg_df["strategy"].values
        means = agg_df[f"{metric}_mean"].values
        stds = agg_df[f"{metric}_std"].values
        colors = [STRATEGY_COLORS.get(s, "#999") for s in strategies]
        labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

        bars = ax.bar(range(len(strategies)), means, yerr=stds,
                      color=colors, capsize=3, edgecolor="white")
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([l.split()[0] for l in labels],
                          rotation=45, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9)

    fig.suptitle("Fig 10: Performance Comparison Across All Methods", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig10_performance_comparison")
