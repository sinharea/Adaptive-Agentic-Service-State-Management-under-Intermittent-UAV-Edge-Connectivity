"""
Generate publication-quality comparison plots for the research paper.

Produces authentic graphs from actual experiment data, focusing on:
1. Recovery Time Comparison across strategies
2. Overall Cost Comparison across strategies  
3. Service Availability vs Connectivity Reliability
4. State Loss vs Connectivity Reliability
5. SLA Violations Comparison
6. Multi-metric Performance Radar
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ------------------------------------------------------------------ #
# Style configuration
# ------------------------------------------------------------------ #

STRATEGY_COLORS = {
    "local_execution": "#4C72B0",
    "threshold_based": "#DD8452",
    "mobility_aware": "#55A868",
    "rl_based": "#C44E52",
    "agentic_manager": "#8172B3",
}

STRATEGY_LABELS = {
    "local_execution": "Local Execution",
    "threshold_based": "Threshold-Based",
    "mobility_aware": "Mobility-Aware",
    "rl_based": "RL-Based (Q-Learning)",
    "agentic_manager": "Agentic Manager\n(Proposed)",
}

STRATEGY_HATCHES = {
    "local_execution": "//",
    "threshold_based": "\\\\",
    "mobility_aware": "..",
    "rl_based": "xx",
    "agentic_manager": "",
}

STRATEGY_ORDER = [
    "local_execution", "threshold_based", "mobility_aware",
    "rl_based", "agentic_manager",
]

STRATEGY_MARKERS = {
    "local_execution": "o",
    "threshold_based": "s",
    "mobility_aware": "^",
    "rl_based": "D",
    "agentic_manager": "*",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ------------------------------------------------------------------ #
# Plot 1: Recovery Time Comparison (Bar chart, averaged over conditions)
# ------------------------------------------------------------------ #

def plot_recovery_time(df, output_dir):
    """Bar chart: Mean Recovery Time per disruption event by strategy."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Average across failure probabilities > 0 (conditions with disruptions)
    df_stress = df[df["failure_probability"] > 0].copy()
    
    # Compute recovery time per disruption: total_recovery / total_interruption
    # This gives us "how fast does the strategy recover per disruption event"
    grouped = df_stress.groupby("strategy").agg(
        avg_recovery=("total_recovery_time", "mean"),
        std_recovery=("total_recovery_time", "std"),
        avg_interruption=("total_interruption_time", "mean"),
    ).reindex(STRATEGY_ORDER)

    x = np.arange(len(STRATEGY_ORDER))
    bars = ax.bar(x, grouped["avg_recovery"], yerr=grouped["std_recovery"],
                  capsize=5, width=0.6, edgecolor="white", linewidth=0.8,
                  color=[STRATEGY_COLORS[s] for s in STRATEGY_ORDER],
                  hatch=[STRATEGY_HATCHES[s] for s in STRATEGY_ORDER])

    # Highlight the best (lowest)
    best_idx = grouped["avg_recovery"].values.argmin()
    bars[best_idx].set_edgecolor("#2d2d2d")
    bars[best_idx].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                       fontsize=9, ha="center")
    ax.set_ylabel("Mean Recovery Time (steps)", fontsize=12)
    ax.set_title("Recovery Time Comparison Under Intermittent Connectivity",
                 fontsize=13, fontweight="bold", pad=12)

    # Add value labels
    for bar, val in zip(bars, grouped["avg_recovery"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    # Annotation
    ax.annotate("Lower is better ↓", xy=(0.02, 0.96),
                xycoords="axes fraction", fontsize=8, fontstyle="italic",
                color="#666")

    fig.tight_layout()
    save_fig(fig, output_dir, "recovery_time_comparison")


# ------------------------------------------------------------------ #
# Plot 2: Overall Cost Comparison (grouped bar by failure level)
# ------------------------------------------------------------------ #

def plot_cost_comparison(df, output_dir):
    """Grouped bar chart: Objective cost by strategy and failure level."""
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    fp_levels = sorted(df["failure_probability"].unique())
    fp_levels = [fp for fp in fp_levels if fp > 0]  # Skip 0 (no difference)

    n_strategies = len(STRATEGY_ORDER)
    n_groups = len(fp_levels)
    bar_width = 0.15
    group_width = n_strategies * bar_width

    for i, strategy in enumerate(STRATEGY_ORDER):
        sdf = df[df["strategy"] == strategy]
        means = []
        stds = []
        for fp in fp_levels:
            fdf = sdf[sdf["failure_probability"] == fp]
            means.append(fdf["objective_cost"].mean())
            stds.append(fdf["objective_cost"].std())

        x = np.arange(n_groups) + i * bar_width
        ax.bar(x, means, bar_width, yerr=stds, capsize=3,
               label=STRATEGY_LABELS[strategy].replace("\n", " "),
               color=STRATEGY_COLORS[strategy],
               hatch=STRATEGY_HATCHES[strategy],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(np.arange(n_groups) + group_width / 2 - bar_width / 2)
    ax.set_xticklabels([f"p = {fp}" for fp in fp_levels], fontsize=10)
    ax.set_xlabel("Link Failure Probability", fontsize=12)
    ax.set_ylabel("Weighted Objective Cost", fontsize=12)
    ax.set_title("Total Cost Comparison Across Connectivity Conditions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax.annotate("Lower is better ↓", xy=(0.02, 0.96),
                xycoords="axes fraction", fontsize=8, fontstyle="italic",
                color="#666")

    fig.tight_layout()
    save_fig(fig, output_dir, "cost_comparison")


# ------------------------------------------------------------------ #
# Plot 3: Service Availability vs Failure Probability (line chart)
# ------------------------------------------------------------------ #

def plot_availability_vs_failure(df, output_dir):
    """Line chart: Service availability vs failure probability."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for strategy in STRATEGY_ORDER:
        sdf = df[df["strategy"] == strategy]
        grouped = sdf.groupby("failure_probability")["service_availability"].agg(
            ["mean", "std"]).reset_index()

        ax.errorbar(grouped["failure_probability"], grouped["mean"],
                    yerr=grouped["std"],
                    label=STRATEGY_LABELS[strategy].replace("\n", " "),
                    color=STRATEGY_COLORS[strategy],
                    marker=STRATEGY_MARKERS[strategy],
                    markersize=8, linewidth=2, capsize=4)

    ax.set_xlabel("Link Failure Probability", fontsize=12)
    ax.set_ylabel("Service Availability", fontsize=12)
    ax.set_title("Service Availability Under Varying Connectivity Reliability",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax.annotate("Higher is better ↑", xy=(0.02, 0.04),
                xycoords="axes fraction", fontsize=8, fontstyle="italic",
                color="#666")

    fig.tight_layout()
    save_fig(fig, output_dir, "availability_vs_failure")


# ------------------------------------------------------------------ #
# Plot 4: State Loss vs Failure Probability
# ------------------------------------------------------------------ #

def plot_state_loss_vs_failure(df, output_dir):
    """Line chart: Cumulative state loss vs failure probability."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for strategy in STRATEGY_ORDER:
        sdf = df[df["strategy"] == strategy]
        grouped = sdf.groupby("failure_probability")["total_state_loss"].agg(
            ["mean", "std"]).reset_index()

        ax.errorbar(grouped["failure_probability"], grouped["mean"],
                    yerr=grouped["std"],
                    label=STRATEGY_LABELS[strategy].replace("\n", " "),
                    color=STRATEGY_COLORS[strategy],
                    marker=STRATEGY_MARKERS[strategy],
                    markersize=8, linewidth=2, capsize=4)

    ax.set_xlabel("Link Failure Probability", fontsize=12)
    ax.set_ylabel("Total State Loss (version deltas)", fontsize=12)
    ax.set_title("Cumulative State Loss Under Intermittent Connectivity",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)

    ax.annotate("Lower is better ↓", xy=(0.02, 0.96),
                xycoords="axes fraction", fontsize=8, fontstyle="italic",
                color="#666")

    fig.tight_layout()
    save_fig(fig, output_dir, "state_loss_vs_failure")


# ------------------------------------------------------------------ #
# Plot 5: SLA Violations Comparison
# ------------------------------------------------------------------ #

def plot_sla_violations(df, output_dir):
    """Bar chart: Total SLA violations by strategy."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    df_stress = df[df["failure_probability"] > 0]
    grouped = df_stress.groupby("strategy")["sla_violations"].agg(
        ["mean", "std"]).reindex(STRATEGY_ORDER)

    x = np.arange(len(STRATEGY_ORDER))
    bars = ax.bar(x, grouped["mean"], yerr=grouped["std"],
                  capsize=5, width=0.6, edgecolor="white", linewidth=0.8,
                  color=[STRATEGY_COLORS[s] for s in STRATEGY_ORDER],
                  hatch=[STRATEGY_HATCHES[s] for s in STRATEGY_ORDER])

    # Highlight best
    best_idx = grouped["mean"].values.argmin()
    bars[best_idx].set_edgecolor("#2d2d2d")
    bars[best_idx].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                       fontsize=9, ha="center")
    ax.set_ylabel("Mean SLA Violations", fontsize=12)
    ax.set_title("SLA Violation Count Under Intermittent Connectivity",
                 fontsize=13, fontweight="bold", pad=12)

    for bar, val in zip(bars, grouped["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.annotate("Lower is better ↓", xy=(0.02, 0.96),
                xycoords="axes fraction", fontsize=8, fontstyle="italic",
                color="#666")

    fig.tight_layout()
    save_fig(fig, output_dir, "sla_violations_comparison")


# ------------------------------------------------------------------ #
# Plot 6: Multi-metric Summary Table/Heatmap
# ------------------------------------------------------------------ #

def plot_summary_heatmap(df, output_dir):
    """Normalized performance heatmap across all key metrics."""
    setup_style()
    
    df_stress = df[df["failure_probability"] > 0]
    
    metrics = {
        "Service\nAvailability": ("service_availability", True),    # higher=better
        "Recovery\nTime": ("total_recovery_time", False),           # lower=better
        "State\nLoss": ("total_state_loss", False),                 # lower=better
        "SLA\nViolations": ("sla_violations", False),               # lower=better
        "Objective\nCost": ("objective_cost", False),               # lower=better
    }
    
    # Build matrix
    rows = []
    for strategy in STRATEGY_ORDER:
        sdf = df_stress[df_stress["strategy"] == strategy]
        row = []
        for name, (col, higher_better) in metrics.items():
            val = sdf[col].mean()
            row.append(val)
        rows.append(row)
    
    raw = np.array(rows)
    
    # Normalize each column to [0, 1] where 1 = best
    normalized = np.zeros_like(raw)
    for j, (name, (col, higher_better)) in enumerate(metrics.items()):
        col_vals = raw[:, j]
        if col_vals.max() == col_vals.min():
            normalized[:, j] = 1.0
        elif higher_better:
            normalized[:, j] = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min())
        else:
            normalized[:, j] = 1.0 - (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    im = ax.imshow(normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()), fontsize=10)
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels([STRATEGY_LABELS[s].replace("\n", " ") 
                        for s in STRATEGY_ORDER], fontsize=10)
    
    # Add text annotations with raw values
    for i in range(len(STRATEGY_ORDER)):
        for j in range(len(metrics)):
            val = raw[i, j]
            text_color = "white" if normalized[i, j] < 0.3 or normalized[i, j] > 0.8 else "black"
            if val > 10:
                txt = f"{val:.0f}"
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Score (1.0 = Best)", fontsize=10)
    
    ax.set_title("Multi-Metric Performance Comparison (Averaged Over Stress Conditions)",
                 fontsize=12, fontweight="bold", pad=12)
    
    fig.tight_layout()
    save_fig(fig, output_dir, "performance_heatmap")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    output_dir = "results/plots"
    
    # Load connectivity experiment data
    csv_path = "results/connectivity_results.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run the connectivity experiment first:")
        print("  python main.py --experiment connectivity")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Failure levels: {sorted(df['failure_probability'].unique())}")
    print()
    
    # Verify data correctness
    print("=== Data Summary (mean over seeds) ===")
    summary = df.groupby(["strategy", "failure_probability"])[
        ["service_availability", "total_recovery_time", 
         "total_state_loss", "sla_violations", "objective_cost"]
    ].mean().round(2)
    print(summary.to_string())
    print()
    
    # Generate all plots
    print("Generating plots...")
    plot_recovery_time(df, output_dir)
    plot_cost_comparison(df, output_dir)
    plot_availability_vs_failure(df, output_dir)
    plot_state_loss_vs_failure(df, output_dir)
    plot_sla_violations(df, output_dir)
    plot_summary_heatmap(df, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")
    print("Files: recovery_time_comparison, cost_comparison, "
          "availability_vs_failure, state_loss_vs_failure, "
          "sla_violations_comparison, performance_heatmap")


if __name__ == "__main__":
    main()
