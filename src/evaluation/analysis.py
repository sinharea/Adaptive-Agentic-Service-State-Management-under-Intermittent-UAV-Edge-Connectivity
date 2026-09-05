"""
Statistical analysis for experimental results.

Computes mean, standard deviation, confidence intervals across seeds.
"""

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import SimulationMetrics, compute_objective_cost


def summarize_runs(results: List[SimulationMetrics],
                   weights: Dict[str, float]) -> pd.DataFrame:
    """Summarize multiple simulation runs into a DataFrame.

    Each row corresponds to one simulation run (strategy + seed).
    """
    rows = []
    for sim in results:
        summary = sim.compute_summary()
        summary["objective_cost"] = compute_objective_cost(summary, weights)
        rows.append(summary)
    return pd.DataFrame(rows)


def aggregate_by_strategy(df: pd.DataFrame,
                          confidence: float = 0.95) -> pd.DataFrame:
    """Aggregate results by strategy, computing mean, std, CI.

    Args:
        df: DataFrame with one row per run.
        confidence: Confidence level for CI (default 0.95).

    Returns:
        DataFrame with strategy as index, columns for each metric
        with suffixes _mean, _std, _ci_low, _ci_high.
    """
    metrics_cols = [c for c in df.columns
                    if c not in ("strategy", "seed")]
    grouped = df.groupby("strategy")
    records = []

    for strategy, group in grouped:
        record = {"strategy": strategy, "num_seeds": len(group)}
        for col in metrics_cols:
            values = group[col].values.astype(float)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std

            if len(values) > 1:
                se = std / np.sqrt(len(values))
                t_val = stats.t.ppf((1 + confidence) / 2, df=len(values) - 1)
                record[f"{col}_ci_low"] = mean - t_val * se
                record[f"{col}_ci_high"] = mean + t_val * se
            else:
                record[f"{col}_ci_low"] = mean
                record[f"{col}_ci_high"] = mean

        records.append(record)

    return pd.DataFrame(records)


def save_results(df: pd.DataFrame, path: str) -> None:
    """Save results DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
