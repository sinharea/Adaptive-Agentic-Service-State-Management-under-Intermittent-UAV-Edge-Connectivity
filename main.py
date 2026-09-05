"""
Experiment runner for UAV-edge state management research prototype.

Usage:
    python main.py --config config/default.yaml
    python main.py --config config/default.yaml --experiment connectivity
    python main.py --config config/default.yaml --strategies local_execution threshold_based agentic_manager
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from src.environment.simulator import Simulator, load_config
from src.evaluation.metrics import SimulationMetrics, compute_objective_cost
from src.evaluation.analysis import summarize_runs, aggregate_by_strategy, save_results
from src.evaluation.plots import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ALL_STRATEGIES = [
    "local_execution",
    "threshold_based",
    "mobility_aware",
    "rl_based",
    "agentic_manager",
]


def run_default_experiment(config: dict,
                           strategies: List[str],
                           output_dir: str = "results") -> pd.DataFrame:
    """Run all strategies with multiple seeds on default config.

    Args:
        config: Configuration dict.
        strategies: List of strategy names.
        output_dir: Output directory.

    Returns:
        DataFrame with one row per (strategy, seed).
    """
    sim = Simulator(config)
    num_seeds = config["simulation"].get("num_seeds", 5)
    base_seed = config["simulation"].get("random_seed", 42)
    weights = config.get("objective", {})

    all_results: List[SimulationMetrics] = []

    for strategy in strategies:
        for i in range(num_seeds):
            seed = base_seed + i
            logger.info(f"Running {strategy} with seed={seed}...")
            start = time.time()
            metrics = sim.run(strategy, seed)
            elapsed = time.time() - start
            logger.info(f"  Completed in {elapsed:.1f}s "
                       f"({metrics.compute_summary().get('service_availability', 0):.3f} avail)")
            all_results.append(metrics)

    df = summarize_runs(all_results, weights)
    return df


def run_variable_experiment(config: dict,
                            strategies: List[str],
                            variable_name: str,
                            variable_values: list,
                            config_path: str,
                            output_dir: str = "results") -> pd.DataFrame:
    """Run experiment varying a single parameter.

    Args:
        config: Base configuration dict.
        strategies: List of strategy names.
        variable_name: Name of variable to vary.
        variable_values: List of values for the variable.
        config_path: Path for variable mapping.
        output_dir: Output directory.

    Returns:
        DataFrame with results including the variable column.
    """
    sim = Simulator(config)
    num_seeds = config["simulation"].get("num_seeds", 5)
    base_seed = config["simulation"].get("random_seed", 42)
    weights = config.get("objective", {})

    all_results: List[SimulationMetrics] = []
    variable_values_list: List[float] = []

    # Map variable name to config override
    variable_map = {
        "failure_probability": lambda v: {"_failure_probability": v},
        "state_size": lambda v: {"service": {"state_size": v}},
        "uav_velocity": lambda v: {"uav": {"velocity_range": [v * 0.5, v]}},
        "bandwidth": lambda v: {"network": {"max_bandwidth": v}},
        "uav_count": lambda v: {"uav": {"count": int(v)}},
    }

    override_fn = variable_map.get(variable_name)
    if override_fn is None:
        raise ValueError(f"Unknown variable: {variable_name}")

    for value in variable_values:
        overrides = override_fn(value)
        for strategy in strategies:
            for i in range(num_seeds):
                seed = base_seed + i
                logger.info(f"Running {strategy} | {variable_name}={value} | seed={seed}")
                metrics = sim.run(strategy, seed, config_overrides=overrides)
                all_results.append(metrics)
                variable_values_list.append(value)

    df = summarize_runs(all_results, weights)
    df[variable_name] = variable_values_list
    return df


def main():
    parser = argparse.ArgumentParser(
        description="UAV-Edge State Management Experiment Runner"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--strategies", nargs="+", default=None,
                       help="Strategies to evaluate (default: all)")
    parser.add_argument("--experiment", type=str, default="default",
                       choices=["default", "connectivity", "state_size",
                               "velocity", "bandwidth", "all"],
                       help="Experiment to run")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    strategies = args.strategies or ALL_STRATEGIES
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Configuration: {args.config}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Experiment: {args.experiment}")

    weights = config.get("objective", {})

    if args.experiment == "default":
        df = run_default_experiment(config, strategies, output_dir)
        save_results(df, os.path.join(output_dir, "default_results.csv"))
        agg_df = aggregate_by_strategy(df)
        save_results(agg_df, os.path.join(output_dir, "default_aggregated.csv"))
        generate_all_plots(df, agg_df, os.path.join(output_dir, "plots"))

    elif args.experiment == "connectivity":
        values = config.get("experiments", {}).get(
            "failure_probabilities", [0.0, 0.1, 0.2, 0.3, 0.5])
        df = run_variable_experiment(config, strategies, "failure_probability",
                                     values, args.config, output_dir)
        save_results(df, os.path.join(output_dir, "connectivity_results.csv"))
        agg_df = aggregate_by_strategy(df)
        save_results(agg_df, os.path.join(output_dir, "connectivity_aggregated.csv"))
        generate_all_plots(df, agg_df, os.path.join(output_dir, "plots"))

    elif args.experiment == "state_size":
        values = config.get("experiments", {}).get(
            "state_sizes", [50, 100, 250, 500, 1000])
        df = run_variable_experiment(config, strategies, "state_size",
                                     values, args.config, output_dir)
        save_results(df, os.path.join(output_dir, "state_size_results.csv"))
        agg_df = aggregate_by_strategy(df)
        generate_all_plots(df, agg_df, os.path.join(output_dir, "plots"))

    elif args.experiment == "velocity":
        values = config.get("experiments", {}).get(
            "uav_velocities", [5, 10, 15, 20, 30])
        df = run_variable_experiment(config, strategies, "uav_velocity",
                                     values, args.config, output_dir)
        save_results(df, os.path.join(output_dir, "velocity_results.csv"))
        agg_df = aggregate_by_strategy(df)
        generate_all_plots(df, agg_df, os.path.join(output_dir, "plots"))

    elif args.experiment == "bandwidth":
        values = config.get("experiments", {}).get(
            "bandwidth_levels", [10, 25, 50, 100, 200])
        df = run_variable_experiment(config, strategies, "bandwidth",
                                     values, args.config, output_dir)
        save_results(df, os.path.join(output_dir, "bandwidth_results.csv"))
        agg_df = aggregate_by_strategy(df)
        generate_all_plots(df, agg_df, os.path.join(output_dir, "plots"))

    elif args.experiment == "all":
        logger.info("Running all experiments sequentially...")

        # Default
        df = run_default_experiment(config, strategies, output_dir)
        save_results(df, os.path.join(output_dir, "default_results.csv"))
        agg_df = aggregate_by_strategy(df)
        save_results(agg_df, os.path.join(output_dir, "default_aggregated.csv"))
        generate_all_plots(df, agg_df, os.path.join(output_dir, "plots"))

        # Connectivity
        fp_values = config.get("experiments", {}).get(
            "failure_probabilities", [0.0, 0.1, 0.2, 0.3, 0.5])
        df_conn = run_variable_experiment(config, strategies, "failure_probability",
                                          fp_values, args.config, output_dir)
        save_results(df_conn, os.path.join(output_dir, "connectivity_results.csv"))
        generate_all_plots(df_conn, aggregate_by_strategy(df_conn),
                          os.path.join(output_dir, "plots_connectivity"))

        # State size
        ss_values = config.get("experiments", {}).get(
            "state_sizes", [50, 100, 250, 500, 1000])
        df_ss = run_variable_experiment(config, strategies, "state_size",
                                        ss_values, args.config, output_dir)
        save_results(df_ss, os.path.join(output_dir, "state_size_results.csv"))
        generate_all_plots(df_ss, aggregate_by_strategy(df_ss),
                          os.path.join(output_dir, "plots_state_size"))

        logger.info("All experiments completed.")

    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
