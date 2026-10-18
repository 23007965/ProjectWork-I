"""
Unified evaluation framework for all models.
Computes metrics, confidence intervals, and statistical tests.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "n_samples": len(y_true)}


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
            bootstrap_values.append(val)
        except Exception:
            continue

    if not bootstrap_values:
        return (np.nan, np.nan)

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_values, alpha * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha) * 100)
    return (lower, upper)


def paired_statistical_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    test_name: str = "wilcoxon",
) -> dict:
    """Paired statistical test comparing two models' absolute errors."""
    errors_a = np.abs(y_true - y_pred_a)
    errors_b = np.abs(y_true - y_pred_b)

    diff = errors_a - errors_b

    if test_name == "wilcoxon":
        try:
            stat, p_value = stats.wilcoxon(diff, alternative="two-sided")
            test_used = "Wilcoxon signed-rank"
        except ValueError:
            stat, p_value = stats.ttest_rel(errors_a, errors_b)
            test_used = "Paired t-test (fallback)"
    elif test_name == "ttest":
        stat, p_value = stats.ttest_rel(errors_a, errors_b)
        test_used = "Paired t-test"
    else:
        stat, p_value = stats.wilcoxon(diff, alternative="two-sided")
        test_used = "Wilcoxon signed-rank"

    return {
        "test": test_used,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "mean_error_a": float(errors_a.mean()),
        "mean_error_b": float(errors_b.mean()),
        "improvement": float(errors_a.mean() - errors_b.mean()),
    }


def aggregate_multi_seed_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across multiple random seeds."""
    group_cols = ["modality", "model"]
    available_cols = [c for c in group_cols if c in results_df.columns]

    if not available_cols:
        return results_df

    metric_cols = [c for c in results_df.columns if c.startswith("test_")]

    summary = results_df.groupby(available_cols).agg(
        {col: ["mean", "std", "count"] for col in metric_cols if col in results_df.columns}
    )

    # Flatten multi-level columns
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary = summary.reset_index()

    return summary


def generate_comparison_table(
    experiment_dir: str,
    output_path: str,
) -> pd.DataFrame:
    """Generate the final comparison table from all experiment results."""
    results_files = list(Path(experiment_dir).rglob("*results*.csv"))

    all_results = []
    for f in results_files:
        try:
            df = pd.read_csv(f)
            all_results.append(df)
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not all_results:
        logger.warning("No result files found")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    summary = aggregate_multi_seed_results(combined)

    summary.to_csv(output_path, index=False)
    logger.info(f"Comparison table saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--experiment-dir", default="experiments")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary = generate_comparison_table(
        args.experiment_dir,
        str(Path(args.output_dir) / "final_results.csv"),
    )

    if not summary.empty:
        print("\n=== Final Model Comparison ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
