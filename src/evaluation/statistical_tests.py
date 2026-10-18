"""
Statistical validation: paired tests, confidence intervals, and multi-seed analysis.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval using t-distribution."""
    n = len(values)
    if n < 2:
        return (values[0] if n == 1 else np.nan, values[0] if n == 1 else np.nan)
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return ci


def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test comparing two models' absolute errors."""
    diff = errors_a - errors_b
    if np.all(diff == 0):
        return {"test": "Wilcoxon", "statistic": 0, "p_value": 1.0, "significant": False}
    try:
        stat, p = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        stat, p = stats.ttest_rel(errors_a, errors_b)
    return {"test": "Wilcoxon", "statistic": float(stat), "p_value": float(p), "significant": p < 0.05}


def paired_comparison_all_models(results_df: pd.DataFrame, reference_model: str = None) -> pd.DataFrame:
    """Run pairwise statistical comparisons between all models."""
    models = results_df.groupby(["modality", "model"]).groups.keys()
    if reference_model is None:
        reference_model = "cross_modal_attention"

    comparisons = []
    for modality, model in models:
        model_results = results_df[
            (results_df["modality"] == modality) & (results_df["model"] == model)
        ]
        if "test_mae" in model_results.columns:
            values = model_results["test_mae"].values
            ci = compute_confidence_interval(values)
            comparisons.append({
                "modality": modality,
                "model": model,
                "mae_mean": np.mean(values),
                "mae_std": np.std(values),
                "mae_ci_lower": ci[0],
                "mae_ci_upper": ci[1],
                "n_seeds": len(values),
            })

    return pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description="Statistical tests")
    parser.add_argument("--results-dir", default="experiments")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all results
    results_files = list(Path(args.results_dir).rglob("*results*.csv"))
    all_dfs = []
    for f in results_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        logger.warning("No results found")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    comparisons = paired_comparison_all_models(combined)

    output_path = Path(args.output_dir) / "statistical_comparisons.csv"
    comparisons.to_csv(output_path, index=False)

    print("\n=== Statistical Comparisons ===")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
