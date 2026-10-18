"""
SHAP-based explainability analysis for feature and modality importance.

Computes SHAP values for both tree-based baselines and the neural attention model,
providing feature-level and modality-level attribution.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_shap_tree(model, X_test: pd.DataFrame, model_name: str) -> dict:
    """Compute SHAP values for a tree-based model."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_importance": feature_importance,
    }


def compute_modality_importance(feature_importance: pd.DataFrame, feature_groups: dict) -> pd.DataFrame:
    """Aggregate SHAP importance by modality."""
    modality_importance = {}

    for modality, features in feature_groups.items():
        if modality in ["all", "modality_flags", "spatial_temporal"]:
            continue
        mask = feature_importance["feature"].isin(features)
        if mask.any():
            modality_importance[modality] = feature_importance.loc[mask, "mean_abs_shap"].sum()

    total = sum(modality_importance.values()) or 1.0
    result = pd.DataFrame([
        {"modality": k, "total_shap": v, "relative_importance": v / total}
        for k, v in modality_importance.items()
    ]).sort_values("relative_importance", ascending=False)

    return result


def save_shap_summary(
    shap_results: dict,
    feature_groups: dict,
    output_dir: str,
    model_name: str,
):
    """Save SHAP analysis results."""
    os.makedirs(output_dir, exist_ok=True)

    fi = shap_results["feature_importance"]
    fi.to_csv(Path(output_dir) / f"shap_feature_importance_{model_name}.csv", index=False)

    modality_imp = compute_modality_importance(fi, feature_groups)
    modality_imp.to_csv(Path(output_dir) / f"shap_modality_importance_{model_name}.csv", index=False)

    logger.info(f"SHAP results saved to {output_dir}")
    logger.info(f"Top 10 features:\n{fi.head(10).to_string(index=False)}")
    logger.info(f"Modality importance:\n{modality_imp.to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--feature-dir", default="data/processed/features/early_fusion")
    parser.add_argument("--feature-groups", default="data/processed/features/feature_groups.json")
    parser.add_argument("--output-dir", default="reports/explainability")
    parser.add_argument("--model-name", default="xgboost_early_fusion")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.feature_groups) as f:
        feature_groups = json.load(f)

    X_test = pd.read_csv(Path(args.feature_dir) / "X_test.csv")

    if args.n_samples < len(X_test):
        X_sample = X_test.sample(n=args.n_samples, random_state=42)
    else:
        X_sample = X_test

    model = joblib.load(args.model_path)
    logger.info(f"Loaded model from {args.model_path}")

    shap_results = compute_shap_tree(model, X_sample, args.model_name)
    save_shap_summary(shap_results, feature_groups, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
