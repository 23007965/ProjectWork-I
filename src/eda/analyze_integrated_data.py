"""
Exploratory data analysis for the integrated multimodal dataset.

Generates summary statistics, distribution plots, correlation analysis,
and spatial-temporal coverage maps for each modality and the integrated data.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def analyze_oceanographic(df: pd.DataFrame, output_dir: str):
    """Analyze oceanographic variables."""
    ocean_cols = [c for c in df.columns if c in [
        "sea_surface_temperature", "salinity", "dissolved_oxygen", "chlorophyll_a"
    ]]
    if not ocean_cols:
        logger.warning("No oceanographic columns found")
        return {}

    stats = {}
    for col in ocean_cols:
        vals = df[col].dropna()
        stats[col] = {
            "count": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
            "missing_pct": (df[col].isna().sum() / len(df)) * 100,
        }

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(ocean_cols[:4]):
        ax = axes[i // 2, i % 2]
        df[col].dropna().hist(bins=30, ax=ax, alpha=0.7, color="steelblue")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "oceanographic_distributions.png", dpi=150)
    plt.close()

    # Seasonal patterns
    if "month" in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, col in enumerate(ocean_cols[:4]):
            ax = axes[i // 2, i % 2]
            monthly = df.groupby("month")[col].mean()
            ax.plot(monthly.index, monthly.values, "o-", color="steelblue")
            ax.set_title(f"Monthly Mean {col.replace('_', ' ').title()}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Value")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "oceanographic_seasonal.png", dpi=150)
        plt.close()

    return stats


def analyze_fisheries(df: pd.DataFrame, output_dir: str):
    """Analyze fisheries variables."""
    fish_cols = [c for c in df.columns if c in ["fishing_hours", "vessel_count"]]
    if not fish_cols:
        logger.warning("No fisheries columns found")
        return {}

    stats = {}
    for col in fish_cols:
        vals = df[col].dropna()
        stats[col] = {
            "count": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
            "missing_pct": (df[col].isna().sum() / len(df)) * 100,
        }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if "fishing_hours" in df.columns:
        df["fishing_hours"].dropna().hist(bins=30, ax=ax, alpha=0.7, color="coral")
        ax.set_title("Distribution of Monthly Fishing Hours per Grid Cell")
        ax.set_xlabel("Fishing Hours")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "fisheries_distribution.png", dpi=150)
    plt.close()

    return stats


def analyze_edna(df: pd.DataFrame, output_dir: str):
    """Analyze eDNA/richness variables."""
    edna_cols = [c for c in df.columns if "edna" in c.lower() or "richness" in c.lower()]
    if not edna_cols:
        logger.warning("No eDNA columns found")
        return {}

    stats = {}
    for col in edna_cols:
        vals = df[col].dropna()
        stats[col] = {
            "count": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
            "missing_pct": (df[col].isna().sum() / len(df)) * 100,
        }

    rich_col = next((c for c in edna_cols if "richness" in c.lower()), None)
    if rich_col:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        df[rich_col].dropna().hist(bins=20, ax=ax, alpha=0.7, color="forestgreen")
        ax.set_title("Distribution of Fish Species Richness")
        ax.set_xlabel("Species Richness")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "edna_richness_distribution.png", dpi=150)
        plt.close()

    return stats


def analyze_modality_coverage(df: pd.DataFrame, output_dir: str):
    """Analyze spatial-temporal coverage and overlap of modalities."""
    coverage = {}
    for flag in ["has_oceanographic", "has_fisheries", "has_edna"]:
        if flag in df.columns:
            coverage[flag] = {
                "present": int(df[flag].sum()),
                "missing": int((1 - df[flag]).sum()),
                "coverage_pct": float(df[flag].mean() * 100),
            }

    if "n_modalities_present" in df.columns:
        counts = df["n_modalities_present"].value_counts().sort_index()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        counts.plot(kind="bar", ax=ax, color=["red", "orange", "steelblue", "green"][:len(counts)])
        ax.set_title("Number of Modalities Present per Observation")
        ax.set_xlabel("Number of Modalities")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "modality_coverage.png", dpi=150)
        plt.close()

    return coverage


def analyze_correlations(df: pd.DataFrame, output_dir: str):
    """Correlation analysis between predictors and target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["lat_grid", "lon_grid", "year", "n_modalities_present"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    if len(numeric_cols) < 3:
        return

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "correlation_matrix.png", dpi=150)
    plt.close()

    # Correlation with target
    target_col = "future_fish_species_richness"
    if target_col in df.columns:
        target_corr = corr[target_col].drop(target_col, errors="ignore").sort_values(ascending=False)
        target_corr.to_csv(Path(output_dir) / "target_correlations.csv")

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        target_corr.head(15).plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Top Correlations with {target_col}")
        ax.set_xlabel("Pearson Correlation")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "target_correlations.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--aligned-data", default="data/processed/aligned_multimodal_data.csv")
    parser.add_argument("--prediction-data", default="data/processed/prediction_dataset.csv")
    parser.add_argument("--output-dir", default="reports/eda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze aligned data
    if os.path.exists(args.aligned_data):
        logger.info("Analyzing aligned multimodal data...")
        aligned = pd.read_csv(args.aligned_data)

        ocean_stats = analyze_oceanographic(aligned, args.output_dir)
        fish_stats = analyze_fisheries(aligned, args.output_dir)
        edna_stats = analyze_edna(aligned, args.output_dir)
        coverage = analyze_modality_coverage(aligned, args.output_dir)

        # Save summary
        summary = {
            "total_records": len(aligned),
            "oceanographic": ocean_stats,
            "fisheries": fish_stats,
            "edna": edna_stats,
            "coverage": coverage,
        }
        with open(Path(args.output_dir) / "eda_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # Analyze prediction dataset
    if os.path.exists(args.prediction_data):
        logger.info("Analyzing prediction dataset...")
        pred = pd.read_csv(args.prediction_data)
        analyze_correlations(pred, args.output_dir)

        # Target distribution by split
        if "split" in pred.columns and "future_fish_species_richness" in pred.columns:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, split in enumerate(["train", "val", "test"]):
                subset = pred[pred["split"] == split]["future_fish_species_richness"]
                axes[i].hist(subset, bins=20, alpha=0.7, color=["steelblue", "coral", "green"][i])
                axes[i].set_title(f"{split.title()} Set (n={len(subset)})")
                axes[i].set_xlabel("Future Species Richness")
                axes[i].set_ylabel("Count")
            plt.suptitle("Target Distribution by Data Split")
            plt.tight_layout()
            plt.savefig(Path(args.output_dir) / "target_by_split.png", dpi=150)
            plt.close()

    logger.info(f"EDA complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
