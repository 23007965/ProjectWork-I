"""
Feature engineering: separate features by modality for baseline and multimodal models.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


OCEAN_FEATURE_PREFIXES = [
    "hist_sea_surface_temperature",
    "hist_salinity",
    "hist_dissolved_oxygen",
    "hist_chlorophyll_a",
]

FISH_FEATURE_PREFIXES = [
    "hist_fishing_hours",
    "hist_vessel_count",
]

EDNA_FEATURE_PREFIXES = [
    "hist_edna_species_richness",
    "hist_edna_total_detections",
]

SPATIAL_TEMPORAL_FEATURES = [
    "latitude",
    "longitude",
    "origin_month_sin",
    "origin_month_cos",
]


def get_features_by_modality(df: pd.DataFrame) -> dict:
    """Identify feature columns belonging to each modality."""
    all_cols = set(df.columns)

    ocean_feats = [c for c in all_cols if any(c.startswith(p) for p in OCEAN_FEATURE_PREFIXES)]
    fish_feats = [c for c in all_cols if any(c.startswith(p) for p in FISH_FEATURE_PREFIXES)]
    edna_feats = [c for c in all_cols if any(c.startswith(p) for p in EDNA_FEATURE_PREFIXES)]
    spatiotemporal = [c for c in SPATIAL_TEMPORAL_FEATURES if c in all_cols]

    modality_flags = [c for c in all_cols if c.startswith("hist_has_")]

    return {
        "oceanographic": sorted(ocean_feats),
        "fisheries": sorted(fish_feats),
        "edna": sorted(edna_feats),
        "spatial_temporal": sorted(spatiotemporal),
        "modality_flags": sorted(modality_flags),
        "all": sorted(ocean_feats + fish_feats + edna_feats + spatiotemporal + modality_flags),
    }


def prepare_split_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "future_fish_species_richness",
    fit_scaler: bool = True,
    scaler: StandardScaler = None,
) -> tuple:
    """Prepare features and target, handling missing values."""
    X = df[feature_cols].copy()
    y = df[target_col].values if target_col in df.columns else None

    # Fill NaN with 0 for missing modality features
    X = X.fillna(0.0)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    elif scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
    else:
        X_scaled = X

    return X_scaled, y, scaler


def build_all_feature_sets(
    dataset_path: str,
    output_dir: str,
    config_path: str = "configs/default_config.yaml",
):
    """Build and save feature sets for all modality combinations."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} samples")

    feature_groups = get_features_by_modality(df)
    target_col = "future_fish_species_richness"

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Build feature sets for each modality and combinations
    modality_configs = {
        "ocean_only": feature_groups["oceanographic"] + feature_groups["spatial_temporal"],
        "fish_only": feature_groups["fisheries"] + feature_groups["spatial_temporal"],
        "edna_only": feature_groups["edna"] + feature_groups["spatial_temporal"],
        "early_fusion": feature_groups["all"],
        "ocean_fish": feature_groups["oceanographic"] + feature_groups["fisheries"] + feature_groups["spatial_temporal"],
        "ocean_edna": feature_groups["oceanographic"] + feature_groups["edna"] + feature_groups["spatial_temporal"],
        "fish_edna": feature_groups["fisheries"] + feature_groups["edna"] + feature_groups["spatial_temporal"],
    }

    for name, features in modality_configs.items():
        if not features:
            logger.warning(f"No features for {name}, skipping")
            continue

        X_train, y_train, scaler = prepare_split_data(train, features, target_col, fit_scaler=True)
        X_val, y_val, _ = prepare_split_data(val, features, target_col, fit_scaler=False, scaler=scaler)
        X_test, y_test, _ = prepare_split_data(test, features, target_col, fit_scaler=False, scaler=scaler)

        feature_dir = Path(output_dir) / name
        os.makedirs(feature_dir, exist_ok=True)

        X_train.to_csv(feature_dir / "X_train.csv", index=False)
        X_val.to_csv(feature_dir / "X_val.csv", index=False)
        X_test.to_csv(feature_dir / "X_test.csv", index=False)
        pd.Series(y_train, name=target_col).to_csv(feature_dir / "y_train.csv", index=False)
        pd.Series(y_val, name=target_col).to_csv(feature_dir / "y_val.csv", index=False)
        pd.Series(y_test, name=target_col).to_csv(feature_dir / "y_test.csv", index=False)
        joblib.dump(scaler, feature_dir / "scaler.joblib")

        logger.info(f"{name}: {len(features)} features, saved to {feature_dir}")

    # Save feature group definitions
    import json
    with open(Path(output_dir) / "feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    logger.info("All feature sets built successfully")


def main():
    parser = argparse.ArgumentParser(description="Build feature sets")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--dataset", default="data/processed/prediction_dataset.csv")
    parser.add_argument("--output-dir", default="data/processed/features")
    args = parser.parse_args()

    build_all_feature_sets(args.dataset, args.output_dir, args.config)


if __name__ == "__main__":
    main()
