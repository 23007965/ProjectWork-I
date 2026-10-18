"""
Build prediction target: future_fish_species_richness (optimized version).

Vectorized approach: for each location with eDNA, find pairs of
(historical window, future window) where both have observations.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_prediction_dataset(
    aligned_path: str,
    history_months: int = 6,
    horizon_months: int = 6,
) -> pd.DataFrame:
    """Build prediction samples efficiently."""
    aligned = pd.read_csv(aligned_path)
    logger.info(f"Loaded {len(aligned)} aligned records")

    aligned["year_month_dt"] = pd.to_datetime(aligned["year_month"])

    # Focus on locations that have eDNA observations
    has_edna = aligned["edna_species_richness"].notna()
    edna_locs = aligned.loc[has_edna, ["lat_grid", "lon_grid"]].drop_duplicates()
    logger.info(f"Locations with eDNA: {len(edna_locs)}")

    # For efficiency, only keep data from locations that have eDNA
    loc_set = set(zip(edna_locs["lat_grid"], edna_locs["lon_grid"]))
    relevant = aligned[
        aligned.apply(lambda r: (r["lat_grid"], r["lon_grid"]) in loc_set, axis=1)
    ].copy()
    logger.info(f"Relevant records (locations with eDNA): {len(relevant)}")

    ocean_features = [c for c in aligned.columns if c in [
        "sea_surface_temperature", "salinity", "dissolved_oxygen", "chlorophyll_a"
    ]]
    fish_features = [c for c in aligned.columns if c in ["fishing_hours", "vessel_count"]]
    edna_features = [c for c in aligned.columns if c in [
        "edna_species_richness", "edna_total_detections"
    ]]
    modality_flags = [c for c in aligned.columns if c.startswith("has_")]

    # Group by location
    samples = []
    loc_groups = relevant.groupby(["lat_grid", "lon_grid"])

    for (lat, lon), loc_data in loc_groups:
        loc_data = loc_data.sort_values("year_month_dt")
        months = loc_data["year_month_dt"].values

        # Find months where future eDNA exists
        edna_months = loc_data.loc[loc_data["edna_species_richness"].notna(), "year_month_dt"].values

        for future_month in edna_months:
            # Prediction origin: horizon_months before future observation
            origin = future_month - np.timedelta64(horizon_months * 30, "D")
            history_start = origin - np.timedelta64(history_months * 30, "D")

            # History window
            history = loc_data[
                (loc_data["year_month_dt"] >= history_start)
                & (loc_data["year_month_dt"] <= origin)
            ]

            # Future target
            future = loc_data[
                (loc_data["year_month_dt"] > origin)
                & (loc_data["year_month_dt"] <= future_month)
            ]

            future_richness_vals = future["edna_species_richness"].dropna()
            if len(future_richness_vals) == 0:
                continue

            future_richness = future_richness_vals.max()

            sample = {
                "lat_grid": lat,
                "lon_grid": lon,
                "prediction_origin": str(pd.Timestamp(origin))[:10],
                "future_fish_species_richness": future_richness,
                "n_history_months": len(history),
            }

            # Historical oceanographic
            for feat in ocean_features:
                vals = history[feat].dropna()
                sample[f"hist_{feat}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_std"] = vals.std() if len(vals) > 1 else 0.0
                sample[f"hist_{feat}_trend"] = (
                    np.polyfit(range(len(vals)), vals.values, 1)[0] if len(vals) > 2 else 0.0
                )

            # Historical fisheries
            for feat in fish_features:
                vals = history[feat].dropna()
                sample[f"hist_{feat}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                sample[f"hist_{feat}_max"] = vals.max() if len(vals) > 0 else np.nan

            # Historical eDNA
            for feat in edna_features:
                vals = history[feat].dropna()
                sample[f"hist_{feat}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_max"] = vals.max() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_latest"] = vals.iloc[-1] if len(vals) > 0 else np.nan

            # Modality flags
            for flag in modality_flags:
                sample[f"hist_{flag}_frac"] = history[flag].mean() if len(history) > 0 else 0.0

            # Spatial/temporal
            sample["latitude"] = lat
            sample["longitude"] = lon
            origin_dt = pd.Timestamp(origin)
            sample["origin_month_sin"] = np.sin(2 * np.pi * origin_dt.month / 12)
            sample["origin_month_cos"] = np.cos(2 * np.pi * origin_dt.month / 12)
            sample["origin_year"] = origin_dt.year

            samples.append(sample)

    result = pd.DataFrame(samples)
    # Remove duplicates (same location-origin)
    if len(result) > 0:
        result = result.drop_duplicates(subset=["lat_grid", "lon_grid", "prediction_origin"])

    logger.info(f"Built {len(result)} prediction samples")
    if len(result) > 0:
        target = result["future_fish_species_richness"]
        logger.info(f"Target: mean={target.mean():.1f}, std={target.std():.1f}, range=[{target.min():.0f}, {target.max():.0f}]")

    return result


def create_splits(df: pd.DataFrame, train_frac=0.6, val_frac=0.2) -> pd.DataFrame:
    """Chronological train/val/test split."""
    df = df.sort_values("prediction_origin").copy()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    df["split"] = "test"
    df.iloc[:train_end, df.columns.get_loc("split")] = "train"
    df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"
    logger.info(f"Splits: {df['split'].value_counts().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build prediction target (v2)")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--aligned-data", default="data/processed/aligned_multimodal_data.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    pred_cfg = config["prediction"]
    dataset = build_prediction_dataset(
        args.aligned_data,
        history_months=pred_cfg["history_window_months"],
        horizon_months=pred_cfg["prediction_horizon_months"],
    )

    if dataset.empty:
        logger.error("No valid prediction samples")
        return

    dataset = create_splits(dataset)
    output_path = Path(args.output_dir) / "prediction_dataset.csv"
    dataset.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    print(f"\n=== Prediction Dataset ===")
    print(f"Samples: {len(dataset)}")
    print(f"Splits: {dataset['split'].value_counts().to_dict()}")
    print(f"Target: {dataset['future_fish_species_richness'].describe()}")
    print(f"Features: {len([c for c in dataset.columns if c.startswith('hist_')])}")


if __name__ == "__main__":
    main()
