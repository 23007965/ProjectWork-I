"""
Build the prediction target: future_fish_species_richness.

For each spatial location and time period, construct:
- Historical features: oceanographic, fisheries, and eDNA data from the history window
- Target: species richness in the future evaluation period

Enforces strict temporal separation to prevent data leakage.
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


def build_temporal_pairs(
    aligned_df: pd.DataFrame,
    history_months: int = 6,
    horizon_months: int = 6,
) -> pd.DataFrame:
    """
    Build (history_window, future_target) pairs for each location.

    For a given location and prediction origin month T:
    - History window: [T - history_months + 1, T]
    - Future window: [T + 1, T + horizon_months]
    - Target: species richness observed in the future window
    """
    logger.info(
        f"Building temporal pairs: history={history_months}mo, horizon={horizon_months}mo"
    )

    aligned_df = aligned_df.copy()
    aligned_df["year_month_dt"] = pd.to_datetime(aligned_df["year_month"])
    aligned_df = aligned_df.sort_values(["lat_grid", "lon_grid", "year_month_dt"])

    # Only process locations that have eDNA observations (target requires them)
    edna_mask = aligned_df["edna_species_richness"].notna() if "edna_species_richness" in aligned_df.columns else pd.Series(False, index=aligned_df.index)
    locations = aligned_df.loc[edna_mask, ["lat_grid", "lon_grid"]].drop_duplicates()
    logger.info(f"Processing {len(locations)} locations with eDNA observations (out of {aligned_df[['lat_grid','lon_grid']].drop_duplicates().shape[0]} total)")

    # Get all unique months
    all_months = sorted(aligned_df["year_month_dt"].unique())
    if len(all_months) < history_months + horizon_months:
        logger.error(
            f"Not enough months ({len(all_months)}) for history({history_months})+horizon({horizon_months})"
        )
        return pd.DataFrame()

    # Identify oceanographic, fisheries, and eDNA feature columns
    ocean_features = [
        c for c in aligned_df.columns
        if c in ["sea_surface_temperature", "salinity", "dissolved_oxygen", "chlorophyll_a"]
    ]
    fish_features = [c for c in aligned_df.columns if c in ["fishing_hours", "vessel_count"]]
    edna_features = [
        c for c in aligned_df.columns
        if c in ["edna_species_richness", "edna_total_detections"]
    ]
    modality_flags = [c for c in aligned_df.columns if c.startswith("has_")]

    # Pre-index by location for faster lookup
    loc_groups = {
        (lat, lon): grp.sort_values("year_month_dt")
        for (lat, lon), grp in aligned_df.groupby(["lat_grid", "lon_grid"])
        if (lat, lon) in set(zip(locations["lat_grid"], locations["lon_grid"]))
    }
    logger.info(f"Pre-indexed {len(loc_groups)} location groups")

    samples = []
    for _, loc in locations.iterrows():
        lat, lon = loc["lat_grid"], loc["lon_grid"]
        loc_data = loc_groups.get((lat, lon))
        if loc_data is None or len(loc_data) == 0:
            continue

        for origin_idx in range(history_months - 1, len(all_months) - horizon_months):
            origin_month = all_months[origin_idx]

            history_start = all_months[origin_idx - history_months + 1]
            history_end = origin_month

            future_start = all_months[origin_idx + 1]
            future_end_idx = min(origin_idx + horizon_months, len(all_months) - 1)
            future_end = all_months[future_end_idx]

            # Historical data
            history = loc_data[
                (loc_data["year_month_dt"] >= history_start)
                & (loc_data["year_month_dt"] <= history_end)
            ]

            # Future eDNA observations (for target)
            future = loc_data[
                (loc_data["year_month_dt"] > history_end)
                & (loc_data["year_month_dt"] <= future_end)
            ]

            # Target: future species richness
            if "edna_species_richness" not in future.columns or future["edna_species_richness"].isna().all():
                future_richness = np.nan
            else:
                future_richness = future["edna_species_richness"].max()

            if np.isnan(future_richness):
                continue

            # Aggregate historical features
            sample = {
                "lat_grid": lat,
                "lon_grid": lon,
                "prediction_origin": str(origin_month)[:10],
                "history_start": str(history_start)[:10],
                "future_end": str(future_end)[:10],
                "future_fish_species_richness": future_richness,
                "n_history_months": len(history),
            }

            # Historical oceanographic features (mean and std over history window)
            for feat in ocean_features:
                vals = history[feat].dropna()
                sample[f"hist_{feat}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_std"] = vals.std() if len(vals) > 1 else 0.0
                sample[f"hist_{feat}_trend"] = (
                    np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 2 else 0.0
                )

            # Historical fisheries features
            for feat in fish_features:
                vals = history[feat].dropna()
                sample[f"hist_{feat}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                sample[f"hist_{feat}_max"] = vals.max() if len(vals) > 0 else np.nan

            # Historical eDNA features
            for feat in edna_features:
                vals = history[feat].dropna()
                sample[f"hist_{feat}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_max"] = vals.max() if len(vals) > 0 else np.nan
                sample[f"hist_{feat}_latest"] = vals.iloc[-1] if len(vals) > 0 else np.nan

            # Modality availability in history
            for flag in modality_flags:
                sample[f"hist_{flag}_frac"] = (
                    history[flag].mean() if len(history) > 0 else 0.0
                )

            # Spatial features
            sample["latitude"] = lat
            sample["longitude"] = lon

            # Temporal features
            origin_dt = pd.Timestamp(origin_month)
            sample["origin_month_sin"] = np.sin(2 * np.pi * origin_dt.month / 12)
            sample["origin_month_cos"] = np.cos(2 * np.pi * origin_dt.month / 12)
            sample["origin_year"] = origin_dt.year

            samples.append(sample)

    result = pd.DataFrame(samples)
    logger.info(
        f"Built {len(result)} prediction samples from {len(locations)} locations"
    )

    if len(result) > 0 and "future_fish_species_richness" in result.columns:
        target = result["future_fish_species_richness"]
        logger.info(
            f"Target distribution: mean={target.mean():.1f}, "
            f"std={target.std():.1f}, "
            f"min={target.min():.0f}, max={target.max():.0f}"
        )

    return result


def create_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    method: str = "chronological",
) -> pd.DataFrame:
    """Create chronological or random train/val/test splits."""
    df = df.copy()

    if method == "chronological":
        df = df.sort_values("prediction_origin")
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        df["split"] = "test"
        df.iloc[:train_end, df.columns.get_loc("split")] = "train"
        df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"

        logger.info(
            f"Chronological split: train={train_end}, "
            f"val={val_end - train_end}, test={n - val_end}"
        )
    else:
        np.random.seed(42)
        rand = np.random.rand(len(df))
        df["split"] = np.where(
            rand < train_frac,
            "train",
            np.where(rand < train_frac + val_frac, "val", "test"),
        )
        logger.info(f"Random split: {df['split'].value_counts().to_dict()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build prediction target")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--aligned-data", default="data/processed/aligned_multimodal_data.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    aligned = pd.read_csv(args.aligned_data)
    logger.info(f"Loaded aligned data: {len(aligned)} records")

    prediction_cfg = config["prediction"]
    dataset = build_temporal_pairs(
        aligned,
        history_months=prediction_cfg["history_window_months"],
        horizon_months=prediction_cfg["prediction_horizon_months"],
    )

    if dataset.empty:
        logger.error("No valid prediction samples could be constructed")
        return

    split_cfg = config["splits"]
    dataset = create_train_val_test_split(
        dataset,
        train_frac=split_cfg["train_fraction"],
        val_frac=split_cfg["val_fraction"],
        method=split_cfg["method"],
    )

    output_path = Path(args.output_dir) / "prediction_dataset.csv"
    dataset.to_csv(output_path, index=False)
    logger.info(f"Prediction dataset saved to {output_path}")

    print(f"\n=== Prediction Dataset Summary ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Split distribution:\n{dataset['split'].value_counts()}")
    print(f"Target stats:\n{dataset['future_fish_species_richness'].describe()}")
    print(f"Feature columns: {len([c for c in dataset.columns if c.startswith('hist_')])}")


if __name__ == "__main__":
    main()
