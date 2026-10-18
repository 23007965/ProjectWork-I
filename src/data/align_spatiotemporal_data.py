"""
Spatial-temporal alignment of oceanographic, fisheries, and eDNA datasets.

Aligns all three modalities to a common spatial grid and temporal resolution,
creating a unified dataset for model training with explicit missing-modality indicators.
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


def assign_grid_cell(df: pd.DataFrame, resolution: float, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Assign each record to the nearest grid cell center."""
    df = df.copy()
    df["lat_grid"] = (np.round(df[lat_col] / resolution) * resolution).round(4)
    df["lon_grid"] = (np.round(df[lon_col] / resolution) * resolution).round(4)
    return df


def assign_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extract year-month from a date column."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["year_month"] = dt.dt.to_period("M").astype(str)
    return df


def aggregate_oceanographic(df: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """Aggregate oceanographic data to monthly grid cells."""
    logger.info("Aggregating oceanographic data to monthly grid cells...")

    lat_col = next((c for c in ["latitude", "lat", "lat_grid"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude", "lon", "lon_grid"] if c in df.columns), None)
    date_col = next((c for c in ["date", "time", "date_parsed"] if c in df.columns), None)

    if not lat_col or not lon_col or not date_col:
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    df = assign_grid_cell(df, resolution, lat_col, lon_col)
    df = assign_month(df, date_col)

    ocean_vars = [
        c for c in ["sea_surface_temperature", "salinity", "dissolved_oxygen", "chlorophyll_a",
                     "thetao", "so", "o2", "chl", "sst"]
        if c in df.columns
    ]

    agg_dict = {var: "mean" for var in ocean_vars}
    agg_dict["year"] = "first"
    agg_dict["month"] = "first"

    result = df.groupby(["lat_grid", "lon_grid", "year_month"]).agg(agg_dict).reset_index()

    # Rename standard variables
    rename_map = {"thetao": "sea_surface_temperature", "so": "salinity", "o2": "dissolved_oxygen", "chl": "chlorophyll_a"}
    result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

    logger.info(f"Oceanographic: {len(result)} grid-month records")
    return result


def aggregate_fisheries(df: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """Aggregate fisheries data to monthly grid cells."""
    logger.info("Aggregating fisheries data to monthly grid cells...")

    lat_col = next((c for c in ["latitude", "lat", "lat_grid"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude", "lon", "lon_grid"] if c in df.columns), None)
    date_col = next((c for c in ["date", "time", "date_parsed"] if c in df.columns), None)

    if not lat_col or not lon_col or not date_col:
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    df = assign_grid_cell(df, resolution, lat_col, lon_col)
    df = assign_month(df, date_col)

    fish_vars = [c for c in ["fishing_hours", "vessel_count"] if c in df.columns]

    agg_dict = {}
    for var in fish_vars:
        agg_dict[var] = "sum"
    agg_dict["year"] = "first"
    agg_dict["month"] = "first"

    result = df.groupby(["lat_grid", "lon_grid", "year_month"]).agg(agg_dict).reset_index()

    logger.info(f"Fisheries: {len(result)} grid-month records")
    return result


def aggregate_edna(df: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """Aggregate eDNA data to monthly grid cells with richness counts."""
    logger.info("Aggregating eDNA data to monthly grid cells...")

    lat_col = next((c for c in ["lat_grid", "decimalLatitude", "latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["lon_grid", "decimalLongitude", "longitude"] if c in df.columns), None)
    date_col = next((c for c in ["date_parsed", "eventDate", "date"] if c in df.columns), None)

    if not lat_col or not lon_col:
        raise ValueError(f"Missing coordinate columns. Found: {list(df.columns)}")

    df = assign_grid_cell(df, resolution, lat_col, lon_col)

    if date_col:
        df = assign_month(df, date_col)
    elif "year_month" in df.columns or "year_month_str" in df.columns:
        ym_col = "year_month" if "year_month" in df.columns else "year_month_str"
        df["year_month"] = df[ym_col].astype(str)
    else:
        raise ValueError("No date information available in eDNA data")

    species_col = next((c for c in ["species", "scientificName"] if c in df.columns), None)

    if species_col:
        result = (
            df.groupby(["lat_grid", "lon_grid", "year_month"])
            .agg(
                edna_species_richness=(species_col, "nunique"),
                edna_total_detections=(species_col, "count"),
            )
            .reset_index()
        )
    elif "species_richness" in df.columns:
        result = (
            df.groupby(["lat_grid", "lon_grid", "year_month"])
            .agg(
                edna_species_richness=("species_richness", "max"),
                edna_total_detections=("n_records", "sum") if "n_records" in df.columns else ("species_richness", "count"),
            )
            .reset_index()
        )
    else:
        raise ValueError("No species or richness column found in eDNA data")

    logger.info(f"eDNA: {len(result)} grid-month records")
    return result


def merge_modalities(
    ocean_df: pd.DataFrame,
    fish_df: pd.DataFrame,
    edna_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge three modalities on spatial-temporal keys with missing-modality indicators."""
    logger.info("Merging modalities...")

    merge_keys = ["lat_grid", "lon_grid", "year_month"]

    # Start from all unique grid-month combinations across all datasets
    all_keys = set()
    for df in [ocean_df, fish_df, edna_df]:
        if not df.empty:
            keys = df[merge_keys].drop_duplicates()
            all_keys.update([tuple(row) for _, row in keys.iterrows()])

    master = pd.DataFrame(list(all_keys), columns=merge_keys)
    logger.info(f"Total unique grid-month locations: {len(master)}")

    # Merge each modality
    if not ocean_df.empty:
        ocean_cols = [c for c in ocean_df.columns if c not in merge_keys or c in merge_keys]
        master = master.merge(ocean_df, on=merge_keys, how="left", suffixes=("", "_ocean"))
        master["has_oceanographic"] = master[
            [c for c in ocean_df.columns if c not in merge_keys]
        ].notna().any(axis=1).astype(int)
    else:
        master["has_oceanographic"] = 0

    if not fish_df.empty:
        master = master.merge(fish_df, on=merge_keys, how="left", suffixes=("", "_fish"))
        master["has_fisheries"] = master[
            [c for c in fish_df.columns if c not in merge_keys]
        ].notna().any(axis=1).astype(int)
    else:
        master["has_fisheries"] = 0

    if not edna_df.empty:
        master = master.merge(edna_df, on=merge_keys, how="left", suffixes=("", "_edna"))
        master["has_edna"] = master[
            [c for c in edna_df.columns if c not in merge_keys]
        ].notna().any(axis=1).astype(int)
    else:
        master["has_edna"] = 0

    master["n_modalities_present"] = (
        master["has_oceanographic"] + master["has_fisheries"] + master["has_edna"]
    )

    logger.info(
        f"Merged dataset: {len(master)} rows, "
        f"3-modality: {(master['n_modalities_present'] == 3).sum()}, "
        f"2-modality: {(master['n_modalities_present'] == 2).sum()}, "
        f"1-modality: {(master['n_modalities_present'] == 1).sum()}"
    )
    return master


def main():
    parser = argparse.ArgumentParser(description="Align spatial-temporal data")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--ocean-csv", default="data/raw/oceanographic_data.csv")
    parser.add_argument("--fish-csv", default="data/raw/fisheries_effort_data.csv")
    parser.add_argument("--edna-csv", default="data/raw/obis_fish_occurrences_clean.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = load_config(args.config)
    resolution = config["study_region"]["spatial_resolution_deg"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    ocean_df = pd.read_csv(args.ocean_csv) if os.path.exists(args.ocean_csv) else pd.DataFrame()
    fish_df = pd.read_csv(args.fish_csv) if os.path.exists(args.fish_csv) else pd.DataFrame()
    edna_df = pd.read_csv(args.edna_csv) if os.path.exists(args.edna_csv) else pd.DataFrame()

    logger.info(
        f"Loaded: ocean={len(ocean_df)}, fisheries={len(fish_df)}, edna={len(edna_df)} records"
    )

    # Aggregate each modality
    ocean_agg = aggregate_oceanographic(ocean_df, resolution) if not ocean_df.empty else pd.DataFrame()
    fish_agg = aggregate_fisheries(fish_df, resolution) if not fish_df.empty else pd.DataFrame()
    edna_agg = aggregate_edna(edna_df, resolution) if not edna_df.empty else pd.DataFrame()

    # Merge
    merged = merge_modalities(ocean_agg, fish_agg, edna_agg)

    output_path = Path(args.output_dir) / "aligned_multimodal_data.csv"
    merged.to_csv(output_path, index=False)
    logger.info(f"Aligned data saved to {output_path}")

    print(f"\n=== Alignment Summary ===")
    print(f"Total records: {len(merged)}")
    print(f"Grid cells with all 3 modalities: {(merged['n_modalities_present'] == 3).sum()}")
    print(f"Grid cells with 2 modalities: {(merged['n_modalities_present'] == 2).sum()}")
    print(f"Grid cells with 1 modality: {(merged['n_modalities_present'] == 1).sum()}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
