"""
Download fisheries data from Global Fishing Watch (GFW).

GFW provides gridded fishing effort data (fishing hours by gear type)
derived from AIS vessel tracking data.

Data access: https://globalfishingwatch.org/data-download/
API: https://gateway.api.globalfishingwatch.org/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_gfw_data(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Attempt to download GFW fishing effort data.

    GFW provides pre-gridded data at 0.01 and 0.1 degree resolution.
    The public bulk download covers daily fishing effort by flag state and gear type.
    """
    logger.info("Attempting Global Fishing Watch data download...")
    logger.info(
        "Note: GFW bulk data requires registration at globalfishingwatch.org"
    )

    # Check for pre-downloaded GFW CSV files
    gfw_files = list(Path(output_dir).glob("*fishing*effort*.csv"))
    if gfw_files:
        logger.info(f"Found existing GFW files: {[f.name for f in gfw_files]}")
        dfs = [pd.read_csv(f) for f in gfw_files]
        return pd.concat(dfs, ignore_index=True)

    logger.warning(
        "No pre-downloaded GFW data found.\n"
        "Download instructions:\n"
        "1. Register at https://globalfishingwatch.org/data-download/\n"
        "2. Download 'Daily Fishing Effort at 10th Degree Resolution by Flag State and Gear Type'\n"
        "3. Filter for the study region and place CSV files in data/raw/\n"
        "Alternative: Use the GFW API with an API key"
    )
    return pd.DataFrame()


def generate_synthetic_fisheries_data(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    spatial_res: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Generate realistic fisheries effort data based on known North Sea patterns.
    Used ONLY when real data cannot be obtained.
    """
    np.random.seed(42)

    lats = np.arange(lat_min, lat_max, spatial_res)
    lons = np.arange(lon_min, lon_max, spatial_res)
    months = pd.date_range(start_date, end_date, freq="MS")

    records = []
    for month in months:
        month_of_year = month.month
        # Fishing effort is lower in winter (rough seas) and higher in summer
        seasonal_effort = 0.7 + 0.3 * np.sin(2 * np.pi * (month_of_year - 2) / 12)

        for lat in lats:
            for lon in lons:
                # Higher effort in southern North Sea (more accessible)
                lat_factor = 1.0 - 0.5 * (lat - lat_min) / (lat_max - lat_min)
                # Higher effort closer to ports (coast)
                coast_factor = max(0.3, 1.0 - abs(lon) / 10.0)

                base_effort = 50.0 * seasonal_effort * lat_factor * coast_factor
                fishing_hours = max(
                    0.0, base_effort + np.random.exponential(base_effort * 0.3)
                )
                vessel_count = max(0, int(fishing_hours / 8 + np.random.poisson(2)))

                if fishing_hours < 1.0:
                    fishing_hours = 0.0
                    vessel_count = 0

                records.append(
                    {
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "date": month.strftime("%Y-%m-%d"),
                        "year": month.year,
                        "month": month_of_year,
                        "fishing_hours": round(fishing_hours, 1),
                        "vessel_count": vessel_count,
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} fisheries records "
        f"({len(lats)} lats x {len(lons)} lons x {len(months)} months)"
    )
    return df


def validate_fisheries_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean fisheries data."""
    initial = len(df)

    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])
        df = df[
            (df["latitude"] >= -90) & (df["latitude"] <= 90)
            & (df["longitude"] >= -180) & (df["longitude"] <= 180)
        ]

    if "fishing_hours" in df.columns:
        df = df[df["fishing_hours"] >= 0]

    logger.info(f"Validation: {initial} -> {len(df)} records")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download fisheries data")
    parser.add_argument(
        "--config", default="configs/default_config.yaml", help="Config file path"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    args = parser.parse_args()

    config = load_config(args.config)
    region = config["study_region"]
    output_dir = args.output_dir or config["data"]["raw_dir"]
    os.makedirs(output_dir, exist_ok=True)

    df = download_gfw_data(
        lat_min=region["lat_min"],
        lat_max=region["lat_max"],
        lon_min=region["lon_min"],
        lon_max=region["lon_max"],
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=output_dir,
    )

    if df.empty:
        if args.use_synthetic:
            logger.warning("GENERATING SYNTHETIC DATA for methodological demonstration.")
            df = generate_synthetic_fisheries_data(
                lat_min=region["lat_min"],
                lat_max=region["lat_max"],
                lon_min=region["lon_min"],
                lon_max=region["lon_max"],
                spatial_res=region["spatial_resolution_deg"],
                start_date=args.start_date,
                end_date=args.end_date,
            )
        else:
            logger.error("No fisheries data available. Use --use-synthetic or download manually.")
            sys.exit(1)

    df = validate_fisheries_data(df)

    csv_path = Path(output_dir) / "fisheries_effort_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} records to {csv_path}")

    print(f"\n=== Fisheries Data Summary ===")
    print(f"Records: {len(df)}")
    if "fishing_hours" in df.columns:
        print(f"Total fishing hours: {df['fishing_hours'].sum():.0f}")
        print(f"Mean monthly fishing hours per cell: {df['fishing_hours'].mean():.1f}")
    if "date" in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
