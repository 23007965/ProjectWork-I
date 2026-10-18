"""
Download oceanographic data from Copernicus Marine Service (CMEMS).

Variables: SST, salinity, dissolved oxygen, chlorophyll-a.
Data product: Global Ocean Physics and Biogeochemistry reanalysis/analysis.

Requires: copernicusmarine Python package or manual download.
This script provides both API-based and manual download instructions.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def try_copernicus_download(
    dataset_id: str,
    variables: list,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
    output_path: str,
) -> bool:
    """Attempt download via copernicusmarine Python package."""
    try:
        import copernicusmarine as cm

        cm.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            start_datetime=start_date,
            end_datetime=end_date,
            output_filename=output_path,
            output_directory=str(Path(output_path).parent),
        )
        return True
    except ImportError:
        logger.warning(
            "copernicusmarine package not installed. "
            "Install with: pip install copernicusmarine"
        )
        return False
    except Exception as e:
        logger.error(f"Copernicus download failed: {e}")
        return False


def generate_synthetic_oceanographic_data(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    spatial_res: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Generate realistic oceanographic data based on known North Sea patterns.
    Used ONLY when real data cannot be downloaded.
    Each value is drawn from scientifically plausible distributions with
    spatial and seasonal variability matching published climatologies.
    """
    np.random.seed(42)

    lats = np.arange(lat_min, lat_max, spatial_res)
    lons = np.arange(lon_min, lon_max, spatial_res)
    months = pd.date_range(start_date, end_date, freq="MS")

    records = []
    for month in months:
        month_of_year = month.month
        for lat in lats:
            for lon in lons:
                lat_factor = (lat - lat_min) / (lat_max - lat_min)
                seasonal = np.sin(2 * np.pi * (month_of_year - 3) / 12)

                sst = 8.0 + 4.0 * seasonal - 2.0 * lat_factor + np.random.normal(0, 0.5)
                salinity = 34.5 + 0.5 * lat_factor + np.random.normal(0, 0.3)
                dissolved_oxygen = (
                    7.0 - 0.5 * seasonal + 0.3 * lat_factor + np.random.normal(0, 0.2)
                )
                chlorophyll = max(
                    0.1,
                    1.5
                    + 2.0 * np.sin(2 * np.pi * (month_of_year - 4) / 12)
                    - 0.3 * lat_factor
                    + np.random.normal(0, 0.4),
                )

                records.append(
                    {
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "date": month.strftime("%Y-%m-%d"),
                        "year": month.year,
                        "month": month_of_year,
                        "sea_surface_temperature": round(sst, 2),
                        "salinity": round(salinity, 2),
                        "dissolved_oxygen": round(dissolved_oxygen, 2),
                        "chlorophyll_a": round(chlorophyll, 3),
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} oceanographic records "
        f"({len(lats)} lats x {len(lons)} lons x {len(months)} months)"
    )
    return df


def load_netcdf_to_dataframe(nc_path: str, variables: list) -> pd.DataFrame:
    """Load a NetCDF file into a flat DataFrame."""
    try:
        import xarray as xr

        ds = xr.open_dataset(nc_path)
        df = ds.to_dataframe().reset_index()
        logger.info(f"Loaded {len(df)} records from {nc_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {nc_path}: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download oceanographic data")
    parser.add_argument(
        "--config", default="configs/default_config.yaml", help="Config file path"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Generate synthetic data if real data unavailable",
    )
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    args = parser.parse_args()

    config = load_config(args.config)
    region = config["study_region"]
    output_dir = args.output_dir or config["data"]["raw_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Attempt real data download first
    nc_output = str(Path(output_dir) / "oceanographic_data.nc")
    success = try_copernicus_download(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
        variables=["thetao", "so"],
        lat_min=region["lat_min"],
        lat_max=region["lat_max"],
        lon_min=region["lon_min"],
        lon_max=region["lon_max"],
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=nc_output,
    )

    if success and os.path.exists(nc_output):
        df = load_netcdf_to_dataframe(nc_output, config["data"]["oceanographic"]["variables"])
    else:
        logger.warning("Real data download failed. Checking for existing files...")
        existing_nc = list(Path(output_dir).glob("*.nc"))
        if existing_nc:
            logger.info(f"Found existing file: {existing_nc[0]}")
            df = load_netcdf_to_dataframe(str(existing_nc[0]), config["data"]["oceanographic"]["variables"])
        elif args.use_synthetic:
            logger.warning(
                "GENERATING SYNTHETIC DATA. Results should be interpreted as methodological demonstration only."
            )
            df = generate_synthetic_oceanographic_data(
                lat_min=region["lat_min"],
                lat_max=region["lat_max"],
                lon_min=region["lon_min"],
                lon_max=region["lon_max"],
                spatial_res=region["spatial_resolution_deg"],
                start_date=args.start_date,
                end_date=args.end_date,
            )
        else:
            logger.error(
                "No data available. Use --use-synthetic or download manually.\n"
                "Manual download instructions:\n"
                "1. Visit https://data.marine.copernicus.eu/\n"
                "2. Search for 'Global Ocean Physics Reanalysis' (GLORYS12V1)\n"
                "3. Select variables: temperature, salinity\n"
                f"4. Set region: lat [{region['lat_min']}, {region['lat_max']}], "
                f"lon [{region['lon_min']}, {region['lon_max']}]\n"
                "5. Download as NetCDF and place in data/raw/"
            )
            sys.exit(1)

    csv_path = Path(output_dir) / "oceanographic_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} records to {csv_path}")

    print(f"\n=== Oceanographic Data Summary ===")
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if "date" in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    for col in ["sea_surface_temperature", "salinity", "dissolved_oxygen", "chlorophyll_a"]:
        if col in df.columns:
            print(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")


if __name__ == "__main__":
    main()
