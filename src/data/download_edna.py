"""
Download marine fish eDNA/occurrence data from OBIS (Ocean Biodiversity Information System).

OBIS provides species occurrence records with coordinates, dates, and taxonomic information.
We query for Actinopterygii (ray-finned fishes) within the study region.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OBIS_API_BASE = "https://api.obis.org/v3"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_obis_occurrences(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    taxon_name: str = "Actinopterygii",
    start_date: str = None,
    end_date: str = None,
    max_records: int = 50000,
) -> pd.DataFrame:
    """Fetch fish occurrence records from the OBIS API."""

    logger.info(
        f"Querying OBIS for {taxon_name} in [{lat_min},{lat_max}] x [{lon_min},{lon_max}]"
    )

    # First resolve taxon ID
    taxon_url = f"{OBIS_API_BASE}/taxon/{taxon_name}"
    resp = requests.get(taxon_url, timeout=30)
    if resp.status_code != 200:
        # Try searching by name
        search_url = f"{OBIS_API_BASE}/taxon/autocomplete"
        resp = requests.get(search_url, params={"q": taxon_name}, timeout=30)
        if resp.status_code == 200:
            results = resp.json()
            if results:
                taxon_id = results[0].get("taxonID")
                logger.info(f"Resolved {taxon_name} to taxonID={taxon_id}")
            else:
                logger.warning(f"Could not resolve taxon: {taxon_name}")
                taxon_id = None
        else:
            taxon_id = None
    else:
        data = resp.json()
        taxon_id = data.get("taxonID")

    all_records = []
    offset = 0
    page_size = 5000

    params = {
        "geometry": f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))",
        "size": page_size,
        "offset": offset,
    }

    if taxon_name:
        params["scientificname"] = taxon_name

    if start_date:
        params["startdate"] = start_date
    if end_date:
        params["enddate"] = end_date

    while offset < max_records:
        params["offset"] = offset
        try:
            resp = requests.get(
                f"{OBIS_API_BASE}/occurrence", params=params, timeout=60
            )
            if resp.status_code != 200:
                logger.error(f"OBIS API returned status {resp.status_code}")
                break

            data = resp.json()
            results = data.get("results", [])
            total = data.get("total", 0)

            if not results:
                break

            all_records.extend(results)
            offset += page_size
            logger.info(f"Fetched {len(all_records)}/{total} records")

            if len(all_records) >= total or len(all_records) >= max_records:
                break

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            break

    if not all_records:
        logger.warning("No records returned from OBIS")
        return pd.DataFrame()

    df = pd.json_normalize(all_records)

    keep_cols = [
        c
        for c in [
            "scientificName",
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
            "date_mid",
            "date_year",
            "date_start",
            "dataset_id",
            "datasetName",
            "basisOfRecord",
            "occurrenceID",
            "taxonID",
            "depth",
            "sst",
            "sss",
        ]
        if c in df.columns
    ]

    df = df[keep_cols].copy()
    logger.info(
        f"Downloaded {len(df)} occurrence records with {df['species'].nunique() if 'species' in df.columns else 'unknown'} unique species"
    )
    return df


def validate_edna_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean eDNA occurrence data."""
    initial_count = len(df)

    # Remove records without coordinates
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])

    # Remove records without species identification
    if "species" in df.columns:
        df = df.dropna(subset=["species"])
        df = df[df["species"].str.strip() != ""]

    # Remove records without dates
    date_col = None
    for col in ["eventDate", "date_mid", "date_start"]:
        if col in df.columns and df[col].notna().sum() > 0:
            date_col = col
            break

    if date_col:
        df = df.dropna(subset=[date_col])
        df["date_parsed"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.dropna(subset=["date_parsed"])
    else:
        if "date_year" in df.columns:
            df = df.dropna(subset=["date_year"])
            df["date_parsed"] = pd.to_datetime(
                df["date_year"].astype(int).astype(str) + "-07-01", errors="coerce"
            )
        else:
            logger.warning("No usable date column found")
            return df

    # Validate coordinate ranges
    df = df[
        (df["decimalLatitude"] >= -90)
        & (df["decimalLatitude"] <= 90)
        & (df["decimalLongitude"] >= -180)
        & (df["decimalLongitude"] <= 180)
    ]

    logger.info(
        f"Validation: {initial_count} -> {len(df)} records ({initial_count - len(df)} removed)"
    )
    return df


def compute_richness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute species richness per location-month for downstream analysis."""
    if "date_parsed" not in df.columns:
        logger.error("date_parsed column required")
        return pd.DataFrame()

    df["year_month"] = df["date_parsed"].dt.to_period("M")
    df["lat_grid"] = (df["decimalLatitude"] * 4).round() / 4  # 0.25 degree grid
    df["lon_grid"] = (df["decimalLongitude"] * 4).round() / 4

    richness = (
        df.groupby(["lat_grid", "lon_grid", "year_month"])
        .agg(
            species_richness=("species", "nunique"),
            n_records=("species", "count"),
            species_list=("species", lambda x: "|".join(sorted(x.unique()))),
            n_datasets=("dataset_id", "nunique") if "dataset_id" in df.columns else ("species", "count"),
        )
        .reset_index()
    )

    richness["year_month_str"] = richness["year_month"].astype(str)
    logger.info(
        f"Richness summary: {len(richness)} location-month observations, "
        f"richness range [{richness['species_richness'].min()}, {richness['species_richness'].max()}]"
    )
    return richness


def main():
    parser = argparse.ArgumentParser(description="Download eDNA data from OBIS")
    parser.add_argument(
        "--config", default="configs/default_config.yaml", help="Config file path"
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument(
        "--max-records", type=int, default=50000, help="Maximum records to fetch"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    region = config["study_region"]
    output_dir = args.output_dir or config["data"]["raw_dir"]
    os.makedirs(output_dir, exist_ok=True)

    df = fetch_obis_occurrences(
        lat_min=region["lat_min"],
        lat_max=region["lat_max"],
        lon_min=region["lon_min"],
        lon_max=region["lon_max"],
        taxon_name=config["data"]["edna"].get("target_group", "Actinopterygii"),
        max_records=args.max_records,
    )

    if df.empty:
        logger.error("No data downloaded. Check API availability and parameters.")
        sys.exit(1)

    raw_path = Path(output_dir) / "obis_fish_occurrences_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to {raw_path}")

    df_clean = validate_edna_data(df)
    clean_path = Path(output_dir) / "obis_fish_occurrences_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Clean data saved to {clean_path}")

    richness = compute_richness_summary(df_clean)
    richness_path = Path(output_dir) / "fish_species_richness_by_location_month.csv"
    richness.to_csv(richness_path, index=False)
    logger.info(f"Richness summary saved to {richness_path}")

    print(f"\n=== eDNA Data Summary ===")
    print(f"Total raw records: {len(df)}")
    print(f"Valid records after cleaning: {len(df_clean)}")
    if "species" in df_clean.columns:
        print(f"Unique species: {df_clean['species'].nunique()}")
    if "date_parsed" in df_clean.columns:
        print(
            f"Date range: {df_clean['date_parsed'].min()} to {df_clean['date_parsed'].max()}"
        )
    print(f"Location-month observations: {len(richness)}")


if __name__ == "__main__":
    main()
