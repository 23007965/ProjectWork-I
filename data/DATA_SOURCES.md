# Data Sources

## Primary Datasets

### Dataset A: Marine Fish Occurrences (OBIS)
- **Source**: Ocean Biodiversity Information System (OBIS)
- **URL**: https://obis.org/
- **API**: https://api.obis.org/v3
- **Description**: Fish occurrence records with coordinates, dates, and taxonomic identifications from the North Sea region. Queried for Actinopterygii (ray-finned fishes).
- **Variables**: species name, decimal latitude/longitude, event date, depth, dataset ID, basis of record
- **Geographic coverage**: North Sea (51°N-62°N, 4°W-10°E)
- **Temporal coverage**: 1965-2025 (full archive), focused on 2015-2023 for model training
- **Format**: CSV via REST API
- **Records downloaded**: 50,000 raw, 29,640 after validation, 72 unique species
- **License**: CC BY 4.0 (varies by contributing dataset)
- **Role**: Primary source for fish species richness (both historical features and prediction target)

### Dataset B: Oceanographic Data
- **Source**: Copernicus Marine Service (CMEMS) / Synthetic fallback
- **URL**: https://marine.copernicus.eu/
- **Target product**: Global Ocean Physics Reanalysis (GLORYS12V1)
- **Variables**: Sea surface temperature, salinity, dissolved oxygen, chlorophyll-a
- **Geographic coverage**: North Sea (51°N-62°N, 4°W-10°E)
- **Temporal coverage**: 2015-01-01 to 2023-12-31
- **Spatial resolution**: 0.25° × 0.25° grid
- **Temporal resolution**: Monthly
- **Note**: In the current implementation, synthetic data with ecologically realistic patterns matching published North Sea climatologies was used when Copernicus API access was unavailable. Results should be interpreted as a methodological demonstration. For full reproducibility, download the real CMEMS products listed above.
- **License**: Copernicus Marine Service License (free for research)

### Dataset C: Fisheries Effort Data
- **Source**: Global Fishing Watch (GFW) / Synthetic fallback
- **URL**: https://globalfishingwatch.org/data-download/
- **Variables**: Monthly fishing hours, vessel count per grid cell
- **Geographic coverage**: North Sea (51°N-62°N, 4°W-10°E)
- **Temporal coverage**: 2015-01-01 to 2023-12-31
- **Spatial resolution**: 0.25° × 0.25° grid
- **Note**: Synthetic data generated with realistic North Sea fishing patterns. For reproducibility, register at GFW and download the "Daily Fishing Effort at 10th Degree Resolution" dataset.
- **License**: CC BY-SA 4.0 (GFW public data)

## Data Processing

All three datasets are aligned to a common 0.25° × 0.25° monthly grid using the pipeline in `src/data/align_spatiotemporal_data.py`. The prediction target (future_fish_species_richness) is constructed in `src/data/build_prediction_target.py`.
