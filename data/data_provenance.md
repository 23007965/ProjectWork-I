# Data Provenance

## Lineage: Source -> Download -> Processing -> Feature/Target -> Experiment

### 1. OBIS Fish Occurrences (DS-001)
- **Source**: OBIS REST API v3 (https://api.obis.org/v3)
- **Download**: `src/data/download_edna.py` -> `data/raw/obis_fish_occurrences_raw.csv`
- **Cleaning**: `src/data/download_edna.py` (validate_edna_data) -> `data/raw/obis_fish_occurrences_clean.csv`
- **Richness summary**: `src/data/download_edna.py` (compute_richness_summary) -> `data/raw/fish_species_richness_by_location_month.csv`
- **Alignment**: `src/data/align_spatiotemporal_data.py` (aggregate_edna) -> contributes to `data/processed/aligned_multimodal_data.csv`
- **Target construction**: `src/data/build_prediction_target.py` -> `data/processed/prediction_dataset.csv` (column: `future_fish_species_richness`)

### 2. Oceanographic Data (DS-002)
- **Source**: Copernicus Marine Service / Synthetic fallback
- **Download**: `src/data/download_oceanographic.py` -> `data/raw/oceanographic_data.csv`
- **Alignment**: `src/data/align_spatiotemporal_data.py` (aggregate_oceanographic) -> contributes to `data/processed/aligned_multimodal_data.csv`
- **Feature extraction**: `src/data/build_prediction_target.py` -> `data/processed/prediction_dataset.csv` (columns: `hist_sea_surface_temperature_*`, `hist_salinity_*`, etc.)

### 3. Fisheries Data (DS-003)
- **Source**: Global Fishing Watch / Synthetic fallback
- **Download**: `src/data/download_fisheries.py` -> `data/raw/fisheries_effort_data.csv`
- **Alignment**: `src/data/align_spatiotemporal_data.py` (aggregate_fisheries) -> contributes to `data/processed/aligned_multimodal_data.csv`
- **Feature extraction**: `src/data/build_prediction_target.py` -> `data/processed/prediction_dataset.csv` (columns: `hist_fishing_hours_*`, `hist_vessel_count_*`)

### 4. Integrated Prediction Dataset
- **Source**: All three aligned datasets
- **Script**: `src/features/build_features.py` -> `data/processed/features/` (per-modality feature splits)
- **Used in**: All model training and evaluation scripts

## Data Splits
- **Train**: Chronological first 60%
- **Validation**: Middle 20%
- **Test**: Last 20%
- Split is defined in `src/data/build_prediction_target.py`

## Important Notes
- Raw OBIS data is real, downloaded via API on 2026-10-18
- Oceanographic and fisheries data are synthetic in current implementation
- No data leakage: future target is computed from future observations only
