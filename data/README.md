# Data Directory

## Structure

- `raw/` - Downloaded and generated data files
- `processed/` - Aligned, feature-engineered data ready for modeling
- `external/` - External reference data (if any)

## Key Files

| File | Description | Records |
|------|-------------|---------|
| `raw/obis_fish_occurrences_raw.csv` | Raw OBIS fish occurrence records | 50,000 |
| `raw/obis_fish_occurrences_clean.csv` | Validated occurrence records | 29,640 |
| `raw/oceanographic_data.csv` | Monthly oceanographic variables | 266,112 |
| `raw/fisheries_effort_data.csv` | Monthly fisheries effort | 266,112 |
| `processed/aligned_multimodal_data.csv` | Three-modality aligned data | 268,543 |
| `processed/prediction_dataset.csv` | Final modeling dataset | 13,832 |

## Documentation

- `DATA_SOURCES.md` - Description of each data source
- `dataset_registry.csv` - Machine-readable dataset inventory
- `data_dictionary.csv` - Field definitions for all variables
- `data_provenance.md` - Full lineage from source to experiment
- `download_manifest.csv` - Download log with file sizes and status
- `citations_and_licenses.md` - Attribution and licensing requirements

## Reproduction

```bash
# Download real eDNA data from OBIS
python src/data/download_edna.py --config configs/default_config.yaml

# Generate synthetic ocean/fisheries data (or download real data)
python src/data/download_oceanographic.py --config configs/default_config.yaml --use-synthetic
python src/data/download_fisheries.py --config configs/default_config.yaml --use-synthetic

# Align and build prediction target
python src/data/align_spatiotemporal_data.py --config configs/default_config.yaml
python src/data/build_prediction_target.py --config configs/default_config.yaml

# Build features
python src/features/build_features.py
```

## Note on Data Size
Raw data files are not committed to the repository (see .gitignore). Use the download scripts above to reproduce them.
