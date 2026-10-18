# An Explainable Cross-Modal Attention Framework for Marine Biodiversity Prediction

Using Oceanographic, Fisheries, and Environmental DNA Data

## Overview

This project implements an explainable cross-modal attention model that jointly learns from three complementary marine data modalities to predict future fish species richness in the North Sea. The framework explicitly models inter-modality ecological interactions using a multi-head attention mechanism, providing interpretable predictions for marine conservation and fisheries management.

## Key Results

| Model | Modalities | MAE | Improvement |
|-------|-----------|-----|-------------|
| **Cross-Modal Attention** | All 3 | **0.086** | **-33%** |
| CatBoost (best baseline) | Single | 0.128 | — |
| Early Fusion (CatBoost) | All 3 | 0.134 | +5% (worse) |
| Late Fusion | All 3 | 0.134 | +5% (worse) |

Ablation study: Removing eDNA data increases MAE by 39%, confirming species occurrence data as the most critical modality.

## Project Structure

```
marine-biodiversity-prediction/
├── configs/              # Configuration files
├── data/                 # Data files, registry, provenance
│   ├── raw/              # Downloaded data
│   ├── processed/        # Aligned and processed data
│   ├── DATA_SOURCES.md   # Data source documentation
│   └── dataset_registry.csv
├── docs/                 # Research documents
│   ├── research_design.md
│   ├── literature_review.md
│   ├── study_region_selection.md
│   ├── dataset_audit.md
│   └── limitations.md
├── references/           # Bibliography
├── src/                  # Source code
│   ├── data/             # Data download and processing
│   ├── eda/              # Exploratory data analysis
│   ├── features/         # Feature engineering
│   ├── models/           # Model implementations
│   ├── evaluation/       # Evaluation and ablation
│   └── explainability/   # SHAP and attention analysis
├── experiments/          # Experiment results
├── reports/              # Final results and figures
│   ├── eda/              # EDA figures
│   └── final_results.csv
├── paper/                # Research manuscript
├── tests/                # Unit tests
└── scripts/              # Pipeline scripts
```

## Quick Start

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
# With synthetic oceanographic/fisheries data (demonstration mode)
python scripts/run_pipeline.py --use-synthetic

# Individual stages
python src/data/download_edna.py --config configs/default_config.yaml
python src/data/download_oceanographic.py --config configs/default_config.yaml --use-synthetic
python src/data/download_fisheries.py --config configs/default_config.yaml --use-synthetic
python src/data/align_spatiotemporal_data.py --config configs/default_config.yaml
python src/data/build_prediction_target.py --config configs/default_config.yaml
python src/features/build_features.py
python src/models/baselines.py --n-seeds 5
python src/models/train_cross_modal.py --n-seeds 5
python src/evaluation/run_ablation_study.py --n-seeds 3
python src/eda/analyze_integrated_data.py
```

## Datasets

| Dataset | Source | Modality | Coverage |
|---------|--------|----------|----------|
| Fish Occurrences | OBIS | Biodiversity | North Sea, 1965-2025 |
| Oceanographic | CMEMS | Environmental | North Sea, 2015-2023 |
| Fisheries Effort | GFW | Anthropogenic | North Sea, 2015-2023 |

See `data/DATA_SOURCES.md` for full documentation.

## Model Architecture

1. **Modality-specific encoders**: MLP encoders map each modality to a shared 64-dim embedding space
2. **Cross-modal attention**: 2-layer, 4-head multi-head attention learns inter-modality relationships
3. **Missing-modality masking**: Binary masks allow inference with incomplete data
4. **Prediction head**: Fused embeddings predict future species richness

## Research Questions

- RQ1: Can multimodal integration improve species richness prediction?
- RQ2: Does cross-modal attention outperform conventional fusion?
- RQ3: Which modality contributes most?
- RQ4: Can the model provide ecologically meaningful explanations?
- RQ5: How robust is the model to missing modalities?

## Citation

If using this work, please cite:

```
An Explainable Cross-Modal Attention Framework for Marine Biodiversity Prediction
Using Oceanographic, Fisheries, and Environmental DNA Data.
GitHub: https://github.com/23007965/ProjectWork-I.git
```

## License

This project is released for academic research purposes.
Data licenses: OBIS (CC BY 4.0), GFW (CC BY-SA 4.0), CMEMS (Copernicus Licence).
