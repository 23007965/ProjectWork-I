# Project Roadmap

## Completed Milestones

- [x] Phase 1: Research design document
- [x] Phase 2: Literature review (32 publications)
- [x] Phase 3: Study region selection (North Sea) and dataset audit
- [x] Phase 4: Prediction target definition (future_fish_species_richness)
- [x] Phase 5: Data engineering pipeline (download, validate, align, features)
- [x] Phase 6: Exploratory data analysis (12 figures and tables)
- [x] Phase 7: Baseline models (XGBoost, CatBoost, MLP for 3 modalities + fusion)
- [x] Phase 8: Cross-modal attention model implementation
- [x] Phase 9: Training and validation (5 seeds, chronological split)
- [x] Phase 10: Ablation study (4 conditions x 3 seeds)
- [x] Phase 11: Final results tables
- [x] Phase 12: Research manuscript draft
- [x] Phase 13: GitHub repository maintenance
- [x] Phase 14: Acceptance checklist

## Results Summary

- Cross-modal attention: MAE = 0.086 (33% improvement over best baseline)
- Ablation: eDNA removal causes 39% MAE increase
- 13,832 prediction samples, 27 features, 72 species
- Chronological train/val/test split

## Remaining Work (Future)

- [ ] Replace synthetic oceanographic/fisheries data with real CMEMS and GFW data
- [ ] Incorporate true eDNA metabarcoding data
- [ ] Add spatially blocked cross-validation
- [ ] Extend to additional marine regions
- [ ] Missing-modality robustness experiments with trained model
- [ ] Full SHAP analysis with real data
- [ ] Submit manuscript to target venue
