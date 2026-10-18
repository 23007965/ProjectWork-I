# Limitations and Threats to Validity

## Data Limitations

1. **Synthetic oceanographic and fisheries data**: The oceanographic and fisheries datasets used in this demonstration are synthetically generated from ecologically realistic distributions. While they capture known North Sea spatial and seasonal patterns (e.g., SST seasonality, latitudinal fishing effort gradients), they lack the real-world noise, extreme events, and fine-grained spatial structure present in actual CMEMS and GFW data. Full validation requires replacing these with real datasets.

2. **OBIS occurrence data vs. true eDNA**: The species occurrence data from OBIS includes records from various observation methods (trawl surveys, visual observations, museum specimens), not exclusively eDNA metabarcoding. True eDNA data would provide more consistent, comparable detection probabilities. The OBIS data is treated as a proxy for biodiversity observations.

3. **Narrow target range**: Species richness aggregated to 0.25° monthly grid cells ranges from 1 to 4, with 93.5% of samples having richness=1. This highly skewed distribution limits the model's ability to demonstrate regression performance improvements and makes R² a poor evaluation metric. Alternative approaches (finer spatial resolution, species-level classification, or different study region) could provide richer targets.

4. **Spatial autocorrelation**: Despite the chronological train/test split, nearby spatial locations in the training and test sets may share similar ecological conditions, potentially inflating apparent model performance. Spatially blocked cross-validation would provide more conservative and realistic performance estimates.

5. **Temporal coverage mismatch**: OBIS records span 1965-2025 with highly irregular coverage, while the oceanographic/fisheries data cover 2015-2023 monthly. The effective overlap for three-modality samples is limited.

## Methodological Limitations

6. **Small effective sample size for rare events**: Only 6.5% of samples have richness > 1, meaning the models have limited examples of biodiversity variation to learn from. The cross-modal attention model's improvement may reflect better handling of these rare events rather than robust ecological learning.

7. **Attention is not causation**: Attention weights indicate what the model focuses on, not causal ecological relationships. We complement attention analysis with SHAP values and ablation studies, but none of these methods establish causal mechanisms.

8. **Architecture calibration**: The cross-modal attention architecture (2 layers, 4 heads, 64-dim embeddings) was designed for moderate sample sizes but not exhaustively tuned. Different configurations may yield different results.

## External Validity

9. **Single region**: Results from the North Sea may not generalize to tropical, polar, or deep-sea ecosystems with different species compositions, environmental drivers, and data availability.

10. **Temporal generalization**: The chronological split evaluates forward prediction within the study period. Performance under future climate scenarios or regime shifts is not assessed.

## Reproducibility

11. **API-dependent data**: OBIS API results may vary over time as new records are added or existing ones are updated. The exact dataset used in this study is documented but not frozen.

12. **Copernicus and GFW registration**: Accessing real oceanographic and fisheries data requires account registration, which may limit immediate reproducibility.
