# Research Design Document

## An Explainable Cross-Modal Attention Framework for Marine Biodiversity Prediction Using Oceanographic, Fisheries, and Environmental DNA Data

---

## 1. Research Background and Motivation

Marine biodiversity faces escalating threats from climate change, overfishing, pollution, and habitat degradation. Effective conservation and sustainable fisheries management depend on accurate, forward-looking predictions of ecosystem health and biodiversity change. Marine organizations routinely collect heterogeneous datasets spanning three complementary domains:

- **Oceanographic observations**: Sea surface temperature (SST), dissolved oxygen, salinity, chlorophyll-a, and ocean currents describe the physical and chemical state of the marine environment.
- **Fisheries records**: Fishing effort, vessel activity, and catch data quantify anthropogenic exploitation pressure on fish communities.
- **Environmental DNA (eDNA)**: Water samples analyzed for DNA traces provide direct molecular evidence of species presence and community composition.

Each data source captures a different dimension of ecosystem dynamics, yet they are predominantly analyzed independently or in limited pairwise combinations. This fragmented approach may miss critical ecological interactions. For example, rising SST may reduce dissolved oxygen, triggering fish migration, which alters catch patterns and ultimately manifests as reduced eDNA diversity. Understanding such cascading effects requires joint modeling of all three information streams.

## 2. Problem Statement

Marine biodiversity is governed by complex interactions among environmental conditions, human activities, and biological communities rather than by any single factor. Current prediction models generally analyze only one or two of these components:

- **Oceanographic models** explain environmental changes but not species responses.
- **Fisheries models** quantify exploitation but ignore environmental drivers.
- **eDNA-based approaches** identify species presence but cannot adequately explain the ecological processes responsible for biodiversity changes.

Consequently, existing approaches provide only a partial representation of marine ecosystem dynamics. The challenge is: *How can heterogeneous marine datasets be jointly modeled so that their interactions improve marine biodiversity prediction while providing interpretable ecological insights for conservation and fisheries management?*

## 3. Research Questions and Hypotheses

### Primary Research Question

Can an explainable cross-modal attention model integrating oceanographic observations, fisheries pressure, and historical eDNA data improve predictions of future marine fish species richness compared with single-modality and conventional multimodal models?

### Sub-Questions

- **RQ1**: Can integrating oceanographic, fisheries, and eDNA information improve marine biodiversity prediction compared with single-modality models?
- **RQ2**: Do cross-modal attention mechanisms learn more informative interactions among heterogeneous marine datasets than conventional feature concatenation?
- **RQ3**: Which data modality contributes most under different environmental and ecological conditions?
- **RQ4**: Can explainable AI reveal meaningful ecological relationships between environmental conditions, fishing activities, and biodiversity patterns?
- **RQ5**: Can the proposed framework generate interpretable predictions suitable for real-world marine conservation and fisheries management?

### Research Hypothesis

Jointly learning the interactions among oceanographic conditions, fisheries activities, and environmental DNA through an explainable cross-modal attention mechanism will improve marine biodiversity prediction compared with models trained on individual data modalities or simple feature-level fusion.

## 4. Related Work (Preliminary — to be expanded in literature review)

The literature review (docs/literature_review.md) will systematically verify and expand the following preliminary assessment from the Ideas PDF:

- Oceanographic AI studies primarily focus on predicting coral bleaching, habitat suitability, or fish abundance using environmental variables.
- Fisheries studies concentrate on stock assessment, fishing effort estimation, or catch prediction.
- eDNA studies mainly address biodiversity monitoring and species detection.
- Recent research has explored integrating eDNA with environmental variables or fisheries information, demonstrating improved predictive capability over single-source models.
- Explainable AI techniques have been introduced in environmental applications to improve model transparency.
- Cross-modal attention and multimodal transformers have been applied in computer vision and NLP but their application to ecological prediction with heterogeneous environmental data remains limited.

**Note**: The claim that no predictive AI framework simultaneously integrates all three modalities with explainable cross-modal attention requires verification through the systematic literature review.

## 5. Research Gap (To Be Verified)

The PDF identifies the following gap: *No predictive AI framework simultaneously integrates oceanographic observations, fisheries information, and environmental DNA while explicitly modeling cross-modal ecological interactions through an explainable attention-based architecture.*

This gap must be verified through the literature review. The novelty should be framed not merely as "combining three datasets" but as:

**Proposed framing**: We propose an explainable cross-modal attention framework that explicitly learns ecological interactions among heterogeneous marine data sources and evaluates whether modeling these interactions leads to more accurate and interpretable biodiversity prediction than existing multimodal fusion approaches.

This shifts the contribution from data integration to *how the AI learns relationships*, which is the type of contribution IEEE and strong Scopus journals look for.

## 6. Proposed Methodology

### 6.1 Study Region Selection

The study region will be selected based on verified spatial-temporal compatibility across all three modalities (see docs/study_region_selection.md). Candidate regions include the North Sea, Mediterranean Sea, Arabian Sea, Bay of Bengal, and Northwest Pacific.

### 6.2 Data Collection and Alignment

Three datasets will be acquired and aligned:

1. **Oceanographic data** from Copernicus Marine Service or NOAA (SST, dissolved oxygen, salinity, chlorophyll-a, currents).
2. **Fisheries data** from Global Fishing Watch or FAO (fishing effort, vessel activity).
3. **eDNA data** from OBIS, NCBI SRA, or ENA (fish species detections with coordinates and dates).

**Spatial alignment**: All datasets will be converted to a common spatial grid (resolution determined by data inspection).

**Temporal alignment**: All datasets will be converted to a common time interval (e.g., monthly), with oceanographic data averaged, fisheries data summed, and eDNA assigned to the nearest time window.

**Missing modality handling**: Missing modalities will be masked rather than deleted, allowing the attention network to learn with incomplete inputs.

### 6.3 Prediction Target

**Primary target**: `future_fish_species_richness` — the number of distinct eligible fish taxa detected by eDNA in a future evaluation period.

**Forecasting formulation**: R_future_hat = f(oceanographic_history, fisheries_history, historical_eDNA, model_parameters)

Only information available at or before the prediction origin will be used as input.

### 6.4 Model Architecture

#### Baselines (Category A — Single-modality)
- Oceanography-only: XGBoost, CatBoost, MLP
- Fisheries-only: XGBoost, CatBoost, MLP
- eDNA-only: XGBoost, CatBoost, MLP

#### Baselines (Category B — Conventional multimodal)
- **Early fusion**: Concatenate features from all modalities, train a single model.
- **Late fusion**: Train separate models per modality, combine predictions (averaging or learned weighting).

#### Proposed Model (Category C — Cross-Modal Attention)
- Modality-specific encoders for oceanographic, fisheries, and eDNA features.
- Cross-modal attention mechanism to learn adaptive inter-modality relationships.
- Missing-modality masking with reliability-aware attention.
- Prediction head for future fish species richness.
- The architecture complexity will be calibrated to the actual sample size.

### 6.5 Evaluation Strategy

- Chronological train/validation/test split to prevent temporal leakage.
- Spatially independent holdout locations where feasible.
- Metrics: MAE, RMSE, R², baseline-relative improvement.
- Multiple random seeds with confidence intervals.
- Paired statistical tests (e.g., Wilcoxon signed-rank or paired t-test).

### 6.6 Explainability Analysis

- **SHAP values**: Feature-level and modality-level importance.
- **Attention weight analysis**: Cross-modal attention patterns (interpreted cautiously, not as causal evidence).
- **Ablation studies**: Systematic removal of each modality and the attention mechanism.
- **Ecological validation**: Comparison of model-identified drivers with established ecological knowledge from the literature.

### 6.7 Robustness Testing

- **Missing-modality robustness**: Evaluate performance when one or more modalities are unavailable.
- **Sensitivity analysis**: Test model stability under different hyperparameter settings.

## 7. Expected Scientific and Practical Contributions

### Scientific Contributions
1. An explainable cross-modal attention architecture for marine biodiversity prediction integrating three complementary data modalities.
2. A multimodal learning strategy tailored to heterogeneous marine datasets with missing-modality robustness.
3. An empirical comparison of single-modality, conventional fusion, and attention-based fusion methods for fish species richness prediction.
4. Interpretable analysis of ecological drivers using combined attention, SHAP, and ablation evidence.

### Practical Contributions
1. Early identification of biodiversity change hotspots.
2. Improved monitoring of marine ecosystem health.
3. Decision support for sustainable fisheries management.
4. Evidence-based recommendations for marine conservation policies.

**Note**: These are *intended* contributions. Whether they are achieved depends on experimental outcomes, which will be reported honestly regardless of whether the hypothesis is confirmed.

## 8. Limitations and Potential Threats to Validity

### Data Limitations
- eDNA sampling is sparse and irregularly distributed; the number of eligible aligned observations may be small.
- eDNA detections are not direct measurements of abundance; detection probability varies with sampling method, water volume, and environmental conditions.
- Fisheries data may have reporting biases and spatial gaps.
- Temporal and spatial alignment introduces aggregation uncertainty.

### Methodological Limitations
- Cross-modal attention on small ecological datasets may not show advantages over simpler models.
- Attention weights are not causal explanations and should be interpreted cautiously.
- The prediction horizon is constrained by the actual eDNA sampling frequency, which may limit the "future prediction" framing.
- Spatial and temporal autocorrelation may inflate apparent model performance if not properly handled in evaluation.

### External Validity
- Results from one study region may not generalize to other marine ecosystems.
- Model performance depends on data quality and completeness, which vary by region.

### Reproducibility
- Some datasets may have access restrictions or version changes.
- Large oceanographic datasets may require substantial storage and computation.

---

**Document Status**: Phase 1 complete. The literature review (Phase 2) will verify claims about the research gap and refine this design as needed.

**Next Step**: Systematic literature review (Phase 2) and dataset investigation (Phase 3).
