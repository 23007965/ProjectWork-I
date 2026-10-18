# Systematic Literature Review

## An Explainable Cross-Modal Attention Framework for Marine Biodiversity Prediction Using Oceanographic, Fisheries, and Environmental DNA Data

**Review date**: 2026-10-18
**Search sources**: Google Scholar, publisher websites (Elsevier, Wiley, Springer, IEEE Xplore, Nature, PNAS, ACS, Frontiers, MDPI, Oxford Academic)
**Search period**: Publications from 2019–2026

---

## 1. eDNA-Based Fish Biodiversity Measurement

Environmental DNA (eDNA) metabarcoding has become a transformative tool for monitoring fish biodiversity in marine and freshwater systems.

**Djurhuus et al. (2020)** published a landmark study in *Nature Communications* demonstrating that eDNA reveals seasonal shifts and potential interactions in a marine community. Using eDNA metabarcoding from the Monterey Bay region, they showed eDNA can capture temporal dynamics of fish community composition that align with known ecological patterns. This study established eDNA as a viable complementary tool to traditional survey methods.

**He et al. (2023)** demonstrated in *ICES Journal of Marine Science* that eDNA metabarcoding enriches traditional trawl survey data for monitoring biodiversity in the marine environment. Working with Canadian Atlantic waters, they showed that eDNA detects species missed by trawls and vice versa, providing a more complete biodiversity picture when combined. This is directly relevant to our project as it demonstrates the complementary nature of eDNA with other data sources.

**Mathon et al. (2022)** used eDNA metabarcoding in *Proceedings of the Royal Society B* to study cross-ocean patterns in fish biodiversity on coral reefs. They demonstrated that eDNA can capture large-scale biogeographic patterns comparable to traditional underwater visual census, with 92 citations indicating strong community adoption.

**Valdivia-Carrillo et al. (2021)** published in *Molecular Ecology Resources* on integrating eDNA metabarcoding with simultaneous underwater visual surveys to describe complex fish communities in a marine biodiversity hotspot. This study validated eDNA against traditional methods, finding substantial concordance but also complementary detections.

**Mathon et al. (2023)** investigated the distribution of coastal fish eDNA sequences in the Anthropocene in *Global Ecology and Biogeography*, demonstrating that eDNA-derived fish diversity patterns correlate with environmental variables including temperature and human impacts.

**Collins et al. (2022)** showed in *Limnology and Oceanography: Letters* that reproduction influences seasonal eDNA variation in a temperate marine fish community, highlighting the importance of accounting for biological seasonality when interpreting eDNA-based richness estimates.

**He et al. (2022)** compared eDNA metabarcoding with seining in eelgrass beds in *Canadian Journal of Fisheries and Aquatic Sciences*, demonstrating implications for biodiversity monitoring in the coastal zone and confirming that eDNA captures different but overlapping community aspects.

## 2. Oceanographic Predictors of Marine Fish Biodiversity

The relationship between oceanographic conditions and fish diversity is well-established in the literature.

**Rutterford et al. (2023)** published a key study in *Global Change Biology* demonstrating that sea surface temperature is the primary driver of recent and predicted fish community structure across Northeast Atlantic shelf seas. This directly supports including SST as a primary oceanographic predictor in our framework.

**Jones et al. (2023)** disentangled the effects of fishing and temperature on increasing fish species richness in the North Sea in *Biodiversity and Conservation*. They found temperature rather than fishing mortality was the primary driver of species richness increase since 1991, providing important context for interpreting the relative importance of oceanographic vs. fisheries predictors.

**Chaudhary et al. (2021)** demonstrated in *PNAS* (335 citations) that global warming is causing a more pronounced dip in marine species richness around the equator, establishing SST as a major determinant of large-scale biodiversity patterns.

**Gordó-Vilaseca et al. (2023)** documented three decades of increasing fish biodiversity across the northeast Atlantic and Arctic Ocean in *PNAS* (70 citations), linking these changes to warming temperatures and providing a temporal context for our prediction framework.

**Barneche et al. (2019)** showed in *Global Ecology and Biogeography* that body size, reef area, and temperature predict global reef-fish species richness across spatial scales, establishing the multi-scale relationship between environmental variables and fish diversity.

**Deutsch et al. (2024)** reviewed climate, oxygen, and the future of marine biodiversity in *Annual Review of Marine Science* (101 citations), identifying dissolved oxygen alongside temperature as key determinants of future fish distributions.

## 3. Fisheries Pressure and Fish Community Changes

**Fontoura et al. (2022)** published in *Science* (121 citations) on connectivity and conservation, demonstrating that fishing pressure interacts with habitat connectivity to shape fish community composition. This high-impact study underscores the importance of including fisheries data in biodiversity models.

**Danet et al. (2024)** showed in *Nature Ecology & Evolution* (66 citations) how human activities reshape riverine fish communities, providing evidence that anthropogenic pressures including fishing effort significantly alter species richness patterns.

**Hilborn et al. (2021)** examined the trade-off between biodiversity and sustainable fish harvest with area-based management in *ICES Journal of Marine Science*, demonstrating the complex relationship between fishing effort and species diversity.

**Rousseau et al. (2024)** published a database of mapped global fishing activity 1950–2017 in *Scientific Data* (41 citations), providing a key data resource linking fishing effort patterns to marine ecosystems.

**Erm et al. (2023)** investigated the biodiversity benefits of marine protected areas in well-regulated fisheries in *Biological Conservation* (29 citations), showing that reduced fishing pressure correlates with higher species richness.

## 4. Multimodal Environmental Machine Learning

**Zhang et al. (2026)** published a directly relevant paper in *Environmental Science & Technology*: "Spatiotemporal Prediction of River Fish Biodiversity by Multimodal Deep Learning of eDNA and Satellite Imagery." This study used multimodal deep learning combining eDNA with remote sensing imagery for fish biodiversity prediction. This is the closest direct precedent to our work, though it uses only two modalities (eDNA + satellite imagery) rather than three, and focuses on rivers rather than marine systems.

**Zhao et al. (2024)** published "Deep multimodal data fusion" in *ACM Computing Surveys* (831 citations), providing a comprehensive review of multimodal fusion techniques including early fusion, late fusion, and attention-based fusion. This is a key methodological reference for our fusion strategy.

**Effrosynidis et al. (2020)** demonstrated species distribution modelling via feature engineering and machine learning for pelagic fishes in the Mediterranean Sea in *Applied Sciences*, using oceanographic variables from Copernicus Marine Service — the same data source we plan to use.

**Liu (2026)** developed a "Computational Intelligence Framework for Multimodal Oceanic Data Analysis" published in *Frontiers in Marine Science*, demonstrating the growing interest in multimodal approaches for marine data.

**Nguyen et al. (2025)** published on "Multimodal sequential cross-modal transformer for predicting plant available water capacity from time series of weather and crop biological data" in *Agricultural Water Management*. While focused on agriculture, this paper demonstrates the cross-modal transformer approach applied to heterogeneous environmental data, providing architectural precedent.

## 5. Cross-Modal Attention and Multimodal Transformers

**Chen & Li (2026)** published "A multimodal transformer neural network framework for fine-scale wildfire occurrence forecasting" in *Engineering with Computers*, applying multimodal transformers to environmental prediction — closely related methodologically to our approach.

**Alizadehsani et al. (2026)** published "Transforming multimodal data into decision assets: A vision-language Model for ecological data using cross-modal alignment" in *Environmental Modelling & Software*, demonstrating cross-modal alignment techniques for ecological applications.

**Bi et al. (2025)** published "Hybrid water quality prediction with multimodal low-rank fusion and localized attention" in *IEEE Internet of Things Journal*, demonstrating multimodal attention-based fusion for environmental monitoring.

**Demiray & Demir (2025)** presented "Unified Cross-Modal Learning for Hydrological Processes Using Multi-Task Transformer Framework" on EarthArXiv, applying cross-modal transformers to hydrological data — another environmental domain with heterogeneous data sources.

**Liu et al. (2025)** published "Cross-Modal and Contrastive Optimization for Explainable Multimodal Recognition" in *Insects* (MDPI), demonstrating explainable cross-modal approaches in a biological context.

## 6. Explainable AI for Ecological Predictions

**Ryo et al. (2021)** published the seminal paper "Explainable artificial intelligence enhances the ecological interpretability of black-box species distribution models" in *Ecography* (248 citations). This is the most influential paper on XAI for ecological modeling, demonstrating how SHAP values can reveal ecologically meaningful relationships from machine learning models.

**He et al. (2022)** applied explainable AI to seagrass distribution in *Ecological Indicators*, using SHAP to reveal environmental constraints. This demonstrates the approach we plan to use for interpreting oceanographic variable importance.

**Başağaoğlu et al. (2022)** published a review on "Interpretable and explainable artificial intelligence in hydroclimatic applications" in *Water* (155 citations), providing methodological guidance for applying XAI in environmental contexts.

**Buebos-Esteve & Dagamac (2025)** evaluated model-agnostic post-hoc methods in XAI for species distribution models in *Biologia futura*, providing a systematic comparison of SHAP, LIME, and other methods.

## 7. Missing-Modality Learning

**Zhao et al. (2021)** introduced the "Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities" at ACL (441 citations), establishing key approaches for handling missing modalities through imagination/imputation networks.

**Woo et al. (2023)** published "Towards Good Practices for Missing Modality Robust Action Recognition" at AAAI (123 citations), providing practical guidance for designing robust multimodal architectures.

**Guo et al. (2024)** presented "Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition" at ACL (198 citations), demonstrating that prompt-based approaches can handle missing modalities gracefully.

**Lan et al. (2024)** published "Robust Multimodal Representation Under Uncertain Missing Modalities" in *ACM Transactions on Multimedia Computing*, providing theoretical foundations for missing-modality robustness.

While these papers come from NLP/vision domains, the architectural principles (modality masking, reliability-aware attention, imagination networks) are directly transferable to our ecological setting where eDNA data is frequently unavailable.

## 8. Spatial and Temporal Validation of Ecological Models

**Meyer & Pebesma (2022)** published "Machine learning-based global maps of ecological variables and the challenge of assessing them" in *Nature Communications* (377 citations). This influential paper demonstrated that standard random cross-validation greatly overestimates predictive performance for spatial data, arguing for spatial cross-validation — directly relevant to our evaluation design.

**Zurell et al. (2020)** established "A standard protocol for reporting species distribution models" in *Ecography* (1,164 citations), providing the community standard for transparent model reporting that our paper should follow.

**Koldasbayeva & Zaytsev (2025)** published "Foundation for unbiased cross-validation of spatio-temporal models for species distribution modeling" in *Ecological Informatics*, providing updated guidance on spatio-temporal cross-validation.

**Luan et al. (2020)** examined "The predictive performances of random forest models with limited sample size" in *Fisheries Research* (307 citations), directly relevant to our situation where aligned multimodal samples may be limited.

## 9. Future Marine Biodiversity Forecasting

**Hodapp et al. (2023)** published "Climate change disrupts core habitats of marine species" in *Global Change Biology* (146 citations), projecting fish species richness changes under climate scenarios.

**Hu et al. (2022)** examined "Effects of climate change in the seas of China" in *Ecological Indicators* (117 citations), predicting changes in fish species distribution and diversity under future climate scenarios.

**Thompson et al. (2023)** studied groundfish biodiversity change in northeastern Pacific waters in *Philosophical Transactions of the Royal Society B* (44 citations), using historical data to project future biodiversity shifts.

**Moullec et al. (2022)** argued in *Ecological Modelling* (64 citations) that using species distribution models alone may underestimate climate change impacts on marine biodiversity, supporting our multimodal approach.

## 10. eDNA + Environmental Data Integration

**Cicala et al. (2024)** published "Spatial analysis of demersal food webs through integration of eDNA metabarcoding with fishing activities" in *Frontiers in Marine Science* (22 citations). This is a key precedent as it integrates eDNA with fisheries data, though it focuses on food web analysis rather than biodiversity prediction.

**Kvalheim et al. (2024)** demonstrated "Integration of citizen science and eDNA reveals novel ecological insights for marine fish conservation" in *Environmental DNA*, showing the value of integrating eDNA with other data sources.

**Schultz et al. (2026)** presented "Interpretable and uncertainty-aware mapping of coastal fish biodiversity in the era of eDNA surveys and machine learning" as a preprint, combining eDNA with machine learning for interpretable biodiversity mapping — closely related to our explainability goals.

---

## Comparison Matrix: Studies by Number and Type of Data Modalities

| Study | Year | Modalities Used | Prediction Target | Fusion Method | XAI? |
|-------|------|----------------|-------------------|---------------|------|
| Djurhuus et al. | 2020 | eDNA (1) | Community composition | N/A | No |
| He et al. | 2023 | eDNA + trawl surveys (2) | Species detection | Comparison | No |
| Mathon et al. | 2022 | eDNA (1) | Fish biodiversity | N/A | No |
| Rutterford et al. | 2023 | Oceanographic (1) | Fish community structure | N/A | No |
| Jones et al. | 2023 | Oceanographic + fisheries (2) | Species richness | Statistical | No |
| Fontoura et al. | 2022 | Fisheries + habitat (2) | Community composition | Statistical | No |
| Zhang et al. | 2026 | eDNA + satellite imagery (2) | Fish biodiversity | Multimodal DL | Limited |
| Effrosynidis et al. | 2020 | Oceanographic (1) | Species distribution | ML (feature eng.) | No |
| Cicala et al. | 2024 | eDNA + fisheries (2) | Food web structure | Statistical | No |
| Ryo et al. | 2021 | Environmental (1) | Species distribution | ML + SHAP | Yes |
| He et al. | 2022 | Environmental (1) | Seagrass distribution | ML + SHAP | Yes |
| Chen & Li | 2026 | Multimodal environmental (3+) | Wildfire occurrence | Transformer | Limited |
| Nguyen et al. | 2025 | Weather + crop biology (2) | Water capacity | Cross-modal transformer | No |
| Bi et al. | 2025 | Multi-source environmental (2+) | Water quality | Attention fusion | No |
| **Proposed (Ours)** | **2026** | **Oceanographic + fisheries + eDNA (3)** | **Future fish species richness** | **Cross-modal attention** | **Yes (SHAP + attention + ablation)** |

## Gap Analysis and Novelty Assessment

### What is established:
1. **eDNA as a biodiversity monitoring tool**: Well-established with hundreds of publications confirming its utility for fish community assessment.
2. **Oceanographic drivers of fish diversity**: SST and dissolved oxygen are well-characterized predictors of fish species richness.
3. **Fisheries impacts on biodiversity**: The relationship between fishing pressure and fish community changes is documented, though complex.
4. **XAI for ecological models**: SHAP-based interpretation of ecological models is mature (Ryo et al. 2021 has 248 citations).
5. **Missing-modality learning**: Architectural approaches are well-developed in NLP/vision (Zhao et al. 2021 has 441 citations).

### What is partially established:
1. **Pairwise integration**: eDNA + environmental variables (Mathon et al. 2023), eDNA + fisheries (Cicala et al. 2024), and environmental + fisheries (Jones et al. 2023) have each been explored in pairs.
2. **Multimodal DL for biodiversity**: Zhang et al. (2026) demonstrated multimodal deep learning for fish biodiversity using eDNA + satellite imagery, but only for rivers and with two modalities.
3. **Cross-modal transformers for environment**: Applied to wildfire (Chen & Li 2026), hydrology (Demiray & Demir 2025), water quality (Bi et al. 2025), but not yet to marine fish biodiversity.

### What remains novel (verified gap):
1. **Three-modality integration (oceanographic + fisheries + eDNA) for fish species richness prediction**: No study identified integrates all three modalities specifically for fish species richness prediction using machine learning.
2. **Cross-modal attention for marine ecological data**: While cross-modal transformers exist for other environmental domains, their application to heterogeneous marine ecological data (point-sampled eDNA, gridded oceanographic, event-based fisheries) has not been demonstrated.
3. **Explainable cross-modal attention with ecological validation**: Combining attention analysis, SHAP, and ablation studies with ecological literature validation for a three-modality marine system has not been published.
4. **Missing-modality robustness for ecological prediction**: Adapting missing-modality learning approaches to ecological contexts where eDNA is intermittently available is novel.
5. **Future fish species richness as a prediction target**: Most studies perform descriptive or concurrent analysis rather than genuine future prediction using temporal holdout.

### Caveats:
- The novelty of combining three modalities is incremental rather than fundamental — the scientific contribution lies in *how* the modalities are integrated (cross-modal attention) and *what insights* the model reveals (explainability), not merely in the data combination.
- The practical feasibility depends critically on data availability and alignment, which remains the highest-risk aspect of the project.
- Zhang et al. (2026) demonstrates that the general approach of multimodal DL for eDNA-based biodiversity prediction works; our contribution extends this to a three-modality marine setting with a stronger explainability framework.

---

## References Summary

This review identified and analyzed 30+ publications across the 10 topic areas. The full machine-readable bibliography is available at `references/bibliography.csv`.

**Key finding**: The proposed research addresses a genuine gap at the intersection of multimodal marine data fusion, cross-modal attention mechanisms, and explainable ecological prediction. The novelty is strongest in the architectural approach (cross-modal attention for heterogeneous ecological data) and the explainability framework (combined attention + SHAP + ablation with ecological validation), rather than in the data combination alone.
