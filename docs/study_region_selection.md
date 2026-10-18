# Study Region Selection

## Objective

Select the marine region with the strongest spatial-temporal data compatibility across three modalities:
1. Fish species occurrence/biodiversity observations (originally specified as eDNA; see note below)
2. Oceanographic conditions (SST, dissolved oxygen, salinity, chlorophyll-a, currents)
3. Fisheries pressure (fishing effort, vessel activity)

## Important Note on eDNA Data Availability

The original research design specified environmental DNA (eDNA) as the biodiversity modality. A systematic investigation of eDNA data availability across all candidate regions revealed that:

- **OBIS** contains only 2 dedicated eDNA datasets globally (Australia 2019, Colombia 2022-2023), both small (<4,000 records each) and geographically limited
- **NCBI SRA** has ~61 BioProjects for marine fish eDNA metabarcoding, mostly raw sequence data requiring bioinformatics processing, with limited temporal coverage (typically single-expedition snapshots)
- **No region has publicly available, processed eDNA fish species detection tables with sufficient repeated temporal sampling** to construct a meaningful future_fish_species_richness prediction target

Given this limitation, we adopt a pragmatic approach: use **systematic fish trawl survey data** (ICES DATRAS NS-IBTS) as the biodiversity modality. This provides:
- Standardized, repeated, species-level fish community observations
- Quarterly surveys across decades (1965-2026)
- Fixed station networks enabling true spatial-temporal prediction
- Well-documented sampling methodology and quality controls

The research framing adapts to "fish biodiversity observations" rather than strictly "eDNA." This is scientifically defensible because the core research question is about multimodal fusion for biodiversity prediction, not about the specific biological sampling method.

## Candidate Region Comparison

| Candidate Region | Fish Biodiversity Coverage | Oceanographic Data Overlap | Fisheries Data Overlap | Repeated Fish Observations | Estimated Eligible Aligned Observations | Feasibility |
|---|---|---|---|---|---|---|
| **North Sea** | **5.5M+ OBIS records, 312+ species; ICES NS-IBTS: 62 years, quarterly, 280+ species per quarter, ~350 hauls/quarter** | **Copernicus NWSHELF_MULTIYEAR_PHY_004_009: 7km, daily/monthly, 1993-2026; BGC_004_011: chlorophyll, O2, nutrients** | **GFW: 0.01° resolution, daily, 2012-present; ICES also has fishing effort data** | **Q1+Q3 every year 2000-2024 (verified); stations revisited systematically** | **~700 hauls/year × 24 years = ~17,000 station-quarter observations** | **STRONGEST. All three modalities verified with extensive overlap.** |
| Mediterranean Sea | 5.1M OBIS records, 1,261 species, 177 datasets | Copernicus Mediterranean MFC products available (MEDSEA products) | GFW coverage available; high fishing pressure | MEDITS trawl survey exists but not verified for this analysis | Requires verification | Good potential but not verified to same depth as North Sea |
| Northeast Pacific (California) | 11M OBIS records, 1,161 species | NOAA products available | GFW coverage available | CalCOFI surveys exist | Requires verification | Strong potential, not verified |
| Northwest Pacific (Japan/Korea) | 158K OBIS records, 3,689 species | Copernicus global products available | GFW coverage moderate | 3 eDNA BioProjects found (small scale) | Low - limited repeatable observations | Moderate - eDNA data exists but very small scale |
| Arabian Sea / Bay of Bengal | 32K OBIS records, 2,980 species | Copernicus Indian Ocean products | GFW coverage variable | Very limited repeated sampling | Low | Weak - sparse biodiversity data |
| Great Barrier Reef | 764K OBIS records, 4,089 species | Copernicus/IMOS products | GFW coverage available | AIMS monitoring exists | Requires verification | Moderate - coral reef focus may differ from fish richness prediction |

## Selected Region: North Sea

### Justification

The **North Sea** (approximately 51°N-62°N, 5°W-10°E) is selected based on verified evidence of the strongest three-modality alignment:

#### 1. Fish Biodiversity Data (ICES DATRAS NS-IBTS)
- **Source**: ICES Database of Trawl Surveys (DATRAS)
- **Survey**: North Sea International Bottom Trawl Survey (NS-IBTS)
- **Temporal coverage**: 1965-2026 (62 years verified via ICES web service)
- **Quarterly surveys**: Q1 (winter) and Q3 (summer) available every year from 2000-2024 (verified)
- **~350 hauls per quarter** (348 verified for Q1 2020)
- **280+ fish species per quarter** (280 verified for Q1 2020)
- **47,663 species-length records per quarter** (verified for Q1 2020)
- **Fixed station grid**: stations are revisited, enabling temporal comparison
- **License**: CC BY 4.0 (open access)

#### 2. Oceanographic Data (Copernicus Marine Service)
- **Physics**: NWSHELF_MULTIYEAR_PHY_004_009
  - 7 km horizontal resolution
  - Daily, monthly, hourly datasets
  - Variables: temperature, salinity, sea level, currents, mixed layer depth, bottom temperature
  - Coverage: 40°N-65°N, 20°W-13°E (fully covers North Sea)
  - Temporal: 1993-2026
- **Biogeochemistry**: NWSHELF_MULTIYEAR_BGC_004_011
  - 7 km horizontal resolution
  - Daily and monthly datasets
  - Variables: chlorophyll-a, dissolved oxygen, nitrate, phosphate, pH, primary production, plankton groups
  - Coverage: same as physics product
  - Temporal: 1993-2026
- **License**: Copernicus Marine Service license (free for research)

#### 3. Fisheries Data (Global Fishing Watch)
- **Source**: Global Fishing Watch public fishing effort dataset
- **Method**: AIS-based apparent fishing effort detection
- **Resolution**: 0.01° daily or 0.1° monthly
- **Temporal**: 2012-present
- **Variables**: fishing hours by flag state, gear type, vessel identity
- **Coverage**: Global (strong AIS coverage in North Sea due to European regulations)
- **License**: CC BY-SA 4.0

### Spatial-Temporal Overlap Assessment

| Aspect | Status |
|---|---|
| Geographic overlap | All three datasets cover the North Sea (51-62°N, 5°W-10°E) |
| Temporal overlap | 2012-2024 (intersection of all three) |
| Repeated biodiversity observations | Q1 and Q3 every year (verified 2000-2024) |
| Station-level matching | NS-IBTS stations can be matched to oceanographic grid cells and GFW fishing effort cells |
| Alignment feasibility | Oceanographic data at 7km can be extracted at trawl station locations; GFW at 0.01° (~1km) can be aggregated around stations |

### Estimated Eligible Aligned Observations

For the 2012-2024 overlap period (all three modalities available):
- **13 years × 2 quarters × ~350 hauls = ~9,100 station-quarter observations**
- Each haul has coordinates, date, species list, and can be matched to oceanographic and fisheries data
- This is sufficient for train/validation/test splits with spatial and temporal holdouts

### Key Advantages of North Sea Selection
1. **Verified data compatibility** at all three modality levels
2. **Longest temporal overlap** among verified candidates
3. **Largest number of repeated, standardized biodiversity observations**
4. **Well-documented survey methodology** (ICES quality controls)
5. **Strong AIS coverage** for fisheries data (European vessel tracking requirements)
6. **Rich oceanographic reanalysis** at appropriate resolution
7. **Established ecological understanding** for validating explainability results
