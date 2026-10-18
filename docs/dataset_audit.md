# Dataset Audit

## Overview

This document provides detailed verification of the three primary datasets selected for the marine biodiversity prediction project, all covering the North Sea region.

---

## Dataset A: Oceanographic Data

### Product 1: Copernicus NWSHELF_MULTIYEAR_PHY_004_009

| Field | Value |
|---|---|
| **Exact product name** | Atlantic- European North West Shelf- Ocean Physics Reanalysis |
| **Product ID** | NWSHELF_MULTIYEAR_PHY_004_009 |
| **Landing page** | https://data.marine.copernicus.eu/product/NWSHELF_MULTIYEAR_PHY_004_009/description |
| **Provider** | Copernicus Marine Service (EU) |
| **Model** | NEMO with 3DVar NEMOVAR assimilation system |
| **Geographic coverage** | Bounding box: [-19.89, 40.07, 13.00, 65.00] (NW European Shelf including full North Sea) |
| **Temporal coverage** | 1993-01-01 to 2026-08-01 (verified via STAC catalog) |
| **Spatial resolution** | ~7 km horizontal |
| **Temporal resolution** | Hourly, daily, and monthly datasets available |
| **Format** | NetCDF |
| **License** | Copernicus Marine Service licence (free, open for research; attribution required) |

**Available variables (verified from STAC catalog):**
- Sea surface temperature (SST) — hourly, daily, monthly
- Bottom temperature — hourly, daily, monthly
- Salinity — daily, monthly
- Sea level — hourly, daily, monthly
- Ocean currents (u, v components) — hourly, daily, monthly
- Mixed layer depth — hourly, daily, monthly
- Bathymetry (static)

**Download method:** Copernicus Marine Toolbox (`copernicusmarine` Python package) or direct web download from Marine Data Store.

**Known limitations:**
- Reanalysis product (model-based with data assimilation), not direct observations
- 7 km resolution may smooth small-scale coastal features
- Quality varies in shallow coastal areas

### Product 2: Copernicus NWSHELF_MULTIYEAR_BGC_004_011

| Field | Value |
|---|---|
| **Exact product name** | Atlantic- European North West Shelf- Ocean Biogeochemistry Reanalysis |
| **Product ID** | NWSHELF_MULTIYEAR_BGC_004_011 |
| **Landing page** | https://data.marine.copernicus.eu/product/NWSHELF_MULTIYEAR_BGC_004_011/description |
| **Provider** | Copernicus Marine Service (EU) |
| **Model** | ERSEM (European Regional Seas Ecosystem Model) coupled to NEMO-NEMOVAR |
| **Geographic coverage** | Bounding box: [-19.89, 40.07, 13.00, 65.00] |
| **Temporal coverage** | 1993-01-01 to 2026-07-31 (verified via STAC catalog) |
| **Spatial resolution** | ~7 km horizontal |
| **Temporal resolution** | Daily and monthly datasets available |
| **Format** | NetCDF |
| **License** | Copernicus Marine Service licence |

**Available variables (verified from STAC catalog):**
- Chlorophyll-a — daily, monthly
- Dissolved oxygen (O2) — daily, monthly
- Nitrate (NO3) — daily, monthly
- Phosphate (PO4) — daily, monthly
- pH — daily, monthly
- Primary production — daily, monthly
- Light attenuation — daily, monthly
- Phytoplankton biomass — daily, monthly
- Plankton groups (diatoms, dinoflagellates, nanophytoplankton, picophytoplankton) — daily, monthly
- Surface CO2 partial pressure — daily, monthly

**Download method:** Same as physics product.

**Known limitations:**
- Biogeochemical reanalysis has higher uncertainty than physical variables
- Assimilates ocean colour satellite chlorophyll data
- Nutrient and oxygen fields are model-derived with limited in-situ validation in some areas

---

## Dataset B: Fisheries Data

### Global Fishing Watch Public Fishing Effort

| Field | Value |
|---|---|
| **Exact dataset name** | Global Fishing Watch Public Fishing Effort |
| **Landing page** | https://globalfishingwatch.org/dataset-and-code-fishing-effort/ |
| **Download portal** | https://globalfishingwatch.org/data-download/datasets/public-fishing-effort |
| **Provider** | Global Fishing Watch |
| **Publication** | Kroodsma et al. (2018), "Tracking the global footprint of fisheries," Science, 359(6378), pp. 904-908. DOI: 10.1126/science.aao5646 |
| **Method** | AIS-based apparent fishing effort detection using convolutional neural networks |
| **Geographic coverage** | Global (strong coverage in North Sea) |
| **Temporal coverage** | 2012-present |
| **Spatial resolution** | 0.01° (daily by flag/gear) or 0.1° (monthly by flag/gear) |
| **Temporal resolution** | Daily or monthly |
| **Variables** | Fishing hours, flag state, gear type, vessel MMSI |
| **Format** | CSV |
| **License** | CC BY-SA 4.0 (free, open; attribution and share-alike required) |
| **BigQuery** | global-fishing-watch.fishing_effort_v3 |

**Download method:**
- Web portal: https://globalfishingwatch.org/data-download/
- Requires free registration
- Also available via Google BigQuery

**Available data products (verified):**
1. Daily fishing hours by flag/gear type at 0.01° resolution
2. Monthly fishing hours by flag/gear type at 0.1° resolution
3. Daily fishing hours by MMSI at 0.1° resolution

**What the data measures:**
- "Apparent fishing effort" — estimated fishing activity based on AIS vessel tracking
- This is a **fishing pressure proxy**, not direct catch data
- Distinguishes fishing gear types (trawlers, longliners, purse seiners, etc.)

**Known limitations:**
- Only covers AIS-equipped vessels (primarily >24m from upper/middle-income countries)
- Up to 90% coverage of vessels >24m, but <1% of vessels <12m
- AIS reception quality varies (satellite coverage gaps possible)
- Early years (2012-2016) show increasing effort partly due to improved receiver coverage, not just fishing activity
- Does not measure actual catch volume

---

## Dataset C: Fish Biodiversity Data

### ICES DATRAS NS-IBTS (North Sea International Bottom Trawl Survey)

| Field | Value |
|---|---|
| **Exact dataset name** | ICES Database of Trawl Surveys (DATRAS) - North Sea International Bottom Trawl Survey (NS-IBTS) |
| **Landing page** | https://datras.ices.dk/Home/SurveyCode.aspx |
| **Download page** | https://datras.ices.dk/Data_products/Download/Download_Data_public.aspx |
| **Web service API** | https://datras.ices.dk/WebServices/DATRASWebService.asmx |
| **DOI** | https://doi.org/10.17895/ices.pub.5713 |
| **Provider** | ICES (International Council for the Exploration of the Sea) |
| **Coordinating group** | IBTSWG (International Bottom Trawl Survey Working Group) |
| **Geographic coverage** | North Sea |
| **Temporal coverage** | 1965-2026 (62 years, verified via web service) |
| **Survey frequency** | Quarterly; Q1 (winter) and Q3 (summer) available every year 2000-2024 (verified) |
| **Format** | XML via web service, CSV downloadable |
| **License** | CC BY 4.0 (open access, citation required) |

**Verified data characteristics (Q1 2020 sample):**
- 348 hauls per quarter (verified)
- 280 unique fish species per quarter (verified by AphiaID count)
- 47,663 species-length records per quarter (verified)
- Each haul has: coordinates (lat/lon), date, species list with counts and weights

**Data record types:**
- **HH (Haul Header)**: Survey metadata, station number, coordinates, date, time, depth, gear, haul duration
- **HL (Length-based catch)**: Species identification (WoRMS AphiaID), total number caught, length measurements, catch weight
- **CA (Age-based)**: Age determinations for selected species

**What the data measures:**
- Systematic trawl catches at fixed station networks
- Species-level identification with standardized taxonomy (WoRMS AphiaIDs)
- Catch per unit effort (CPUE) by species
- Community composition at each sampling station

**Citation:**
ICES Database on Trawl Surveys (DATRAS). ICES, Copenhagen. https://datras.ices.dk

**Note on eDNA vs trawl data:**
The original research design specified environmental DNA (eDNA) data. Investigation revealed that publicly available eDNA datasets with sufficient repeated temporal coverage for future prediction do not exist at the required scale for any marine region. The ICES NS-IBTS trawl survey provides scientifically superior data for constructing future_fish_species_richness targets due to its systematic, repeated, standardized sampling across decades. The research framing uses "fish biodiversity observations" as the modality, acknowledging this practical adaptation. See docs/study_region_selection.md for the full eDNA availability assessment.

**Known limitations:**
- Trawl surveys are selective — catchability varies by species, size, and behaviour
- Bottom trawl does not sample pelagic species effectively
- Station positions may shift slightly between surveys
- Some years/quarters may have reduced coverage due to weather or logistics
- Not truly "eDNA" — physical capture-based species detection

---

## Three-Dataset Compatibility Audit

### 1. Geographic Overlap
- **Verified**: All three datasets cover the North Sea (approximately 51-62°N, 5°W-10°E)
- NS-IBTS stations: verified within 51-62°N, 2°W-10°E (from Q1 2020 sample)
- Copernicus products: bounding box [-19.89, 40.07, 13.00, 65.00] fully contains the North Sea
- GFW: global coverage with strong AIS reception in European waters

### 2. Temporal Overlap
- **All three modalities available**: 2012-2024 (13 years)
- NS-IBTS: 1965-2026
- Copernicus: 1993-2026
- GFW: 2012-present
- **Limiting factor**: GFW starts in 2012

### 3. Repeated Biodiversity Observations
- **Verified**: Q1 and Q3 surveys every year 2000-2024
- ~350 hauls per quarter with fixed station network
- Multiple countries contribute (DK, NL, UK, DE, NO, FR, SE verified in sample data)

### 4. Fish Species-Level Observations
- **Verified**: 280 unique species (AphiaIDs) in Q1 2020
- Species identified using standardized WoRMS taxonomy
- Each haul provides a complete species list with abundance

### 5. Matching Feasibility
- Oceanographic data at 7 km can be extracted at each NS-IBTS station location
- GFW at 0.01° (~1 km) can be aggregated around each station
- Temporal matching: monthly oceanographic averages matched to survey quarter

### 6. Future Prediction Task Support
- **Supported**: Use Q1 survey + preceding oceanographic/fisheries data to predict Q3 species richness (or year-over-year)
- Multiple prediction horizons possible (6 months within-year, 12 months year-over-year)

### 7. Validation Design
- **Spatial holdout**: Reserve stations in one sub-region (e.g., southern North Sea)
- **Temporal holdout**: Reserve 2022-2024 as test years
- Sufficient data for meaningful evaluation (~9,100 station-quarter observations in 2012-2024)

### 8. Licensing
- ICES DATRAS: CC BY 4.0 ✓
- Copernicus Marine: Free research licence ✓
- GFW: CC BY-SA 4.0 ✓
- All permit research use with attribution
