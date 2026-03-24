# How Will Mid-Century Climate Change Alter the Spatial Suitability of Renewable Energy Across Great Britain? A Machine Learning Assessment Using UKCP18 Projections (2040 - 2060)

**Author:** Lina Abrahams
**Programme:** BSc Geography, King's College London, 2025-26

## Overview

This project models the suitability of solar, onshore wind, and biomass energy across the UK under baseline and future (RCP8.5, 2040-2060) climate scenarios. Random Forest classifiers are trained on REPD site locations and environmental predictors resampled to a 12 km British National Grid, then applied to UKCP18 climate projections to map how suitability shifts under climate change.

## Repository Structure

```
Analysis/
├── config.py                         Configuration: grid specs, paths, hyperparameters
├── 01_prepare_training_data.py       Presence/pseudo-absence sampling and predictor extraction
├── 02_train_models.py                Random Forest training with ensemble seeds
├── 03_validate_models.py             Accuracy metrics, spatial CV, SHAP, error maps, correlation matrix
├── 04_generate_suitability_maps.py   Baseline/future suitability and delta maps
├── 05_generate_figures.py            Publication-ready figures and tables
└── 06_ensemble_and_sensitivity.py    Ensemble uncertainty analysis and threshold sensitivity
└── outputs/                          Generated maps, figures, and tables
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run scripts in order:

```bash
python 01_prepare_training_data.py
python 02_train_models.py
python 03_validate_models.py
python 04_generate_suitability_maps.py
python 05_generate_figures.py
python 06_ensemble_and_sensitivity.py
```

All outputs are written to `outputs/`.

## Data Sources

| Dataset | Source | Resolution |
|---|---|---|
| UKCP18 climate projections | Met Office / CEDA Archive | 12 km RCM grid |
| OS Terrain 50 (elevation) | Ordnance Survey | 50 m DTM |
| CEH Land Cover Map 2024 | UK Centre for Ecology & Hydrology | 25 m |
| BGS soil texture | British Geological Survey | 1 km |
| National Grid / road network | ESO ETYS / OS OpenRoads | Vector |
| REPD (operational sites) | DESNZ | Point locations |
| Protected areas | Natural England (England Only) | Vector |


## License

MIT — see [LICENSE](LICENSE).
