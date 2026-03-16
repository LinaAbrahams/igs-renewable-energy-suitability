"""
Central configuration for all analysis scripts.

All constants, file paths, and hyperparameters are defined here.
No other script should contain hardcoded values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"

RAW_DIR = PROJECT_ROOT / "01_Raw_Data"
PROCESSED_DIR = PROJECT_ROOT / "02_Processed"
TRAINING_DIR = PROJECT_ROOT / "03_Training_Data_WithRoads"
MODELS_DIR = PROJECT_ROOT / "04_Models_WithRoads"
RESULTS_DIR = PROJECT_ROOT / "05_Outputs_WithRoads"

# ---------------------------------------------------------------------------
# Grid specification (OSGB 1936 / British National Grid)
# ---------------------------------------------------------------------------

CRS = "EPSG:27700"
GRID_NCOLS = 82
GRID_NROWS = 112
CELL_SIZE = 12_000
GRID_ORIGIN_X = -216_000
GRID_ORIGIN_Y = 1_236_000

# ---------------------------------------------------------------------------
# Technologies
# ---------------------------------------------------------------------------

TECHNOLOGIES = ["solar", "wind", "bio"]

# ---------------------------------------------------------------------------
# Random seeds and train/test split
# ---------------------------------------------------------------------------

PRIMARY_SEED = 42
ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]
TEST_SIZE = 0.3
TRAIN_SIZE = 0.7

# ---------------------------------------------------------------------------
# Random Forest hyperparameters
# ---------------------------------------------------------------------------

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": PRIMARY_SEED,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Predictor variables (10 total, including distance to roads)
# ---------------------------------------------------------------------------

CLIMATE_VARS = ["tas", "pr", "sfcWind", "rss"]
STATIC_VARS = ["elevation", "slope", "landcover", "soil", "dist_grid", "dist_roads"]
PREDICTOR_COLS = CLIMATE_VARS + STATIC_VARS

# ---------------------------------------------------------------------------
# Pseudo-absence generation
# ---------------------------------------------------------------------------

ABSENCE_BUFFER_M = 10_000

# ---------------------------------------------------------------------------
# Raw data paths
# ---------------------------------------------------------------------------

REPD_CSV = RAW_DIR / "REPD" / "REPD_Publication_Q3_2025 (1).csv"

RAW_BOUNDARIES_DIR = RAW_DIR / "Boundaries"
RAW_CONSTRAINTS_DIR = RAW_DIR / "Constraints"

# ---------------------------------------------------------------------------
# Processed rasters — static
# ---------------------------------------------------------------------------

STATIC_RASTERS = {
    var: PROCESSED_DIR / "Static" / f"{var}.tif"
    for var in STATIC_VARS
}

PROTECTED_MASK = PROCESSED_DIR / "Constraints" / "protected_mask.tif"

# ---------------------------------------------------------------------------
# Processed rasters — climate (primary ensemble member)
# ---------------------------------------------------------------------------

CLIMATE_BASELINE = {
    var: PROCESSED_DIR / "Climate" / "Baseline" / f"{var}_baseline.tif"
    for var in CLIMATE_VARS
}

CLIMATE_FUTURE = {
    var: PROCESSED_DIR / "Climate" / "Future" / f"{var}_future.tif"
    for var in CLIMATE_VARS
}

# ---------------------------------------------------------------------------
# Processed rasters — additional ensemble members
# ---------------------------------------------------------------------------

CLIMATE_MEMBERS = {}
for member in ("Member04", "Member08"):
    CLIMATE_MEMBERS[member] = {
        period: {
            var: PROCESSED_DIR / f"Climate_{member}" / f"{var}_{period}.tif"
            for var in CLIMATE_VARS
        }
        for period in ("baseline", "future")
    }

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

TRAINING_CSV = {tech: TRAINING_DIR / f"train_{tech}.csv" for tech in TECHNOLOGIES}

# ---------------------------------------------------------------------------
# Trained models
# ---------------------------------------------------------------------------

MODEL_FILES = {tech: MODELS_DIR / f"rf_{tech}.joblib" for tech in TECHNOLOGIES}

# ---------------------------------------------------------------------------
# Output maps
# ---------------------------------------------------------------------------

SUITABILITY_BASELINE = {
    tech: RESULTS_DIR / "Maps" / f"suit_baseline_{tech}.tif"
    for tech in TECHNOLOGIES
}

SUITABILITY_FUTURE = {
    tech: RESULTS_DIR / "Maps" / f"suit_future_{tech}.tif"
    for tech in TECHNOLOGIES
}

SUITABILITY_DELTA = {
    tech: RESULTS_DIR / "Maps" / f"delta_{tech}.tif"
    for tech in TECHNOLOGIES
}

DEVELOPABLE_SUITABILITY = {
    tech: RESULTS_DIR / "Maps" / f"dev_suit_{tech}.tif"
    for tech in TECHNOLOGIES
}

# ---------------------------------------------------------------------------
# Ensemble outputs
# ---------------------------------------------------------------------------

ENSEMBLE_DIR = RESULTS_DIR / "Ensemble"

ENSEMBLE_MEAN = {
    tech: ENSEMBLE_DIR / f"ensemble_mean_{tech}.tif"
    for tech in TECHNOLOGIES
}

ENSEMBLE_STD = {
    tech: ENSEMBLE_DIR / f"ensemble_std_{tech}.tif"
    for tech in TECHNOLOGIES
}

ENSEMBLE_RANGE = {
    tech: ENSEMBLE_DIR / f"ensemble_range_{tech}.tif"
    for tech in TECHNOLOGIES
}

# ---------------------------------------------------------------------------
# Original presence data (before predictor extraction)
# ---------------------------------------------------------------------------

ORIGINAL_TRAINING_DIR = PROJECT_ROOT / "03_Training_Data"

PRESENCE_CSV = {
    tech: ORIGINAL_TRAINING_DIR / f"{tech}_presence.csv"
    for tech in TECHNOLOGIES
}

# ---------------------------------------------------------------------------
# Clean analysis outputs (grid-level de-duplicated)
# ---------------------------------------------------------------------------

TRAINING_DIR_CLEAN = PROJECT_ROOT / "03_Training_Data_Clean"
MODELS_DIR_CLEAN = PROJECT_ROOT / "04_Models_Clean"
RESULTS_DIR_CLEAN = PROJECT_ROOT / "05_Outputs_Clean"
MAPS_DIR_CLEAN = RESULTS_DIR_CLEAN / "Maps"
FIGURES_DIR_CLEAN = RESULTS_DIR_CLEAN / "Figures"

TRAINING_CSV_CLEAN = {
    tech: TRAINING_DIR_CLEAN / f"train_{tech}.csv"
    for tech in TECHNOLOGIES
}

MODEL_FILES_CLEAN = {
    tech: MODELS_DIR_CLEAN / f"rf_{tech}.joblib"
    for tech in TECHNOLOGIES
}

LR_MODEL_FILES = {
    tech: MODELS_DIR_CLEAN / f"lr_{tech}.joblib"
    for tech in TECHNOLOGIES
}

# ---------------------------------------------------------------------------
# Spatial CV configuration
# ---------------------------------------------------------------------------

SPATIAL_CV_REGIONS = ["Scotland", "North England", "Wales", "Midlands", "South England"]
ENGLAND_NORTH_Y = 400_000
ENGLAND_MIDLANDS_Y = 250_000

COUNTRIES_SHP = (
    RAW_DIR / "Boundaries"
    / "Countries_December_2024_Boundaries_UK_BFC_6983126662299524946"
)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

TECH_COLORS = {"solar": "#F4A300", "wind": "#2E86C1", "bio": "#27AE60"}
TECH_LABELS = {"solar": "Solar PV", "wind": "Onshore Wind", "bio": "Biomass"}
