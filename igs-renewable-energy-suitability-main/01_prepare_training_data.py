"""
Step 1: Prepare training data for Random Forest suitability models.

Loads REPD operational sites, de-duplicates to one record per 12 km
grid cell, generates pseudo-absence points with a 10 km buffer,
extracts 10 predictor values from baseline rasters, and exports
clean training CSVs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from scipy.spatial import cKDTree

import config


def site_to_cell(x, y):
    """Convert BNG coordinates to grid cell indices (row, col)."""
    col = ((x - config.GRID_ORIGIN_X) / config.CELL_SIZE).astype(int)
    row = ((config.GRID_ORIGIN_Y - y) / config.CELL_SIZE).astype(int)
    return row, col


def cell_center(row, col):
    """Return the BNG coordinates of a cell's centre."""
    x = config.GRID_ORIGIN_X + (col + 0.5) * config.CELL_SIZE
    y = config.GRID_ORIGIN_Y - (row + 0.5) * config.CELL_SIZE
    return x, y


def get_valid_mask():
    """Return boolean array (nrows, ncols) where all predictors have data."""
    valid = np.ones((config.GRID_NROWS, config.GRID_NCOLS), dtype=bool)
    all_paths = {
        **{v: config.CLIMATE_BASELINE[v] for v in config.CLIMATE_VARS},
        **{v: config.STATIC_RASTERS[v] for v in config.STATIC_VARS},
    }
    for var, path in all_paths.items():
        with rasterio.open(path) as src:
            band = src.read(1)
            if src.nodata is not None:
                valid &= (band != src.nodata)
            valid &= ~np.isnan(band)
    return valid


def deduplicate_to_cells(df):
    """Convert site coordinates to grid cells and keep one per cell."""
    x = pd.to_numeric(df["X-coordinate"], errors="coerce")
    y = pd.to_numeric(df["Y-coordinate"], errors="coerce")
    df = df.loc[x.notna() & y.notna()].copy()
    rows, cols = site_to_cell(
        df["X-coordinate"].values.astype(float),
        df["Y-coordinate"].values.astype(float),
    )
    df["cell_row"] = rows
    df["cell_col"] = cols
    df["cell_id"] = rows * config.GRID_NCOLS + cols
    return df.drop_duplicates(subset=["cell_id"])


def generate_absences(pres_rows, pres_cols, n_absences, valid_mask, rng):
    """Sample pseudo-absence cells from valid land at least 10 km from presence."""
    pres_x, pres_y = cell_center(pres_rows, pres_cols)
    tree = cKDTree(np.column_stack([pres_x, pres_y]))

    land_rows, land_cols = np.where(valid_mask)
    land_x, land_y = cell_center(land_rows, land_cols)
    dists, _ = tree.query(np.column_stack([land_x, land_y]), k=1)

    pres_ids = set(pres_rows * config.GRID_NCOLS + pres_cols)
    land_ids = land_rows * config.GRID_NCOLS + land_cols
    pool = (dists >= config.ABSENCE_BUFFER_M) & ~np.isin(land_ids, list(pres_ids))

    idx = rng.choice(np.sum(pool), size=n_absences, replace=False)
    return land_rows[pool][idx], land_cols[pool][idx]


def extract_predictors(rows, cols):
    """Sample all 10 baseline predictor values at given cell positions."""
    data = {}
    for var in config.CLIMATE_VARS:
        with rasterio.open(config.CLIMATE_BASELINE[var]) as src:
            data[var] = src.read(1)[rows, cols]
    for var in config.STATIC_VARS:
        with rasterio.open(config.STATIC_RASTERS[var]) as src:
            data[var] = src.read(1)[rows, cols]
    return pd.DataFrame(data)


def assign_regions(xs, ys):
    """Assign spatial CV regions using UK country boundaries."""
    countries = gpd.read_file(config.COUNTRIES_SHP)
    pts = gpd.GeoDataFrame(
        {"y_bng": ys},
        geometry=gpd.points_from_xy(xs, ys),
        crs=config.CRS,
    )
    joined = gpd.sjoin(
        pts, countries[["CTRY24NM", "geometry"]], how="left", predicate="within"
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    regions = pd.Series("", index=joined.index)
    regions.loc[joined["CTRY24NM"] == "Scotland"] = "Scotland"
    regions.loc[joined["CTRY24NM"] == "Wales"] = "Wales"
    eng = joined["CTRY24NM"] == "England"
    regions.loc[eng & (joined["y_bng"] >= config.ENGLAND_NORTH_Y)] = "North England"
    regions.loc[
        eng
        & (joined["y_bng"] >= config.ENGLAND_MIDLANDS_Y)
        & (joined["y_bng"] < config.ENGLAND_NORTH_Y)
    ] = "Midlands"
    regions.loc[eng & (joined["y_bng"] < config.ENGLAND_MIDLANDS_Y)] = "South England"
    return regions.values


def main():
    config.TRAINING_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
    valid_mask = get_valid_mask()
    rng = np.random.default_rng(config.PRIMARY_SEED)

    for tech in config.TECHNOLOGIES:
        df = pd.read_csv(config.PRESENCE_CSV[tech])
        deduped = deduplicate_to_cells(df)
        keep = valid_mask[deduped["cell_row"].values, deduped["cell_col"].values]
        deduped = deduped.loc[keep]

        n_raw = len(df)
        n_unique = len(deduped)
        print(f"{tech.capitalize()}: {n_raw} sites -> {n_unique} unique cells")

        pres_rows = deduped["cell_row"].values
        pres_cols = deduped["cell_col"].values
        abs_rows, abs_cols = generate_absences(
            pres_rows, pres_cols, n_unique, valid_mask, rng
        )

        all_rows = np.concatenate([pres_rows, abs_rows])
        all_cols = np.concatenate([pres_cols, abs_cols])
        labels = np.concatenate([np.ones(n_unique), np.zeros(n_unique)])
        xs, ys = cell_center(all_rows, all_cols)

        predictors = extract_predictors(all_rows, all_cols)
        regions = assign_regions(xs, ys)

        out = pd.DataFrame({"X-coordinate": xs, "Y-coordinate": ys, "label": labels.astype(int)})
        out = pd.concat([out, predictors], axis=1)
        out["region"] = regions
        out.to_csv(config.TRAINING_CSV_CLEAN[tech], index=False)
        print(f"  Saved {len(out)} rows to {config.TRAINING_CSV_CLEAN[tech].name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
