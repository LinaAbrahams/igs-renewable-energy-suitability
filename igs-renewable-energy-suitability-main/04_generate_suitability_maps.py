"""
Step 4: Generate suitability probability maps.

Applies trained RF models to baseline and future climate rasters,
produces delta (change) maps, applies protected-area masking for
developable suitability, and saves statistics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import rasterio
import joblib
import geopandas as gpd
import matplotlib.pyplot as plt

import config


def load_predictor_stack(climate_dict):
    """Load 10 predictor bands and return (X, rows, cols, valid_mask)."""
    bands = {}
    valid = np.ones((config.GRID_NROWS, config.GRID_NCOLS), dtype=bool)

    for var in config.CLIMATE_VARS:
        with rasterio.open(climate_dict[var]) as src:
            data = src.read(1).astype(np.float64)
            if src.nodata is not None:
                valid &= (data != src.nodata)
            valid &= ~np.isnan(data)
            bands[var] = data

    for var in config.STATIC_VARS:
        with rasterio.open(config.STATIC_RASTERS[var]) as src:
            data = src.read(1).astype(np.float64)
            if src.nodata is not None:
                valid &= (data != src.nodata)
            valid &= ~np.isnan(data)
            bands[var] = data

    rows, cols = np.where(valid)
    X = np.column_stack([bands[v][rows, cols] for v in config.PREDICTOR_COLS])
    return X, rows, cols, valid


def predict_map(model, climate_dict):
    """Predict suitability probability for all valid cells."""
    X, rows, cols, _ = load_predictor_stack(climate_dict)
    proba = model.predict_proba(X)[:, 1]
    suit = np.full((config.GRID_NROWS, config.GRID_NCOLS), np.nan, dtype=np.float32)
    suit[rows, cols] = proba
    return suit


def save_geotiff(data, path):
    """Write a 2D array as a single-band GeoTIFF on the project grid."""
    transform = rasterio.transform.from_origin(
        config.GRID_ORIGIN_X, config.GRID_ORIGIN_Y,
        config.CELL_SIZE, config.CELL_SIZE,
    )
    with rasterio.open(
        path, "w", driver="GTiff",
        height=config.GRID_NROWS, width=config.GRID_NCOLS,
        count=1, dtype="float32",
        crs=config.CRS, transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def assign_cell_regions(valid_mask):
    """Return a (nrows, ncols) array of region labels for valid cells."""
    rows, cols = np.where(valid_mask)
    xs = config.GRID_ORIGIN_X + (cols + 0.5) * config.CELL_SIZE
    ys = config.GRID_ORIGIN_Y - (rows + 0.5) * config.CELL_SIZE

    countries = gpd.read_file(config.COUNTRIES_SHP)
    pts = gpd.GeoDataFrame(
        {"y_bng": ys},
        geometry=gpd.points_from_xy(xs, ys), crs=config.CRS,
    )
    joined = gpd.sjoin(
        pts, countries[["CTRY24NM", "geometry"]], how="left", predicate="within",
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    region = np.full(len(joined), "", dtype="U20")
    ctry = joined["CTRY24NM"].values.astype(str)
    yv = joined["y_bng"].values
    region[ctry == "Scotland"] = "Scotland"
    region[ctry == "Wales"] = "Wales"
    eng = ctry == "England"
    region[eng & (yv >= config.ENGLAND_NORTH_Y)] = "North England"
    region[eng & (yv >= config.ENGLAND_MIDLANDS_Y) & (yv < config.ENGLAND_NORTH_Y)] = "Midlands"
    region[eng & (yv < config.ENGLAND_MIDLANDS_Y)] = "South England"

    grid = np.full((config.GRID_NROWS, config.GRID_NCOLS), "", dtype="U20")
    grid[rows, cols] = region
    return grid


def main():
    config.MAPS_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_CLEAN.mkdir(parents=True, exist_ok=True)

    with rasterio.open(config.PROTECTED_MASK) as src:
        protected = src.read(1)

    stats = []
    dev_table = []
    area_km2 = (config.CELL_SIZE / 1000) ** 2

    for tech in config.TECHNOLOGIES:
        print(f"Processing {tech}...")
        model = joblib.load(config.MODEL_FILES_CLEAN[tech])

        baseline = predict_map(model, config.CLIMATE_BASELINE)
        future = predict_map(model, config.CLIMATE_FUTURE)
        delta = future - baseline
        dev = np.where(protected == 1, np.nan, future)

        save_geotiff(baseline, config.MAPS_DIR_CLEAN / f"suit_baseline_{tech}.tif")
        save_geotiff(future, config.MAPS_DIR_CLEAN / f"suit_future_{tech}.tif")
        save_geotiff(delta, config.MAPS_DIR_CLEAN / f"delta_{tech}.tif")
        save_geotiff(dev, config.MAPS_DIR_CLEAN / f"dev_suit_{tech}.tif")

        mean_b = np.nanmean(baseline)
        mean_f = np.nanmean(future)
        pct = (mean_f - mean_b) / mean_b * 100
        high_b = int(np.nansum(baseline >= 0.7))
        high_f = int(np.nansum(future >= 0.7))
        high_d = int(np.nansum(dev >= 0.7))

        stats.append({
            "technology": tech,
            "mean_baseline": mean_b, "mean_future": mean_f,
            "mean_delta": np.nanmean(delta), "pct_change": pct,
            "high_suit_baseline_cells": high_b,
            "high_suit_future_cells": high_f,
            "high_suit_baseline_km2": int(high_b * area_km2),
            "high_suit_future_km2": int(high_f * area_km2),
            "high_suit_dev_cells": high_d,
            "high_suit_dev_km2": int(high_d * area_km2),
        })
        dev_table.append({
            "technology": tech,
            "high_suit_before_masking_km2": int(high_f * area_km2),
            "high_suit_after_masking_km2": int(high_d * area_km2),
            "removed_km2": int((high_f - high_d) * area_km2),
        })
        print(f"  Baseline: {mean_b:.3f}  Future: {mean_f:.3f}  Change: {pct:+.1f}%")

    pd.DataFrame(stats).to_csv(config.MAPS_DIR_CLEAN / "suitability_statistics.csv", index=False)
    pd.DataFrame(dev_table).to_csv(config.MAPS_DIR_CLEAN / "table_11_developable_areas.csv", index=False)

    # --- Climate impact summary ---
    fig, ax = plt.subplots(figsize=(7, 5))
    pcts = [s["pct_change"] for s in stats]
    colors = [config.TECH_COLORS[t] for t in config.TECHNOLOGIES]
    bars = ax.bar(
        [config.TECH_LABELS[t] for t in config.TECHNOLOGIES],
        pcts, color=colors, width=0.5,
    )
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.3 * np.sign(v) if v != 0 else 0.3,
                f"{v:+.1f}%", ha="center", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Change in Mean Suitability (%)")
    ax.set_title("Climate Change Impact on Suitability (2040-2060)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR_CLEAN / "climate_impact_summary.png", dpi=300)
    plt.close(fig)

    # --- Regional suitability ---
    print("Computing regional breakdown...")
    _, _, _, valid_mask = load_predictor_stack(config.CLIMATE_BASELINE)
    region_grid = assign_cell_regions(valid_mask)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(config.SPATIAL_CV_REGIONS))
    w = 0.25
    for i, tech in enumerate(config.TECHNOLOGIES):
        model = joblib.load(config.MODEL_FILES_CLEAN[tech])
        future = predict_map(model, config.CLIMATE_FUTURE)
        means = []
        for region in config.SPATIAL_CV_REGIONS:
            mask = region_grid == region
            vals = future[mask]
            means.append(np.nanmean(vals) if np.any(~np.isnan(vals)) else 0)
        ax.bar(x + i * w, means, w, label=config.TECH_LABELS[tech],
               color=config.TECH_COLORS[tech])

    ax.set_xticks(x + w)
    ax.set_xticklabels(config.SPATIAL_CV_REGIONS, rotation=15, ha="right")
    ax.set_ylabel("Mean Future Suitability")
    ax.set_title("Regional Suitability (2040-2060)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR_CLEAN / "regional_suitability.png", dpi=300)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
