"""
Step 6: Ensemble uncertainty analysis and threshold sensitivity.

Part A predicts future suitability under three UKCP18 ensemble members,
computes pixel-wise mean and standard deviation, and produces panel maps.
Part B evaluates how high-suitability area changes across probability
thresholds.
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


MEMBERS = {
    "Member01": config.CLIMATE_FUTURE,
    "Member04": config.CLIMATE_MEMBERS["Member04"]["future"],
    "Member08": config.CLIMATE_MEMBERS["Member08"]["future"],
}

CELL_AREA_KM2 = (config.CELL_SIZE / 1000) ** 2


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


def ensemble_analysis():
    """Part A: predict under 3 ensemble members, compute mean and std."""
    countries = gpd.read_file(config.COUNTRIES_SHP)
    extent = [
        config.GRID_ORIGIN_X,
        config.GRID_ORIGIN_X + config.GRID_NCOLS * config.CELL_SIZE,
        config.GRID_ORIGIN_Y - config.GRID_NROWS * config.CELL_SIZE,
        config.GRID_ORIGIN_Y,
    ]

    all_maps = {}
    summary_rows = []

    for tech in config.TECHNOLOGIES:
        model = joblib.load(config.MODEL_FILES_CLEAN[tech])
        member_maps = []
        for member_name, climate_dict in MEMBERS.items():
            suit = predict_map(model, climate_dict)
            member_maps.append(suit)
            summary_rows.append({
                "technology": tech,
                "member": member_name,
                "mean_suitability": float(np.nanmean(suit)),
            })
        all_maps[tech] = member_maps

        stack = np.stack(member_maps, axis=0)
        ens_mean = np.nanmean(stack, axis=0)
        ens_std = np.nanstd(stack, axis=0)

        save_geotiff(ens_mean, config.MAPS_DIR_CLEAN / f"ensemble_mean_{tech}.tif")
        save_geotiff(ens_std, config.MAPS_DIR_CLEAN / f"ensemble_std_{tech}.tif")
        print(f"{tech.capitalize():8s}  mean={np.nanmean(ens_mean):.3f}  "
              f"mean_std={np.nanmean(ens_std):.4f}")

    pd.DataFrame(summary_rows).to_csv(
        config.MAPS_DIR_CLEAN / "ensemble_summary.csv", index=False,
    )

    # --- 3×3 panel: rows = technology, cols = member ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 16))
    member_names = list(MEMBERS.keys())
    for i, tech in enumerate(config.TECHNOLOGIES):
        for j, member_name in enumerate(member_names):
            ax = axes[i, j]
            countries.boundary.plot(ax=ax, color="grey", linewidth=0.5)
            im = ax.imshow(
                all_maps[tech][j], extent=extent, origin="upper",
                cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest",
            )
            ax.set_xlim(-50_000, 700_000)
            ax.set_ylim(0, 1_200_000)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(member_name, fontsize=13)
            if j == 0:
                ax.set_ylabel(config.TECH_LABELS[tech], fontsize=13)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Suitability Probability")
    fig.suptitle("Future Suitability by Ensemble Member", fontsize=15, y=0.95)
    fig.savefig(config.FIGURES_DIR_CLEAN / "ensemble_suitability_3x3.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 1×3 std panel ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, tech in enumerate(config.TECHNOLOGIES):
        ax = axes[i]
        countries.boundary.plot(ax=ax, color="grey", linewidth=0.5)
        stack = np.stack(all_maps[tech], axis=0)
        std_map = np.nanstd(stack, axis=0)
        im = ax.imshow(
            std_map, extent=extent, origin="upper",
            cmap="Purples", vmin=0, vmax=np.nanmax(std_map),
            interpolation="nearest",
        )
        ax.set_xlim(-50_000, 700_000)
        ax.set_ylim(0, 1_200_000)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(config.TECH_LABELS[tech], fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Std Dev")

    fig.suptitle("Ensemble Uncertainty (Standard Deviation)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(config.FIGURES_DIR_CLEAN / "ensemble_uncertainty_std.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def threshold_sensitivity():
    """Part B: area exceeding suitability thresholds."""
    thresholds = [0.5, 0.6, 0.7, 0.8]
    rows = []

    for tech in config.TECHNOLOGIES:
        with rasterio.open(config.MAPS_DIR_CLEAN / f"suit_future_{tech}.tif") as src:
            suit = src.read(1)
        for thresh in thresholds:
            n_cells = int(np.nansum(suit >= thresh))
            rows.append({
                "technology": tech,
                "threshold": thresh,
                "n_cells": n_cells,
                "area_km2": int(n_cells * CELL_AREA_KM2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(config.MAPS_DIR_CLEAN / "threshold_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(thresholds))
    w = 0.25
    for i, tech in enumerate(config.TECHNOLOGIES):
        sub = df[df["technology"] == tech]
        ax.bar(x + i * w, sub["area_km2"].values, w,
               label=config.TECH_LABELS[tech], color=config.TECH_COLORS[tech])
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"≥ {t}" for t in thresholds])
    ax.set_xlabel("Suitability Threshold")
    ax.set_ylabel("Area (km²)")
    ax.set_title("High-Suitability Area by Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR_CLEAN / "threshold_sensitivity.png", dpi=300)
    plt.close(fig)

    print("\n--- Threshold Sensitivity ---")
    for tech in config.TECHNOLOGIES:
        sub = df[df["technology"] == tech]
        vals = "  ".join(f"≥{r['threshold']}: {r['area_km2']:,} km²"
                         for _, r in sub.iterrows())
        print(f"  {tech.capitalize():8s}  {vals}")


def main():
    config.MAPS_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR_CLEAN.mkdir(parents=True, exist_ok=True)

    print("=== Part A: Ensemble Uncertainty ===")
    ensemble_analysis()

    print("\n=== Part B: Threshold Sensitivity ===")
    threshold_sensitivity()

    print("\nDone.")


if __name__ == "__main__":
    main()
