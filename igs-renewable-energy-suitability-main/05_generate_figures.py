"""
Step 5: Generate SHAP explanations, feature importance, and error maps.

Produces beeswarm SHAP plots, feature importance bar charts, and
spatial error maps (false positives / false negatives) for each
technology.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import shap
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

import config


def shap_analysis(model, X_test, feature_names, tech, fig_dir):
    """Generate SHAP beeswarm and feature importance plots."""
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_test)
    if isinstance(sv_raw, list):
        sv = sv_raw[1]
    elif sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = sv_raw

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv, X_test, feature_names=feature_names,
        show=False, plot_size=(8, 6),
    )
    plt.title(f"SHAP Summary — {config.TECH_LABELS[tech]}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_summary_{tech}.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)
    sorted_names = [feature_names[i] for i in order]
    sorted_imp = importance[order]
    plt.figure(figsize=(7, 5))
    plt.barh(
        sorted_names,
        sorted_imp,
        color=config.TECH_COLORS[tech], alpha=0.8,
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Feature Importance — {config.TECH_LABELS[tech]}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"feature_importance_{tech}.png", dpi=300)
    plt.close()


def error_map(model, X_test, y_test, test_rows, test_cols, tech, fig_dir):
    """Plot spatial error map with FP (red) and FN (blue)."""
    y_pred = model.predict(X_test)
    fp = (y_pred == 1) & (y_test == 0)
    fn = (y_pred == 0) & (y_test == 1)

    error_grid = np.full((config.GRID_NROWS, config.GRID_NCOLS), np.nan)
    error_grid[test_rows[fp], test_cols[fp]] = 1
    error_grid[test_rows[fn], test_cols[fn]] = -1

    countries = gpd.read_file(config.COUNTRIES_SHP)

    fig, ax = plt.subplots(figsize=(6, 8))
    countries.boundary.plot(ax=ax, color="grey", linewidth=0.6)

    cmap = ListedColormap(["#2E86C1", "#E74C3C"])
    extent = [
        config.GRID_ORIGIN_X,
        config.GRID_ORIGIN_X + config.GRID_NCOLS * config.CELL_SIZE,
        config.GRID_ORIGIN_Y - config.GRID_NROWS * config.CELL_SIZE,
        config.GRID_ORIGIN_Y,
    ]
    ax.imshow(
        error_grid, extent=extent, origin="upper",
        cmap=cmap, vmin=-1, vmax=1, alpha=0.7, interpolation="nearest",
    )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#E74C3C", label="False Positive"),
                 Patch(color="#2E86C1", label="False Negative")],
        loc="lower left",
    )
    ax.set_title(f"Prediction Errors — {config.TECH_LABELS[tech]}")
    ax.set_xlim(-50000, 700000)
    ax.set_ylim(0, 1200000)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(fig_dir / f"error_map_{tech}.png", dpi=300)
    plt.close(fig)


def main():
    config.FIGURES_DIR_CLEAN.mkdir(parents=True, exist_ok=True)

    for tech in config.TECHNOLOGIES:
        print(f"Processing {tech}...")
        df = pd.read_csv(config.TRAINING_CSV_CLEAN[tech])
        X = df[config.PREDICTOR_COLS]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.PRIMARY_SEED, stratify=y,
        )

        model = joblib.load(config.MODEL_FILES_CLEAN[tech])

        shap_analysis(
            model, X_test.values, config.PREDICTOR_COLS,
            tech, config.FIGURES_DIR_CLEAN,
        )

        df_test = df.iloc[X_test.index]
        xs = df_test["X-coordinate"].values
        ys = df_test["Y-coordinate"].values
        test_rows = ((config.GRID_ORIGIN_Y - ys) / config.CELL_SIZE).astype(int)
        test_cols = ((xs - config.GRID_ORIGIN_X) / config.CELL_SIZE).astype(int)

        error_map(
            model, X_test, y_test.values,
            test_rows, test_cols,
            tech, config.FIGURES_DIR_CLEAN,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
