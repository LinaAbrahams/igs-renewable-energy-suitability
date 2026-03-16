"""
Step 3: Validate trained models and assess reliability.

Runs multi-seed sensitivity analysis, leave-one-region-out spatial
cross-validation, and generates validation figures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

import config


def seed_sensitivity(X, y, seeds):
    """Train RF with multiple seeds and return AUC per seed."""
    aucs = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=seed, stratify=y,
        )
        rf = RandomForestClassifier(**{**config.RF_PARAMS, "random_state": seed})
        rf.fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]))
    return aucs


def spatial_cv(X, y, regions):
    """Leave-one-region-out cross-validation."""
    results = []
    for region in config.SPATIAL_CV_REGIONS:
        mask = regions == region
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        rf = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(X[~mask], y[~mask])
        prob = rf.predict_proba(X[mask])[:, 1]
        results.append({
            "region": region,
            "auc": roc_auc_score(y[mask], prob),
            "n_test": int(mask.sum()),
        })
    return results


def plot_stability(all_aucs, path):
    """Box plot of AUC across seeds for each technology."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [all_aucs[t] for t in config.TECHNOLOGIES]
    colors = [config.TECH_COLORS[t] for t in config.TECHNOLOGIES]
    bp = ax.boxplot(
        data,
        labels=[config.TECH_LABELS[t] for t in config.TECHNOLOGIES],
        patch_artist=True, widths=0.5,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("AUC")
    ax.set_title("Model Stability Across Random Seeds")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_spatial_cv(all_cv, path):
    """Heatmap of spatial CV AUC per region and technology."""
    techs = config.TECHNOLOGIES
    regions = config.SPATIAL_CV_REGIONS
    matrix = np.full((len(regions), len(techs)), np.nan)
    for j, tech in enumerate(techs):
        for row in all_cv[tech]:
            if row["region"] in regions:
                matrix[regions.index(row["region"]), j] = row["auc"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(techs)))
    ax.set_xticklabels([config.TECH_LABELS[t] for t in techs])
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    for i in range(len(regions)):
        for j in range(len(techs)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold")
    fig.colorbar(im, label="AUC")
    ax.set_title("Leave-One-Region-Out Spatial CV")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_model_comparison(metrics_df, path):
    """Grouped bar chart comparing RF and LR AUC."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(config.TECHNOLOGIES))
    w = 0.35
    rf = [metrics_df.loc[metrics_df["technology"] == t, "rf_auc"].values[0]
          for t in config.TECHNOLOGIES]
    lr = [metrics_df.loc[metrics_df["technology"] == t, "lr_auc"].values[0]
          for t in config.TECHNOLOGIES]
    ax.bar(x - w / 2, rf, w, label="Random Forest", color="#2E86C1")
    ax.bar(x + w / 2, lr, w, label="Logistic Regression", color="#E67E22")
    ax.set_xticks(x)
    ax.set_xticklabels([config.TECH_LABELS[t] for t in config.TECHNOLOGIES])
    ax.set_ylabel("AUC")
    ax.set_title("Model Comparison: RF vs Logistic Regression")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc_curves(path):
    """Combined ROC curves for all three technologies."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for tech in config.TECHNOLOGIES:
        df = pd.read_csv(config.TRAINING_CSV_CLEAN[tech])
        X, y = df[config.PREDICTOR_COLS], df["label"]
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.PRIMARY_SEED, stratify=y,
        )
        model = joblib.load(config.MODEL_FILES_CLEAN[tech])
        prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        auc = roc_auc_score(y_te, prob)
        ax.plot(fpr, tpr, color=config.TECH_COLORS[tech],
                label=f"{config.TECH_LABELS[tech]} (AUC = {auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_auc_bars(metrics_df, path):
    """AUC bar chart for each technology."""
    fig, ax = plt.subplots(figsize=(7, 5))
    aucs = [metrics_df.loc[metrics_df["technology"] == t, "rf_auc"].values[0]
            for t in config.TECHNOLOGIES]
    colors = [config.TECH_COLORS[t] for t in config.TECHNOLOGIES]
    bars = ax.bar(
        [config.TECH_LABELS[t] for t in config.TECHNOLOGIES],
        aucs, color=colors, width=0.5,
    )
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("AUC")
    ax.set_title("Random Forest AUC by Technology")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    config.FIGURES_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.read_csv(config.MODELS_DIR_CLEAN / "metrics_summary.csv")

    all_aucs = {}
    all_cv = {}

    for tech in config.TECHNOLOGIES:
        df = pd.read_csv(config.TRAINING_CSV_CLEAN[tech])
        X = df[config.PREDICTOR_COLS].values
        y = df["label"].values
        regions = df["region"].values

        aucs = seed_sensitivity(X, y, config.ENSEMBLE_SEEDS)
        all_aucs[tech] = aucs
        mean_auc = np.mean(aucs)
        cv_pct = np.std(aucs) / mean_auc * 100
        print(f"{tech.capitalize():8s}  Mean AUC: {mean_auc:.4f}  CV: {cv_pct:.1f}%")

        cv_results = spatial_cv(X, y, regions)
        all_cv[tech] = cv_results
        for r in cv_results:
            print(f"  {r['region']:15s}  AUC: {r['auc']:.4f}  (n={r['n_test']})")

    cv_rows = []
    for tech in config.TECHNOLOGIES:
        for r in all_cv[tech]:
            cv_rows.append({"technology": tech, **r})
    pd.DataFrame(cv_rows).to_csv(
        config.MODELS_DIR_CLEAN / "spatial_cv_results.csv", index=False,
    )

    figs = config.FIGURES_DIR_CLEAN
    plot_stability(all_aucs, figs / "model_stability.png")
    plot_spatial_cv(all_cv, figs / "spatial_cv_results.png")
    plot_model_comparison(metrics_df, figs / "model_comparison.png")
    plot_roc_curves(figs / "roc_curves_combined.png")
    plot_auc_bars(metrics_df, figs / "model_performance_auc.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
