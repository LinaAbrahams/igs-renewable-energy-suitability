"""
Step 2: Train Random Forest and Logistic Regression classifiers.

Loads de-duplicated training data, verifies no duplicate predictor rows
exist across the train/test split, trains and evaluates models, and
prints a comparison with the previous (WithRoads) analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

import config


def count_leakage(X_train, X_test):
    """Return number of test rows with an identical predictor vector in train."""
    train_set = set(map(tuple, X_train.values))
    return sum(1 for row in X_test.values if tuple(row) in train_set)


def evaluate(model, X_test, y_test):
    """Compute classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def main():
    config.MODELS_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
    results = []

    for tech in config.TECHNOLOGIES:
        df = pd.read_csv(config.TRAINING_CSV_CLEAN[tech])
        X = df[config.PREDICTOR_COLS]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.PRIMARY_SEED, stratify=y,
        )

        leakage = count_leakage(X_train, X_test)
        print(
            f"{tech.capitalize():8s} {len(X_train):>4d} train / {len(X_test):>4d} test | "
            f"Zero duplicate predictor rows in test set: "
            f"{'YES' if leakage == 0 else f'NO ({leakage})'}"
        )

        rf = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(X_train, y_train)
        rf_m = evaluate(rf, X_test, y_test)
        joblib.dump(rf, config.MODEL_FILES_CLEAN[tech])

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                random_state=config.PRIMARY_SEED, max_iter=1000,
            )),
        ])
        lr.fit(X_train, y_train)
        lr_m = evaluate(lr, X_test, y_test)
        joblib.dump(lr, config.LR_MODEL_FILES[tech])

        top3 = pd.Series(rf.feature_importances_, index=config.PREDICTOR_COLS)
        top3 = top3.nlargest(3).index.tolist()

        results.append({
            "technology": tech,
            "n_train": len(X_train), "n_test": len(X_test),
            "leakage_rows": leakage,
            "rf_accuracy": rf_m["accuracy"], "rf_auc": rf_m["auc"],
            "rf_precision": rf_m["precision"], "rf_recall": rf_m["recall"],
            "rf_f1": rf_m["f1"],
            "lr_accuracy": lr_m["accuracy"], "lr_auc": lr_m["auc"],
            "lr_precision": lr_m["precision"], "lr_recall": lr_m["recall"],
            "lr_f1": lr_m["f1"],
            "top_feature_1": top3[0], "top_feature_2": top3[1],
            "top_feature_3": top3[2],
        })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(config.MODELS_DIR_CLEAN / "metrics_summary.csv", index=False)

    print("\n--- AUC Comparison: Old (WithRoads) vs New (Clean) ---")
    print(f"{'Tech':>10s}  {'Old AUC':>8s}  {'New AUC':>8s}  {'Old n':>6s}  {'New n':>6s}")
    old = pd.read_csv(config.MODELS_DIR / "metrics_summary.csv")
    for tech in config.TECHNOLOGIES:
        old_row = old.loc[old["technology"] == tech].iloc[0]
        new_row = metrics_df.loc[metrics_df["technology"] == tech].iloc[0]
        print(
            f"{tech.capitalize():>10s}  {old_row['auc_score']:>8.4f}  {new_row['rf_auc']:>8.4f}"
            f"  {int(old_row['train_samples'] + old_row['test_samples']):>6d}"
            f"  {int(new_row['n_train'] + new_row['n_test']):>6d}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
