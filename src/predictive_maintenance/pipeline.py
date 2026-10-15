"""Cleaning, feature engineering, model training, evaluation, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from predictive_maintenance.simulator import REQUIRED_COLUMNS, SENSOR_COLUMNS, simulate_sensor_data


ROLLING_WINDOW = 5
TEST_FRACTION = 0.20


def load_sensor_data(input_path: Path | None, **simulation_options: int) -> pd.DataFrame:
    """Load a documented CSV or generate the built-in virtual telemetry."""
    if input_path is None:
        data = simulate_sensor_data(**simulation_options)
    else:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
        data = pd.read_csv(input_path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    return data[REQUIRED_COLUMNS].copy()


def clean_sensor_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps, remove invalid records, and fill sensor gaps."""
    cleaned = data.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)
    cleaned["asset_id"] = cleaned["asset_id"].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["asset_id", "timestamp", "failure_within_horizon"])
    cleaned = cleaned[cleaned["asset_id"] != ""]
    cleaned = cleaned.drop_duplicates(subset=["asset_id", "timestamp"], keep="last")

    for column in SENSOR_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["failure_within_horizon"] = pd.to_numeric(
        cleaned["failure_within_horizon"], errors="coerce"
    )
    cleaned = cleaned.dropna(subset=["failure_within_horizon"])
    cleaned["failure_within_horizon"] = cleaned["failure_within_horizon"].astype(int)
    if not set(cleaned["failure_within_horizon"].unique()).issubset({0, 1}):
        raise ValueError("failure_within_horizon must contain only 0 and 1")

    cleaned = cleaned.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    for column in SENSOR_COLUMNS:
        cleaned[column] = cleaned.groupby("asset_id", observed=True)[column].ffill()
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    cleaned = cleaned.dropna(subset=SENSOR_COLUMNS).reset_index(drop=True)
    if cleaned["asset_id"].nunique() < 2:
        raise ValueError("At least two devices are required for a chronological holdout")
    if cleaned["failure_within_horizon"].nunique() < 2:
        raise ValueError("The dataset must include both warning and normal examples")
    return cleaned


def engineer_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build current, trailing-window, and short-term change features."""
    ordered = data.sort_values(["asset_id", "timestamp"]).copy()
    features = ordered[SENSOR_COLUMNS].copy()
    grouped = ordered.groupby("asset_id", sort=False, observed=True)

    for column in SENSOR_COLUMNS:
        rolling = grouped[column].rolling(ROLLING_WINDOW, min_periods=1)
        features[f"{column}_mean_{ROLLING_WINDOW}"] = (
            rolling.mean().reset_index(level=0, drop=True).sort_index()
        )
        features[f"{column}_std_{ROLLING_WINDOW}"] = (
            rolling.std().reset_index(level=0, drop=True).fillna(0).sort_index()
        )
        features[f"{column}_change_{ROLLING_WINDOW}"] = grouped[column].diff(ROLLING_WINDOW).fillna(0)

    return features.astype(float), list(features.columns)


def chronological_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Hold out later shared timestamps so both periods can contain failures."""
    ordered_times = np.sort(data["timestamp"].unique())
    split_at = max(1, int(len(ordered_times) * (1.0 - TEST_FRACTION)))
    if split_at >= len(ordered_times):
        raise ValueError("At least two distinct timestamps are required for a time split")

    cutoff = ordered_times[split_at]
    train_indices = data.index[data["timestamp"] < cutoff].to_numpy()
    test_indices = data.index[data["timestamp"] >= cutoff].to_numpy()
    for split_name, indices in (("training", train_indices), ("test", test_indices)):
        if data.loc[indices, "failure_within_horizon"].nunique() < 2:
            raise ValueError(
                f"The {split_name} period must include normal and warning examples; "
                "check the label horizon or choose a different time range."
            )
    return train_indices, test_indices


def _save_figures(
    output_dir: Path,
    y_test: pd.Series,
    predictions: np.ndarray,
    model: RandomForestClassifier,
    feature_names: list[str],
    test_data: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set(title="Failure-warning confusion matrix", xlabel="Predicted label", ylabel="Actual label")
    ax.set_xticklabels(["Normal", "Warning"])
    ax.set_yticklabels(["Normal", "Warning"], rotation=0)
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=feature_names).nlargest(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    importance.plot.barh(ax=ax, color="#168c83")
    ax.set(title="Most influential model features", xlabel="Random Forest importance", ylabel="")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    example_asset = test_data["asset_id"].iloc[0]
    example = test_data[test_data["asset_id"] == example_asset].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(example["timestamp"], example["vibration_mm_s"], color="#1f77b4", label="Vibration")
    warning_points = example[example["failure_within_horizon"] == 1]
    ax.scatter(
        warning_points["timestamp"], warning_points["vibration_mm_s"],
        color="#d1495b", s=36, label="Failure within horizon", zorder=3,
    )
    ax.set(title=f"Sensor trend and warning window: {example_asset}", xlabel="Timestamp", ylabel="Vibration (mm/s)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_dir / "sensor_warning_timeline.png", dpi=160)
    plt.close(fig)


def run_pipeline(
    input_path: Path | None,
    assets: int,
    readings_per_asset: int,
    horizon: int,
    seed: int,
    output_dir: Path,
    model_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    """Execute the full workflow and persist all reproducible deliverables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_sensor_data(
        input_path,
        assets=assets,
        readings_per_asset=readings_per_asset,
        horizon=horizon,
        seed=seed,
    )
    if input_path is None:
        raw_data.to_csv(data_dir / "simulated_sensor_readings.csv", index=False)
    data = clean_sensor_data(raw_data)
    features, feature_names = engineer_features(data)
    target = data["failure_within_horizon"]
    train_indices, test_indices = chronological_split(data)

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features.loc[train_indices], target.loc[train_indices])
    probabilities = model.predict_proba(features.loc[test_indices])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y_test = target.loc[test_indices]
    test_data = data.loc[test_indices].copy()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["normal", "warning"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "rows": int(len(data)),
        "devices": int(data["asset_id"].nunique()),
        "training_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "warning_horizon_readings": horizon,
        "seed": seed,
        "data_source": "synthetic simulation" if input_path is None else str(input_path),
    }

    prediction_output = test_data[["asset_id", "timestamp", *SENSOR_COLUMNS]].copy()
    prediction_output["actual_failure_within_horizon"] = y_test.to_numpy()
    prediction_output["predicted_failure_within_horizon"] = predictions
    prediction_output["failure_probability"] = probabilities
    prediction_output.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    model_path = model_dir / "failure_warning_model.joblib"
    joblib.dump({"model": model, "feature_names": feature_names, "horizon": horizon}, model_path)
    _save_figures(output_dir, y_test, predictions, model, feature_names, test_data)

    return {
        "rows": metrics["rows"],
        "devices": metrics["devices"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "predicted_alerts": int(predictions.sum()),
        "actual_warnings": int(y_test.sum()),
    }