from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predictive_maintenance.pipeline import (  # noqa: E402
    chronological_split,
    clean_sensor_data,
    engineer_features,
    run_pipeline,
)
from predictive_maintenance.simulator import REQUIRED_COLUMNS, simulate_sensor_data  # noqa: E402


class PredictiveMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = simulate_sensor_data(seed=42)

    def test_simulation_is_repeatable_and_labeled(self) -> None:
        repeated = simulate_sensor_data(seed=42)
        self.assertEqual(list(self.data.columns), REQUIRED_COLUMNS)
        self.assertTrue(self.data.equals(repeated))
        self.assertEqual(set(self.data["failure_within_horizon"].unique()), {0, 1})

    def test_cleaning_deduplicates_and_fills_sensor_gaps(self) -> None:
        damaged = self.data.copy()
        damaged.loc[0, "temperature_c"] = np.nan
        damaged = pd.concat([damaged, damaged.iloc[[1]]], ignore_index=True)

        cleaned = clean_sensor_data(damaged)

        self.assertEqual(len(cleaned), len(self.data))
        self.assertFalse(cleaned.isna().any().any())

    def test_features_and_shared_time_split_are_valid(self) -> None:
        cleaned = clean_sensor_data(self.data)
        features, names = engineer_features(cleaned)
        train_indices, test_indices = chronological_split(cleaned)

        self.assertEqual(len(features), len(cleaned))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertNotIn("failure_within_horizon", names)
        self.assertNotIn("asset_id", names)
        self.assertLess(cleaned.loc[train_indices, "timestamp"].max(), cleaned.loc[test_indices, "timestamp"].min())
        self.assertEqual(set(cleaned.loc[train_indices, "failure_within_horizon"].unique()), {0, 1})
        self.assertEqual(set(cleaned.loc[test_indices, "failure_within_horizon"].unique()), {0, 1})

    def test_pipeline_persists_model_metrics_predictions_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = run_pipeline(
                input_path=None,
                assets=20,
                readings_per_asset=150,
                horizon=10,
                seed=42,
                output_dir=root / "outputs",
                model_dir=root / "models",
                data_dir=root / "data",
            )

            self.assertGreater(summary["recall"], 0)
            self.assertTrue((root / "models" / "failure_warning_model.joblib").is_file())
            self.assertTrue((root / "outputs" / "predictions.csv").is_file())
            metrics_path = root / "outputs" / "metrics.json"
            self.assertGreater(json.loads(metrics_path.read_text(encoding="utf-8"))["test_rows"], 0)
            for chart in (
                "confusion_matrix.png",
                "feature_importance.png",
                "sensor_warning_timeline.png",
            ):
                self.assertTrue((root / "outputs" / "figures" / chart).is_file())


if __name__ == "__main__":
    unittest.main()