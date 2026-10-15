"""Run the predictive-maintenance training and simulation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predictive_maintenance.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate IoT sensor histories and train a failure-warning model."
    )
    parser.add_argument("--input", type=Path, help="Optional CSV using the documented schema.")
    parser.add_argument("--assets", type=int, default=30, help="Simulated device count (default: 30).")
    parser.add_argument(
        "--readings-per-asset", type=int, default=180,
        help="Maximum simulated readings per device (default: 180).",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Warning horizon in readings (default: 10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.assets < 2:
        raise SystemExit("--assets must be at least 2 so a chronological holdout can be evaluated.")
    if args.readings_per_asset < 30:
        raise SystemExit("--readings-per-asset must be at least 30.")
    if args.horizon < 1 or args.horizon >= args.readings_per_asset:
        raise SystemExit("--horizon must be positive and smaller than --readings-per-asset.")

    summary = run_pipeline(
        input_path=args.input,
        assets=args.assets,
        readings_per_asset=args.readings_per_asset,
        horizon=args.horizon,
        seed=args.seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
    )
    print("\nPredictive maintenance run complete")
    print(f"Rows: {summary['rows']:,} | devices: {summary['devices']}")
    print(f"Accuracy: {summary['accuracy']:.3f} | precision: {summary['precision']:.3f} | recall: {summary['recall']:.3f}")
    print(f"Test alerts: {summary['predicted_alerts']} | actual warnings: {summary['actual_warnings']}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    print(f"Predictions: {args.output_dir / 'predictions.csv'}")
    print(f"Charts: {args.output_dir / 'figures'}")
    print(f"Model: {args.model_dir / 'failure_warning_model.joblib'}")


if __name__ == "__main__":
    main()