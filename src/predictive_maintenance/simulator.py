"""Generate deterministic, simulated sensor histories for multiple devices."""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "current_a",
    "rpm",
]
REQUIRED_COLUMNS = ["asset_id", "timestamp", *SENSOR_COLUMNS, "failure_within_horizon"]


def simulate_sensor_data(
    assets: int = 30,
    readings_per_asset: int = 180,
    horizon: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create pre-failure telemetry with an explicit future-failure label.

    Each device runs until its simulated failure point. Sensor drift becomes
    stronger during the final part of the cycle, and the target is one when
    failure is due within the next ``horizon`` readings.
    """
    if assets < 2:
        raise ValueError("assets must be at least 2")
    if readings_per_asset < 30:
        raise ValueError("readings_per_asset must be at least 30")
    if not 1 <= horizon < readings_per_asset:
        raise ValueError("horizon must be positive and smaller than readings_per_asset")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    lower_failure = max(2 * horizon + 1, int(readings_per_asset * 0.72))
    upper_failure = max(lower_failure + 1, int(readings_per_asset * 0.96))
    upper_failure = min(upper_failure, readings_per_asset)

    for asset_number in range(1, assets + 1):
        asset_id = f"PUMP-{asset_number:03d}"
        failure_step = int(rng.integers(lower_failure, upper_failure))
        offsets = rng.normal(0.0, [2.2, 0.08, 0.25, 0.7, 28.0])

        for step in range(failure_step):
            degradation = np.clip((step - (failure_step - 48)) / 48, 0.0, 1.0)
            temperature = 66.0 + 13.0 * degradation + offsets[0] + rng.normal(0, 1.8)
            vibration = 0.42 + 2.7 * degradation + offsets[1] + rng.normal(0, 0.12)
            pressure = 8.2 - 1.35 * degradation + offsets[2] + rng.normal(0, 0.2)
            current = 11.8 + 4.2 * degradation + offsets[3] + rng.normal(0, 0.55)
            rpm = 1490.0 - 170.0 * degradation + offsets[4] + rng.normal(0, 18.0)
            readings_to_failure = failure_step - step

            rows.append(
                {
                    "asset_id": asset_id,
                    "timestamp": start + pd.Timedelta(minutes=step),
                    "temperature_c": temperature,
                    "vibration_mm_s": vibration,
                    "pressure_bar": pressure,
                    "current_a": current,
                    "rpm": rpm,
                    "failure_within_horizon": int(readings_to_failure <= horizon),
                }
            )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)