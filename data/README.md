# Data

The application generates `simulated_sensor_readings.csv` on the first run. The file is ignored by Git because it is reproducible. No hardware or external download is required.

To use another labeled dataset, pass `--input path/to/data.csv`. The CSV must contain `asset_id`, `timestamp`, `temperature_c`, `vibration_mm_s`, `pressure_bar`, `current_a`, `rpm`, and `failure_within_horizon`. The target must be 0/1 and mean that failure is expected within the chosen number of future readings. Do not include post-failure rows unless they are intentionally part of the prediction problem.