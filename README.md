# AI-Powered Predictive Maintenance for IoT Devices
A reproducible, hardware-free demonstration of machine-health monitoring. It simulates temperature, vibration, pressure, motor current, and RPM for multiple pump assets, engineers trailing sensor features, trains a Random Forest warning model, and saves test predictions, evaluation metrics, and charts.

## 📸 Output Screenshots
<img width="1920" height="1020" alt="Screenshot 2026-10-14 220224" src="https://github.com/user-attachments/assets/a44f7c2f-d2c0-4a88-8ce9-56e476a6970d" />
<img width="1920" height="1020" alt="Screenshot 2026-10-14 220259" src="https://github.com/user-attachments/assets/c84666bd-4581-414c-af4d-7225d444f0fa" />

## 🎥 Demo Video
https://youtu.be/pAaO1j6McSQ?si=FOrnhODDs61vC5Gr

> **Scope:** The built-in dataset is synthetic. This project demonstrates an end-to-end workflow; its scores are not evidence of performance on real equipment. Industrial deployment needs representative validated data, domain review, safety controls, monitoring, and an appropriate alert policy.

## ⚠️ Problem and Industry Context

Reactive maintenance finds a fault after a machine has stopped; fixed-interval preventive maintenance replaces parts on a schedule, sometimes too early and sometimes too late. Predictive maintenance uses condition data and failure history to estimate whether intervention is becoming necessary, giving a maintenance team time to inspect and plan.

Factories and manufacturing plants use it to reduce unplanned line stoppages; power plants monitor rotating equipment and thermal systems; automotive operations monitor production machinery and fleet components; aviation teams monitor engine and component health under strict certification and safety processes. Typical benefits are fewer surprise outages, better spare-parts planning, longer useful asset life, and more focused maintenance. The size of those benefits depends on data quality, costs, and operations; this demo makes no quantified savings claim.

In simple terms, sensors report how a machine is behaving. The program looks for patterns that appeared before known failures and raises a warning when the pattern resembles an upcoming failure. Technically, timestamped telemetry is cleaned, transformed into trailing-window features, split by time, used to fit a binary classifier, evaluated on later readings, and converted into a probability and a warning decision.

## 👩‍💻 Workflow

```text
Sensors / virtual simulator
        -> timestamped asset telemetry
        -> validation, sorting, missing-value handling
        -> trailing means, variation, and short-term changes
        -> per-asset chronological train/test split
        -> balanced Random Forest classifier
        -> warning probability and binary decision
        -> metrics, CSV predictions, and PNG charts
```

The virtual simulator varies five sensors and gives each device a gradual degradation phase before a simulated failure. `failure_within_horizon=1` means the failure point is within the next configured number of readings. The pipeline uses a 0.5 probability threshold for its example warning. A production alert should instead be selected with operators according to the relative cost of missed failures and unnecessary inspections.

## 📃 Architecture and Modules

| Component | Responsibility |
| --- | --- |
| `src/predictive_maintenance/simulator.py` | Reproducibly creates labeled device histories and documents the sensor schema. |
| `src/predictive_maintenance/pipeline.py` | Loads CSV or simulated data, cleans it, builds features, performs a time split, trains/evaluates the model, and writes artifacts. |
| `main.py` | Validates command-line options and runs the complete workflow. |
| `dashboard.py` | Provides a local Streamlit interface for launching runs and inspecting saved metrics, charts, and predictions. |
| `data/` | Generated or user-provided source data. Generated CSV is reproducible and Git-ignored. |
| `models/` | Serialized fitted model and feature metadata. Model binaries are Git-ignored. |
| `outputs/` | Metrics, holdout predictions, and charts from a run. Generated files are Git-ignored. |
| `images/` | Curated screenshots/plots checked into Git for the README and project proof. |
| `docs/` | Practical guide, limitations, and presentation notes. |

## 💻 Tech Stack Options

| Option | Tools and data | Models | Difficulty / GPU | Expected output |
| --- | --- | --- | --- | --- |
| A: Beginner (selected) | Python, Pandas, NumPy, scikit-learn, Streamlit; simulated or labeled tabular sensor histories | Logistic Regression baseline, Random Forest | Beginner; no GPU | Binary warning, metrics, feature importance, confusion matrix, local dashboard |
| B: Intermediate | Python, Pandas, XGBoost or LightGBM, MLflow; NASA C-MAPSS or a labeled industrial dataset | Gradient boosting, survival / remaining-useful-life model | Intermediate; CPU usually sufficient | Calibrated failure risk or remaining useful life, experiment tracking |
| C: Advanced | PyTorch, streaming stack (MQTT/Kafka), time-series store and dashboard | LSTM/Temporal CNN/Transformer and anomaly detection | Advanced; GPU recommended for deep learning at scale | Online ingestion, sequence forecasts, monitored alert service |

The selected option is easiest to run locally while still demonstrating a useful engineering workflow. Logistic Regression is a reasonable baseline extension; the submitted pipeline uses Random Forest to capture nonlinear sensor interactions without a GPU. LSTM is not necessary for this small proof-of-work project.

## 🧩 Repository Layout

```text
AI-Powered Predictive Maintenance for IoT Devices/
├── data/
│   └── README.md
├── docs/
│   └── PROJECT_GUIDE.md
├── images/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── outputs/
│   ├── README.md
│   └── .gitkeep
├── src/predictive_maintenance/
│   ├── __init__.py
│   ├── pipeline.py
│   └── simulator.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── dashboard.py
├── main.py
├── README.md
└── requirements.txt
```

## 📂 Setup

Use Python 3.10–3.13 for a stable beginner environment. Python 3.14 prereleases may not have compatible binary wheels for every scientific library yet. No GPU, cloud account, sensor, or external dataset download is needed.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script activation is blocked, run the environment's interpreter directly: `\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🎯 Run

From the repository root with the virtual environment activated:

```bash
python main.py
```

The default run simulates 30 devices with up to 180 readings each and a 10-reading warning horizon. It creates:

```text
data/simulated_sensor_readings.csv
models/failure_warning_model.joblib
outputs/metrics.json
outputs/predictions.csv
outputs/figures/confusion_matrix.png
outputs/figures/feature_importance.png
outputs/figures/sensor_warning_timeline.png
```

Change simulation size or seed for a repeatable experiment:

```bash
python main.py --assets 40 --readings-per-asset 220 --horizon 12 --seed 7
```

Run a compatible, already labeled CSV:

```bash
python main.py --input data/my_labeled_sensors.csv --horizon 10
```

The CSV schema and label meaning are described in [`data/README.md`](data/README.md). The passed horizon documents the meaning of your label; the external CSV target is not regenerated by the program.

Launch the interactive dashboard from the repository root:

```bash
python -m streamlit run dashboard.py
```

Open `http://localhost:8501`, choose simulation settings in the sidebar, and select **Run simulation and train**. The dashboard shows holdout metrics, charts, a prediction preview, and a CSV download.

Run the automated regression checks with:

```bash
python -m unittest discover -s tests -v
```

## 📚 Results and Interpretation

`metrics.json` reports accuracy, precision, recall, ROC-AUC when both classes occur in the holdout, the confusion matrix, split sizes, and a class-wise report. Accuracy alone can be misleading when failures are rare. Recall measures how many warning cases were found; precision measures how many raised warnings were true warnings. Compare both alongside false negatives, false positives, alert lead time, and maintenance costs before choosing a threshold.

The test set is the later 20% of the shared timestamp range. Trailing features use current and past readings only. This makes the demonstration more realistic than randomly mixing neighboring timestamps, though it is not a substitute for validation on separate real assets and operating conditions. Small changes in the seed, synthetic generator, and library versions can change metrics.

## 📸 Screenshots and Demo Assets

The checked-in charts below were generated by this project run. Capture your own terminal run, dataset preview, prediction table, and GitHub repository page before presenting the project; do not commit the serialized model or generated full dataset.

![Confusion matrix](images/confusion-matrix.png)

![Feature importance](images/feature-importance.png)

![Sensor warning timeline](images/sensor-warning-timeline.png)

## 📝 Learning Outcomes

- Sensor-data schema design and deterministic virtual simulation.
- Time-aware validation and feature engineering for telemetry.
- Classification metrics under class imbalance.
- Saving reproducible model, prediction, and visualization artifacts.
- Communicating data limitations and a path from prototype to operational validation.

## ⚙️ GitHub

Suggested repository name: `ai-predictive-maintenance-iot`. Description: `A reproducible Python demo that simulates IoT sensor telemetry, predicts near-term equipment failure, and reports model metrics and charts.` Suggested topics: `predictive-maintenance`, `iot`, `machine-learning`, `scikit-learn`, `python`, `industrial-analytics`, `time-series`.

```bash
git init
git add README.md requirements.txt .gitignore main.py dashboard.py src data docs notebooks images outputs tests
git commit -m "Build simulated IoT predictive maintenance pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-predictive-maintenance-iot.git
git push -u origin main
```

Make focused commits as work progresses: `docs: define project scope`, `feat: add deterministic sensor simulator`, `feat: train and evaluate warning model`, `feat: add evaluation charts`, `docs: add run guide and proof assets`. Inspect `git status` before every commit so generated files and secrets are not added.

See [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) for the ten implementation phases, simulation walkthrough, GitHub proof schedule, and final audit checklist.
