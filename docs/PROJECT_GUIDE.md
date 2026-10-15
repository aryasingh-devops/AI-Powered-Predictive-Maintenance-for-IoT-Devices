# Project Guide: From Setup to Portfolio Proof

## A. What Predictive Maintenance Means

**Simple explanation:** A machine gives clues about its health. Predictive maintenance watches those clues and estimates whether a breakdown is becoming likely, so a team can inspect or service the machine before an unplanned stop.

**Technical explanation:** Condition-monitoring telemetry (temperature, vibration, pressure, electrical current, speed, acoustic signals, and operating context) is time-aligned with inspection or failure events. A supervised model estimates a defined target such as failure within a horizon or remaining useful life. Its output is combined with operating limits, maintenance policy, and human review to create an actionable work order or alert.

Manufacturing and factory teams use condition estimates to protect throughput and coordinate planned stops. Power plants monitor critical rotating and thermal assets where outages are costly. Automotive operations can prioritize production-line equipment or fleet inspections. Aviation uses highly controlled health-monitoring processes to support inspections and maintenance planning; a student classifier is not qualified for aviation decisions. Across all sectors the central value is better-timed intervention, not a promise that failures disappear.

The end-to-end workflow is: acquire and time-stamp sensors; validate units, device identity, and quality; label historical windows from maintenance records; clean and align data; engineer past-only features; split by time and preferably by asset; train a baseline and candidate model; assess recall, precision, false alarms, calibration, and lead time; generate an alert with an explicit threshold and escalation path; monitor the deployed model and data drift. Visualization helps people inspect trends and model behavior; it is not itself an alert-management system.

## B. Architecture

```text
                     +----------------------+
                     | CSV dataset or       |
                     | virtual sensor source|
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Validate / clean /   |
                     | order by asset/time  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Past-only rolling    |
                     | features             |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Shared-time split    |
                     | and Random Forest    |
                     +----------+-----------+
                                |
                +---------------+----------------+
                v                                v
       +------------------+             +-------------------+
       | Metrics and      |             | Warning score and |
       | confusion matrix |             | prediction CSV    |
       +------------------+             +--------+----------+
                                                   |
                                                   v
                                       +----------------------+
                                       | Human-reviewed alert|
                                       | in this demo only   |
                                       +----------------------+
```

The sensor data is one row per device and timestamp. The simulator creates five continuous signals and a binary target: `failure_within_horizon=1` means the simulated failure occurs in the next configured number of readings. In the project, the model emits a probability; a 0.5 threshold produces the demonstration's binary warning. The plots, CSV, and local Streamlit dashboard provide visualization and alert evidence. No alert is sent to a real device or operator.

## C. Implementation Phases

| Phase | What and why | Expected output | Common mistake | Verify |
| --- | --- | --- | --- | --- |
| 1. Setup | Create a Python 3.10–3.13 venv and install pinned-range dependencies. | Importable environment | Installing into another interpreter | `python --version`; `python -m pip list` |
| 2. Dataset loading | Use built-in virtual data or a labeled CSV with the documented schema. | One timestamped table | Mixing units or ambiguous labels | Inspect header and target counts |
| 3. Data cleaning | Parse timestamps, reject invalid IDs/labels, deduplicate asset-time records, fill sensor gaps. | Sorted, usable records | Dropping asset/time identity or silently coercing target labels | Check missing values and binary classes |
| 4. Feature engineering | Calculate trailing mean, standard deviation, and five-reading change for each sensor. | Model-ready numeric matrix | Using future/centered rolling windows | Confirm each feature uses current/past only |
| 5. Model building | Fit class-balanced Random Forest on earlier shared timestamps. | Fitted model artifact | Training on the holdout or passing timestamp/target as a feature | Check saved feature names and split counts |
| 6. Evaluation | Compute accuracy, precision, recall, ROC-AUC, report, and confusion matrix on later readings. | `metrics.json` and plot | Reporting accuracy alone for rare failures | Review warning-class recall and false negatives |
| 7. Failure prediction | Convert probability to a 0.5 warning decision and save each test row. | `predictions.csv` | Treating probability as certainty | Inspect probabilities and threshold implications |
| 8. Visualization | Plot confusion matrix, feature importance, and an asset sensor timeline; review them in the dashboard. | Three PNGs and local dashboard | Calling feature importance causal evidence | Check labels, axes, and saved image files |
| 9. GitHub publishing | Commit source and curated proof, ignore generated data/model artifacts. | Reproducible public repo | Committing secrets or a huge dataset | Review `git status` and README links |
| 10. Final output | Run from a clean venv and verify all expected artifacts. | Reproducible end-to-end demo | Claiming simulated scores generalize to plants | Repeat run with fixed seed and inspect output |

## D. Virtual Simulation Walkthrough

1. The simulator creates device IDs and a sequence of timestamped readings for each virtual pump.
2. A deterministic random generator chooses when each device reaches its simulated failure point.
3. During the final 48 readings, temperature, vibration, and current tend to rise, while pressure and RPM tend to fall; noise and device-specific offsets prevent identical traces.
4. The target is 1 for the last `horizon` readings before the failure point and 0 earlier. There are no post-failure observations.
5. Cleaning orders each device by timestamp. Feature engineering computes raw measurements, trailing five-reading averages and standard deviations, and change over five readings.
6. Earlier 80% of the shared timestamp range is used for training; later 20% is held out. Different devices reach their simulated failure points at different times, so both periods can include warning examples without randomly mixing adjacent timestamps.
7. The classifier estimates warning probability. The example marks probabilities of at least 0.5 as warnings.
8. The run writes a raw generated CSV, serialized model, metrics JSON, holdout prediction CSV, and three plots.

This is a virtual simulation, not a physical digital twin. It captures simple degradation patterns for software practice but omits real failure modes, operating regimes, maintenance interventions, sensor faults, and plant-specific costs. Never use it to control machinery.

## E. Run, Train, and Inspect

```bash
python main.py
python main.py --assets 40 --readings-per-asset 220 --horizon 12 --seed 7
python main.py --input data/my_labeled_sensors.csv --horizon 10
python -m streamlit run dashboard.py
```

The default command performs training and evaluation in one run. The dashboard opens at `http://localhost:8501`; its sidebar can launch a new simulation. Open `outputs/metrics.json` for summary scores, `outputs/predictions.csv` for holdout rows and probabilities, and the PNGs under `outputs/figures/` for visual review. The actual values vary with simulation settings and dependency versions; use the values produced by your own run in the portfolio, never copied sample scores.

If a warning appears in the test CSV, it is a model decision, not a maintenance instruction. Check its actual label and probability. To test a specific custom sensor dataset, keep its target definition aligned with the horizon; the app expects already labeled rows.

## F. GitHub Proof Strategy: Seven Days

| Day | Work and commit | Proof to capture |
| --- | --- | --- |
| 1. Setup | Create the repository structure, venv instructions, requirements, and scope. Commit `docs: define project scope and setup`. | Root folders, Python version, successful dependency install |
| 2. Dataset | Implement simulator and schema guide. Commit `feat: simulate labeled device telemetry`. | CSV header and a small table preview; sensor trend |
| 3. Preprocessing | Add validation, cleanup, chronological features/split. Commit `feat: prepare time-aware sensor features`. | Row counts, class distribution, feature list |
| 4. Model | Train and save Random Forest. Commit `feat: train failure warning classifier`. | Terminal run and model file existence (do not need to commit model) |
| 5. Evaluation | Report metrics and confusion matrix. Commit `feat: report holdout model metrics`. | `metrics.json` and confusion-matrix chart |
| 6. Visualization | Add feature ranking and sensor warning timeline. Commit `feat: visualize failure warning results`. | All three reviewed charts in `images/` |
| 7. Publish | Polish README, topics, screenshots, and push. Commit `docs: publish run guide and project proof`. | GitHub repo page, commit history, README rendered correctly |

Every commit should be small enough to explain in an interview. Keep generated data and model binary out of Git by default; add only curated small screenshots under `images/`. Use a clear description and topics such as `predictive-maintenance`, `iot`, `machine-learning`, `python`, and `scikit-learn`.

## G. Screenshot and Demo Asset Checklist

| Evidence | Suggested filename | Where it belongs |
| --- | --- | --- |
| Small sensor-data preview | `sensor-data-preview.png` | `images/` |
| Preprocessing / feature counts | `preprocessing-summary.png` | `images/` |
| Training terminal output | `training-run.png` | `images/` |
| Prediction rows and probabilities | `prediction-preview.png` | `images/` |
| Failure warning timeline | `sensor-warning-timeline.png` | `images/` |
| Confusion matrix | `confusion-matrix.png` | `images/` |
| Feature importance | `feature-importance.png` | `images/` |
| GitHub repository page | `github-repository.png` | portfolio or `images/` |

Capture legible windows without personal information or secrets. Keep original outputs under `outputs/` locally and copy only selected, reviewed images into `images/`; embed them in the README with Markdown such as `![Confusion matrix](images/confusion-matrix.png)`. Make sure the plotted scores match the same code/data version described in the commit.

## H. GitHub Publishing Commands

Create an empty repository named `ai-predictive-maintenance-iot`, then from the project root:

```bash
git init
git add README.md requirements.txt .gitignore main.py dashboard.py src data docs notebooks images outputs tests
git status
git commit -m "Build simulated IoT predictive maintenance pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-predictive-maintenance-iot.git
git push -u origin main
```

Before every `git add`, inspect `git status`. Do not commit passwords, tokens, private data, full environment folders, or unexplained generated artifacts. Make sure screenshots are your own run's evidence.

## I. Final Audit

- [ ] Fresh environment installs from `requirements.txt`.
- [ ] `python main.py` completes with a fixed seed.
- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] Generated CSV has multiple devices and both target classes.
- [ ] Training timestamps precede holdout timestamps, and both splits contain both labels.
- [ ] Features exclude target, asset ID, and future/centered information.
- [ ] Metrics, predictions, model, and all three charts are produced.
- [ ] README commands and CSV schema match the implementation.
- [ ] Claims distinguish a synthetic prototype from a validated industrial system.
- [ ] Git status contains no credentials, virtual environment, or unintended large artifacts.

## J. Next Steps Toward Industry Relevance

After this proof, compare a simple Logistic Regression baseline with Random Forest, test on NASA C-MAPSS or another dataset with a clearly defined target, validate by leaving entire assets out, tune the threshold using the cost of missed failures and false alarms, and track experiments. Extend the dashboard with threshold comparison and alert review only after the evaluation is understood. For an operational system, add sensor-quality checks, model monitoring, access control, alert acknowledgement, retraining governance, and human safety review.