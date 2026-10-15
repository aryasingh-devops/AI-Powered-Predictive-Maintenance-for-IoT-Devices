"""Local Streamlit dashboard for reviewing a predictive-maintenance run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predictive_maintenance.pipeline import run_pipeline  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="IoT Predictive Maintenance", page_icon="📡", layout="wide")
st.title("IoT Predictive Maintenance")
st.caption("Virtual sensor telemetry · near-term failure warnings · local model evaluation")
st.warning(
    "This dashboard uses synthetic demonstration data. It is not validated for real equipment "
    "and must not be used to make safety or maintenance decisions."
)

with st.sidebar:
    st.header("Simulation")
    asset_count = st.slider("Virtual devices", min_value=10, max_value=60, value=30, step=5)
    readings = st.slider("Maximum readings per device", min_value=100, max_value=300, value=180, step=20)
    horizon = st.slider("Warning horizon (readings)", min_value=5, max_value=20, value=10)
    seed = st.number_input("Random seed", min_value=0, max_value=100_000, value=42)
    run_clicked = st.button("Run simulation and train", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Generating telemetry, training the model, and saving evaluation outputs..."):
        try:
            summary = run_pipeline(
                input_path=None,
                assets=asset_count,
                readings_per_asset=readings,
                horizon=horizon,
                seed=int(seed),
                output_dir=OUTPUT_DIR,
                model_dir=MODEL_DIR,
                data_dir=DATA_DIR,
            )
            st.success(
                f"Run complete: {summary['rows']:,} readings across {summary['devices']} devices."
            )
        except Exception as error:
            st.exception(error)

metrics_path = OUTPUT_DIR / "metrics.json"
predictions_path = OUTPUT_DIR / "predictions.csv"
if not metrics_path.is_file() or not predictions_path.is_file():
    st.info("No saved run is available yet. Choose simulation settings in the sidebar and run the pipeline.")
    st.stop()

with metrics_path.open(encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)

st.subheader("Holdout performance")
metric_columns = st.columns(4)
metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
metric_columns[1].metric("Precision", f"{metrics['precision']:.3f}")
metric_columns[2].metric("Recall", f"{metrics['recall']:.3f}")
auc_value = metrics.get("roc_auc")
metric_columns[3].metric("ROC-AUC", f"{auc_value:.3f}" if auc_value is not None else "N/A")
st.caption(
    f"{metrics['rows']:,} rows · {metrics['devices']} devices · "
    f"{metrics['training_rows']:,} training rows · {metrics['test_rows']:,} later holdout rows · "
    f"warning horizon: {metrics['warning_horizon_readings']} readings"
)

st.subheader("Evaluation charts")
chart_columns = st.columns(2)
chart_paths = [
    OUTPUT_DIR / "figures" / "confusion_matrix.png",
    OUTPUT_DIR / "figures" / "feature_importance.png",
    OUTPUT_DIR / "figures" / "sensor_warning_timeline.png",
]
chart_titles = ["Confusion matrix", "Feature importance", "Sensor warning timeline"]
for index, (chart_path, chart_title) in enumerate(zip(chart_paths, chart_titles, strict=True)):
    if chart_path.is_file():
        with chart_columns[index % 2]:
            st.image(str(chart_path), caption=chart_title, use_container_width=True)

st.subheader("Holdout predictions")
predictions = pd.read_csv(predictions_path)
st.dataframe(predictions.head(50), use_container_width=True, hide_index=True)
st.download_button(
    "Download holdout predictions",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)