"""
web_server.py
-------------
Flask Server for AI-Driven Real-Time Patient Monitoring System.
Implements Gap 1: PhysioNet 2019 40-Variable Temporal Forecast & Hysteresis Triage Dashboard.
"""

import os
import glob
import time
import random
import numpy as np
import threading
from flask import Flask, render_template_string, jsonify, request

from data_loader import CSVPatient
from physionet_dataset_loader import PhysioNet2019Dataset
from temporal_feature_engineering import normalize_vital_keys
from temporal_risk_prediction import TemporalRiskPredictor
from temporal_patient_stream import TemporalMonitoringManager


app = Flask(__name__)

# Initialize Predictor and Monitoring Manager
predictor = TemporalRiskPredictor(model_type="lightgbm")
predictor.load_or_train()
manager = TemporalMonitoringManager(predictor=predictor)

# Active patient streams
csv_patients_stream = []
physionet_patient_streams = []

chart_history = {
    "timestamps": [],
    "heart_rate": {},
    "resp_rate": {},
    "spo2": {},
    "temperature": {},
    "sbp": {},
    "news2": {},
    "qsofa": {},
    "ml_prob": {},
    "lead_time": {}
}

current_patients_snapshots = []


def initialize_streams():
    """Initializes active patient streams from the 6 clinical patient CSV files."""
    global csv_patients_stream
    csv_patients_stream.clear()

    csv_metadata = {
        "patient1.csv": ("Patient P01 (Sepsis Progression)", "ICU-01", "Sepsis Trajectory (Shock Risk)", "Hospital A (Academic)"),
        "patient2.csv": ("Patient P02 (Acute Hypoxia/ARDS)", "ICU-02", "Acute Respiratory Failure", "Hospital B (Community)"),
        "patient3.csv": ("Patient P03 (Hemorrhagic Shock)", "ICU-03", "Post-Op Hemorrhage", "Hospital C (Tertiary ICU)"),
        "patient4.csv": ("Patient P04 (Hypertensive Emergency)", "ICU-04", "Neuro / Autonomic Risk", "Hospital A (Academic)"),
        "patient5.csv": ("Patient P05 (Responding to Rx)", "ICU-05", "ICU De-escalation & Recovery", "Hospital B (Community)"),
        "patient6.csv": ("Patient P06 (Stable Telemetry)", "ICU-06", "General Ward Monitoring", "Hospital A (Academic)"),
    }

    csv_files = sorted(glob.glob("patient*.csv"))
    for idx, fp in enumerate(csv_files[:6], 1):
        pid = f"P{idx:02d}"
        base_name = os.path.basename(fp)
        meta = csv_metadata.get(base_name, (f"Patient {pid} ({base_name})", f"ICU-{idx:02d}", "ICU Time-Series", "Hospital A (ICU)"))
        p_obj = CSVPatient(fp, pid)
        csv_patients_stream.append(p_obj)
        manager.register_patient(
            patient_id=pid,
            name=meta[0],
            bed=meta[1],
            archetype=meta[2],
            hospital=meta[3]
        )


def update_tick():
    """Performs one monitoring tick across all 6 active patient CSV streams."""
    global current_patients_snapshots, chart_history

    snapshots = []

    # Update all active CSV patients
    for cp in csv_patients_stream:
        vital = cp.get_next_vital()
        if vital is None:
            cp.index = 0
            vital = cp.get_next_vital()
        if vital:
            snap = manager.feed_vital_and_predict(cp.patient_id, vital)
            snapshots.append(snap)

    # Update Chart History
    cur_step = len(chart_history["timestamps"]) + 1
    chart_history["timestamps"].append(f"T{cur_step}")
    if len(chart_history["timestamps"]) > 30:
        chart_history["timestamps"] = chart_history["timestamps"][-30:]

    for p in snapshots:
        pid = str(p["id"])
        for metric in ["heart_rate", "resp_rate", "spo2", "temperature", "sbp", "news2", "qsofa", "ml_prob", "lead_time"]:
            if pid not in chart_history[metric]:
                chart_history[metric][pid] = []
            val = p.get(metric, 0)
            if metric == "lead_time":
                val = p.get("estimated_lead_time_hours", 6.0)
            chart_history[metric][pid].append(val)
            if len(chart_history[metric][pid]) > 30:
                chart_history[metric][pid] = chart_history[metric][pid][-30:]

    current_patients_snapshots = snapshots
    return snapshots


@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return render_template_string(html_content)


@app.route("/api/patients")
def get_patients():
    if not current_patients_snapshots:
        update_tick()

    force_reorder = (request.args.get("reorder") == "true")
    grouped = manager.get_tier_grouped_patients(current_patients_snapshots, force_reorder=force_reorder)

    return jsonify({
        "all_patients": grouped["all_patients"],
        "critical_tier": grouped["critical_tier"],
        "early_warning_tier": grouped["early_warning_tier"],
        "stable_tier": grouped["stable_tier"],
        "reorder_countdown_sec": grouped["reorder_countdown_sec"],
        "count": len(current_patients_snapshots),
        "critical_count": len(grouped["critical_tier"]),
        "early_warnings_count": len(grouped["early_warning_tier"]),
        "stable_count": len(grouped["stable_tier"])
    })


@app.route("/api/charts")
def get_charts():
    return jsonify(chart_history)


@app.route("/api/benchmark")
def get_benchmark():
    return jsonify(predictor.benchmark_results)


@app.route("/api/reorder", methods=["POST"])
def force_reorder_triage():
    """Explicitly triggers a triage re-sort."""
    grouped = manager.get_tier_grouped_patients(current_patients_snapshots, force_reorder=True)
    return jsonify({"status": "triage_reordered", "critical_count": len(grouped["critical_tier"])})


@app.route("/api/tick", methods=["POST"])
def manual_tick():
    snaps = update_tick()
    return jsonify({"status": "ok", "updated_count": len(snaps)})


@app.route("/api/reset", methods=["POST"])
def reset_simulation():
    for cp in csv_patients_stream:
        cp.index = 0
    initialize_streams()
    update_tick()
    return jsonify({"status": "reset_successful"})


def background_monitoring_thread():
    """Periodically streams patient vitals every 4 seconds."""
    while True:
        try:
            update_tick()
        except Exception as e:
            print(f"[Monitoring Error] {e}")
        time.sleep(4)


if __name__ == "__main__":
    initialize_streams()
    for _ in range(6):
        update_tick()

    t = threading.Thread(target=background_monitoring_thread, daemon=True)
    t.start()

    print("[Web Server] Starting Gap 1 Patient Monitoring System on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)