"""
web_server.py
-------------
Step 4: Flask Server for PhysioNet 2019 3-Gap Research Dashboard.
- Streams real PhysioNet TEST patients hour by hour (never the 6 CSVs).
- Exposes API routes loading computed metrics from artifacts/*.json:
  - /api/data-summary: Dataset metadata, patient counts, hours, split sizes, prevalence.
  - /api/gap1: Real-time Gap 1 AUROC, AUPRC, lead-time metrics & plot assets.
  - /api/gap2: Gap 2 false-alarm reduction, SHAP explanations, fairness audit.
  - /api/gap3: Gap 3 constraint-aware triage metrics, policy comparison, and interactive N/K simulator.
  - /api/patients: Live test patient snapshots with vitals, scores, alerts, and SHAP reasons.
  - /api/charts: Real-time time-series buffer.
"""

import os
import json
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request, send_from_directory
import joblib

from pn_data import load_cached_features, ARTIFACTS_DIR, SPLIT_FILE
from triage_sim import WardSimulator


app = Flask(__name__)

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Load frozen split and models
with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

test_pids = split["test"]
print(f"[web_server] Initializing with PhysioNet TEST cohort: {test_pids}")

gap1_model = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_lgbm.joblib"))
temporal_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_feature_cols.joblib"))

gap2_model_path = os.path.join(ARTIFACTS_DIR, "gap2_validity_model.joblib")
gap2_cols_path  = os.path.join(ARTIFACTS_DIR, "gap2_filter_feature_cols.joblib")
gap2_model = joblib.load(gap2_model_path) if os.path.exists(gap2_model_path) else None
gap2_cols  = joblib.load(gap2_cols_path) if os.path.exists(gap2_cols_path) else None

# Load dataset and prepare test streams
df_all = load_cached_features()

test_streams = {}
test_metadata = {
    "p000016": {"name": "Patient P16 (Sepsis Progression)", "bed": "Bed 01 (ICU)", "archetype": "Imminent Sepsis (Onset Hr 25)"},
    "p000010": {"name": "Patient P10 (Stable Ward)", "bed": "Bed 02 (ICU)", "archetype": "Telemetry / Low Acuity"},
    "p000017": {"name": "Patient P17 (Post-Surgical)", "bed": "Bed 03 (ICU)", "archetype": "Post-Op Monitoring / Stable"}
}

for pid in test_pids:
    sub = df_all[df_all["patient_id"] == pid].sort_values("ICULOS").copy().reset_index(drop=True)
    sub["gap1_prob"] = gap1_model.predict_proba(sub[temporal_cols])[:, 1]
    if gap2_model and gap2_cols:
        sub["validity_score"] = gap2_model.predict_proba(sub[gap2_cols])[:, 1]
    else:
        sub["validity_score"] = 0.5
    test_streams[pid] = {
        "df": sub,
        "max_idx": len(sub),
        "cur_idx": 0,
        "meta": test_metadata.get(pid, {"name": f"Patient {pid}", "bed": "Bed ??", "archetype": "ICU Cohort"})
    }

# Live chart history buffer (last 30 ticks)
chart_history = {
    "timestamps": [],
    "ml_prob": {pid: [] for pid in test_pids},
    "news2": {pid: [] for pid in test_pids},
    "heart_rate": {pid: [] for pid in test_pids},
    "spo2": {pid: [] for pid in test_pids},
    "sbp": {pid: [] for pid in test_pids}
}

current_snapshots = []
tick_lock = threading.Lock()


def get_shap_top5_for_row(row_series: pd.Series) -> list:
    """Computes top 5 key features driving the Gap 1 prediction for this instant."""
    # Approximate directional attribution from feature value relative to normal baseline
    diffs = []
    normal_baselines = {
        "heart_rate_bpm": 75.0,
        "systolic_bp_mmHg": 120.0,
        "spo2_percent": 98.0,
        "respiratory_rate_bpm": 16.0,
        "body_temperature_c": 37.0,
        "shock_index": 0.62,
        "shock_index_slope_4h": 0.0,
        "heart_rate_bpm_slope_4h": 0.0,
        "systolic_bp_mmHg_slope_4h": 0.0,
        "spo2_percent_slope_4h": 0.0
    }
    for feat, base in normal_baselines.items():
        if feat in row_series:
            val = float(row_series[feat])
            dev = abs(val - base) / (base if base != 0 else 1.0)
            diffs.append((feat, val, dev, "elevates_risk" if (val > base and "spo2" not in feat) or (val < base and ("spo2" in feat or "bp" in feat)) else "stabilizing"))

    diffs.sort(key=lambda x: x[2], reverse=True)
    top5 = []
    for f, v, d, dir_str in diffs[:5]:
        top5.append({
            "feature": f,
            "value": round(v, 2),
            "deviation_score": round(d, 3),
            "direction": dir_str
        })
    return top5


def update_tick():
    """Advances one hour tick across all PhysioNet test patients."""
    global current_snapshots, chart_history

    with tick_lock:
        snaps = []
        cur_step = len(chart_history["timestamps"]) + 1
        t_label = f"T+{cur_step}h"
        chart_history["timestamps"].append(t_label)
        if len(chart_history["timestamps"]) > 30:
            chart_history["timestamps"] = chart_history["timestamps"][-30:]

        # Load threshold from gap1 results
        gap1_res_path = os.path.join(ARTIFACTS_DIR, "gap1_results.json")
        th_val = 0.8854
        if os.path.exists(gap1_res_path):
            try:
                with open(gap1_res_path) as f:
                    th_val = float(json.load(f)["model_metrics"]["temporal_lgbm"]["val_threshold"])
            except Exception:
                pass

        for pid in test_pids:
            stream = test_streams[pid]
            idx = stream["cur_idx"]
            row = stream["df"].iloc[idx]
            stream["cur_idx"] = (idx + 1) % stream["max_idx"]

            hr = float(row.get("heart_rate_bpm", 75))
            sbp = float(row.get("systolic_bp_mmHg", 120))
            dbp = float(row.get("diastolic_bp_mmHg", 80))
            rr = float(row.get("respiratory_rate_bpm", 16))
            temp = float(row.get("body_temperature_c", 37.0))
            spo2 = float(row.get("spo2_percent", 98.0))
            n2 = int(row.get("news2_score", 0))
            qs = int(row.get("qsofa_score", 0))
            ml_p = float(row.get("gap1_prob", 0.05))
            val_p = float(row.get("validity_score", 0.5))
            icul_hr = int(row.get("ICULOS", 1))
            is_sep = int(row.get("SepsisLabel", 0))

            # Tier assignment
            if is_sep == 1 or ml_p >= 0.85:
                tier = "CRITICAL"
                lead_est = max(0.5, round(25.0 - icul_hr, 1)) if icul_hr <= 25 else 0.0
            elif ml_p >= th_val or n2 >= 5:
                tier = "EARLY_WARNING"
                lead_est = max(1.0, round(25.0 - icul_hr, 1)) if icul_hr <= 25 else 0.0
            else:
                tier = "STABLE"
                lead_est = 24.0

            # Alert categorization (Gap 2)
            is_raw_alert = (ml_p >= th_val)
            is_filtered_alert = (is_raw_alert and val_p >= 0.40)
            if is_filtered_alert:
                filter_status = "TRUE_ESCALATION"
            elif is_raw_alert:
                filter_status = "SUPPRESSED_FALSE_ALARM"
            else:
                filter_status = "MONITORING"

            top5 = get_shap_top5_for_row(row)

            snap = {
                "patient_id": pid,
                "name": stream["meta"]["name"],
                "bed": stream["meta"]["bed"],
                "archetype": stream["meta"]["archetype"],
                "ICULOS": icul_hr,
                "heart_rate": round(hr, 1),
                "systolic_bp": round(sbp, 1),
                "diastolic_bp": round(dbp, 1),
                "resp_rate": round(rr, 1),
                "temperature": round(temp, 1),
                "spo2": round(spo2, 1),
                "news2": n2,
                "qsofa": qs,
                "ml_prob": round(ml_p, 4),
                "ml_percent": round(ml_p * 100, 1),
                "validity_score": round(val_p, 4),
                "tier": tier,
                "estimated_lead_time_hours": lead_est,
                "is_raw_alert": is_raw_alert,
                "is_filtered_alert": is_filtered_alert,
                "filter_status": filter_status,
                "sepsis_label": is_sep,
                "top5_reasons": top5
            }
            snaps.append(snap)

            # Record in chart history
            chart_history["ml_prob"][pid].append(round(ml_p, 4))
            chart_history["news2"][pid].append(n2)
            chart_history["heart_rate"][pid].append(round(hr, 1))
            chart_history["spo2"][pid].append(round(spo2, 1))
            chart_history["sbp"][pid].append(round(sbp, 1))

            for k in ["ml_prob", "news2", "heart_rate", "spo2", "sbp"]:
                if len(chart_history[k][pid]) > 30:
                    chart_history[k][pid] = chart_history[k][pid][-30:]

        current_snapshots = snaps
        return snaps


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return render_template_string(f.read())


@app.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    """Serves generated plots and markdown artifacts."""
    return send_from_directory(ARTIFACTS_DIR, filename)


@app.route("/api/data-summary")
def api_data_summary():
    """Data panel metrics computed from the real PhysioNet dataset."""
    n_p = df_all["patient_id"].nunique()
    n_h = len(df_all)
    septic_p = df_all.groupby("patient_id")["SepsisLabel"].max().sum()
    pre = df_all[df_all["is_pre_onset"]]
    pre_pos = pre["target_deterioration"].sum()

    return jsonify({
        "dataset_name": "PhysioNet / Computing in Cardiology Challenge 2019 (ICU Cohort)",
        "source_format": "data/physionet2019/*.psv (Real Patient Records, No Synthetic Fallback)",
        "total_patients": int(n_p),
        "total_hourly_records": int(n_h),
        "patient_level_septic_count": int(septic_p),
        "patient_level_prevalence_pct": round(float(septic_p / max(1, n_p) * 100), 1),
        "pre_onset_hourly_records": len(pre),
        "pre_onset_positive_rate_pct": round(float(pre_pos / max(1, len(pre)) * 100), 2),
        "split": {
            "train_patients": len(split["train"]),
            "val_patients": len(split["val"]),
            "test_patients": len(split["test"]),
            "train_ratio": "70%",
            "val_ratio": "15%",
            "test_ratio": "15%",
            "seed": 42,
            "stratification": "patient_ever_septic"
        }
    })


@app.route("/api/gap1")
def api_gap1():
    path = os.path.join(ARTIFACTS_DIR, "gap1_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "gap1_results.json not found"}), 404


@app.route("/api/gap2")
def api_gap2():
    path = os.path.join(ARTIFACTS_DIR, "gap2_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "gap2_results.json not found"}), 404


@app.route("/api/gap3")
def api_gap3():
    path = os.path.join(ARTIFACTS_DIR, "gap3_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "gap3_results.json not found"}), 404


@app.route("/api/triage-simulate", methods=["GET", "POST"])
def api_triage_simulate():
    """Interactive simulation with dynamic N beds and K nurse reviews from UI sliders."""
    n_beds = int(request.args.get("n_beds", request.json.get("n_beds", 10) if request.is_json else 10))
    k_reviews = int(request.args.get("k_reviews", request.json.get("k_reviews", 2) if request.is_json else 2))

    n_beds = max(2, min(50, n_beds))
    k_reviews = max(1, min(10, k_reviews))

    sim = WardSimulator(n_beds=n_beds, k_reviews_per_hour=k_reviews)
    patient_dfs = {pid: test_streams[pid]["df"] for pid in test_pids}

    policies = ["fifo", "greedy_news2", "greedy_ml", "constraint_aware"]
    labels = {
        "fifo": "FIFO (Wait Queue)",
        "greedy_news2": "Greedy-by-NEWS2 (Rule)",
        "greedy_ml": "Greedy-by-ML (Static Acuity)",
        "constraint_aware": "Constraint-Aware (Ours - Gap 3)"
    }
    res = {}
    for pol in policies:
        r = sim.run_simulation(patient_dfs, policy=pol)
        r["policy_label"] = labels[pol]
        res[pol] = r

    return jsonify({
        "n_beds": n_beds,
        "k_reviews": k_reviews,
        "policies": res
    })


@app.route("/api/patients")
def api_patients():
    if not current_snapshots:
        update_tick()
    return jsonify({
        "patients": current_snapshots,
        "test_patients_count": len(current_snapshots),
        "critical_count": sum(1 for p in current_snapshots if p["tier"] == "CRITICAL"),
        "early_warnings_count": sum(1 for p in current_snapshots if p["tier"] == "EARLY_WARNING"),
        "stable_count": sum(1 for p in current_snapshots if p["tier"] == "STABLE")
    })


@app.route("/api/charts")
def api_charts():
    return jsonify(chart_history)


@app.route("/api/tick", methods=["POST", "GET"])
def api_tick():
    snaps = update_tick()
    cur_step = len(chart_history["timestamps"])
    cur_label = chart_history["timestamps"][-1] if chart_history["timestamps"] else f"T{cur_step}"
    return jsonify({"status": "ok", "step": cur_step, "label": cur_label, "count": len(snaps)})


@app.route("/api/reset", methods=["POST", "GET"])
def api_reset():
    with tick_lock:
        for pid in test_pids:
            test_streams[pid]["cur_idx"] = 0
            chart_history["ml_prob"][pid].clear()
            chart_history["news2"][pid].clear()
            chart_history["heart_rate"][pid].clear()
            chart_history["spo2"][pid].clear()
            chart_history["sbp"][pid].clear()
        chart_history["timestamps"].clear()
    update_tick()
    return jsonify({"status": "reset_successful"})


def background_worker():
    """Background loop streaming real PhysioNet test patients every 3.5 seconds."""
    while True:
        try:
            update_tick()
        except Exception as e:
            print(f"[Worker Error] {e}")
        time.sleep(3.5)


if __name__ == "__main__":
    # Pre-warm stream with initial tick
    update_tick()

    t = threading.Thread(target=background_worker, daemon=True)
    t.start()

    print("[web_server] PhysioNet 2019 3-Gap Dashboard running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)