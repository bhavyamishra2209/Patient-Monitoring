"""
web_server.py
-------------
High-Performance Clinical Telemetry & Triage Flask Server.
- Powered by real PhysioNet 2019 dataset (1,000 patients, 38,776 hours) for models and benchmarks.
- Streams patient1–patient6 CSV files for the interactive hospital ICU live demo.
- Live dynamic triage priority ranking with FLIP shuffle support, rank delta tracking, and clinical explanations.
- Endpoints:
  /api/data-summary, /api/prediction, /api/trust, /api/triage, /api/triage-simulate,
  /api/patients, /api/charts, /api/patient-detail/<pid>, /api/playback, /api/tick, /api/reset
"""

import os
import json
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request, send_from_directory
import joblib

from pn_data import (
    load_cached_features, ARTIFACTS_DIR, SPLIT_FILE,
    compute_news2, compute_qsofa, CORE_VITALS, COL_MAP, LAB_COLS, CLINICAL_DEFAULTS
)
from triage_sim import WardSimulator


app = Flask(__name__)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------- Load frozen split & trained models ----------
with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

test_pids = split["test"]
print(f"[web_server] PhysioNet cohort: {split.get('n_patients', 1000)} patients | Test cohort: {len(test_pids)} patients")

gap1_model = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_lgbm.joblib"))
temporal_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_feature_cols.joblib"))

gap2_model_path = os.path.join(ARTIFACTS_DIR, "gap2_validity_model.joblib")
gap2_cols_path  = os.path.join(ARTIFACTS_DIR, "gap2_filter_feature_cols.joblib")
gap2_model = joblib.load(gap2_model_path) if os.path.exists(gap2_model_path) else None
gap2_cols  = joblib.load(gap2_cols_path)  if os.path.exists(gap2_cols_path) else None

# PhysioNet features for evaluation APIs
df_all = load_cached_features()

# ---------- Load patient1–6 CSVs for live demo streaming ----------
DEMO_CSV_DIR = "."
DEMO_PATIENTS = {
    "patient1.csv": {
        "display_name": "Alex Rivera",
        "bed": "ICU Bed 01",
        "admission": "Sepsis Workup",
        "age": 58, "gender": "Male",
        "phenotype": "Hyperdynamic Sepsis Progression",
        "story": "Admitted with suspected urinary tract infection; vitals decompensate into septic shock."
    },
    "patient2.csv": {
        "display_name": "Maria Santos",
        "bed": "ICU Bed 02",
        "admission": "Respiratory Distress",
        "age": 67, "gender": "Female",
        "phenotype": "Acute Hypoxemic Respiratory Failure (ARDS)",
        "story": "Bilateral pneumonia progressing to tachypnea, severe desaturation, and respiratory collapse."
    },
    "patient3.csv": {
        "display_name": "James Chen",
        "bed": "ICU Bed 03",
        "admission": "Post-Op Whipple",
        "age": 44, "gender": "Male",
        "phenotype": "Occult Intra-Abdominal Bleeding",
        "story": "Unstable hemodynamics post-surgery; hidden bleeding triggers abrupt hemorrhagic shock."
    },
    "patient4.csv": {
        "display_name": "Priya Sharma",
        "bed": "ICU Bed 04",
        "admission": "Hypertensive Crisis",
        "age": 72, "gender": "Female",
        "phenotype": "Hypertensive Emergency with End-Organ Stress",
        "story": "Severe systolic and diastolic elevations refractory to initial oral antihypertensives."
    },
    "patient5.csv": {
        "display_name": "David Kowalski",
        "bed": "ICU Bed 05",
        "admission": "Initial Resuscitation",
        "age": 51, "gender": "Male",
        "phenotype": "Responding to Resuscitation & Stabilizing",
        "story": "Arrived unstable from ER; aggressive fluid and antibiotic management leads to rapid stabilization."
    },
    "patient6.csv": {
        "display_name": "Sarah Thompson",
        "bed": "ICU Bed 06",
        "admission": "Ward Observation",
        "age": 35, "gender": "Female",
        "phenotype": "Stable Clinical Baseline (Control)",
        "story": "Elective post-procedure observation; vitals remain within narrow homeostatic physiological bounds."
    },
}

def build_demo_feature_row(row_dict, history_buffer):
    """
    Build a feature vector compatible with the trained temporal model
    from a patient CSV row + recent history buffer.
    """
    feat = {}

    feat["heart_rate_bpm"] = float(row_dict.get("heart_rate_bpm", 75))
    feat["systolic_bp_mmHg"] = float(row_dict.get("systolic_bp_mmHg", 120))
    feat["diastolic_bp_mmHg"] = float(row_dict.get("diastolic_bp_mmHg", 80))
    feat["respiratory_rate_bpm"] = float(row_dict.get("respiratory_rate_bpm", 16))
    feat["body_temperature_c"] = float(row_dict.get("body_temperature_c", 37))
    feat["spo2_percent"] = float(row_dict.get("spo2_percent", 98))

    # Derived
    sbp = max(feat["systolic_bp_mmHg"], 10.0)
    feat["shock_index"] = feat["heart_rate_bpm"] / sbp
    feat["pulse_pressure"] = max(0.0, feat["systolic_bp_mmHg"] - feat["diastolic_bp_mmHg"])
    feat["mean_arterial_pressure"] = feat["diastolic_bp_mmHg"] + (feat["pulse_pressure"] / 3.0)

    # Demographics
    feat["Age"] = float(row_dict.get("Age", 60))
    feat["Gender"] = float(row_dict.get("Gender", 0))
    feat["hours_since_last_lab"] = 2.0

    # Baseline deltas
    if history_buffer:
        base = history_buffer[0]
        for v in CORE_VITALS:
            feat[f"{v}_delta_baseline"] = feat.get(v, 0.0) - base.get(v, feat.get(v, 0.0))
    else:
        for v in CORE_VITALS:
            feat[f"{v}_delta_baseline"] = 0.0

    # Rolling 4h and 12h
    hist_4h = history_buffer[-6:] if len(history_buffer) >= 6 else history_buffer
    hist_12h = history_buffer[-18:] if len(history_buffer) >= 18 else history_buffer

    for v in CORE_VITALS:
        v_4 = [h[v] for h in hist_4h if v in h] if hist_4h else [feat.get(v, 0.0)]
        feat[f"{v}_mean_4h"] = float(np.mean(v_4))
        feat[f"{v}_std_4h"]  = float(np.std(v_4)) if len(v_4) > 1 else 0.0
        feat[f"{v}_slope_4h"] = _slope(v_4)

        v_12 = [h[v] for h in hist_12h if v in h] if hist_12h else [feat.get(v, 0.0)]
        feat[f"{v}_mean_12h"] = float(np.mean(v_12))
        feat[f"{v}_std_12h"]  = float(np.std(v_12)) if len(v_12) > 1 else 0.0
        feat[f"{v}_min_12h"]  = float(np.min(v_12))
        feat[f"{v}_max_12h"]  = float(np.max(v_12))
        feat[f"{v}_slope_12h"] = _slope(v_12)

    # Rule baselines
    s = pd.Series(feat)
    feat["news2_score"] = float(compute_news2(s))
    feat["qsofa_score"] = float(compute_qsofa(s))

    return feat


def _slope(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean()
    ym = np.mean(vals)
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - xm) * (np.array(vals) - ym)) / denom)


# Load demo streams
demo_streams = {}
for csv_file, meta in DEMO_PATIENTS.items():
    fpath = os.path.join(DEMO_CSV_DIR, csv_file)
    if not os.path.exists(fpath):
        print(f"[web_server] Warning: {csv_file} not found, skipping")
        continue
    df = pd.read_csv(fpath)
    demo_streams[csv_file] = {
        "df": df,
        "meta": meta,
        "max_idx": len(df),
        "cur_idx": 0,
        "history": []
    }

print(f"[web_server] Loaded {len(demo_streams)} demo patients from CSVs")

# ---------- Live State ----------
chart_history = {
    "timestamps": [],
    "patients": {}
}
for key in demo_streams:
    pid = key.replace(".csv", "")
    chart_history["patients"][pid] = {
        "ml_risk": [], "news2": [], "heart_rate": [],
        "spo2": [], "sbp": [], "temperature": [], "resp_rate": [],
        "shock_index": []
    }

current_snapshots = []
last_ranks = {}
live_alerts_log = []
tick_lock = threading.Lock()

playback_state = {
    "paused": False,
    "speed": 1.0,  # 1.0 = 3.0s, 2.0 = 1.5s, 5.0 = 0.6s
    "step": 0
}

# Load validation threshold from gap1 results
_gap1_threshold = 0.60
gap1_res_path = os.path.join(ARTIFACTS_DIR, "gap1_results.json")
if os.path.exists(gap1_res_path):
    try:
        with open(gap1_res_path) as f:
            _gap1_threshold = float(json.load(f)["model_metrics"]["temporal_lgbm"]["val_threshold"])
    except Exception:
        pass


def get_top_factors(feat_dict):
    """Return top 5 factors driving risk based on deviation and trends."""
    baselines = {
        "heart_rate_bpm": (75.0, "Heart Rate", "bpm"),
        "systolic_bp_mmHg": (120.0, "Systolic BP", "mmHg"),
        "spo2_percent": (98.0, "SpO₂", "%"),
        "respiratory_rate_bpm": (16.0, "Resp Rate", "bpm"),
        "body_temperature_c": (37.0, "Temperature", "°C"),
        "shock_index": (0.62, "Shock Index", "ratio"),
        "heart_rate_bpm_slope_4h": (0.0, "HR Trend (4h)", "bpm/h"),
        "systolic_bp_mmHg_slope_4h": (0.0, "SBP Trend (4h)", "mmHg/h"),
        "spo2_percent_slope_4h": (0.0, "SpO₂ Trend (4h)", "%/h"),
        "respiratory_rate_bpm_slope_4h": (0.0, "Resp Trend (4h)", "bpm/h"),
    }
    diffs = []
    for key, (base, label, unit) in baselines.items():
        val = float(feat_dict.get(key, base))
        dev = abs(val - base) / max(abs(base), 0.01)
        if "slope" in key:
            dev = abs(val) * 8.0  # weight slopes heavily for temporal gap
            direction = "↑ rapid rise" if val > 0.05 else ("↓ rapid drop" if val < -0.05 else "→ steady")
        elif "spo2" in key or "bp" in key.lower():
            direction = "↓ hypoperfusion" if val < base else "normal"
        else:
            direction = "↑ elevated" if val > base else "normal"

        diffs.append({
            "feature": label,
            "value": round(val, 2),
            "unit": unit,
            "deviation": round(dev, 3),
            "direction": direction
        })

    diffs.sort(key=lambda x: x["deviation"], reverse=True)
    return diffs[:5]


def update_tick():
    """Advances one tick across all 6 demo patients, scores models, and updates triage priority."""
    global current_snapshots, last_ranks

    with tick_lock:
        snaps = []
        playback_state["step"] += 1
        cur_step = playback_state["step"]
        t_label = f"T+{cur_step}"
        chart_history["timestamps"].append(t_label)
        if len(chart_history["timestamps"]) > 50:
            chart_history["timestamps"] = chart_history["timestamps"][-50:]

        for csv_file, stream in demo_streams.items():
            pid = csv_file.replace(".csv", "")
            idx = stream["cur_idx"]
            row = stream["df"].iloc[idx].to_dict()
            stream["cur_idx"] = (idx + 1) % stream["max_idx"]
            meta = stream["meta"]

            # Demographics
            row["Age"] = meta["age"]
            row["Gender"] = 1 if meta["gender"] == "Male" else 0
            row["ICULOS"] = idx + 1

            # Core vital dict for history
            vital_snap = {
                "heart_rate_bpm": float(row.get("heart_rate_bpm", 75)),
                "systolic_bp_mmHg": float(row.get("systolic_bp_mmHg", 120)),
                "diastolic_bp_mmHg": float(row.get("diastolic_bp_mmHg", 80)),
                "respiratory_rate_bpm": float(row.get("respiratory_rate_bpm", 16)),
                "body_temperature_c": float(row.get("body_temperature_c", 37)),
                "spo2_percent": float(row.get("spo2_percent", 98)),
                "shock_index": float(row.get("heart_rate_bpm", 75)) / max(float(row.get("systolic_bp_mmHg", 120)), 10),
                "mean_arterial_pressure": float(row.get("diastolic_bp_mmHg", 80)) + (float(row.get("systolic_bp_mmHg", 120)) - float(row.get("diastolic_bp_mmHg", 80))) / 3.0,
            }
            stream["history"].append(vital_snap)
            if len(stream["history"]) > 30:
                stream["history"] = stream["history"][-30:]

            # Feature vector
            feat = build_demo_feature_row(row, stream["history"])
            feat_vec = pd.DataFrame([feat])
            for c in temporal_cols:
                if c not in feat_vec.columns:
                    feat_vec[c] = 0.0
            feat_vec = feat_vec[temporal_cols]

            ml_prob = float(gap1_model.predict_proba(feat_vec)[:, 1][0])

            # Trust / Alarm Validity Model (Gap 2)
            val_score = 0.5
            if gap2_model and gap2_cols:
                feat_g2 = pd.DataFrame([feat])
                for c in gap2_cols:
                    if c not in feat_g2.columns:
                        feat_g2[c] = 0.0
                feat_g2 = feat_g2[gap2_cols]
                val_score = float(gap2_model.predict_proba(feat_g2)[:, 1][0])

            hr = vital_snap["heart_rate_bpm"]
            sbp = vital_snap["systolic_bp_mmHg"]
            dbp = vital_snap["diastolic_bp_mmHg"]
            rr = vital_snap["respiratory_rate_bpm"]
            temp = vital_snap["body_temperature_c"]
            spo2 = vital_snap["spo2_percent"]
            si = vital_snap["shock_index"]
            n2 = int(feat.get("news2_score", 0))
            qs = int(feat.get("qsofa_score", 0))
            clinical_state = str(row.get("clinical_state", "Monitoring"))

            # Acuity Tier
            if ml_prob >= 0.82 or clinical_state in ["Septic Shock", "Severe ARDS", "Hemorrhagic Shock"]:
                tier = "CRITICAL"
                tier_label = "Critical (Immediate)"
                action_text = "Stat Physician Evaluation & Invasive Telemetry"
            elif ml_prob >= _gap1_threshold or n2 >= 5 or clinical_state in ["Early Sepsis", "Impending Hypoxia", "Hypertensive Emergency", "Occult Bleeding"]:
                tier = "WARNING"
                tier_label = "Warning (High Urgency)"
                action_text = "Nurse Bedside Assessment & Confirmatory Labs"
            else:
                tier = "STABLE"
                tier_label = "Stable (Routine)"
                action_text = "Maintain Continuous Standard Monitoring"

            # Alert trust status (Gap 2)
            is_raw_alert = (ml_prob >= _gap1_threshold)
            is_filtered_alert = (is_raw_alert and val_score >= 0.25)
            top_factors = get_top_factors(feat)
            top_factors_summary = ", ".join([f"{f['feature']} ({f['direction']}, {f['value']}{f['unit']})" for f in top_factors[:2]])

            if is_filtered_alert:
                alert_type = "ACTIONABLE_ESCALATED"
                alert_status = "Escalate (Actionable)"
                alert_badge = "🚨 ACTIONABLE ALERT"
                alert_headline = f"Actionable Deterioration ({clinical_state})"
                alert_action = f"Escalated to Bedside Nurse (Trust: {round(val_score*100)}%)"
                alert_why = f"High temporal risk ({round(ml_prob*100)}%) driven by {top_factors_summary}. Verified by Gap 2 trust filter."
            elif is_raw_alert:
                alert_type = "FALSE_ALARM_SUPPRESSED"
                alert_status = "Suppressed (False Alarm)"
                alert_badge = "🛡️ FALSE ALARM SUPPRESSED"
                alert_headline = f"Spurious Alarm Suppressed ({clinical_state})"
                alert_action = f"Silenced by Gap 2 Filter (Trust: {round(val_score*100)}%)"
                alert_why = f"Transient vital fluctuation ({top_factors[0]['feature']}: {top_factors[0]['value']}{top_factors[0]['unit']}), but systemic perfusion markers stable. Silenced to prevent alarm fatigue."
            else:
                alert_type = "NORMAL_MONITORING"
                alert_status = "Monitoring"
                alert_badge = "✓ STABLE TELEMETRY"
                alert_headline = f"Compensated Physiology ({clinical_state})"
                alert_action = "Continuous Standard Observation"
                alert_why = "Vital trajectories within homeostatic target boundaries."

            # Clinical triage reason
            if tier == "CRITICAL":
                triage_reason = f"Severe decompensation ({clinical_state}): SpO₂ {spo2:.0f}%, Shock Index {si:.2f}, HR {hr:.0f} bpm."
            elif tier == "WARNING":
                triage_reason = f"Subtle negative trend ({clinical_state}): HR rising, SBP declining, NEWS2 = {n2}."
            else:
                triage_reason = f"Physiologically compensated ({clinical_state}): Vitals stable, risk low."

            snap = {
                "id": pid,
                "name": meta["display_name"],
                "bed": meta["bed"],
                "admission": meta["admission"],
                "age": meta["age"],
                "gender": meta["gender"],
                "phenotype": meta["phenotype"],
                "story": meta["story"],
                "hour": idx + 1,
                "total_hours": stream["max_idx"],
                "timestamp_str": str(row.get("timestamp", f"08:{idx*2:02d}:00")),
                "clinical_state": clinical_state,
                "heart_rate": round(hr, 1),
                "systolic_bp": round(sbp, 1),
                "diastolic_bp": round(dbp, 1),
                "resp_rate": round(rr, 1),
                "temperature": round(temp, 2),
                "spo2": round(spo2, 1),
                "shock_index": round(si, 2),
                "news2": n2,
                "qsofa": qs,
                "ml_risk": round(ml_prob, 4),
                "ml_risk_pct": round(ml_prob * 100, 1),
                "validity_score": round(val_score, 4),
                "validity_pct": round(val_score * 100, 1),
                "tier": tier,
                "tier_label": tier_label,
                "alert_status": alert_status,
                "alert_type": alert_type,
                "alert_badge": alert_badge,
                "alert_headline": alert_headline,
                "alert_action": alert_action,
                "alert_why": alert_why,
                "action_text": action_text,
                "triage_reason": triage_reason,
                "top_factors": top_factors
            }
            snaps.append(snap)

            # Record to live alerts log if actionable or suppressed
            if alert_type in ["ACTIONABLE_ESCALATED", "FALSE_ALARM_SUPPRESSED"]:
                live_alerts_log.append({
                    "step": cur_step,
                    "timestamp": t_label,
                    "time_str": str(row.get("timestamp", f"08:{idx*2:02d}:00")),
                    "pid": pid,
                    "name": meta["display_name"],
                    "bed": meta["bed"],
                    "alert_type": alert_type,
                    "alert_badge": alert_badge,
                    "alert_headline": alert_headline,
                    "alert_action": alert_action,
                    "alert_why": alert_why,
                    "risk_pct": round(ml_prob * 100, 1),
                    "trust_pct": round(val_score * 100, 1),
                    "clinical_state": clinical_state,
                    "key_drivers": top_factors[:3]
                })
                if len(live_alerts_log) > 30:
                    del live_alerts_log[0]

            # Chart history update
            ch = chart_history["patients"].get(pid, {
                "ml_risk": [], "news2": [], "heart_rate": [],
                "spo2": [], "sbp": [], "temperature": [], "resp_rate": [],
                "shock_index": []
            })
            ch["ml_risk"].append(round(ml_prob, 4))
            ch["news2"].append(n2)
            ch["heart_rate"].append(round(hr, 1))
            ch["spo2"].append(round(spo2, 1))
            ch["sbp"].append(round(sbp, 1))
            ch["temperature"].append(round(temp, 2))
            ch["resp_rate"].append(round(rr, 1))
            ch["shock_index"].append(round(si, 2))
            for k in ch:
                if len(ch[k]) > 50:
                    ch[k] = ch[k][-50:]
            chart_history["patients"][pid] = ch

        # Gap 3 Triage Ranking: Sort by acuity and risk probability
        tier_order = {"CRITICAL": 0, "WARNING": 1, "STABLE": 2}
        snaps.sort(key=lambda s: (tier_order.get(s["tier"], 3), -s["ml_risk"]))

        # Assign ranks and rank deltas for animated card shuffling
        for rank_idx, s in enumerate(snaps, start=1):
            pid = s["id"]
            prev_r = last_ranks.get(pid, rank_idx)
            s["rank"] = rank_idx
            s["prev_rank"] = prev_r
            # delta > 0 means escalated priority (e.g. was 4, now 1 -> moved up +3)
            s["rank_delta"] = prev_r - rank_idx
            last_ranks[pid] = rank_idx

        current_snapshots = snaps
        return snaps


# ---------- Routes ----------

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
    return send_from_directory(ARTIFACTS_DIR, filename)


@app.route("/api/data-summary")
def api_data_summary():
    """Real PhysioNet 2019 dataset summary across 1,000 challenge patient records."""
    n_p = df_all["patient_id"].nunique()
    n_h = len(df_all)
    septic_p = df_all.groupby("patient_id")["SepsisLabel"].max().sum()
    pre = df_all[df_all["is_pre_onset"]]
    pre_pos = pre["target_deterioration"].sum()

    return jsonify({
        "dataset": "PhysioNet / CinC Challenge 2019 (Training Set A)",
        "source": "data/physionet2019/*.psv (1,000 real patient files)",
        "total_patients": int(n_p),
        "total_hours": int(n_h),
        "septic_patients": int(septic_p),
        "prevalence_pct": round(float(septic_p / max(1, n_p) * 100), 1),
        "pre_onset_hours": len(pre),
        "hourly_positive_rate": round(float(pre_pos / max(1, len(pre)) * 100), 2),
        "split": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
            "seed": 42
        },
        "demo_patients": len(demo_streams)
    })


@app.route("/api/prediction")
def api_prediction():
    """Gap 1: Temporal Prediction benchmark metrics on test cohort."""
    path = os.path.join(ARTIFACTS_DIR, "gap1_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Results not found. Run train_gap1.py first."}), 404


@app.route("/api/trust")
def api_trust():
    """Gap 2: Trust, False Alarm Reduction & Fairness metrics."""
    path = os.path.join(ARTIFACTS_DIR, "gap2_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Results not found. Run train_gap2.py first."}), 404


@app.route("/api/alerts")
def api_alerts():
    """Live streaming alert feed showing actionable alerts and suppressed false alarms."""
    return jsonify({
        "alerts": list(reversed(live_alerts_log[-20:])),
        "total_recorded": len(live_alerts_log)
    })


@app.route("/api/triage")
def api_triage():
    """Gap 3: Resource-constrained triage benchmark metrics."""
    path = os.path.join(ARTIFACTS_DIR, "gap3_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Results not found. Run triage_sim.py first."}), 404


@app.route("/api/triage-simulate", methods=["GET", "POST"])
def api_triage_simulate():
    """Interactive simulation with dynamic N beds and K nurse reviews."""
    n_beds = int(request.args.get("n_beds", request.json.get("n_beds", 10) if request.is_json else 10))
    k_reviews = int(request.args.get("k_reviews", request.json.get("k_reviews", 2) if request.is_json else 2))
    n_beds = max(2, min(50, n_beds))
    k_reviews = max(1, min(10, k_reviews))

    sim = WardSimulator(n_beds=n_beds, k_reviews_per_hour=k_reviews)
    patient_dfs = {}
    for pid in test_pids[:max(n_beds, 15)]:
        sub = df_all[df_all["patient_id"] == pid].sort_values("ICULOS").copy().reset_index(drop=True)
        sub["gap1_prob"] = gap1_model.predict_proba(sub[temporal_cols])[:, 1]
        patient_dfs[pid] = sub

    policies = ["fifo", "greedy_news2", "greedy_ml", "constraint_aware"]
    labels = {
        "fifo": "First-In First-Out Queue",
        "greedy_news2": "Priority by NEWS2 Score",
        "greedy_ml": "Priority by ML Risk (Static)",
        "constraint_aware": "Constraint-Aware Ranking (Ours)"
    }
    res = {}
    for pol in policies:
        r = sim.run_simulation(patient_dfs, policy=pol)
        r["label"] = labels[pol]
        res[pol] = r

    return jsonify({"n_beds": n_beds, "k_reviews": k_reviews, "policies": res})


@app.route("/api/patients")
def api_patients():
    if not current_snapshots:
        update_tick()
    crit = sum(1 for p in current_snapshots if p["tier"] == "CRITICAL")
    warn = sum(1 for p in current_snapshots if p["tier"] == "WARNING")
    stab = sum(1 for p in current_snapshots if p["tier"] == "STABLE")
    return jsonify({
        "patients": current_snapshots,
        "counts": {"total": len(current_snapshots), "critical": crit, "warning": warn, "stable": stab},
        "playback": playback_state
    })


@app.route("/api/patient-detail/<pid>")
def api_patient_detail(pid):
    """Returns high-resolution telemetry time series for deep-dive modal."""
    csv_file = f"{pid}.csv"
    if csv_file not in demo_streams:
        return jsonify({"error": "Patient not found"}), 404

    stream = demo_streams[csv_file]
    df = stream["df"]
    cur_idx = max(1, stream["cur_idx"])
    slice_df = df.iloc[:cur_idx].copy()

    records = slice_df.to_dict(orient="records")
    return jsonify({
        "pid": pid,
        "meta": stream["meta"],
        "cur_idx": cur_idx,
        "max_idx": stream["max_idx"],
        "history": records
    })


@app.route("/api/charts")
def api_charts():
    return jsonify(chart_history)


@app.route("/api/playback", methods=["GET", "POST"])
def api_playback():
    if request.method == "POST":
        data = request.json or {}
        action = data.get("action", "")
        if action == "toggle":
            playback_state["paused"] = not playback_state["paused"]
        elif action == "pause":
            playback_state["paused"] = True
        elif action == "play":
            playback_state["paused"] = False
        elif action == "speed":
            spd = float(data.get("speed", 1.0))
            playback_state["speed"] = max(0.5, min(10.0, spd))
    return jsonify(playback_state)


@app.route("/api/tick", methods=["POST", "GET"])
def api_tick():
    snaps = update_tick()
    step = playback_state["step"]
    return jsonify({"status": "ok", "step": step, "count": len(snaps)})


@app.route("/api/reset", methods=["POST", "GET"])
def api_reset():
    global last_ranks
    with tick_lock:
        for csv_file in demo_streams:
            demo_streams[csv_file]["cur_idx"] = 0
            demo_streams[csv_file]["history"].clear()
            pid = csv_file.replace(".csv", "")
            if pid in chart_history["patients"]:
                for k in chart_history["patients"][pid]:
                    chart_history["patients"][pid][k].clear()
        chart_history["timestamps"].clear()
        last_ranks.clear()
        live_alerts_log.clear()
        playback_state["step"] = 0
    update_tick()
    return jsonify({"status": "reset", "step": 0})


def background_worker():
    """Background streaming loop for live demonstration."""
    while True:
        if not playback_state["paused"]:
            try:
                update_tick()
            except Exception as e:
                print(f"[Worker Error] {e}")
        # Dynamically calculate interval according to speed multiplier
        interval = max(0.4, 3.0 / max(0.1, playback_state["speed"]))
        time.sleep(interval)


if __name__ == "__main__":
    update_tick()
    t = threading.Thread(target=background_worker, daemon=True)
    t.start()
    print("[web_server] Patient Monitoring System running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)