"""
train_gap2.py
-------------
Step 2: Gap 2 - Trust, False-Alarm Reduction, SHAP Explainability & Fairness Audit.
- Alert = Gap 1 temporal model probability >= validation threshold.
- False alarm = alert with no SepsisLabel==1 in the next 6h.
- Trains an alarm-validity classifier (LightGBM) on candidate alerts on the frozen split.
- Uses SHAP (TreeExplainer) to compute global feature importance and top-5 local explanations per alert.
- Reports alerts per patient-day, PPV, and sensitivity retained BEFORE vs. AFTER filtering.
- Performs demographic fairness audit: AUROC and false-alarm rate by Gender and Age band (<50, 50-70, >70).
- Saves artifacts/gap2_results.json, artifacts/gap2_shap_summary.png, and artifacts/gap2_fairness.png.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
from pn_data import load_cached_features, ARTIFACTS_DIR, SPLIT_FILE


def compute_alarm_labels(df_pre: pd.DataFrame, df_full: pd.DataFrame) -> np.ndarray:
    """
    Labels each alert as valid (1) if SepsisLabel==1 within next 6 hours, or false alarm (0) otherwise.
    """
    valid_labels = []
    for pid, group in df_pre.groupby("patient_id", sort=False):
        full_sub = df_full[df_full["patient_id"] == pid]
        full_labels = full_sub["SepsisLabel"].values
        full_culos  = full_sub["ICULOS"].values

        for _, row in group.iterrows():
            c_hr = row["ICULOS"]
            future_mask = (full_culos > c_hr) & (full_culos <= c_hr + 6)
            has_sepsis_next_6h = np.any(full_labels[future_mask] == 1) if np.any(future_mask) else False
            valid_labels.append(1 if has_sepsis_next_6h else 0)

    return np.array(valid_labels)


def train_and_evaluate_gap2() -> Dict:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df_full = load_cached_features()

    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    gap1_results_path = os.path.join(ARTIFACTS_DIR, "gap1_results.json")
    with open(gap1_results_path, "r") as f:
        gap1_res = json.load(f)

    th_gap1 = float(gap1_res["model_metrics"]["temporal_lgbm"]["val_threshold"])
    print(f"[train_gap2] Loaded Gap 1 validation threshold: {th_gap1:.4f}")

    gap1_model = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_lgbm.joblib"))
    feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_feature_cols.joblib"))

    # Pre-onset evaluation dataset
    df_pre = df_full[df_full["is_pre_onset"]].copy()
    df_pre["gap1_prob"] = gap1_model.predict_proba(df_pre[feature_cols])[:, 1]
    df_pre["valid_alert"] = compute_alarm_labels(df_pre, df_full)
    df_pre["is_alert"] = (df_pre["gap1_prob"] >= th_gap1)

    df_train = df_pre[df_pre["split"] == "train"].copy()
    df_val   = df_pre[df_pre["split"] == "val"].copy()
    df_test  = df_pre[df_pre["split"] == "test"].copy()

    # Candidate alerts for training: top elevated-risk hours in train (mix of pre-sepsis hours and transient non-septic spikes)
    df_train_sorted = df_train.sort_values("gap1_prob", ascending=False)
    train_alert_candidates = df_train_sorted.iloc[:80].copy()

    # Features that differentiate true imminent decompensation from premature or transient alerts
    filter_features = [
        "gap1_prob", "shock_index", "shock_index_slope_4h", "heart_rate_bpm_slope_4h",
        "systolic_bp_mmHg_slope_4h", "body_temperature_c_slope_4h", "news2_score", "ICULOS"
    ]
    # Filter to existing columns
    filter_features = [c for c in filter_features if c in df_pre.columns]

    print(f"[train_gap2] Training Alarm-Validity Filter on {len(train_alert_candidates)} candidate alerts "
          f"({train_alert_candidates['valid_alert'].sum()} valid, {len(train_alert_candidates) - train_alert_candidates['valid_alert'].sum()} false alarms)...")
    validity_model = lgb.LGBMClassifier(
        n_estimators=30,
        learning_rate=0.08,
        max_depth=3,
        num_leaves=7,
        min_child_samples=4,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    validity_model.fit(train_alert_candidates[filter_features], train_alert_candidates["valid_alert"])
    joblib.dump(validity_model, os.path.join(ARTIFACTS_DIR, "gap2_validity_model.joblib"))
    joblib.dump(filter_features, os.path.join(ARTIFACTS_DIR, "gap2_filter_feature_cols.joblib"))

    # Evaluate on Test alerts
    df_test["validity_score"] = validity_model.predict_proba(df_test[filter_features])[:, 1]

    # Calculate test metrics BEFORE vs AFTER filtering
    test_patient_hours = len(df_test)
    test_patient_days = max(0.1, test_patient_hours / 24.0)

    # 1. BEFORE filtering (Raw Gap 1 alerts)
    raw_alerts = df_test[df_test["is_alert"]].copy()
    n_raw_alerts = len(raw_alerts)
    n_raw_true_alerts = int(raw_alerts["valid_alert"].sum())
    n_raw_false_alarms = n_raw_alerts - n_raw_true_alerts

    raw_alerts_per_day = round(n_raw_alerts / test_patient_days, 2)
    raw_ppv = round(n_raw_true_alerts / max(1, n_raw_alerts), 4)
    raw_sensitivity = 1.0  # reference baseline

    # 2. AFTER filtering
    # Determine validity threshold to filter false alarms while retaining true alarms
    val_cand = df_val.sort_values("gap1_prob", ascending=False).iloc[:20].copy()
    if len(val_cand) > 0 and val_cand["valid_alert"].sum() > 0:
        val_cand_scores = validity_model.predict_proba(val_cand[filter_features])[:, 1]
        valid_val_scores = val_cand_scores[val_cand["valid_alert"] == 1]
        validity_thresh = float(np.min(valid_val_scores) * 0.95) if len(valid_val_scores) > 0 else 0.50
    else:
        validity_thresh = 0.50

    # Ensure validity_thresh is sensible
    validity_thresh = float(np.clip(validity_thresh, 0.30, 0.95))

    filtered_alerts = raw_alerts[raw_alerts["validity_score"] >= validity_thresh].copy()
    n_filt_alerts = len(filtered_alerts)
    n_filt_true_alerts = int(filtered_alerts["valid_alert"].sum())
    n_filt_false_alarms = n_filt_alerts - n_filt_true_alerts

    filt_alerts_per_day = round(n_filt_alerts / test_patient_days, 2)
    filt_ppv = round(n_filt_true_alerts / max(1, n_filt_alerts), 4) if n_filt_alerts > 0 else 0.0
    sensitivity_retained = round(n_filt_true_alerts / max(1, n_raw_true_alerts), 4)
    false_alarm_reduction_pct = round(
        (n_raw_false_alarms - n_filt_false_alarms) / max(1, n_raw_false_alarms) * 100.0, 1
    )

    print("\n[train_gap2] False-Alarm Filter Performance on Test Cohort:")
    print(f"  Raw Alerts: {n_raw_alerts} ({raw_alerts_per_day} alerts/patient-day), PPV: {raw_ppv:.1%}")
    print(f"  Filtered Alerts: {n_filt_alerts} ({filt_alerts_per_day} alerts/patient-day), PPV: {filt_ppv:.1%}")
    print(f"  False Alarms Eliminated: {false_alarm_reduction_pct}% ({n_raw_false_alarms} -> {n_filt_false_alarms})")
    print(f"  True Alert Sensitivity Retained: {sensitivity_retained:.1%}")

    # 3. SHAP TreeExplainer Analysis
    print("[train_gap2] Computing SHAP explanations with TreeExplainer...")
    explainer = shap.TreeExplainer(gap1_model)
    shap_test_values = explainer(df_test[feature_cols])

    # Per-alert top-5 explanations
    alert_explanations = []
    alert_indices = df_test.index[df_test["is_alert"]].tolist()

    for row_loc, orig_idx in enumerate(df_test.index):
        if orig_idx not in alert_indices:
            continue
        row = df_test.loc[orig_idx]
        sv_row = shap_test_values.values[row_loc]

        # Top 5 positive and negative contributing features
        top_order = np.argsort(np.abs(sv_row))[::-1][:5]
        top_features = []
        for feat_idx in top_order:
            feat_name = feature_cols[feat_idx]
            feat_val = float(row[feat_name])
            shap_impact = float(sv_row[feat_idx])
            direction = "escalates_risk" if shap_impact > 0 else "protective"
            top_features.append({
                "feature": feat_name,
                "value": round(feat_val, 2),
                "shap_impact": round(shap_impact, 4),
                "direction": direction
            })

        alert_explanations.append({
            "patient_id": row["patient_id"],
            "ICULOS": int(row["ICULOS"]),
            "gap1_risk_prob": round(float(row["gap1_prob"]), 4),
            "validity_score": round(float(row["validity_score"]), 4),
            "is_valid_alert": bool(row["valid_alert"] == 1),
            "filtered_action": "DISMISSED_FALSE_ALARM" if row["validity_score"] < validity_thresh else "ESCALATED_TRUE_ALERT",
            "top_5_shap_reasons": top_features
        })

    # Plot Global SHAP Summary
    plot_shap_summary(shap_test_values, feature_cols)

    # 4. Fairness Audit (by Gender and Age bands)
    fairness_results = run_fairness_audit(df_test)
    plot_fairness_audit(fairness_results)

    # Compile Final Gap 2 Results JSON
    results = {
        "alert_threshold_gap1": round(th_gap1, 4),
        "validity_threshold_gap2": round(validity_thresh, 4),
        "test_patient_hours": test_patient_hours,
        "test_patient_days": round(test_patient_days, 2),
        "before_filtering": {
            "total_alerts": n_raw_alerts,
            "true_alerts": n_raw_true_alerts,
            "false_alarms": n_raw_false_alarms,
            "alerts_per_patient_day": raw_alerts_per_day,
            "ppv": raw_ppv,
            "sensitivity": raw_sensitivity
        },
        "after_filtering": {
            "total_alerts": n_filt_alerts,
            "true_alerts": n_filt_true_alerts,
            "false_alarms": n_filt_false_alarms,
            "alerts_per_patient_day": filt_alerts_per_day,
            "ppv": filt_ppv,
            "sensitivity_retained": sensitivity_retained,
            "false_alarm_reduction_percent": false_alarm_reduction_pct
        },
        "fairness_audit": fairness_results,
        "sample_alert_explanations": alert_explanations
    }

    gap2_json_path = os.path.join(ARTIFACTS_DIR, "gap2_results.json")
    with open(gap2_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[train_gap2] Saved Gap 2 results to {gap2_json_path}")
    return results


def run_fairness_audit(df_test: pd.DataFrame) -> Dict:
    """Computes AUROC and False Alarm Rate (FAR) by Gender and Age bands."""
    fairness = {"gender": {}, "age_band": {}}

    # Gender audit (0: Female / 1: Male)
    for g in [0, 1]:
        g_name = "Female (0)" if g == 0 else "Male (1)"
        sub = df_test[df_test["Gender"] == g]
        if len(sub) == 0:
            continue
        y_true = sub["target_deterioration"].values
        y_score = sub["gap1_prob"].values
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan

        # FAR among negatives
        neg_sub = sub[sub["target_deterioration"] == 0]
        far = float(np.mean(neg_sub["is_alert"])) if len(neg_sub) > 0 else 0.0

        fairness["gender"][g_name] = {
            "n_hours": len(sub),
            "n_patients": sub["patient_id"].nunique(),
            "auroc": round(auc, 3) if not np.isnan(auc) else "N/A (single class)",
            "false_alarm_rate": round(far, 3)
        }

    # Age band audit (<50, 50-70, >70)
    age_bands = {
        "<50": df_test[df_test["Age"] < 50],
        "50-70": df_test[(df_test["Age"] >= 50) & (df_test["Age"] <= 70)],
        ">70": df_test[df_test["Age"] > 70]
    }

    for band_name, sub in age_bands.items():
        if len(sub) == 0:
            continue
        y_true = sub["target_deterioration"].values
        y_score = sub["gap1_prob"].values
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan

        neg_sub = sub[sub["target_deterioration"] == 0]
        far = float(np.mean(neg_sub["is_alert"])) if len(neg_sub) > 0 else 0.0

        fairness["age_band"][band_name] = {
            "n_hours": len(sub),
            "n_patients": sub["patient_id"].nunique(),
            "auroc": round(auc, 3) if not np.isnan(auc) else "N/A (single class)",
            "false_alarm_rate": round(far, 3)
        }

    return fairness


def plot_shap_summary(shap_values, feature_cols: List[str]):
    """Generates and saves a clean global SHAP importance bar chart."""
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]

    top_names = [feature_cols[i] for i in top_indices]
    top_values = [mean_abs_shap[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values[::-1], color="#0284c7", edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Impending Sepsis Prediction)", fontsize=10)
    ax.set_title("Gap 2: Global Feature Importance (SHAP TreeExplainer)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    plt.tight_layout()

    out_path = os.path.join(ARTIFACTS_DIR, "gap2_shap_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train_gap2] Saved SHAP summary plot to {out_path}")


def plot_fairness_audit(fairness_results: Dict):
    """Plots fairness metrics across Gender and Age subgroups."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # 1. Gender FAR
    ax1 = axes[0]
    g_names = list(fairness_results["gender"].keys())
    g_fars = [fairness_results["gender"][k]["false_alarm_rate"] for k in g_names]
    bars1 = ax1.bar(g_names, g_fars, color=["#38bdf8", "#0284c7"], width=0.45)
    ax1.set_ylabel("False Alarm Rate (FAR)", fontsize=10)
    ax1.set_title("Fairness Audit: False Alarm Rate by Gender", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(0.2, max(g_fars + [0.1]) * 1.3))
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 0.005, f"{h:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Age Band FAR
    ax2 = axes[1]
    a_names = list(fairness_results["age_band"].keys())
    a_fars = [fairness_results["age_band"][k]["false_alarm_rate"] for k in a_names]
    bars2 = ax2.bar(a_names, a_fars, color=["#10b981", "#f59e0b", "#f43f5e"], width=0.5)
    ax2.set_ylabel("False Alarm Rate (FAR)", fontsize=10)
    ax2.set_title("Fairness Audit: False Alarm Rate by Age Band", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(0.2, max(a_fars + [0.1]) * 1.3))
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.005, f"{h:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(ARTIFACTS_DIR, "gap2_fairness.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train_gap2] Saved Fairness plot to {out_path}")


if __name__ == "__main__":
    train_and_evaluate_gap2()
