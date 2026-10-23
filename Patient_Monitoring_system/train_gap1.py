"""
train_gap1.py
-------------
Step 1: Gap 1 - Temporal Prediction of Clinical Deterioration.
Trains and benchmarks on the frozen PhysioNet 2019 split:
  (a) Rule baselines: NEWS2 and qSOFA
  (b) Static LightGBM: point-in-time vitals only (no temporal history)
  (c) Temporal LightGBM: full rolling statistics, slopes, deltas, and missingness indicators
Computes AUROC, AUPRC, sensitivity at 90% specificity with 95% patient-bootstrap CIs.
Measures advance warning lead time before clinical SepsisLabel onset.
Saves artifacts/gap1_results.json, artifacts/gap1_roc_pr.png, and artifacts/gap1_lead_time.png.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import lightgbm as lgb
from pn_data import load_cached_features, ARTIFACTS_DIR, SPLIT_FILE


def find_threshold_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.90) -> Tuple[float, float]:
    """Finds decision threshold achieving at least target_spec (e.g. 90%) specificity on validation data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    max_fpr = 1.0 - target_spec
    finite_mask = np.isfinite(thresholds) & (fpr <= max_fpr)
    if np.any(finite_mask):
        valid_idx = np.where(finite_mask)[0]
        # Choose the threshold that maximizes TPR subject to FPR <= max_fpr
        idx = valid_idx[np.argmax(tpr[valid_idx])]
        chosen_thresh = float(thresholds[idx])
        achieved_spec = float(1.0 - fpr[idx])
    else:
        neg_scores = y_score[y_true == 0]
        chosen_thresh = float(np.percentile(neg_scores, target_spec * 100)) if len(neg_scores) > 0 else 0.5
        achieved_spec = target_spec
    return chosen_thresh, achieved_spec


def bootstrap_patient_ci(
    df_test: pd.DataFrame,
    test_pids: List[str],
    score_col: str,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Computes 95% bootstrap confidence intervals clustered at the patient level."""
    rng = np.random.RandomState(seed)
    aurocs, auprcs, sensitivities = [], [], []

    for _ in range(n_bootstraps):
        sampled_pids = rng.choice(test_pids, size=len(test_pids), replace=True)
        # Gather rows for sampled patients
        dfs = [df_test[df_test["patient_id"] == p] for p in sampled_pids]
        sub = pd.concat(dfs, ignore_index=True)
        y_true = sub["target_deterioration"].values
        y_score = sub[score_col].values

        if len(np.unique(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_score)
            prc = average_precision_score(y_true, y_score)
            pred = (y_score >= threshold).astype(int)
            pos_mask = (y_true == 1)
            sens = float(np.sum((pred == 1) & pos_mask) / np.sum(pos_mask)) if np.sum(pos_mask) > 0 else 0.0

            aurocs.append(auc)
            auprcs.append(prc)
            sensitivities.append(sens)
        except Exception:
            continue

    def get_ci(vals, default_val):
        if len(vals) < 50:
            return [round(default_val, 4), round(default_val, 4)]
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    return {
        "auroc_ci": get_ci(aurocs, 0.5),
        "auprc_ci": get_ci(auprcs, 0.1),
        "sensitivity_ci": get_ci(sensitivities, 0.5)
    }


def train_and_evaluate_gap1() -> Dict:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_cached_features()

    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    # Filter to pre-onset hours to prevent post-onset diagnostic leakage
    df_pre = df[df["is_pre_onset"]].copy()

    df_train = df_pre[df_pre["split"] == "train"].copy()
    df_val   = df_pre[df_pre["split"] == "val"].copy()
    df_test  = df_pre[df_pre["split"] == "test"].copy()

    print(f"[train_gap1] Pre-onset hourly records -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    print(f"  Target positive rate: Train={df_train['target_deterioration'].mean():.1%}, "
          f"Val={df_val['target_deterioration'].mean():.1%}, Test={df_test['target_deterioration'].mean():.1%}")

    # Feature definitions
    static_cols = [
        "heart_rate_bpm", "systolic_bp_mmHg", "diastolic_bp_mmHg",
        "respiratory_rate_bpm", "body_temperature_c", "spo2_percent",
        "shock_index", "mean_arterial_pressure", "Age", "Gender", "hours_since_last_lab"
    ]
    # Filter to existing static cols
    static_cols = [c for c in static_cols if c in df.columns]

    # Curated temporal feature set: core vitals, derived metrics, rolling stats, slopes, deltas
    core_vitals = [
        "heart_rate_bpm", "systolic_bp_mmHg", "diastolic_bp_mmHg",
        "respiratory_rate_bpm", "body_temperature_c", "spo2_percent",
        "shock_index", "mean_arterial_pressure"
    ]
    temporal_cols = []
    for v in core_vitals:
        if v in df.columns:
            temporal_cols.extend([
                v, f"{v}_delta_baseline", f"{v}_mean_4h", f"{v}_std_4h",
                f"{v}_slope_4h", f"{v}_mean_12h", f"{v}_std_12h", f"{v}_slope_12h"
            ])
    for extra in ["Age", "Gender", "hours_since_last_lab"]:
        if extra in df.columns:
            temporal_cols.append(extra)
    # Filter to only existing columns
    temporal_cols = [c for c in temporal_cols if c in df.columns]

    y_train = df_train["target_deterioration"].values
    y_val   = df_val["target_deterioration"].values
    y_test  = df_test["target_deterioration"].values

    # 1. Train Static LightGBM
    print(f"[train_gap1] Training Static LightGBM on {len(static_cols)} features...")
    model_static = lgb.LGBMClassifier(
        n_estimators=60,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=7,
        min_child_samples=5,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    model_static.fit(df_train[static_cols], y_train)

    # 2. Train Temporal LightGBM
    print(f"[train_gap1] Training Temporal LightGBM on {len(temporal_cols)} rolling/trend features...")
    model_temporal = lgb.LGBMClassifier(
        n_estimators=60,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=7,
        min_child_samples=5,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    model_temporal.fit(df_train[temporal_cols], y_train)

    # Save models
    joblib.dump(model_temporal, os.path.join(ARTIFACTS_DIR, "temporal_lgbm.joblib"))
    joblib.dump(model_static, os.path.join(ARTIFACTS_DIR, "static_lgbm.joblib"))
    joblib.dump(temporal_cols, os.path.join(ARTIFACTS_DIR, "temporal_feature_cols.joblib"))

    # Predictions on Val set
    prob_val_temporal = model_temporal.predict_proba(df_val[temporal_cols])[:, 1]
    prob_val_static   = model_static.predict_proba(df_val[static_cols])[:, 1]
    score_val_news2   = df_val["news2_score"].values.astype(float)
    score_val_qsofa   = df_val["qsofa_score"].values.astype(float)

    # Predictions on Test set
    prob_test_temporal = model_temporal.predict_proba(df_test[temporal_cols])[:, 1]
    prob_test_static   = model_static.predict_proba(df_test[static_cols])[:, 1]
    score_test_news2   = df_test["news2_score"].values.astype(float)
    score_test_qsofa   = df_test["qsofa_score"].values.astype(float)

    df_test["prob_temporal"] = prob_test_temporal
    df_test["prob_static"]   = prob_test_static

    # Compute validation thresholds for 90% specificity
    thresh_temporal, spec_temp = find_threshold_at_specificity(y_val, prob_val_temporal, target_spec=0.90)
    thresh_static, spec_stat   = find_threshold_at_specificity(y_val, prob_val_static, target_spec=0.90)
    thresh_news2, spec_n2      = find_threshold_at_specificity(y_val, score_val_news2, target_spec=0.90)
    thresh_qsofa, spec_qs      = find_threshold_at_specificity(y_val, score_val_qsofa, target_spec=0.90)

    print(f"[train_gap1] Validation 90% Specificity Thresholds: "
          f"Temporal={thresh_temporal:.4f}, Static={thresh_static:.4f}, NEWS2={thresh_news2:.1f}, qSOFA={thresh_qsofa:.1f}")

    # Evaluate on Test set
    models_eval = {
        "temporal_lgbm": {
            "name": "Temporal LightGBM (Gap 1)",
            "y_score": prob_test_temporal,
            "col_name": "prob_temporal",
            "val_threshold": thresh_temporal,
            "achieved_val_spec": spec_temp
        },
        "static_lgbm": {
            "name": "Static LightGBM (Point-in-Time)",
            "y_score": prob_test_static,
            "col_name": "prob_static",
            "val_threshold": thresh_static,
            "achieved_val_spec": spec_stat
        },
        "rule_news2": {
            "name": "Rule-Based NEWS2",
            "y_score": score_test_news2,
            "col_name": "news2_score",
            "val_threshold": thresh_news2,
            "achieved_val_spec": spec_n2
        },
        "rule_qsofa": {
            "name": "Rule-Based qSOFA",
            "y_score": score_test_qsofa,
            "col_name": "qsofa_score",
            "val_threshold": thresh_qsofa,
            "achieved_val_spec": spec_qs
        }
    }

    test_pids = split["test"]
    results = {}

    for k, info in models_eval.items():
        y_score = info["y_score"]
        th = info["val_threshold"]
        auc = float(roc_auc_score(y_test, y_score))
        prc = float(average_precision_score(y_test, y_score))

        pred = (y_score >= th).astype(int)
        pos_mask = (y_test == 1)
        sens = float(np.sum((pred == 1) & pos_mask) / np.sum(pos_mask)) if np.sum(pos_mask) > 0 else 0.0
        neg_mask = (y_test == 0)
        spec = float(np.sum((pred == 0) & neg_mask) / np.sum(neg_mask)) if np.sum(neg_mask) > 0 else 0.0

        ci_dict = bootstrap_patient_ci(df_test, test_pids, info["col_name"], th)

        results[k] = {
            "name": info["name"],
            "auroc": round(auc, 4),
            "auroc_ci": ci_dict["auroc_ci"],
            "auprc": round(prc, 4),
            "auprc_ci": ci_dict["auprc_ci"],
            "sensitivity_at_90_spec": round(sens, 4),
            "sensitivity_ci": ci_dict["sensitivity_ci"],
            "test_specificity": round(spec, 4),
            "val_threshold": round(th, 4)
        }

    # Lead Time Analysis on Septic Test Patients
    septic_test_pids = []
    for pid in test_pids:
        sub = df[df["patient_id"] == pid]
        if (sub["SepsisLabel"] == 1).any():
            septic_test_pids.append(pid)

    print(f"[train_gap1] Septic test patients for lead time evaluation: {septic_test_pids}")

    lead_times = {k: [] for k in models_eval.keys()}
    patient_lead_details = []

    for pid in septic_test_pids:
        sub_full = df[df["patient_id"] == pid].sort_values("ICULOS").copy()
        sub_full["prob_temporal"] = model_temporal.predict_proba(sub_full[temporal_cols])[:, 1]
        sub_full["prob_static"]   = model_static.predict_proba(sub_full[static_cols])[:, 1]

        onset_hr = float(sub_full[sub_full["SepsisLabel"] == 1]["ICULOS"].min())
        sub_pre = sub_full[sub_full["ICULOS"] < onset_hr]
        p_res = {"patient_id": pid, "onset_hour": onset_hr, "lead_times": {}}

        for k, info in models_eval.items():
            th = info["val_threshold"]
            col = info["col_name"]
            # Look for earliest trigger before clinical onset
            alert_rows = sub_pre[sub_pre[col] >= th]
            if len(alert_rows) > 0:
                first_fire_hr = float(alert_rows["ICULOS"].min())
                lead = max(0.0, onset_hr - first_fire_hr)
            else:
                lead = 0.0

            lead_times[k].append(round(lead, 2))
            p_res["lead_times"][k] = round(lead, 2)

        patient_lead_details.append(p_res)

    for k in results.keys():
        lt_list = lead_times[k]
        results[k]["mean_lead_time_hours"] = round(float(np.mean(lt_list)), 2) if lt_list else 0.0
        results[k]["lead_times_by_patient"] = lt_list

    # Save Gap 1 Results JSON
    gap1_json_path = os.path.join(ARTIFACTS_DIR, "gap1_results.json")
    with open(gap1_json_path, "w") as f:
        json.dump({
            "model_metrics": results,
            "patient_lead_details": patient_lead_details,
            "n_test_patients": len(test_pids),
            "n_septic_test_patients": len(septic_test_pids),
            "target_definition": "SepsisLabel==1 at any hour in [t+2, t+6] (pre-onset only)"
        }, f, indent=2)
    print(f"[train_gap1] Saved results to {gap1_json_path}")

    # Generate Figures: 1) ROC / PR Curves, 2) Lead Time Comparison
    plot_gap1_figures(y_test, models_eval, results, lead_times)
    return results


def plot_gap1_figures(y_test: np.ndarray, models_eval: Dict, results: Dict, lead_times: Dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)

    # Subplot 1: ROC Curves
    ax1 = axes[0]
    colors = {"temporal_lgbm": "#0284c7", "static_lgbm": "#64748b", "rule_news2": "#f59e0b", "rule_qsofa": "#ef4444"}
    for k, info in models_eval.items():
        fpr, tpr, _ = roc_curve(y_test, info["y_score"])
        auc_val = results[k]["auroc"]
        ci = results[k]["auroc_ci"]
        ax1.plot(fpr, tpr, label=f"{info['name']} (AUC={auc_val:.3f} [{ci[0]}-{ci[1]}])", color=colors[k], lw=2)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax1.set_title("Test Set ROC Curves (Gap 1)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8)

    # Subplot 2: Precision-Recall Curves
    ax2 = axes[1]
    for k, info in models_eval.items():
        precision, recall, _ = precision_recall_curve(y_test, info["y_score"])
        prc_val = results[k]["auprc"]
        ci = results[k]["auprc_ci"]
        ax2.plot(recall, precision, label=f"{info['name']} (AUPRC={prc_val:.3f})", color=colors[k], lw=2)

    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax2.set_ylabel("Precision (PPV)", fontsize=10)
    ax2.set_title("Test Set Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)

    # Subplot 3: Lead Time Bar / Comparison Plot
    ax3 = axes[2]
    model_names = [models_eval[k]["name"].replace(" (Gap 1)", "").replace(" (Point-in-Time)", "") for k in models_eval.keys()]
    mean_leads = [results[k]["mean_lead_time_hours"] for k in models_eval.keys()]
    bar_cols = [colors[k] for k in models_eval.keys()]

    bars = ax3.bar(model_names, mean_leads, color=bar_cols, width=0.55, edgecolor="none")
    ax3.set_ylabel("Mean Advance Warning Notice (Hours)", fontsize=10)
    ax3.set_title("Early Warning Lead Time Before Sepsis Onset", fontsize=11, fontweight="bold")
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.3, axis="y")

    for b in bars:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h + 0.1, f"+{h:.1f}h" if h > 0 else "0.0h", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    roc_pr_path = os.path.join(ARTIFACTS_DIR, "gap1_roc_pr.png")
    plt.savefig(roc_pr_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Lead time boxplot
    fig_lt, ax_lt = plt.subplots(figsize=(7, 4.5), dpi=150)
    box_data = [lead_times[k] for k in models_eval.keys()]
    ax_lt.boxplot(box_data, patch_artist=True,
                  boxprops=dict(facecolor="#bae6fd", color="#0284c7"),
                  medianprops=dict(color="#0f172a", lw=2))
    ax_lt.set_xticks(range(1, len(model_names) + 1))
    ax_lt.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax_lt.set_ylabel("Advance Notice Before Sepsis Onset (Hours)", fontsize=10)
    ax_lt.set_title("Early Warning Lead Time Distribution on Septic Test Cohort", fontsize=11, fontweight="bold")
    ax_lt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    lt_box_path = os.path.join(ARTIFACTS_DIR, "gap1_lead_time.png")
    plt.savefig(lt_box_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[train_gap1] Generated figures: {roc_pr_path} and {lt_box_path}")


if __name__ == "__main__":
    train_and_evaluate_gap1()
