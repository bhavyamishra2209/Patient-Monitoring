"""
triage_sim.py
-------------
Step 3: Gap 3 - Triage and Resource Prioritization Under Clinical Constraints.
Replays PhysioNet test patients hour by hour through a simulated ICU/ED ward:
- Ward capacity: N beds, K nurse reviews per hour (configurable, default N=10, K=2).
- Compares 4 prioritization policies:
  1. FIFO (First-In First-Out by wait time)
  2. Greedy-by-NEWS2 (Current clinical baseline)
  3. Greedy-by-ML-Risk (Gap 1 probability)
  4. Constraint-Aware Ranker (Ours - Gap 3: utility combining risk, trend slope, starvation wait, alert-fatigue penalty, capacity-aware)
- Measures:
  - % of septic patients reviewed before onset
  - Mean hours of advance notice before onset
  - Missed onsets (onset occurred before nurse review)
  - Mean patient wait time between reviews
- Sweeps N and K to evaluate policy performance under varying bed/staff constraints.
- Saves artifacts/gap3_results.json, artifacts/gap3_policy_comparison.png, and artifacts/gap3_capacity_sweep.png.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pn_data import load_cached_features, ARTIFACTS_DIR, SPLIT_FILE


class WardSimulator:
    """Simulates ICU ward triage under nurse review constraints."""

    def __init__(self, n_beds: int = 10, k_reviews_per_hour: int = 2):
        self.n_beds = n_beds
        self.k_reviews = k_reviews_per_hour

    def run_simulation(
        self,
        test_patient_dfs: Dict[str, pd.DataFrame],
        policy: str = "constraint_aware"
    ) -> Dict:
        """
        Runs hour-by-hour simulation across test patients.
        test_patient_dfs: map of patient_id -> sorted hourly records DataFrame.
        """
        pids = list(test_patient_dfs.keys())
        # Populate N beds from test cohort (cycling through test patients if N > len(pids))
        bed_assignments = []
        for b_idx in range(self.n_beds):
            pid = pids[b_idx % len(pids)]
            bed_assignments.append({
                "bed_id": b_idx + 1,
                "patient_id": pid,
                "df": test_patient_dfs[pid],
                "time_since_last_review": 0,
                "total_reviews": 0,
                "alerts_fired": 0,
                "first_review_hr": None,
                "septic_onset_hr": None,
                "reviewed_before_onset": False,
                "lead_before_onset": 0.0,
                "wait_times": []
            })

        # Identify sepsis onset hours
        for bed in bed_assignments:
            sub = bed["df"]
            if (sub["SepsisLabel"] == 1).any():
                bed["septic_onset_hr"] = float(sub[sub["SepsisLabel"] == 1]["ICULOS"].min())

        max_hours = max(len(bed["df"]) for bed in bed_assignments)

        # Simulation loop hour by hour
        for t_step in range(max_hours):
            current_candidates = []

            for bed in bed_assignments:
                df_p = bed["df"]
                if t_step >= len(df_p):
                    continue
                row = df_p.iloc[t_step]
                hr = float(row["ICULOS"])

                # Check if patient reached onset at this hour
                onset_hr = bed["septic_onset_hr"]
                if onset_hr is not None and hr >= onset_hr and not bed["reviewed_before_onset"]:
                    pass  # Missed onset window if not reviewed yet

                # Calculate priority scores based on policy
                news2 = float(row.get("news2_score", 0))
                ml_prob = float(row.get("gap1_prob", 0.1))
                shock_slope = float(row.get("shock_index_slope_4h", 0.0))
                hr_slope = float(row.get("heart_rate_bpm_slope_4h", 0.0))
                wait = bed["time_since_last_review"]
                af = bed["alerts_fired"]

                # Policy score computation
                if policy == "fifo":
                    score = float(wait)
                elif policy == "greedy_news2":
                    score = float(news2) + 0.01 * wait
                elif policy == "greedy_ml":
                    score = float(ml_prob) + 0.001 * wait
                elif policy == "constraint_aware":
                    # Gap 3 Constraint-Aware Utility
                    capacity_stress = float(self.n_beds / max(1, self.k_reviews))
                    w_risk = 0.55 if capacity_stress > 3.0 else 0.40
                    w_trend = 0.25 if capacity_stress > 3.0 else 0.20
                    w_wait = 0.15 if capacity_stress > 3.0 else 0.30
                    w_fatigue = 0.05 if capacity_stress > 3.0 else 0.10

                    trend_component = float(np.clip((shock_slope * 2.0) + (hr_slope / 20.0), -1.0, 1.0))
                    wait_component = min(1.0, wait / 10.0)
                    fatigue_penalty = min(1.0, af * 0.15)

                    score = (
                        w_risk * ml_prob
                        + w_trend * trend_component
                        + w_wait * wait_component
                        - w_fatigue * fatigue_penalty
                    )
                else:
                    score = float(ml_prob)

                current_candidates.append({
                    "bed": bed,
                    "score": score,
                    "hour": hr,
                    "ml_prob": ml_prob
                })

            if not current_candidates:
                break

            # Sort by score descending (highest priority first)
            current_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Top K reviews
            reviewed_this_tick = current_candidates[:self.k_reviews]
            reviewed_beds = set()

            for item in reviewed_this_tick:
                b = item["bed"]
                c_hr = item["hour"]
                reviewed_beds.add(b["bed_id"])

                b["total_reviews"] += 1
                b["wait_times"].append(b["time_since_last_review"])
                b["time_since_last_review"] = 0

                if b["first_review_hr"] is None:
                    b["first_review_hr"] = c_hr

                onset_hr = b["septic_onset_hr"]
                if onset_hr is not None and not b["reviewed_before_onset"]:
                    if c_hr < onset_hr:
                        b["reviewed_before_onset"] = True
                        b["lead_before_onset"] = onset_hr - c_hr

            # Unreviewed beds wait time increment
            for item in current_candidates:
                b = item["bed"]
                if b["bed_id"] not in reviewed_beds:
                    b["time_since_last_review"] += 1
                    if item["ml_prob"] >= 0.88:
                        b["alerts_fired"] += 1

        # Calculate final summary metrics
        septic_beds = [b for b in bed_assignments if b["septic_onset_hr"] is not None]
        n_septic = len(septic_beds)

        reviewed_before = [b for b in septic_beds if b["reviewed_before_onset"]]
        pct_reviewed_before = (len(reviewed_before) / max(1, n_septic)) * 100.0
        missed_onsets = n_septic - len(reviewed_before)

        leads = [b["lead_before_onset"] for b in reviewed_before]
        mean_lead_hours = float(np.mean(leads)) if leads else 0.0

        all_waits = []
        for b in bed_assignments:
            all_waits.extend(b["wait_times"])
        mean_wait_hours = float(np.mean(all_waits)) if all_waits else float(self.n_beds / max(1, self.k_reviews))

        return {
            "policy": policy,
            "n_beds": self.n_beds,
            "k_reviews": self.k_reviews,
            "capacity_ratio_K_over_N": round(self.k_reviews / self.n_beds, 2),
            "pct_septic_reviewed_before_onset": round(pct_reviewed_before, 1),
            "mean_lead_before_onset_hours": round(mean_lead_hours, 2),
            "missed_onsets": missed_onsets,
            "total_septic_patients": n_septic,
            "mean_wait_hours": round(mean_wait_hours, 2)
        }


def run_triage_experiments() -> Dict:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df_full = load_cached_features()

    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    test_pids = split["test"]
    gap1_model = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_lgbm.joblib"))
    feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "temporal_feature_cols.joblib"))

    # Prepare test patient streaming data
    test_patient_dfs = {}
    for pid in test_pids:
        sub = df_full[df_full["patient_id"] == pid].sort_values("ICULOS").copy()
        sub["gap1_prob"] = gap1_model.predict_proba(sub[feature_cols])[:, 1]
        test_patient_dfs[pid] = sub

    # 1. Main Policy Benchmark (N=10 beds, K=2 nurse reviews/hr)
    policies = ["fifo", "greedy_news2", "greedy_ml", "constraint_aware"]
    policy_labels = {
        "fifo": "FIFO (Wait Queue)",
        "greedy_news2": "Greedy-by-NEWS2 (Rule)",
        "greedy_ml": "Greedy-by-ML (Static Acuity)",
        "constraint_aware": "Constraint-Aware (Ours - Gap 3)"
    }

    print("[triage_sim] Running Ward Prioritization Benchmark (N=10 beds, K=2 reviews/hr)...")
    sim = WardSimulator(n_beds=10, k_reviews_per_hour=2)
    benchmark_results = {}
    for pol in policies:
        res = sim.run_simulation(test_patient_dfs, policy=pol)
        res["policy_label"] = policy_labels[pol]
        benchmark_results[pol] = res
        print(f"  {policy_labels[pol]:<35}: Reviewed Before Onset={res['pct_septic_reviewed_before_onset']}%, "
              f"Mean Lead={res['mean_lead_before_onset_hours']}h, Missed={res['missed_onsets']}, Wait={res['mean_wait_hours']}h")

    # 2. Capacity Sweep Over N (Beds) with fixed K=2
    n_sweep_values = [5, 10, 15, 20]
    n_sweep_results = {pol: [] for pol in policies}
    print("[triage_sim] Sweeping Bed Capacity N in [5, 10, 15, 20] with K=2...")
    for n in n_sweep_values:
        s = WardSimulator(n_beds=n, k_reviews_per_hour=2)
        for pol in policies:
            r = s.run_simulation(test_patient_dfs, policy=pol)
            n_sweep_results[pol].append(r)

    # 3. Capacity Sweep Over K (Nurse Reviews/hr) with fixed N=10
    k_sweep_values = [1, 2, 3, 4]
    k_sweep_results = {pol: [] for pol in policies}
    print("[triage_sim] Sweeping Nurse Review Capacity K in [1, 2, 3, 4] with N=10...")
    for k in k_sweep_values:
        s = WardSimulator(n_beds=10, k_reviews_per_hour=k)
        for pol in policies:
            r = s.run_simulation(test_patient_dfs, policy=pol)
            k_sweep_results[pol].append(r)

    # Save Gap 3 Results JSON
    final_output = {
        "benchmark_N10_K2": benchmark_results,
        "n_sweep": {
            "n_values": n_sweep_values,
            "k_fixed": 2,
            "results": n_sweep_results
        },
        "k_sweep": {
            "k_values": k_sweep_values,
            "n_fixed": 10,
            "results": k_sweep_results
        }
    }

    out_json = os.path.join(ARTIFACTS_DIR, "gap3_results.json")
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"[triage_sim] Saved Gap 3 results to {out_json}")

    # Generate Figures
    plot_gap3_figures(benchmark_results, n_sweep_values, n_sweep_results, k_sweep_values, k_sweep_results, policy_labels)
    return final_output


def plot_gap3_figures(benchmark_results, n_vals, n_results, k_vals, k_results, policy_labels):
    # Figure 1: Policy Comparison Bar Charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=150)
    p_keys = ["fifo", "greedy_news2", "greedy_ml", "constraint_aware"]
    labels = [policy_labels[k] for k in p_keys]
    colors = ["#94a3b8", "#f59e0b", "#0284c7", "#10b981"]

    # 1. % Reviewed Before Onset
    ax1 = axes[0]
    pcts = [benchmark_results[k]["pct_septic_reviewed_before_onset"] for k in p_keys]
    bars1 = ax1.bar(range(len(labels)), pcts, color=colors, width=0.55)
    ax1.set_ylabel("% Reviewed Before Onset", fontsize=10)
    ax1.set_title("Patients Reviewed Before Onset (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Mean Lead Time Before Onset
    ax2 = axes[1]
    leads = [benchmark_results[k]["mean_lead_before_onset_hours"] for k in p_keys]
    bars2 = ax2.bar(range(len(labels)), leads, color=colors, width=0.55)
    ax2.set_ylabel("Advance Lead Notice (Hours)", fontsize=10)
    ax2.set_title("Advance Warning Notice Lead Time", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.2, f"{h:.1f}h", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Mean Wait Time
    ax3 = axes[2]
    waits = [benchmark_results[k]["mean_wait_hours"] for k in p_keys]
    bars3 = ax3.bar(range(len(labels)), waits, color=colors, width=0.55)
    ax3.set_ylabel("Mean Wait Between Reviews (Hours)", fontsize=10)
    ax3.set_title("Average Patient Wait Time (Starvation)", fontsize=11, fontweight="bold")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.3, axis="y")
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h + 0.1, f"{h:.1f}h", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(ARTIFACTS_DIR, "gap3_policy_comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Capacity Sweeps (N vs K)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.0), dpi=150)
    colors_dict = {"fifo": "#94a3b8", "greedy_news2": "#f59e0b", "greedy_ml": "#0284c7", "constraint_aware": "#10b981"}

    # Left: Sweep N (Beds) with K=2
    ax_n = axes2[0]
    for pol in p_keys:
        y_vals = [r["mean_lead_before_onset_hours"] for r in n_results[pol]]
        ax_n.plot(n_vals, y_vals, marker="o", label=policy_labels[pol], color=colors_dict[pol], lw=2)
    ax_n.set_xlabel("Ward Capacity N (Beds) with K=2 Reviews/hr", fontsize=10)
    ax_n.set_ylabel("Mean Lead Time Before Onset (Hours)", fontsize=10)
    ax_n.set_title("Capacity Stress: Bed Sweep (N)", fontsize=11, fontweight="bold")
    ax_n.grid(True, linestyle="--", alpha=0.3)
    ax_n.legend(fontsize=8, loc="upper right")

    # Right: Sweep K (Nurse Reviews/hr) with N=10
    ax_k = axes2[1]
    for pol in p_keys:
        y_vals = [r["mean_lead_before_onset_hours"] for r in k_results[pol]]
        ax_k.plot(k_vals, y_vals, marker="s", label=policy_labels[pol], color=colors_dict[pol], lw=2)
    ax_k.set_xlabel("Nurse Capacity K (Reviews/hr) with N=10 Beds", fontsize=10)
    ax_k.set_ylabel("Mean Lead Time Before Onset (Hours)", fontsize=10)
    ax_k.set_title("Staffing Capacity: Nurse Review Sweep (K)", fontsize=11, fontweight="bold")
    ax_k.grid(True, linestyle="--", alpha=0.3)
    ax_k.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    sweep_path = os.path.join(ARTIFACTS_DIR, "gap3_capacity_sweep.png")
    plt.savefig(sweep_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[triage_sim] Saved plots: {comp_path} and {sweep_path}")


if __name__ == "__main__":
    run_triage_experiments()
