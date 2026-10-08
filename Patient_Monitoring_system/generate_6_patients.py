"""
generate_6_patients.py
----------------------
Generates 6 distinct, physiologically realistic, and clinically rich
patient vital time-series CSV files (patient1.csv - patient6.csv).

Each patient represents a distinct clinical archetype:
1. patient1.csv: Sepsis Progression (Gradual deterioration -> Septic shock)
2. patient2.csv: Acute Hypoxia & Respiratory Failure (ARDS)
3. patient3.csv: Post-Operative Hemorrhagic Shock (Internal Bleeding)
4. patient4.csv: Hypertensive Crisis & Neurological Alteration
5. patient5.csv: Responding to Treatment (Recovery / De-escalation)
6. patient6.csv: Stable Ward Telemetry (Low Acuity / Normal Baseline)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_patient_csvs(output_dir=".", n_steps=120):
    np.random.seed(42)
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    base_time = datetime(2026, 10, 6, 8, 0, 0)

    # -------------------------------------------------------------
    # 1. Patient 1: Sepsis Progression (SEP_001)
    # Starts mildly febrile -> HR rises, SBP drops, Temp spikes, Lactate rises
    # -------------------------------------------------------------
    rows_p1 = []
    hr, sbp, dbp, rr, temp, spo2, lactate = 78.0, 122.0, 78.0, 16.0, 37.4, 98.0, 1.2
    for step in range(n_steps):
        t_str = (base_time + timedelta(minutes=step * 2)).strftime("%H:%M:%S")
        if step >= 25:
            # Gradual deterioration
            hr = min(142.0, hr + np.random.normal(0.65, 0.35))
            sbp = max(80.0, sbp - np.random.normal(0.45, 0.3))
            dbp = max(48.0, sbp * 0.62 + np.random.normal(0, 1.5))
            rr = min(34.0, rr + np.random.normal(0.18, 0.12))
            temp = min(40.2, temp + np.random.normal(0.03, 0.015))
            spo2 = max(89.5, spo2 - np.random.normal(0.08, 0.04))
            lactate = min(4.6, lactate + np.random.normal(0.035, 0.015))
            cons = 1 if step > 75 else 0
            state = "Septic Shock" if step > 80 else ("Early Sepsis" if step > 40 else "SIRS")
        else:
            # Baseline with minor noise
            hr += np.random.normal(0, 0.6)
            sbp += np.random.normal(0, 0.8)
            dbp = sbp * 0.65 + np.random.normal(0, 0.5)
            rr += np.random.normal(0, 0.2)
            temp += np.random.normal(0, 0.02)
            spo2 += np.random.normal(0, 0.08)
            cons = 0
            state = "Mild SIRS"

        rows_p1.append({
            "patient_id": "SEP_001",
            "timestamp": t_str,
            "heart_rate_bpm": int(np.clip(hr, 50, 160)),
            "systolic_bp_mmHg": int(np.clip(sbp, 60, 200)),
            "diastolic_bp_mmHg": int(np.clip(dbp, 35, sbp - 15)),
            "respiratory_rate_bpm": int(np.clip(rr, 10, 45)),
            "body_temperature_c": round(float(np.clip(temp, 35.0, 41.5)), 2),
            "spo2_percent": round(float(np.clip(spo2, 80.0, 100.0)), 1),
            "lactate": round(float(np.clip(lactate, 0.5, 6.0)), 2),
            "clinical_state": state,
            "consciousness": cons
        })

    # -------------------------------------------------------------
    # 2. Patient 2: Acute Hypoxia & ARDS (RESP_002)
    # Severe respiratory deterioration: SpO2 plunges, RR surges
    # -------------------------------------------------------------
    rows_p2 = []
    hr, sbp, dbp, rr, temp, spo2, lactate = 82.0, 130.0, 82.0, 18.0, 37.2, 97.0, 1.3
    for step in range(n_steps):
        t_str = (base_time + timedelta(minutes=step * 2)).strftime("%H:%M:%S")
        if step >= 20:
            rr = min(40.0, rr + np.random.normal(0.25, 0.12))
            spo2 = max(80.5, spo2 - np.random.normal(0.22, 0.08))
            hr = min(138.0, hr + np.random.normal(0.55, 0.3))
            sbp = max(88.0, sbp - np.random.normal(0.35, 0.25))
            dbp = max(50.0, sbp * 0.63 + np.random.normal(0, 1.0))
            temp += np.random.normal(0.01, 0.015)
            lactate = min(3.8, lactate + np.random.normal(0.025, 0.01))
            cons = 1 if step > 65 else 0
            state = "Severe ARDS" if step > 70 else ("Impending Hypoxia" if step > 35 else "Tachypnea")
        else:
            hr += np.random.normal(0, 0.5)
            sbp += np.random.normal(0, 0.7)
            dbp = sbp * 0.64 + np.random.normal(0, 0.5)
            rr += np.random.normal(0, 0.25)
            temp += np.random.normal(0, 0.01)
            spo2 += np.random.normal(0, 0.1)
            cons = 0
            state = "Compensated Respiratory"

        rows_p2.append({
            "patient_id": "RESP_002",
            "timestamp": t_str,
            "heart_rate_bpm": int(np.clip(hr, 50, 160)),
            "systolic_bp_mmHg": int(np.clip(sbp, 60, 200)),
            "diastolic_bp_mmHg": int(np.clip(dbp, 35, sbp - 15)),
            "respiratory_rate_bpm": int(np.clip(rr, 10, 45)),
            "body_temperature_c": round(float(np.clip(temp, 35.0, 41.0)), 2),
            "spo2_percent": round(float(np.clip(spo2, 75.0, 100.0)), 1),
            "lactate": round(float(np.clip(lactate, 0.5, 6.0)), 2),
            "clinical_state": state,
            "consciousness": cons
        })

    # -------------------------------------------------------------
    # 3. Patient 3: Post-Op Hemorrhagic Shock (HEM_003)
    # Rapid occult bleeding: Tachycardia, severe hypotension, high Shock Index
    # -------------------------------------------------------------
    rows_p3 = []
    hr, sbp, dbp, rr, temp, spo2, lactate = 74.0, 126.0, 80.0, 15.0, 36.7, 98.5, 0.9
    for step in range(n_steps):
        t_str = (base_time + timedelta(minutes=step * 2)).strftime("%H:%M:%S")
        if step >= 15:
            # Internal hemorrhage
            hr = min(148.0, hr + np.random.normal(0.75, 0.35))
            sbp = max(70.0, sbp - np.random.normal(0.60, 0.3))
            dbp = max(40.0, dbp - np.random.normal(0.42, 0.2))
            rr = min(32.0, rr + np.random.normal(0.18, 0.1))
            temp = max(35.2, temp - np.random.normal(0.015, 0.01))  # hypothermia in hemorrhage
            spo2 = max(88.0, spo2 - np.random.normal(0.09, 0.04))
            lactate = min(5.2, lactate + np.random.normal(0.04, 0.02))
            cons = 1 if step > 55 else 0
            state = "Hemorrhagic Shock" if step > 50 else ("Occult Bleeding" if step > 25 else "Post-Op Stable")
        else:
            hr += np.random.normal(0, 0.5)
            sbp += np.random.normal(0, 0.6)
            dbp = sbp * 0.64 + np.random.normal(0, 0.5)
            rr += np.random.normal(0, 0.2)
            temp += np.random.normal(0, 0.01)
            spo2 += np.random.normal(0, 0.05)
            cons = 0
            state = "Post-Op Recovery"

        rows_p3.append({
            "patient_id": "HEM_003",
            "timestamp": t_str,
            "heart_rate_bpm": int(np.clip(hr, 50, 160)),
            "systolic_bp_mmHg": int(np.clip(sbp, 60, 200)),
            "diastolic_bp_mmHg": int(np.clip(dbp, 35, sbp - 15)),
            "respiratory_rate_bpm": int(np.clip(rr, 10, 45)),
            "body_temperature_c": round(float(np.clip(temp, 34.5, 40.0)), 2),
            "spo2_percent": round(float(np.clip(spo2, 80.0, 100.0)), 1),
            "lactate": round(float(np.clip(lactate, 0.5, 6.0)), 2),
            "clinical_state": state,
            "consciousness": cons
        })

    # -------------------------------------------------------------
    # 4. Patient 4: Hypertensive Emergency & Neuro Risk (NEURO_004)
    # Severe blood pressure spikes > 200 mmHg, altered consciousness
    # -------------------------------------------------------------
    rows_p4 = []
    hr, sbp, dbp, rr, temp, spo2, lactate = 86.0, 172.0, 104.0, 17.0, 37.0, 97.0, 1.4
    for step in range(n_steps):
        t_str = (base_time + timedelta(minutes=step * 2)).strftime("%H:%M:%S")
        if step >= 30:
            sbp = min(228.0, sbp + np.random.normal(0.65, 0.4))
            dbp = min(132.0, dbp + np.random.normal(0.35, 0.25))
            hr = np.clip(hr + np.random.normal(0.2, 1.2), 55.0, 125.0)  # variable/autonomic
            rr = min(28.0, rr + np.random.normal(0.12, 0.1))
            temp += np.random.normal(0.005, 0.015)
            spo2 = max(91.0, spo2 - np.random.normal(0.06, 0.03))
            cons = 1 if step > 45 else 0
            state = "Hypertensive Emergency" if step > 50 else "Severe Hypertension"
        else:
            sbp += np.random.normal(0, 1.2)
            dbp += np.random.normal(0, 0.8)
            hr += np.random.normal(0, 0.6)
            rr += np.random.normal(0, 0.2)
            temp += np.random.normal(0, 0.01)
            spo2 += np.random.normal(0, 0.08)
            cons = 0
            state = "Hypertensive Urgency"

        rows_p4.append({
            "patient_id": "NEURO_004",
            "timestamp": t_str,
            "heart_rate_bpm": int(np.clip(hr, 45, 150)),
            "systolic_bp_mmHg": int(np.clip(sbp, 90, 240)),
            "diastolic_bp_mmHg": int(np.clip(dbp, 50, sbp - 20)),
            "respiratory_rate_bpm": int(np.clip(rr, 10, 40)),
            "body_temperature_c": round(float(np.clip(temp, 35.5, 40.0)), 2),
            "spo2_percent": round(float(np.clip(spo2, 85.0, 100.0)), 1),
            "lactate": round(float(np.clip(lactate, 0.5, 4.0)), 2),
            "clinical_state": state,
            "consciousness": cons
        })

    # -------------------------------------------------------------
    # 5. Patient 5: Responding to Resuscitation & Recovery (REC_005)
    # Starts in moderate distress -> steadily normalizes to healthy
    # -------------------------------------------------------------
    rows_p5 = []
    hr, sbp, dbp, rr, temp, spo2, lactate = 118.0, 96.0, 60.0, 26.0, 38.8, 93.0, 2.7
    for step in range(n_steps):
        t_str = (base_time + timedelta(minutes=step * 2)).strftime("%H:%M:%S")
        if step >= 15:
            # Steady recovery response
            hr = max(70.0, hr - np.random.normal(0.55, 0.25))
            sbp = min(122.0, sbp + np.random.normal(0.35, 0.2))
            dbp = min(78.0, dbp + np.random.normal(0.22, 0.15))
            rr = max(15.0, rr - np.random.normal(0.14, 0.08))
            temp = max(36.8, temp - np.random.normal(0.025, 0.012))
            spo2 = min(99.0, spo2 + np.random.normal(0.08, 0.04))
            lactate = max(1.0, lactate - np.random.normal(0.022, 0.01))
            cons = 0
            state = "Stabilizing (Ward Transfer)" if step > 65 else "Responding to Rx"
        else:
            hr += np.random.normal(0, 0.5)
            sbp += np.random.normal(0, 0.6)
            dbp += np.random.normal(0, 0.4)
            rr += np.random.normal(0, 0.3)
            temp += np.random.normal(0, 0.01)
            spo2 += np.random.normal(0, 0.08)
            cons = 0
            state = "Initial Resuscitation"

        rows_p5.append({
            "patient_id": "REC_005",
            "timestamp": t_str,
            "heart_rate_bpm": int(np.clip(hr, 50, 150)),
            "systolic_bp_mmHg": int(np.clip(sbp, 70, 180)),
            "diastolic_bp_mmHg": int(np.clip(dbp, 40, sbp - 15)),
            "respiratory_rate_bpm": int(np.clip(rr, 10, 40)),
            "body_temperature_c": round(float(np.clip(temp, 35.5, 40.5)), 2),
            "spo2_percent": round(float(np.clip(spo2, 85.0, 100.0)), 1),
            "lactate": round(float(np.clip(lactate, 0.5, 5.0)), 2),
            "clinical_state": state,
            "consciousness": cons
        })

    # -------------------------------------------------------------
    # 6. Patient 6: Stable Ward Telemetry (STAB_006)
    # Healthy baseline physiological drift, zero alert fatigue
    # -------------------------------------------------------------
    rows_p6 = []
    hr, sbp, dbp, rr, temp, spo2, lactate = 68.0, 118.0, 76.0, 14.0, 36.8, 98.8, 0.9
    for step in range(n_steps):
        t_str = (base_time + timedelta(minutes=step * 2)).strftime("%H:%M:%S")
        hr = np.clip(hr + np.random.normal(0, 0.4), 62.0, 76.0)
        sbp = np.clip(sbp + np.random.normal(0, 0.6), 112.0, 126.0)
        dbp = np.clip(dbp + np.random.normal(0, 0.4), 72.0, 82.0)
        rr = np.clip(rr + np.random.normal(0, 0.2), 12.0, 16.0)
        temp = np.clip(temp + np.random.normal(0, 0.015), 36.5, 37.1)
        spo2 = np.clip(spo2 + np.random.normal(0, 0.06), 97.5, 99.8)
        lactate = np.clip(lactate + np.random.normal(0, 0.01), 0.7, 1.1)

        rows_p6.append({
            "patient_id": "STAB_006",
            "timestamp": t_str,
            "heart_rate_bpm": int(hr),
            "systolic_bp_mmHg": int(sbp),
            "diastolic_bp_mmHg": int(dbp),
            "respiratory_rate_bpm": int(rr),
            "body_temperature_c": round(float(temp), 2),
            "spo2_percent": round(float(spo2), 1),
            "lactate": round(float(lactate), 2),
            "clinical_state": "Stable Ward",
            "consciousness": 0
        })

    # Save all 6 CSV files
    dfs = {
        "patient1.csv": pd.DataFrame(rows_p1),
        "patient2.csv": pd.DataFrame(rows_p2),
        "patient3.csv": pd.DataFrame(rows_p3),
        "patient4.csv": pd.DataFrame(rows_p4),
        "patient5.csv": pd.DataFrame(rows_p5),
        "patient6.csv": pd.DataFrame(rows_p6),
    }

    for fname, df in dfs.items():
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        print(f"[Generated] {fname} ({len(df)} rows) -> {fpath}")

if __name__ == "__main__":
    generate_patient_csvs()
