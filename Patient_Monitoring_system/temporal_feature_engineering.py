"""
temporal_feature_engineering.py
---------------------------------
Implements Gap 1: Temporal Feature Engineering for Early Deterioration Prediction.
Extracts rolling-window statistics, linear trend slopes, volatility, baseline deltas,
and cross-vital clinical features (e.g. Shock Index, MAP) from time-series vital streams.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


CORE_VITALS = [
    "heart_rate_bpm",
    "systolic_bp_mmHg",
    "diastolic_bp_mmHg",
    "respiratory_rate_bpm",
    "body_temperature_c",
    "spo2_percent"
]

VITAL_ALIASES = {
    "heart_rate": "heart_rate_bpm",
    "hr": "heart_rate_bpm",
    "sbp": "systolic_bp_mmHg",
    "systolic_bp": "systolic_bp_mmHg",
    "dbp": "diastolic_bp_mmHg",
    "diastolic_bp": "diastolic_bp_mmHg",
    "map": "mean_arterial_pressure",
    "resp_rate": "respiratory_rate_bpm",
    "rr": "respiratory_rate_bpm",
    "resp": "respiratory_rate_bpm",
    "respiratory_rate": "respiratory_rate_bpm",
    "temp": "body_temperature_c",
    "temperature": "body_temperature_c",
    "spo2": "spo2_percent",
    "o2sat": "spo2_percent",
    "consciousness": "consciousness",
    "lactate": "lactate",
    "wbc": "wbc",
    "creatinine": "creatinine",
    "platelets": "platelets",
    "sepsislabel": "sepsis_label"
}


def normalize_vital_keys(vital_dict: Dict) -> Dict:
    """Normalize incoming vital dict keys to standard column names."""
    normalized = {}
    for k, v in vital_dict.items():
        standard_k = VITAL_ALIASES.get(k.lower(), k)
        try:
            normalized[standard_k] = float(v) if v is not None and v != '' else 0.0
        except (ValueError, TypeError):
            normalized[standard_k] = v
    # Ensure derived defaults if missing
    if "diastolic_bp_mmHg" not in normalized and "systolic_bp_mmHg" in normalized:
        normalized["diastolic_bp_mmHg"] = max(40.0, float(normalized["systolic_bp_mmHg"]) * 0.65)
    return normalized


def compute_derived_metrics(row: Dict) -> Dict:
    """Compute standard clinical derived parameters (Shock Index, MAP, Pulse Pressure)."""
    sbp = float(row.get("systolic_bp_mmHg", 120.0))
    dbp = float(row.get("diastolic_bp_mmHg", 80.0))
    hr  = float(row.get("heart_rate_bpm", 75.0))

    safe_sbp = max(sbp, 10.0)
    shock_index = hr / safe_sbp
    pulse_pressure = max(0.0, sbp - dbp)
    mean_arterial_pressure = dbp + (pulse_pressure / 3.0)

    return {
        "shock_index": shock_index,
        "mean_arterial_pressure": mean_arterial_pressure,
        "pulse_pressure": pulse_pressure
    }


def compute_linear_slope(values: np.ndarray) -> float:
    """Computes ordinary least squares slope of values over time steps."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(values)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    nom = np.sum((x - x_mean) * (values - y_mean))
    return float(nom / denom)


def extract_features_from_window(
    window_history: List[Dict],
    short_window_len: int = 4,
    long_window_len: int = 12
) -> Dict[str, float]:
    """
    Given a list of consecutive vital records up to current instant t (ordered chronologically),
    extract comprehensive temporal features capturing:
      - Current point-in-time state
      - Short-term & long-term rolling means
      - Rolling standard deviation (volatility)
      - Linear trend slopes (trajectory)
      - Cumulative delta from admission baseline
      - Clinical cross-vital parameters (Shock Index, MAP)
    """
    if not window_history:
        return {}

    norm_history = [normalize_vital_keys(rec) for rec in window_history]
    current = norm_history[-1]
    baseline = norm_history[0]

    features: Dict[str, float] = {}

    # 1. Instantaneous features
    for col in CORE_VITALS:
        features[f"{col}_val"] = float(current.get(col, 0.0))
    features["consciousness_val"] = float(current.get("consciousness", 0.0))

    # Instantaneous derived
    cur_derived = compute_derived_metrics(current)
    for k, v in cur_derived.items():
        features[f"{k}_val"] = float(v)

    # 2. Rolling window features for each core vital
    for col in CORE_VITALS:
        all_vals = np.array([float(r.get(col, 0.0)) for r in norm_history], dtype=float)
        cur_val = all_vals[-1]
        base_val = all_vals[0]

        # Delta from baseline
        features[f"{col}_delta_baseline"] = float(cur_val - base_val)

        # Short window (e.g. last 4 points)
        short_vals = all_vals[-short_window_len:] if len(all_vals) >= short_window_len else all_vals
        features[f"{col}_mean_short"] = float(np.mean(short_vals))
        features[f"{col}_std_short"]  = float(np.std(short_vals)) if len(short_vals) > 1 else 0.0
        features[f"{col}_slope_short"] = compute_linear_slope(short_vals)
        features[f"{col}_delta_short"] = float(cur_val - short_vals[0])

        # Long window (e.g. last 12 points)
        long_vals = all_vals[-long_window_len:] if len(all_vals) >= long_window_len else all_vals
        features[f"{col}_mean_long"]  = float(np.mean(long_vals))
        features[f"{col}_std_long"]   = float(np.std(long_vals)) if len(long_vals) > 1 else 0.0
        features[f"{col}_slope_long"] = compute_linear_slope(long_vals)
        features[f"{col}_min_long"]   = float(np.min(long_vals))
        features[f"{col}_max_long"]   = float(np.max(long_vals))

        # Momentum: short mean vs long mean difference
        features[f"{col}_mean_diff_short_long"] = float(features[f"{col}_mean_short"] - features[f"{col}_mean_long"])

    # 3. Rolling derived metrics (Shock index slope & mean)
    si_vals = np.array([
        compute_derived_metrics(r)["shock_index"] for r in norm_history
    ], dtype=float)
    features["shock_index_mean_short"]  = float(np.mean(si_vals[-short_window_len:]))
    features["shock_index_slope_short"] = compute_linear_slope(si_vals[-short_window_len:])
    features["shock_index_slope_long"]  = compute_linear_slope(si_vals[-long_window_len:])
    features["shock_index_delta"]       = float(si_vals[-1] - si_vals[0])

    return features


def calculate_news2_score(norm_dict: Dict) -> int:
    """Standard NEWS2 calculation from normalized vital values."""
    score = 0
    rr = float(norm_dict.get("respiratory_rate_bpm", 16))
    if rr >= 25: score += 3
    elif 21 <= rr <= 24: score += 2
    elif 9 <= rr <= 11: score += 1
    elif rr <= 8: score += 3

    spo2 = float(norm_dict.get("spo2_percent", 98))
    if spo2 <= 91: score += 3
    elif 92 <= spo2 <= 93: score += 2
    elif 94 <= spo2 <= 95: score += 1

    temp = float(norm_dict.get("body_temperature_c", 37.0))
    if temp >= 39.1: score += 2
    elif 38.1 <= temp <= 39.0: score += 1
    elif 35.1 <= temp <= 36.0: score += 1
    elif temp <= 35.0: score += 3

    sbp = float(norm_dict.get("systolic_bp_mmHg", 120))
    if sbp <= 90: score += 3
    elif 91 <= sbp <= 100: score += 2
    elif 101 <= sbp <= 110: score += 1
    elif sbp >= 220: score += 3

    hr = float(norm_dict.get("heart_rate_bpm", 75))
    if hr >= 131: score += 3
    elif 111 <= hr <= 130: score += 2
    elif 91 <= hr <= 110: score += 1
    elif 41 <= hr <= 50: score += 1
    elif hr <= 40: score += 3

    cons_raw = norm_dict.get("consciousness", 0)
    try:
        cons_val = int(cons_raw) if (cons_raw is not None and not np.isnan(float(cons_raw))) else 0
    except (ValueError, TypeError):
        cons_val = 0

    if cons_val == 1:
        score += 3

    return score


def calculate_qsofa_score(norm_dict: Dict) -> int:
    """Standard qSOFA score (0-3)."""
    score = 0
    rr_val = float(norm_dict.get("respiratory_rate_bpm", 16) or 16)
    sbp_val = float(norm_dict.get("systolic_bp_mmHg", 120) or 120)
    if rr_val >= 22: score += 1
    if sbp_val <= 100: score += 1

    cons_raw = norm_dict.get("consciousness", 0)
    try:
        cons_val = int(cons_raw) if (cons_raw is not None and not np.isnan(float(cons_raw))) else 0
    except (ValueError, TypeError):
        cons_val = 0

    if cons_val == 1: score += 1
    return score


def build_lookahead_dataset_from_timeseries(
    df: pd.DataFrame,
    patient_id_col: str = "patient_id",
    lookahead_steps: int = 15,    # steps ahead for deterioration window
    min_history_steps: int = 4
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Converts multi-patient time-series DataFrame into feature matrix X and lookahead target y.
    
    Target Definition (Gap 1):
      y_t = 1 if patient deteriorates into acute risk (NEWS2 >= 6, qSOFA >= 2, or critical condition)
               within the lookahead window [t + 2, t + lookahead_steps].
      y_t = 0 otherwise.
    """
    records_X = []
    labels_y = []

    for pid, group in df.groupby(patient_id_col, sort=False):
        rows = group.to_dict("records")
        n = len(rows)
        if n < min_history_steps + 2:
            continue

        for t in range(min_history_steps, n - 2):
            window = rows[: t + 1]
            feat = extract_features_from_window(window)

            # Lookahead window: from t + 2 to t + lookahead_steps
            future_window = rows[t + 2 : min(n, t + lookahead_steps + 1)]
            
            deteriorated = False
            if future_window:
                for fut in future_window:
                    fut_norm = normalize_vital_keys(fut)
                    sep_val = fut.get("SepsisLabel", 0)
                    try:
                        if sep_val is not None and not np.isnan(float(sep_val)) and int(float(sep_val)) == 1:
                            deteriorated = True
                            break
                    except (ValueError, TypeError):
                        pass

                    c_state = str(fut.get("clinical_state", "")).lower()
                    if "critical" in c_state or "decompensated" in c_state or "septic shock" in c_state:
                        deteriorated = True
                        break
                    n2 = calculate_news2_score(fut_norm)
                    qs = calculate_qsofa_score(fut_norm)
                    if n2 >= 7 or qs >= 2:
                        deteriorated = True
                        break

            records_X.append(feat)
            labels_y.append(1 if deteriorated else 0)

    X_df = pd.DataFrame(records_X).fillna(0.0)
    y_ser = pd.Series(labels_y, name="target_deterioration_2_6h")
    return X_df, y_ser
