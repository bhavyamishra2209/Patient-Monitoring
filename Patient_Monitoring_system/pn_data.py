"""
pn_data.py
-----------
Step 0: Load all PhysioNet 2019 .psv files from data/physionet2019/*.psv.
No synthetic fallback, no patient*.csv mixing.
Engineers vectorized temporal rolling features (mean, std, slope, baseline delta, shock index, MAP, hours since last lab, missingness indicators).
Creates a frozen patient-level 70/15/15 stratified split saved to artifacts/split.json.
Caches engineered features to artifacts/features.parquet.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


PHYSIONET_DIR = "data/physionet2019"
ARTIFACTS_DIR = "artifacts"
SPLIT_FILE = os.path.join(ARTIFACTS_DIR, "split.json")
FEATURES_PARQUET = os.path.join(ARTIFACTS_DIR, "features.parquet")

# Mapping from PhysioNet column names to standard app column names
COL_MAP = {
    "HR": "heart_rate_bpm",
    "O2Sat": "spo2_percent",
    "Temp": "body_temperature_c",
    "SBP": "systolic_bp_mmHg",
    "DBP": "diastolic_bp_mmHg",
    "Resp": "respiratory_rate_bpm",
    "MAP": "mean_arterial_pressure",
}

CORE_VITALS = [
    "heart_rate_bpm",
    "systolic_bp_mmHg",
    "diastolic_bp_mmHg",
    "respiratory_rate_bpm",
    "body_temperature_c",
    "spo2_percent",
    "shock_index",
    "mean_arterial_pressure"
]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]

CLINICAL_DEFAULTS = {
    "heart_rate_bpm": 75.0,
    "spo2_percent": 98.0,
    "body_temperature_c": 37.0,
    "systolic_bp_mmHg": 120.0,
    "diastolic_bp_mmHg": 80.0,
    "respiratory_rate_bpm": 16.0,
    "mean_arterial_pressure": 85.0,
    "shock_index": 75.0 / 120.0,
    "Age": 60.0,
    "Gender": 0.0,
    "Unit1": 0.0,
    "Unit2": 0.0,
    "HospAdmTime": -10.0,
    "ICULOS": 1.0,
}


def calc_ols_slope(y: np.ndarray) -> float:
    """Computes ordinary least squares linear slope over equally spaced time steps."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - np.mean(y))) / denom)


def compute_news2(row: pd.Series) -> int:
    """Computes NEWS2 score from patient vitals."""
    score = 0
    rr = float(row.get("respiratory_rate_bpm", 16.0))
    if rr >= 25 or rr <= 8:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    spo2 = float(row.get("spo2_percent", 98.0))
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    temp = float(row.get("body_temperature_c", 37.0))
    if temp >= 39.1 or temp <= 35.0:
        score += 3 if temp <= 35.0 else 2
    elif 38.1 <= temp <= 39.0 or 35.1 <= temp <= 36.0:
        score += 1

    sbp = float(row.get("systolic_bp_mmHg", 120.0))
    if sbp <= 90 or sbp >= 220:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110:
        score += 1

    hr = float(row.get("heart_rate_bpm", 75.0))
    if hr >= 131 or hr <= 40:
        score += 3
    elif 111 <= hr <= 130:
        score += 2
    elif (91 <= hr <= 110) or (41 <= hr <= 50):
        score += 1

    return score


def compute_qsofa(row: pd.Series) -> int:
    """Computes qSOFA score (0-3)."""
    score = 0
    if float(row.get("respiratory_rate_bpm", 16.0)) >= 22:
        score += 1
    if float(row.get("systolic_bp_mmHg", 120.0)) <= 100:
        score += 1
    return score


def load_raw_physionet_data(data_dir: str = PHYSIONET_DIR) -> pd.DataFrame:
    """Loads all .psv files from data_dir. Raises FileNotFoundError if none found."""
    pattern = os.path.join(data_dir, "*.psv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No PhysioNet 2019 .psv files found in '{data_dir}'. "
            "Synthetic data fallback is disabled per strict research requirements."
        )

    dfs = []
    for fp in files:
        pid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, sep='|')
        df["patient_id"] = pid
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def create_or_load_split(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Creates or loads frozen patient-level 70/15/15 split stratified on 'patient ever septic'.
    Saves to artifacts/split.json.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            split = json.load(f)
            print(f"[pn_data] Loaded existing split from {SPLIT_FILE}")
            return split

    patient_septic = df.groupby("patient_id")["SepsisLabel"].max()
    septic_pids = sorted(patient_septic[patient_septic == 1].index.tolist())
    non_septic_pids = sorted(patient_septic[patient_septic == 0].index.tolist())

    rng = np.random.RandomState(42)
    perm_sep = rng.permutation(septic_pids).tolist()
    perm_non = rng.permutation(non_septic_pids).tolist()

    n_sep = len(perm_sep)
    n_non = len(perm_non)

    # 70% train, 15% val, 15% test — uses floor to guarantee test gets remainder
    n_sep_train = max(1, int(n_sep * 0.70))
    n_sep_val   = max(1, int(n_sep * 0.15)) if n_sep > 2 else (1 if n_sep > 1 else 0)
    n_non_train = max(1, int(n_non * 0.70))
    n_non_val   = max(1, int(n_non * 0.15)) if n_non > 2 else (1 if n_non > 1 else 0)

    train_pids = sorted(perm_sep[:n_sep_train] + perm_non[:n_non_train])
    val_pids = sorted(perm_sep[n_sep_train:n_sep_train + n_sep_val] + perm_non[n_non_train:n_non_train + n_non_val])
    test_pids = sorted(perm_sep[n_sep_train + n_sep_val:] + perm_non[n_non_train + n_non_val:])

    split = {
        "train": train_pids,
        "val": val_pids,
        "test": test_pids,
        "stratification": "patient_ever_septic",
        "random_seed": 42,
        "n_patients": len(patient_septic),
        "n_septic": n_sep,
        "n_non_septic": n_non
    }

    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)

    print(f"[pn_data] Created & saved frozen split to {SPLIT_FILE}")
    print(f"  Train: {len(train_pids)} (septic: {len(perm_sep[:n_sep_train])})")
    print(f"  Val:   {len(val_pids)} (septic: {len(perm_sep[n_sep_train:n_sep_train+n_sep_val])})")
    print(f"  Test:  {len(test_pids)} (septic: {len(perm_sep[n_sep_train+n_sep_val:])})")
    return split


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers vectorized temporal rolling features:
    - Forward-fill vitals/labs within each patient.
    - Missingness indicator columns.
    - Hours since last lab.
    - Shock index = HR / SBP, MAP, pulse pressure.
    - 4h and 12h rolling mean, std, slope, min, max.
    - Baseline delta.
    - Target: y_t = 1 if SepsisLabel==1 at any hour in [t+2, t+6].
    """
    df = df_raw.copy()
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)

    # 1. Missingness indicators before forward fill
    for c in list(COL_MAP.keys()) + LAB_COLS:
        if c in df.columns:
            df[f"{c}_isna"] = df[c].isna().astype(float)

    # 2. Hours since last lab calculation
    available_labs = [c for c in LAB_COLS if c in df.columns]
    if available_labs:
        has_any_lab = (~df[available_labs].isna()).any(axis=1)
        # Vectorized cumsum grouping within patient
        df["_lab_seen"] = has_any_lab
        hours_since = []
        for pid, group in df.groupby("patient_id", sort=False):
            hrs = 0
            for seen in group["_lab_seen"]:
                if seen:
                    hrs = 0
                else:
                    hrs += 1
                hours_since.append(hrs)
        df["hours_since_last_lab"] = hours_since
        df.drop(columns=["_lab_seen"], inplace=True)
    else:
        df["hours_since_last_lab"] = 0.0

    # 3. Rename PhysioNet columns to standard app names
    df.rename(columns=COL_MAP, inplace=True)

    # 4. Forward-fill within patient, then backfill, then clinical defaults
    all_numeric_cols = [c for c in df.columns if c not in ["patient_id", "SepsisLabel"]]
    for col in all_numeric_cols:
        df[col] = df.groupby("patient_id")[col].ffill()
        df[col] = df.groupby("patient_id")[col].bfill()
        default_val = CLINICAL_DEFAULTS.get(col, 0.0)
        df[col] = df[col].fillna(default_val)

    # 5. Clinical Derived Features
    safe_sbp = np.maximum(df["systolic_bp_mmHg"].values, 10.0)
    df["shock_index"] = df["heart_rate_bpm"].values / safe_sbp
    df["pulse_pressure"] = np.maximum(0.0, df["systolic_bp_mmHg"].values - df["diastolic_bp_mmHg"].values)
    if "mean_arterial_pressure" not in df.columns or df["mean_arterial_pressure"].isna().all():
        df["mean_arterial_pressure"] = df["diastolic_bp_mmHg"].values + (df["pulse_pressure"].values / 3.0)

    # 6. Baseline Delta & Vectorized Rolling Features (4h and 12h)
    for col in CORE_VITALS:
        # Admission baseline delta
        baseline = df.groupby("patient_id")[col].transform("first")
        df[f"{col}_delta_baseline"] = df[col] - baseline

        # 4h rolling stats
        roll4 = df.groupby("patient_id")[col].rolling(4, min_periods=1)
        df[f"{col}_mean_4h"] = roll4.mean().values
        df[f"{col}_std_4h"]  = roll4.std().fillna(0.0).values
        # 4h slope
        roll4_slope = df.groupby("patient_id")[col].rolling(4, min_periods=2).apply(calc_ols_slope, raw=True)
        df[f"{col}_slope_4h"] = roll4_slope.fillna(0.0).values

        # 12h rolling stats
        roll12 = df.groupby("patient_id")[col].rolling(12, min_periods=1)
        df[f"{col}_mean_12h"] = roll12.mean().values
        df[f"{col}_std_12h"]  = roll12.std().fillna(0.0).values
        df[f"{col}_min_12h"]  = roll12.min().values
        df[f"{col}_max_12h"]  = roll12.max().values
        # 12h slope
        roll12_slope = df.groupby("patient_id")[col].rolling(12, min_periods=2).apply(calc_ols_slope, raw=True)
        df[f"{col}_slope_12h"] = roll12_slope.fillna(0.0).values

    # 7. Rule-based scores for baseline benchmarking
    df["news2_score"] = df.apply(compute_news2, axis=1)
    df["qsofa_score"] = df.apply(compute_qsofa, axis=1)

    # 8. Target Construction:
    # y_t = 1 if SepsisLabel==1 at any hour in [t+2, t+6].
    # Evaluated on pre-onset hours (SepsisLabel==0 at hour t) to avoid post-onset leakage.
    target_list = []
    is_pre_onset_list = []
    hours_to_onset_list = []

    for pid, group in df.groupby("patient_id", sort=False):
        labels = group["SepsisLabel"].values
        culos = group["ICULOS"].values
        n = len(labels)
        onset_idx = np.where(labels == 1)[0]
        first_onset_hr = culos[onset_idx[0]] if len(onset_idx) > 0 else np.nan

        for i in range(n):
            cur_hr = culos[i]
            is_pre = (labels[i] == 0)
            is_pre_onset_list.append(is_pre)

            # Lookahead window [t+2, t+6]
            future_slice = labels[i + 2 : min(n, i + 7)]
            deteriorated = 1 if (len(future_slice) > 0 and np.any(future_slice == 1)) else 0
            target_list.append(deteriorated)

            if not np.isnan(first_onset_hr):
                hours_to_onset_list.append(float(first_onset_hr - cur_hr))
            else:
                hours_to_onset_list.append(np.nan)

    df["target_deterioration"] = target_list
    df["is_pre_onset"] = is_pre_onset_list
    df["hours_to_onset"] = hours_to_onset_list

    return df


def prepare_and_cache_dataset() -> pd.DataFrame:
    """Loads PhysioNet, engineers features, assigns split, and caches to parquet."""
    print("[pn_data] Loading raw PhysioNet 2019 dataset...")
    df_raw = load_raw_physionet_data()
    print(f"[pn_data] Loaded {df_raw['patient_id'].nunique()} patients ({len(df_raw)} total hourly records).")

    print("[pn_data] Engineering vectorized temporal features...")
    df_feats = engineer_features(df_raw)

    split = create_or_load_split(df_feats)
    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])

    def assign_split(pid):
        if pid in train_set:
            return "train"
        if pid in val_set:
            return "val"
        if pid in test_set:
            return "test"
        return "unknown"

    df_feats["split"] = df_feats["patient_id"].apply(assign_split)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df_feats.to_parquet(FEATURES_PARQUET, index=False)
    print(f"[pn_data] Successfully cached {len(df_feats)} records to {FEATURES_PARQUET}")
    return df_feats


def load_cached_features() -> pd.DataFrame:
    """Loads cached features parquet if available, otherwise generates it."""
    if os.path.exists(FEATURES_PARQUET):
        return pd.read_parquet(FEATURES_PARQUET)
    return prepare_and_cache_dataset()


if __name__ == "__main__":
    df = prepare_and_cache_dataset()
    print("\nDataset summary:")
    print("Columns:", len(df.columns))
    print("Split counts:\n", df.groupby("split")["patient_id"].nunique())
    print("Target distribution (pre-onset only):")
    pre = df[df["is_pre_onset"]]
    print(pre.groupby(["split", "target_deterioration"]).size())
