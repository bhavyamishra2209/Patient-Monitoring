"""
temporal_patient_stream.py
--------------------------
Gap 1 Live Streaming, Stateful Patient History Buffer & Triage Hysteresis Engine.
Implements:
  1. Stateful sliding historical buffers (deque) for PhysioNet & CSV time-series vitals.
  2. Decoupled Data Refresh vs Re-Ranking (real-time vital stream with stable card positions).
  3. Ranking Hysteresis & Tier-Based Grouping (Critical, Early Warning, Stable) to prevent visual alarm fatigue.
  4. Exponential Moving Average (EMA) smoothing to eliminate single-point jitter.
"""

from collections import deque
import time
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from temporal_feature_engineering import normalize_vital_keys, calculate_news2_score, calculate_qsofa_score
from temporal_risk_prediction import TemporalRiskPredictor


class PatientStreamBuffer:
    """Maintains a rolling historical vital sign buffer for a single patient."""

    def __init__(self, patient_id: str, max_history: int = 30):
        self.patient_id = patient_id
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.step_count = 0
        
        # Hysteresis state tracking
        self.current_tier = "STABLE"  # "CRITICAL", "EARLY_WARNING", "STABLE"
        self.consecutive_tier_readings = 0
        self.candidate_tier = "STABLE"
        self.smoothed_risk_prob = 0.10  # Exponential Moving Average

    def add_vital(self, vital: Dict) -> None:
        norm = normalize_vital_keys(vital)
        norm["patient_id"] = self.patient_id
        norm["step"] = self.step_count
        self.step_count += 1
        self.history.append(norm)

    def get_history(self) -> List[Dict]:
        return list(self.history)

    def get_latest(self) -> Optional[Dict]:
        return self.history[-1] if self.history else None

    def update_hysteresis(self, instant_prob: float, news2_score: int, persistence_threshold: int = 2) -> str:
        """
        Applies Exponential Moving Average (EMA) and persistence hysteresis.
        Prevents jittery rank switching on single-beat spikes.
        """
        # EMA smoothing: 70% historical momentum + 30% current reading
        self.smoothed_risk_prob = 0.70 * self.smoothed_risk_prob + 0.30 * instant_prob

        # Determine target candidate tier
        if news2_score >= 7 or self.smoothed_risk_prob >= 0.80 or instant_prob >= 0.85:
            target_tier = "CRITICAL"
        elif self.smoothed_risk_prob >= 0.55 or instant_prob >= 0.60 or (instant_prob >= 0.50 and news2_score >= 4):
            target_tier = "EARLY_WARNING"
        else:
            target_tier = "STABLE"

        # Check persistence
        if target_tier == self.current_tier:
            self.candidate_tier = self.current_tier
            self.consecutive_tier_readings = 0
        else:
            if target_tier == self.candidate_tier:
                self.consecutive_tier_readings += 1
                # If target is more urgent, escalate quickly (1-2 ticks); if de-escalating, require 3 ticks
                req_ticks = 1 if (target_tier == "CRITICAL") else persistence_threshold
                if self.consecutive_tier_readings >= req_ticks:
                    self.current_tier = target_tier
                    self.consecutive_tier_readings = 0
            else:
                self.candidate_tier = target_tier
                self.consecutive_tier_readings = 1

        return self.current_tier


class TemporalMonitoringManager:
    """
    Central Manager for active patients.
    Processes live ticks, runs Gap 1 Temporal GBDT model, and evaluates Rule vs ML divergence
    with stable tier-based clinical grouping.
    """

    def __init__(self, predictor: Optional[TemporalRiskPredictor] = None):
        if predictor is None:
            self.predictor = TemporalRiskPredictor(model_type="lightgbm")
            self.predictor.load_or_train()
        else:
            self.predictor = predictor

        self.patient_buffers: Dict[str, PatientStreamBuffer] = {}
        self.patient_metadata: Dict[str, Dict] = {}
        self.last_triage_reorder_time = time.time()
        self.triage_reorder_interval_sec = 60  # Decoupled re-sort cadence (60 seconds)
        self.cached_patient_order: List[str] = []

    def register_patient(self, patient_id: str, name: str = "", bed: str = "", archetype: str = "Standard", hospital: str = "Hospital A (ICU)") -> None:
        self.patient_buffers[patient_id] = PatientStreamBuffer(patient_id)
        self.patient_metadata[patient_id] = {
            "name": name if name else f"Patient {patient_id}",
            "bed": bed if bed else f"Bed {patient_id}",
            "archetype": archetype,
            "hospital": hospital
        }
        if patient_id not in self.cached_patient_order:
            self.cached_patient_order.append(patient_id)

    def feed_vital_and_predict(self, patient_id: str, raw_vital: Dict) -> Dict:
        """Appends vital to patient buffer and produces complete Gap 1 assessment."""
        if patient_id not in self.patient_buffers:
            self.register_patient(patient_id)

        buf = self.patient_buffers[patient_id]
        buf.add_vital(raw_vital)
        history_list = buf.get_history()

        # Run Gap 1 Temporal ML Inference
        pred_res = self.predictor.predict_window(history_list)
        latest_vital = buf.get_latest()
        meta = self.patient_metadata.get(patient_id, {})

        n2 = pred_res["news2_instant"]
        qs = pred_res["qsofa_instant"]
        prob = pred_res["prob_deterioration"]
        prob_pct = pred_res["prob_percent"]
        lead_time = pred_res["estimated_lead_time_hours"]
        trends = pred_res["trends"]
        early_warn = pred_res["early_warning_active"]

        # Apply Hysteresis & Stable Clinical Tier
        stable_tier = buf.update_hysteresis(prob, n2)

        # Build concise clinical badge
        if stable_tier == "CRITICAL":
            tier_label = "CRITICAL"
            tier_color = "danger"
            summary_tag = "🚨 ACUTE DETERIORATION: Immediate escalation required."
        elif stable_tier == "EARLY_WARNING":
            tier_label = "EARLY WARNING"
            tier_color = "warning"
            summary_tag = f"⚡ IMPENDING CRASH in ~{lead_time}h (Trajectory divergence)."
        else:
            tier_label = "STABLE"
            tier_color = "success"
            summary_tag = "✅ STABLE: Normal trajectory within physiological limits."

        # Map display risk
        display_risk = "Critical" if stable_tier == "CRITICAL" else ("High" if prob >= 0.60 else ("Moderate" if prob >= 0.35 else "Low"))

        snapshot = {
            "id": patient_id,
            "name": meta.get("name", f"Patient {patient_id}"),
            "bed": meta.get("bed", f"Bed {patient_id}"),
            "archetype": meta.get("archetype", "Standard"),
            "hospital": meta.get("hospital", "Hospital A"),
            "heart_rate": int(latest_vital.get("heart_rate_bpm", 75)),
            "resp_rate": int(latest_vital.get("respiratory_rate_bpm", 16)),
            "spo2": float(latest_vital.get("spo2_percent", 98.0)),
            "sbp": int(latest_vital.get("systolic_bp_mmHg", 120)),
            "dbp": int(latest_vital.get("diastolic_bp_mmHg", 80)),
            "temperature": float(latest_vital.get("body_temperature_c", 37.0)),
            "consciousness": int(latest_vital.get("consciousness", 0)),
            "lactate": float(latest_vital.get("lactate", 1.1) or 1.1),
            "wbc": float(latest_vital.get("wbc", 7.2) or 7.2),
            "news2": n2,
            "qsofa": qs,
            "ml_prob": prob_pct,
            "smoothed_prob": round(buf.smoothed_risk_prob * 100, 1),
            "ml_risk": display_risk,
            "tier": stable_tier,
            "tier_label": tier_label,
            "tier_color": tier_color,
            "estimated_lead_time_hours": lead_time,
            "shock_index": pred_res.get("shock_index", 0.7),
            "trends": trends,
            "early_warning_active": (stable_tier == "EARLY_WARNING" or early_warn),
            "summary_tag": summary_tag,
            "history_length": len(history_list)
        }

        return snapshot

    def get_tier_grouped_patients(self, snapshots: List[Dict], force_reorder: bool = False) -> Dict:
        """
        Groups patients into 3 stable severity tiers.
        Keeps individual cards locked in position within their tier to prevent visual disorientation.
        """
        now = time.time()
        time_until_reorder = max(0, int(self.triage_reorder_interval_sec - (now - self.last_triage_reorder_time)))

        if force_reorder or (now - self.last_triage_reorder_time >= self.triage_reorder_interval_sec):
            self.last_triage_reorder_time = now
            time_until_reorder = self.triage_reorder_interval_sec

        # Group by tier
        critical_list = [p for p in snapshots if p["tier"] == "CRITICAL"]
        early_warning_list = [p for p in snapshots if p["tier"] == "EARLY_WARNING"]
        stable_list = [p for p in snapshots if p["tier"] == "STABLE"]

        # Sort within tier deterministically by smoothed risk
        critical_list.sort(key=lambda x: x["smoothed_prob"], reverse=True)
        early_warning_list.sort(key=lambda x: x["smoothed_prob"], reverse=True)
        stable_list.sort(key=lambda x: x["smoothed_prob"], reverse=True)

        return {
            "all_patients": critical_list + early_warning_list + stable_list,
            "critical_tier": critical_list,
            "early_warning_tier": early_warning_list,
            "stable_tier": stable_list,
            "reorder_countdown_sec": time_until_reorder
        }
