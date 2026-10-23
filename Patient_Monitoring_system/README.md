# AI-Based Patient Prioritization and Risk Prediction System

An intelligent healthcare system that combines clinical decision-support algorithms with machine learning to prioritize patients based on real-time vital signs and risk assessment.

## Table of Contents

- [Overview](#overview)
- [How to Reproduce (3 Research Gaps)](#how-to-reproduce-3-research-gaps)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Clinical Scoring Systems](#clinical-scoring-systems)
- [Machine Learning Models](#machine-learning-models)
- [Configuration](#configuration)

## How to Reproduce (3 Research Gaps)

All models and evaluations run on ONE real dataset: **PhysioNet/CinC Challenge 2019** (`data/physionet2019/*.psv`). No synthetic data is generated or used as fallback, and `patient*.csv` files are never mixed into training or evaluation.

### Quick Start (One Command Reproduction)

To execute the entire end-to-end pipeline (data preparation, Gap 1, Gap 2, Gap 3, and markdown report compilation):

```powershell
python make_report.py
```

This generates all metrics, 95% bootstrap confidence intervals, and publication-ready figures in `artifacts/`:
- `artifacts/split.json`: Frozen 70/15/15 patient-level stratified split (seed 42).
- `artifacts/features.parquet`: Vectorized rolling temporal feature cache.
- `artifacts/gap1_results.json`, `gap1_roc_pr.png`, `gap1_lead_time.png`
- `artifacts/gap2_results.json`, `gap2_shap_summary.png`, `gap2_fairness.png`
- `artifacts/gap3_results.json`, `gap3_policy_comparison.png`, `gap3_capacity_sweep.png`
- `artifacts/RESEARCH_GAPS_REPORT.md`: Comprehensive traceable Markdown report.

---

### Step-by-Step Execution Commands

If you wish to execute each research gap independently:

#### Step 0: Ingest Data & Engineer Rolling Features
```powershell
python pn_data.py
```
- Loads all `.psv` files from `data/physionet2019/`.
- Forward-fills vitals/labs, adds missingness indicators, and calculates `hours_since_last_lab`.
- Vectorizes 4h & 12h rolling means, standard deviations, OLS trajectory slopes, and baseline deltas.
- Enforces frozen patient-level 70/15/15 stratified split (`artifacts/split.json`).
- Caches features to `artifacts/features.parquet`.

#### Step 1: Gap 1 - Temporal Deterioration Prediction
```powershell
python train_gap1.py
```
- Trains on pre-onset hours ($t < t_{\text{onset}}$) with target $y_t = 1$ if sepsis occurs in $[t+2, t+6]$.
- Evaluates Temporal LightGBM against Static LightGBM, rule-based NEWS2, and rule-based qSOFA.
- Computes AUROC, AUPRC, and sensitivity at 90% validation specificity with 95% patient-bootstrap CIs.
- Measures advance warning lead time before clinical onset.

#### Step 2: Gap 2 - Trust, False-Alarm Filtering & Explainability
```powershell
python train_gap2.py
```
- Evaluates raw alarms ($P_{\text{temporal}} \ge \text{threshold}_{\text{val}}$).
- Trains second-stage False-Alarm Filter (LightGBM) to differentiate true imminent sepsis from premature alarms.
- Uses SHAP (`TreeExplainer`) to compute global importance and per-alert top-5 contributing features.
- Audits demographic fairness across Gender and Age bands (<50, 50-70, >70).

#### Step 3: Gap 3 - Triage and Resource Prioritization Under Constraints
```powershell
python triage_sim.py
```
- Simulates an ICU/ED ward under capacity constraints ($N$ beds, $K$ nurse reviews per hour).
- Benchmarks 4 policies: FIFO, Greedy-by-NEWS2, Greedy-by-ML, and Constraint-Aware Ranker.
- Sweeps $N \in [5, 20]$ and $K \in [1, 4]$ to evaluate policy robustness under staffing shortages.

#### Step 4: Launch Real-Time PhysioNet Web Dashboard
```powershell
python web_server.py
```
- Open `http://localhost:5000` in your web browser.
- Displays 3 dedicated tabs for Gap 1, Gap 2, and Gap 3.
- Streams real PhysioNet TEST patients hour by hour with live trajectory charts, SHAP explanation tags, and interactive $N$/$K$ ward capacity sliders.

---

### Data Leakage Prevention Statement

To ensure rigorous clinical validity:
1. **Pre-Onset Windowing**: Evaluated strictly on pre-onset hours ($t < t_{\text{onset}}$ where `SepsisLabel == 0` at hour $t$). Once sepsis onset occurs, early prediction has either succeeded or failed; including post-onset hours would cause trivial diagnostic leakage (artificially inflating AUROC > 0.98).
2. **Lookahead Separation**: Lookahead window begins at $t+2$ ($[t+2, t+6]$), enforcing a 2-hour buffer so the model must forecast deterioration rather than detect instantaneous abnormalities.
3. **Patient-Level Split**: Patient records are never shared between train, validation, and test splits. All rolling windows are strictly isolated within each patient.

## Features

- **Multi-Model Risk Assessment**: Combines clinical scores with ML predictions for comprehensive evaluation
- **Real-Time Vital Signs Monitoring**: Tracks heart rate, respiratory rate, SpO2, blood pressure, temperature, and consciousness level
- **Dynamic Prioritization**: Adjusts priority weights based on:
  - ICU bed availability
  - Alert fatigue metrics
  - Patient risk levels
  - Clinical stability
- **Web-Based Dashboard**: Real-time visualization of patient data, clinical scores, and priority rankings
- **Bayesian Updates**: Calculates posterior probabilities of critical status
- **Flexible Data Input**: Supports both CSV data and synthetic patient generation
- **Historical Analysis**: Tracks vital signs and scores over time

## System Architecture

### Core Components

1. **data_loader.py** - Loads patient data from CSV files
2. **random_generator.py** - Generates synthetic patient vital signs for simulation
3. **risk_prediction.py** - Machine learning models for risk classification
4. **patient_prioritization.py** - Priority queue and Bayesian calculations
5. **dynamic_weights.py** - Weight adjustment based on hospital conditions
6. **vitals_graph.py** - Visualization of vital signs trends
7. **web_server.py** - Flask-based web dashboard for real-time monitoring
8. **main.py** - Orchestrates the entire system

## Installation

### Requirements

- Python 3.7+
- Flask
- scikit-learn
- NumPy
- Matplotlib

### Setup

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install flask scikit-learn numpy matplotlib
   ```
3. Ensure CSV patient files (patient1.csv - patient5.csv) are in the project directory

## Usage

### Running the System

1. **Start the main system**:
   ```bash
   python main.py
   ```
   This will:
   - Load patient data from CSV files
   - Generate synthetic patients for simulation
   - Train ML models on vital signs data
   - Begin real-time patient monitoring and prioritization
   - Output priority queue and risk assessments

2. **Access the Web Dashboard**:
   ```bash
   python web_server.py
   ```
   Then open `index.html` in a web browser to view:
   - Real-time vital signs for all patients
   - Clinical scores (qSOFA, NEWS2)
   - ML risk predictions
   - Priority queue rankings
   - Historical charts

## Project Structure

```
AI_PROJECT1/
├── main.py                      # Main orchestrator
├── patient_prioritization.py    # Prioritization algorithms
├── risk_prediction.py           # ML models (LR, RF)
├── data_loader.py               # CSV data handling
├── random_generator.py          # Synthetic patient generation
├── dynamic_weights.py           # Dynamic weight adjustment
├── vitals_graph.py              # Visualization utilities
├── web_server.py                # Flask dashboard
├── index.html                   # Web interface
├── patient1.csv - patient5.csv  # Patient vital signs data
└── README.md                    # This file
```

## Clinical Scoring Systems

### qSOFA (Quick SOFA)
Rapid assessment for sepsis risk. Scores 0-3 based on:
- Respiratory rate ≥ 22 (1 point)
- Systolic blood pressure ≤ 100 mmHg (1 point)
- Altered consciousness (1 point)

**Clinical Use**: Quick screening for sepsis-related organ dysfunction

### NEWS2 (National Early Warning Score 2)
Comprehensive acuity score. Weighted scoring for:
- Respiratory rate
- SpO2 level
- Body temperature
- Systolic blood pressure
- Heart rate
- Consciousness level
- Oxygen supplementation

**Clinical Use**: Standardized assessment of patient acuity and deterioration risk

## Machine Learning Models

### Feature Set
- Respiratory rate (breaths per minute)
- Systolic blood pressure (mmHg)
- Heart rate (beats per minute)
- SpO2 (blood oxygen percentage)
- Body temperature (°C)
- Consciousness level (0/1)

### Risk Categories
- **Low**: Score 0-2 (managed on general ward)
- **Medium**: Score 3-5 (close monitoring, potential ICU)
- **High**: Score ≥6 (immediate ICU consideration)

### Model Performance
Both Logistic Regression and Random Forest are trained on patient data with cross-validation. Performance metrics are generated during training for model validation.

## Configuration

### Dynamic Weight Parameters (dynamic_weights.py)

Default weight distribution:
- **stability** (0.4): Patient physiological stability
- **response_time** (0.3): Urgency of intervention needed
- **resource_efficiency** (0.2): ICU bed utilization
- **excess_alert_frequency** (0.1): Alert fatigue management

### Bayesian Priors (patient_prioritization.py)
- Prior probability of critical status: 20%
- Likelihood ratios for risk assessment
- ICU bed availability tracking

### Thresholds
- ICU bed low threshold: ≤20% available
- ICU bed medium threshold: ≤50% available
- Alert fatigue high: >5 alerts per patient
- Alert fatigue medium: >2 alerts per patient

## Clinical Validation

This system is designed to complement clinical judgment, not replace it. All recommendations should be reviewed by qualified healthcare professionals. The combination of multiple assessment methods (clinical scores + ML) provides more robust decision support than any single approach.

## Future Enhancements

- Integration with hospital EHR systems
- Real-time vital signs from monitoring devices
- Improved alert fatigue models
- Additional clinical scoring systems
- Predictive deterioration modeling
- Multi-hospital resource coordination

## License

This is an academic project for AI coursework.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Note**: This system is a demonstration of AI in healthcare. Always follow institutional protocols and clinical guidelines when making patient care decisions.
