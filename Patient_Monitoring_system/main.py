import time
import heapq
import glob
from data_loader import CSVPatient
from random_generator import RandomPatient
from risk_prediction import RiskPredictor
from dynamic_weights import get_weights_for_patient


# qSOFA
def calculate_qsofa(patient):
    score = 0
    if patient["resp_rate"] >= 22:
        score += 1
    if patient["sbp"] <= 100:
        score += 1
    if patient["consciousness"] == 1:
        score += 1
    return score



# NEWS2
def calculate_news2(patient):
    score = 0

    if patient["resp_rate"] >= 25:
        score += 3
    elif 21 <= patient["resp_rate"] <= 24:
        score += 2

    if patient["spo2"] <= 91:
        score += 3
    elif 92 <= patient["spo2"] <= 93:
        score += 2
    elif 94 <= patient["spo2"] <= 95:
        score += 1

    if patient["temperature"] >= 39:
        score += 2
    elif patient["temperature"] <= 35:
        score += 3

    if patient["sbp"] <= 90:
        score += 3
    elif 91 <= patient["sbp"] <= 100:
        score += 2

    if patient["heart_rate"] >= 131:
        score += 3
    elif 111 <= patient["heart_rate"] <= 130:
        score += 2

    if patient["consciousness"] == 1:
        score += 3

    return score


# AI Utility Model 

# the weights correspond to the four terms in the new utility function
weights = {
    "stability": 0.4,              # w1
    "response_time": 0.3,          # w2
    "resource_efficiency": 0.2,    # w3
    "excess_alert_frequency": 0.1, # w4
}

# Bayesian Parameters

# Prior probability that any monitored patient is critical
PRIOR_CRITICAL = 0.2  

# estimates
LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL = 0.85
LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRITICAL = 0.30


alert_fatigue = {}
environment_value = 0.5  # Example ICU stress factor 

# ML Model Initialization and Training
predictor = RiskPredictor(model_type="random_forest")
csv_files = glob.glob("patient*.csv")
if csv_files:
    print("[System] Training ML Risk Predictor on CSV files...")
    predictor.train(csv_files)
    print("[System] ML model training complete.")
else:
    print("[Warning] No patient CSV files found. Model will be untrained.")


def compute_vital_risk(qsofa, news2):
    """Original vital risk helper left in place for backwards compatibility.
    We can reuse it when deriving stability and response-time proxies.
    """
    return (qsofa / 3.0 + news2 / 20.0) / 2.0


# Bayesian Probability Update

def bayesian_update(is_high_risk: bool):
    """
    Computes P(Critical | Evidence) using Bayes Rule.
    Evidence = whether patient is high-risk based on qSOFA/NEWS2
    """

    if is_high_risk:
        likelihood = LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL
        evidence_prob = (
            LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL * PRIOR_CRITICAL
            + LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRITICAL * (1 - PRIOR_CRITICAL)
        )
    else:
        # Complement probabilities
        likelihood = 1 - LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL
        evidence_prob = (
            (1 - LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL) * PRIOR_CRITICAL
            + (1 - LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRITICAL) * (1 - PRIOR_CRITICAL)
        )

    posterior = (likelihood * PRIOR_CRITICAL) / evidence_prob
    return posterior


def compute_utility(patient_id: int,
                    stability: float,
                    response_time: float,
                    resource_efficiency: float,
                    excess_alert_freq: float,
                    weights_dict: dict = None) -> float:
    """Compute utility using the new maximization function:

    U = w1*(Stability) + w2*(1 - Response Time)
        + w3*(Resource Efficiency) - w4*(Excess Alert Frequency)
    
    Note: Response time is inverted so that higher urgency (high NEWS2) 
    increases utility rather than decreasing it.
    Utility is scaled to be more meaningful (0-1 range).
    """
    w = weights_dict if weights_dict else weights
    
    # Invert response_time so high urgency increases utility
    inverted_response_time = 1.0 - response_time
    
    utility = (
        w["stability"] * stability
        + w["response_time"] * inverted_response_time
        + w["resource_efficiency"] * resource_efficiency
        - w["excess_alert_frequency"] * excess_alert_freq
    )
    
    # Scale utility to be more meaningful
    # Typical range would be around 0.5-1.5 with the above components
    return utility


# Initialize Patients

csv_patients = [
    CSVPatient("patient1.csv", 1),
    CSVPatient("patient2.csv", 2),
    CSVPatient("patient3.csv", 3),
    CSVPatient("patient4.csv", 4),
    CSVPatient("patient5.csv", 5),
]

random_patients = [
    RandomPatient(6),
    RandomPatient(7),
    RandomPatient(8),
    RandomPatient(9),
    RandomPatient(10),
]


# GREEDY BEST-FIRST SEARCH
def prioritize_patients(patient_list):
    """
    Implements Greedy Best-First Search.
    Uses utility score as heuristic h(n).
    """
    priority_queue = []
    
    # Create a mapping for quick lookup
    patient_map = {p["patient_id"]: p for p in patient_list}

    for patient in patient_list:
        utility = patient["utility"]
        patient_id = patient["patient_id"]

        # Use negative utility because heapq is min-heap
        # Only store utility and patient_id (not the dict) to avoid comparison issues
        heapq.heappush(priority_queue, (-utility, patient_id))

    ordered_patients = []

    while priority_queue:
        _, patient_id = heapq.heappop(priority_queue)
        ordered_patients.append(patient_map[patient_id])

    return ordered_patients

