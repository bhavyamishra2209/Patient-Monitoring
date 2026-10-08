import csv

class CSVPatient:
    def __init__(self, file_path, patient_id):
        self.file_path = file_path
        self.patient_id = patient_id
        self.data = []
        self.index = 0
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sbp_val = int(float(row.get("systolic_bp_mmHg", 120)))
                dbp_val = row.get("diastolic_bp_mmHg")
                dbp_int = int(float(dbp_val)) if dbp_val not in (None, "") else max(40, int(sbp_val * 0.65))
                lactate_val = row.get("lactate")
                lactate_flt = float(lactate_val) if lactate_val not in (None, "") else 1.1

                self.data.append({
                    "patient_id": self.patient_id,
                    "resp_rate": int(float(row.get("respiratory_rate_bpm", 16))),
                    "sbp": sbp_val,
                    "diastolic_bp_mmHg": dbp_int,
                    "heart_rate": int(float(row.get("heart_rate_bpm", 75))),
                    "spo2": float(row.get("spo2_percent", 98.0)),
                    "temperature": float(row.get("body_temperature_c", 37.0)),
                    "consciousness": int(float(row.get("consciousness", 0))),
                    "lactate": lactate_flt,
                    "clinical_state": row.get("clinical_state", "Monitored")
                })

    def get_next_vital(self):
        if self.index < len(self.data):
            vital = self.data[self.index]
            self.index += 1
            return vital
        else:
            return None
