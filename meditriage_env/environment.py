import pandas as pd
import numpy as np
import random
import uuid
import os
from typing import Tuple, Dict, List
from .models import MedicalObservation, TriageAction, TriageState

class MedicalDataHandler:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        
        print("📂 Loading MediAssist data...")
        
        # Load diseases
        self.diseases = pd.read_csv(os.path.join(base_path, "dia_3.csv"))
        self.diseases['_id'] = pd.to_numeric(self.diseases['_id'], errors='coerce')
        self.diseases = self.diseases.dropna(subset=['_id'])
        self.diseases['_id'] = self.diseases['_id'].astype(int)
        self.diseases.set_index('_id', inplace=True)

        # Load symptoms
        self.symptoms = pd.read_csv(os.path.join(base_path, "symptoms2.csv"))
        sym_id_col = '_id' if '_id' in self.symptoms.columns else 'id'
        self.symptoms[sym_id_col] = pd.to_numeric(self.symptoms[sym_id_col], errors='coerce')
        self.symptoms = self.symptoms.dropna(subset=[sym_id_col])
        self.symptoms[sym_id_col] = self.symptoms[sym_id_col].astype(int)
        self.symptoms.set_index(sym_id_col, inplace=True)

        # Load symptom-disease matrix
        self.matrix = pd.read_csv(os.path.join(base_path, "sym_dis_matrix.csv"), index_col=0)
        self.matrix.index = pd.to_numeric(self.matrix.index, errors='coerce')
        self.matrix = self.matrix.loc[~self.matrix.index.isna()].copy()
        self.matrix.index = self.matrix.index.astype(int)
        self.matrix.columns = pd.to_numeric(pd.Index(self.matrix.columns), errors='coerce')
        self.matrix = self.matrix.loc[:, ~pd.isna(self.matrix.columns)].copy()
        self.matrix.columns = self.matrix.columns.astype(int)

        print(f"✅ Loaded: {len(self.diseases)} diseases, {len(self.symptoms)} symptoms")

        # Critical disease keywords for triage
        self.critical_keywords = [
            "heart", "stroke", "sepsis", "hypertension", "asthma",
            "attack", "failure", "emergency", "critical", "severe",
            "hemorrhage", "shock", "trauma", "poisoning", "overdose"
        ]

    def get_symptoms_for_disease(self, disease_id: int) -> List[str]:
        """Get real symptoms for a disease from the matrix."""
        if disease_id not in self.matrix.columns:
            return ["Generalized weakness", "Fatigue"]
        
        # Get symptom IDs that are linked to this disease
        sym_col = self.matrix[disease_id]
        linked_sym_ids = sym_col[sym_col > 0].index.tolist()
        
        if not linked_sym_ids:
            return ["Generalized weakness", "Fatigue"]
        
        # Get symptom names
        sym_names = []
        for sym_id in linked_sym_ids[:5]:  # Max 5 symptoms
            if sym_id in self.symptoms.index:
                name_col = 'name' if 'name' in self.symptoms.columns else self.symptoms.columns[0]
                sym_names.append(str(self.symptoms.loc[sym_id, name_col]))
        
        return sym_names if sym_names else ["Generalized weakness", "Fatigue"]

    def get_case(self, difficulty: str = "easy") -> Dict:
        """Generate a patient case based on difficulty."""
        
        if difficulty == "easy":
            # Easy: clearly critical OR clearly non-critical
            use_critical = random.random() < 0.5
            if use_critical:
                # Find a critical disease
                critical_diseases = self.diseases[
                    self.diseases['diagnose'].str.lower().str.contains(
                        '|'.join(self.critical_keywords), na=False
                    )
                ]
                if len(critical_diseases) > 0:
                    row = critical_diseases.sample(n=1).iloc[0]
                    disease_id = critical_diseases.sample(n=1).index[0]
                else:
                    disease_id = self.diseases.sample(n=1).index[0]
                    row = self.diseases.loc[disease_id]
            else:
                # Non-critical disease
                non_critical = self.diseases[
                    ~self.diseases['diagnose'].str.lower().str.contains(
                        '|'.join(self.critical_keywords), na=False
                    )
                ]
                if len(non_critical) > 0:
                    disease_id = non_critical.sample(n=1).index[0]
                    row = self.diseases.loc[disease_id]
                else:
                    disease_id = self.diseases.sample(n=1).index[0]
                    row = self.diseases.loc[disease_id]
        else:
            # Medium/Hard: random disease
            disease_id = self.diseases.sample(n=1).index[0]
            row = self.diseases.loc[disease_id]

        disease_name = str(row['diagnose'])
        is_critical = any(k in disease_name.lower() for k in self.critical_keywords)

        # Get real symptoms
        symptoms = self.get_symptoms_for_disease(disease_id)

        # Generate vitals based on criticality
        if is_critical:
            heart_rate = random.randint(110, 160)
            temp = round(random.uniform(38.5, 40.5), 1)
            bp_systolic = random.randint(160, 200) if random.random() < 0.5 else random.randint(70, 90)
        else:
            heart_rate = random.randint(60, 100)
            temp = round(random.uniform(36.5, 37.5), 1)
            bp_systolic = random.randint(110, 130)

        # Hard difficulty: add resource scarcity
        if difficulty == "hard":
            icu_beds = random.randint(0, 1)
            ambulances = random.randint(0, 1)
        elif difficulty == "medium":
            icu_beds = random.randint(1, 3)
            ambulances = random.randint(1, 2)
        else:
            icu_beds = random.randint(2, 5)
            ambulances = random.randint(1, 3)

        return {
            "disease_id": disease_id,
            "disease": disease_name,
            "symptoms": symptoms,
            "vitals": {
                "heart_rate": float(heart_rate),
                "temp": float(temp),
                "bp_systolic": float(bp_systolic)
            },
            "is_critical": is_critical,
            "difficulty": difficulty,
            "resources": {
                "icu_beds": icu_beds,
                "ambulances": ambulances
            }
        }


class MedicalTriageEnv:
    def __init__(self):
        self.data_handler = MedicalDataHandler()
        self.current_case = None
        self.state = None
        self.difficulty = "easy"

    def reset(self, difficulty: str = "easy") -> MedicalObservation:
        """Start a new patient episode."""
        self.difficulty = difficulty
        self.current_case = self.data_handler.get_case(difficulty=difficulty)
        self.state = TriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty
        )

        print(f"✨ New Case [{difficulty}]: {self.current_case['disease']}")
        print(f"   Symptoms: {self.current_case['symptoms'][:3]}")
        print(f"   Critical: {self.current_case['is_critical']}")

        return MedicalObservation(
            patient_id=f"PAT-{random.randint(1000, 9999)}",
            symptoms=self.current_case['symptoms'],
            vitals=self.current_case['vitals'],
            hospital_resources=self.current_case['resources'],
            difficulty=difficulty,
            message=f"New patient arrived. Assess and triage appropriately."
        )

    def step(self, action: TriageAction) -> Tuple[None, float, bool, dict]:
        """Process triage decision and return reward."""
        if self.current_case is None:
            raise ValueError("Call reset() before step()!")

        self.state.step_count += 1
        is_critical = self.current_case['is_critical']
        difficulty = self.current_case['difficulty']
        resources = self.current_case['resources']

        reward = self._calculate_reward(action, is_critical, difficulty, resources)

        self.state.is_done = True
        self.state.last_reward = reward

        return None, reward, True, {
            "actual_condition": self.current_case['disease'],
            "was_critical": is_critical,
            "correct_priority": 1 if is_critical else 3,
            "your_priority": action.priority_level,
            "difficulty": difficulty,
            "feedback": self._get_feedback(action, is_critical, reward)
        }

    def _calculate_reward(
        self, 
        action: TriageAction, 
        is_critical: bool, 
        difficulty: str,
        resources: dict
    ) -> float:
        """
        Meaningful reward function with partial progress signals.
        Not just binary — rewards partial correctness too.
        """
        reward = 0.0

        if difficulty == "easy":
            # Easy: Binary — correct or not
            if is_critical and action.priority_level == 1:
                reward = 1.0  # Correct emergency identification
            elif not is_critical and action.priority_level == 3:
                reward = 1.0  # Correct non-emergency
            elif is_critical and action.priority_level == 2:
                reward = 0.3  # Partially correct (urgent but not emergency)
            elif not is_critical and action.priority_level == 2:
                reward = 0.5  # Slightly overcautious but ok
            else:
                reward = 0.0  # Wrong

        elif difficulty == "medium":
            # Medium: Reward based on how close priority is
            correct_priority = 1 if is_critical else 3
            diff = abs(action.priority_level - correct_priority)
            if diff == 0:
                reward = 1.0
            elif diff == 1:
                reward = 0.4  # One level off
            else:
                reward = 0.0  # Completely wrong

            # Allocation bonus
            if is_critical and action.allocation in ["icu", "emergency"]:
                reward = min(1.0, reward + 0.1)
            elif not is_critical and action.allocation in ["ward", "waiting_room"]:
                reward = min(1.0, reward + 0.1)

        elif difficulty == "hard":
            # Hard: Resource-aware scoring
            correct_priority = 1 if is_critical else 3
            diff = abs(action.priority_level - correct_priority)

            if diff == 0:
                base_reward = 1.0
            elif diff == 1:
                base_reward = 0.4
            else:
                base_reward = 0.0

            # Resource penalty — if ICU is full but you sent them there
            if action.allocation == "icu" and resources["icu_beds"] == 0:
                base_reward *= 0.5  # Penalize impossible allocation

            # Bonus for good reasoning (contains key words)
            reasoning_lower = action.reasoning.lower()
            if any(word in reasoning_lower for word in ["critical", "emergency", "vital", "urgent"]):
                base_reward = min(1.0, base_reward + 0.05)

            reward = base_reward

        return round(reward, 2)

    def _get_feedback(self, action: TriageAction, is_critical: bool, reward: float) -> str:
        """Human-readable feedback for the agent."""
        if reward >= 0.9:
            return "✅ Excellent triage decision!"
        elif reward >= 0.5:
            return "⚠️ Partially correct — review priority assignment"
        else:
            correct = "EMERGENCY (Priority 1)" if is_critical else "NON-URGENT (Priority 3)"
            return f"❌ Incorrect. Patient needed: {correct}"