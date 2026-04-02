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

        print("📂 Loading MediCore AI dataset...")

        new_dataset_path = os.path.join(base_path, "diseases_symptoms.csv")
        old_dataset_path = os.path.join(base_path, "dia_3.csv")

        if os.path.exists(new_dataset_path):
            self.df = pd.read_csv(new_dataset_path)
            print(f"✅ Loaded new dataset: {self.df.shape[0]} rows, {self.df['diseases'].nunique()} diseases")
            self.disease_col = 'diseases'
            self.symptom_cols = [c for c in self.df.columns if c != 'diseases']
            self.use_new_dataset = True
        else:
            self.df = pd.read_csv(old_dataset_path)
            print(f"⚠️ Using old dataset: {len(self.df)} diseases")
            self.use_new_dataset = False

        self.critical_keywords = [
            "heart", "stroke", "sepsis", "hypertension", "attack",
            "failure", "emergency", "hemorrhage", "shock", "poisoning",
            "overdose", "cancer", "tumor", "seizure", "coma",
            "meningitis", "pneumonia", "embolism", "aneurysm"
        ]

        if self.use_new_dataset:
            self.diseases = self.df[self.disease_col].unique().tolist()
            self.disease_symptom_map = self._build_disease_symptom_map()

        print(f"✅ Total diseases available: {len(self.diseases) if self.use_new_dataset else 'N/A'}")

    def _build_disease_symptom_map(self) -> Dict:
        disease_map = {}
        for disease in self.diseases:
            disease_rows = self.df[self.df[self.disease_col] == disease]
            symptom_scores = disease_rows[self.symptom_cols].mean()
            top_symptoms = symptom_scores[symptom_scores > 0.3].index.tolist()
            disease_map[disease] = top_symptoms if top_symptoms else self.symptom_cols[:3]
        return disease_map

    def get_case(self, difficulty: str = "easy") -> Dict:
        is_critical = random.random() < 0.4

        if is_critical:
            critical_diseases = [d for d in self.diseases
                                if any(k in d.lower() for k in self.critical_keywords)]
            disease = random.choice(critical_diseases) if critical_diseases else random.choice(self.diseases)
        else:
            non_critical = [d for d in self.diseases
                           if not any(k in d.lower() for k in self.critical_keywords)]
            disease = random.choice(non_critical) if non_critical else random.choice(self.diseases)

        is_critical = any(k in disease.lower() for k in self.critical_keywords)

        if self.use_new_dataset and disease in self.disease_symptom_map:
            symptoms = self.disease_symptom_map[disease][:5]
            symptoms = [s.replace('_', ' ').title() for s in symptoms]
        else:
            symptoms = ["Generalized weakness", "Fatigue"]

        if not symptoms:
            symptoms = ["Generalized weakness", "Fatigue"]

        if is_critical:
            heart_rate = float(random.randint(110, 160))
            temp = round(random.uniform(38.5, 40.5), 1)
            bp_systolic = float(random.randint(160, 200) if random.random() < 0.5
                               else random.randint(70, 90))
        else:
            heart_rate = float(random.randint(60, 100))
            temp = round(random.uniform(36.5, 37.5), 1)
            bp_systolic = float(random.randint(110, 130))

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
            "disease": disease,
            "symptoms": symptoms,
            "vitals": {
                "heart_rate": heart_rate,
                "temp": temp,
                "bp_systolic": bp_systolic
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
        self.difficulty = difficulty
        self.current_case = self.data_handler.get_case(difficulty=difficulty)
        self.state = TriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty
        )
        print(f"✨ New Case [{difficulty}]: {self.current_case['disease']}")
        print(f"   Critical: {self.current_case['is_critical']}")

        return MedicalObservation(
            patient_id=f"PAT-{random.randint(1000, 9999)}",
            symptoms=self.current_case['symptoms'],
            vitals=self.current_case['vitals'],
            hospital_resources=self.current_case['resources'],
            difficulty=difficulty,
            message="New patient arrived. Assess and triage appropriately."
        )

    def step(self, action: TriageAction) -> Tuple[None, float, bool, dict]:
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

    def _calculate_reward(self, action, is_critical, difficulty, resources) -> float:
        reward = 0.0

        if difficulty == "easy":
            if is_critical and action.priority_level == 1:
                reward = 1.0
            elif not is_critical and action.priority_level == 3:
                reward = 1.0
            elif is_critical and action.priority_level == 2:
                reward = 0.3
            elif not is_critical and action.priority_level == 2:
                reward = 0.5
            elif is_critical and action.priority_level == 3:
                reward = -0.5
            else:
                reward = 0.0

        elif difficulty == "medium":
            correct = 1 if is_critical else 3
            diff = abs(action.priority_level - correct)
            if diff == 0:
                reward = 1.0
            elif diff == 1:
                reward = 0.4
            else:
                reward = -0.3 if is_critical else 0.0

            if is_critical and action.allocation in ["icu", "emergency"]:
                reward = min(1.0, reward + 0.1)
            elif not is_critical and action.allocation in ["ward", "waiting_room"]:
                reward = min(1.0, reward + 0.1)

        elif difficulty == "hard":
            correct = 1 if is_critical else 3
            diff = abs(action.priority_level - correct)

            if diff == 0:
                base = 1.0
            elif diff == 1:
                base = 0.4
            else:
                base = -0.5 if is_critical else 0.0

            if action.allocation == "icu" and resources["icu_beds"] == 0:
                base *= 0.5

            reasoning_lower = action.reasoning.lower()
            if any(w in reasoning_lower for w in ["critical", "emergency", "vital", "urgent"]):
                base = min(1.0, base + 0.05)

            reward = base

        return round(float(reward), 2)

    def _get_feedback(self, action, is_critical, reward) -> str:
        if reward >= 0.9:
            return "✅ Excellent triage decision!"
        elif reward >= 0.5:
            return "⚠️ Partially correct — review priority assignment"
        elif reward < 0:
            return "🚨 CRITICAL ERROR — Patient needed immediate attention!"
        else:
            correct = "EMERGENCY (Priority 1)" if is_critical else "NON-URGENT (Priority 3)"
            return f"❌ Incorrect. Patient needed: {correct}"