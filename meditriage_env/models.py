from pydantic import BaseModel
from typing import List, Dict, Optional

class MedicalObservation(BaseModel):
    patient_id: str
    symptoms: List[str]
    vitals: Dict[str, float]
    hospital_resources: Dict[str, int]
    difficulty: str = "easy"
    message: Optional[str] = None

class TriageAction(BaseModel):
    priority_level: int  # 1: Emergency, 2: Urgent, 3: Non-Urgent
    allocation: str      # "icu", "emergency", "ward", "waiting_room"
    reasoning: str

class TriageState(BaseModel):
    episode_id: str
    step_count: int
    difficulty: str = "easy"
    is_done: bool = False
    last_reward: float = 0.0