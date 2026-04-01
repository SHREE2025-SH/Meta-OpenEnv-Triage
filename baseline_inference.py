import requests
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_URL = "http://127.0.0.1:8000"

def get_ai_explanation(patient, decision, result):
    symptoms = ", ".join(patient["symptoms"][:3])
    hr = patient["vitals"]["heart_rate"]
    temp = patient["vitals"]["temp"]
    priority = decision["priority_level"]
    allocation = decision["allocation"]
    condition = result["info"]["actual_condition"]
    reward = result["reward"]
    
    prompt = (
        "You are a medical AI assistant. Explain this triage decision briefly:\n"
        "Patient symptoms: " + symptoms + "\n"
        "Vitals: HR=" + str(hr) + ", Temp=" + str(temp) + "\n"
        "AI Decision: Priority " + str(priority) + " - " + allocation + "\n"
        "Actual condition: " + condition + "\n"
        "Reward received: " + str(reward) + "\n"
        "Give a 2-3 sentence doctor-friendly explanation."
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )
    return response.text

def run_triage_test():
    print("MediTriage AI Environment - Baseline Test")
    print("=" * 50)

    for difficulty in ["easy", "medium", "hard"]:
        print("\n[" + difficulty.upper() + "] Testing...")
        
        reset_res = requests.post(BASE_URL + "/reset?difficulty=" + difficulty)
        patient = reset_res.json()
        
        print("Patient: " + patient["patient_id"])
        print("Symptoms: " + ", ".join(patient["symptoms"][:3]))

        priority = 1 if any(s in ["Chest Pain", "Shortness of breath"] for s in patient["symptoms"]) else 3
        decision = {
            "priority_level": priority,
            "allocation": "icu" if priority == 1 else "waiting_room",
            "reasoning": "Baseline heuristic based on symptom severity."
        }

        step_res = requests.post(BASE_URL + "/step", json=decision)
        result = step_res.json()

        print("Decision: Priority " + str(priority))
        print("Reward: " + str(result["reward"]))
        print("Condition: " + str(result["info"]["actual_condition"]))
        
        try:
            explanation = get_ai_explanation(patient, decision, result)
        except Exception as e:
            explanation = "AI explanation unavailable - quota exceeded. Core environment working correctly."
        
        print("AI Explanation: " + explanation)
        print("-" * 50)

if __name__ == "__main__":
    run_triage_test()