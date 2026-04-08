import requests
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", "your-groq-api-key")
)

def get_ai_explanation(patient, decision, result):
    symptoms = ", ".join(patient.get("symptoms", [])[:3])
    vitals = patient.get("vitals", {})
    hr = vitals.get("heart_rate", "N/A")
    temp = vitals.get("temp", "N/A")
    priority = decision["priority_level"]
    allocation = decision["allocation"]
    condition = result.get("info", {}).get("actual_condition", "N/A")
    reward = result.get("reward", "N/A")

    prompt = (
        "You are a medical AI assistant. Explain this triage decision briefly:\n"
        "Patient symptoms: " + symptoms + "\n"
        "Vitals: HR=" + str(hr) + ", Temp=" + str(temp) + "\n"
        "AI Decision: Priority " + str(priority) + " - " + allocation + "\n"
        "Actual condition: " + condition + "\n"
        "Reward received: " + str(reward) + "\n"
        "Give a 2-3 sentence doctor-friendly explanation."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content

def run_triage_test():
    print("START")
    print("MediCore AI - Medical Triage Baseline Test")
    print("=" * 50)

    for difficulty in ["easy", "medium", "hard"]:
        try:
            print("STEP difficulty=" + difficulty)

            reset_res = requests.post(API_BASE_URL + "/reset?difficulty=" + difficulty)
            patient = reset_res.json()

            patient_id = patient.get("patient_id") or patient.get("id") or "unknown"
            symptoms = patient.get("symptoms", [])
            vitals = patient.get("vitals", {})

            print("STEP patient_id=" + str(patient_id))
            print("STEP symptoms=" + ", ".join(symptoms[:3]))
            print("STEP vitals HR=" + str(vitals.get("heart_rate", "N/A")) +
                  " Temp=" + str(vitals.get("temp", "N/A")))

            priority = 1 if any(s in ["Chest Pain", "Shortness Of Breath",
                                       "Chest Pressure"]
                               for s in symptoms) else 3
            decision = {
                "priority_level": priority,
                "allocation": "icu" if priority == 1 else "waiting_room",
                "reasoning": "Baseline heuristic based on symptom severity."
            }

            step_res = requests.post(API_BASE_URL + "/step", json=decision)
            result = step_res.json()

            print("STEP decision priority=" + str(priority))
            print("STEP reward=" + str(result.get("reward", "N/A")))
            print("STEP condition=" + str(result.get("info", {}).get("actual_condition", "N/A")))
            print("STEP feedback=" + str(result.get("info", {}).get("feedback", "N/A")))

            try:
                explanation = get_ai_explanation(patient, decision, result)
                print("STEP ai_explanation=" + explanation)
            except Exception as e:
                print("STEP ai_explanation=Unavailable - " + str(e))

        except Exception as e:
            print("STEP error=" + str(e))

        print("-" * 50)

    print("END")

if __name__ == "__main__":
    run_triage_test()