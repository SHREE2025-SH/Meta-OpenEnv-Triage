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
        print("STEP difficulty=" + difficulty)

        reset_res = requests.post(API_BASE_URL + "/reset?difficulty=" + difficulty)
        patient = reset_res.json()

        print("STEP patient_id=" + patient["patient_id"])
        print("STEP symptoms=" + ", ".join(patient["symptoms"][:3]))
        print("STEP vitals HR=" + str(patient["vitals"]["heart_rate"]) +
              " Temp=" + str(patient["vitals"]["temp"]))

        priority = 1 if any(s in ["Chest Pain", "Shortness Of Breath",
                                   "Chest Pressure"]
                           for s in patient["symptoms"]) else 3
        decision = {
            "priority_level": priority,
            "allocation": "icu" if priority == 1 else "waiting_room",
            "reasoning": "Baseline heuristic based on symptom severity."
        }

        step_res = requests.post(API_BASE_URL + "/step", json=decision)
        result = step_res.json()

        print("STEP decision priority=" + str(priority))
        print("STEP reward=" + str(result["reward"]))
        print("STEP condition=" + str(result["info"]["actual_condition"]))
        print("STEP feedback=" + str(result["info"]["feedback"]))

        try:
            explanation = get_ai_explanation(patient, decision, result)
            print("STEP ai_explanation=" + explanation)
        except Exception as e:
            print("STEP ai_explanation=Unavailable - " + str(e))

        print("-" * 50)

    print("END")

if __name__ == "__main__":
    run_triage_test()