import requests
import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://shree3010-meditriage-openenv.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
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
    print("[START] task=MedicalTriage", flush=True)

    total_reward = 0.0
    steps = 0

    for difficulty in ["easy", "medium", "hard"]:
        try:
            reset_res = requests.post(
                API_BASE_URL + "/reset?difficulty=" + difficulty,
                timeout=30
            )
            patient = reset_res.json()

            symptoms = patient.get("symptoms", [])
            vitals = patient.get("vitals", {})

            priority = 1 if any(s in ["Chest Pain", "Shortness Of Breath",
                                       "Chest Pressure"]
                               for s in symptoms) else 3
            decision = {
                "priority_level": priority,
                "allocation": "icu" if priority == 1 else "waiting_room",
                "reasoning": "Baseline heuristic based on symptom severity."
            }

            step_res = requests.post(
                API_BASE_URL + "/step",
                json=decision,
                timeout=30
            )
            result = step_res.json()
            reward = result.get("reward", 0.0)
            total_reward += reward
            steps += 1

            print(f"[STEP] step={steps} difficulty={difficulty} reward={reward} priority={priority}", flush=True)

        except Exception as e:
            steps += 1
            print(f"[STEP] step={steps} difficulty={difficulty} reward=0.0 error={str(e)}", flush=True)

    score = round(total_reward / steps, 4) if steps > 0 else 0.0
    print(f"[END] task=MedicalTriage score={score} steps={steps}", flush=True)

if __name__ == "__main__":
    run_triage_test()