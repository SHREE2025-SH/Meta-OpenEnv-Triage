import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def run_triage_test():
    print("🚀 Connecting to MediAssist AI Environment...")
    
    # 1. Reset the environment to get a new patient
    try:
        reset_res = requests.post(f"{BASE_URL}/reset?difficulty=easy")
        reset_res.raise_for_status()
        patient = reset_res.json()
        
        print(f"\n✅ Patient Checked In: {patient['patient_id']}")
        print(f"📋 Symptoms: {', '.join(patient['symptoms'])}")
        print(f"💓 Vitals: HR: {patient['vitals']['heart_rate']}, Temp: {patient['vitals']['temp']}")

        # 2. Simple "Agent" logic (Baseline)
        # If symptoms include 'Chest Pain', we'll treat it as Priority 1 (Emergency)
        priority = 1 if any(s in ['Chest Pain', 'Shortness of breath'] for s in patient['symptoms']) else 3
        
        decision = {
            "priority_level": priority,
            "allocation": "icu" if priority == 1 else "waiting_room",
            "reasoning": "Baseline heuristic based on symptom severity."
        }

        # 3. Send the decision back to the environment
        step_res = requests.post(f"{BASE_URL}/step", json=decision)
        step_res.raise_for_status()
        result = step_res.json()

        print(f"\n⚖️  AI Decision: Priority {priority}")
        print(f"💰 Reward Received: {result['reward']}")
        print(f"📖 Actual Condition: {result['info']['actual_condition']}")
        print("\n🏁 Simulation Complete.")

    except Exception as e:
        print(f"❌ Error: {e}. Make sure your Uvicorn server is running!")

if __name__ == "__main__":
    run_triage_test()