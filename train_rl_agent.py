import requests
import numpy as np
import random

BASE_URL = "http://127.0.0.1:8000"
ACTIONS = [1, 2, 3] # Priority Levels
EPISODES = 20 # How many patients to learn from

# Simplified Q-Table (Dictionary: {symptom_key: [scores_for_act1, scores_for_act2, scores_for_act3]})
q_table = {}

def get_q_values(symptoms):
    s_key = "-".join(sorted(symptoms))
    if s_key not in q_table:
        q_table[s_key] = np.zeros(len(ACTIONS))
    return s_key, q_table[s_key]

print("🧠 Starting RL Training Session...")

for i in range(EPISODES):
    # 1. Get Patient (State)
    resp = requests.post(f"{BASE_URL}/reset").json()
    symptoms = resp['symptoms']
    s_key, q_values = get_q_values(symptoms)

    # 2. Choose Action (Epsilon-Greedy: Explore vs Exploit)
    if random.random() < 0.2: # Explore
        action_idx = random.randint(0, 2)
    else: # Exploit (Choose best known)
        action_idx = np.argmax(q_values)
    
    chosen_priority = ACTIONS[action_idx]

    # 3. Take Action and get Reward
    step_resp = requests.post(f"{BASE_URL}/step", json={
        "priority_level": chosen_priority,
        "allocation": "ward",
        "reasoning": "RL Training Step"
    }).json()
    
    reward = step_resp['reward']

    # 4. Update Q-Table (The "Learning" Part)
    # Formula: Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
    q_table[s_key][action_idx] += 0.1 * (reward - q_table[s_key][action_idx])

    print(f"Episode {i+1}: Symptoms: {symptoms[:2]}... | Action: {chosen_priority} | Reward: {reward}")

print("\n✅ Training Complete! The Agent has learned from the Medical Environment.")