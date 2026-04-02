import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import requests
from collections import deque

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ─── DQN Neural Network ───────────────────────────────────────────────────────
class DQNetwork(nn.Module):
    """
    Deep Q-Network for Medical Triage
    Input: symptom vector (encoded symptoms)
    Output: Q-values for each action (priority 1, 2, 3)
    """
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 3):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)


# ─── Experience Replay Buffer ─────────────────────────────────────────────────
class ReplayBuffer:
    """
    Stores past experiences for training.
    Agent learns from random samples — prevents overfitting to recent events.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# ─── State Encoder ────────────────────────────────────────────────────────────
class StateEncoder:
    """
    Converts patient observation (symptoms + vitals) into a numeric vector
    that the neural network can understand.
    """
    # Common symptoms vocabulary
    SYMPTOM_VOCAB = [
        "headache", "fever", "chest pain", "shortness of breath", "fatigue",
        "nausea", "vomiting", "dizziness", "cough", "bleeding",
        "weakness", "pain", "swelling", "infection", "inflammation",
        "anxiety", "depression", "insomnia", "seizure", "paralysis",
        "vision", "hearing", "breathing", "heart", "stroke",
        "cancer", "tumor", "diabetes", "hypertension", "pneumonia"
    ]

    def encode(self, observation: dict) -> np.ndarray:
        """Convert observation dict to numpy vector."""
        # Symptom encoding (30 features)
        symptom_vec = np.zeros(len(self.SYMPTOM_VOCAB))
        symptoms_text = " ".join(observation.get("symptoms", [])).lower()
        for i, keyword in enumerate(self.SYMPTOM_VOCAB):
            if keyword in symptoms_text:
                symptom_vec[i] = 1.0

        # Vitals encoding (normalized, 3 features)
        vitals = observation.get("vitals", {})
        hr = vitals.get("heart_rate", 80) / 200.0  # normalize
        temp = (vitals.get("temp", 37.0) - 35.0) / 10.0  # normalize
        bp = vitals.get("bp_systolic", 120) / 250.0  # normalize

        # Resources encoding (2 features)
        resources = observation.get("hospital_resources", {})
        icu = resources.get("icu_beds", 3) / 5.0
        amb = resources.get("ambulances", 2) / 3.0

        # Combine all features
        state = np.concatenate([symptom_vec, [hr, temp, bp, icu, amb]])
        return state.astype(np.float32)

    @property
    def state_size(self):
        return len(self.SYMPTOM_VOCAB) + 5  # symptoms + vitals + resources


# ─── DQN Agent ────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network Agent for Medical Triage
    Uses experience replay and target network for stable training.
    """
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        target_update: int = 10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        # Main network (learns)
        self.policy_net = DQNetwork(state_size, output_size=action_size).to(device)
        # Target network (stable reference)
        self.target_net = DQNetwork(state_size, output_size=action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer()
        self.encoder = StateEncoder()

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values (using target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str = "dqn_model.pth"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"💾 Model saved to {path}")

    def load(self, path: str = "dqn_model.pth"):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"📂 Model loaded from {path}")


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_dqn(episodes: int = 200, base_url: str = "http://127.0.0.1:8000"):
    """Train the DQN agent on the Medical Triage environment."""

    encoder = StateEncoder()
    agent = DQNAgent(state_size=encoder.state_size)

    print(f"\n🧠 Starting DQN Training — {episodes} episodes")
    print(f"📊 State size: {encoder.state_size}")
    print(f"🎯 Actions: Priority 1 (Emergency), 2 (Urgent), 3 (Non-urgent)")
    print("=" * 60)

    rewards_history = []
    losses_history = []

    for episode in range(episodes):
        # Get difficulty — start easy, progress to harder
        if episode < episodes * 0.4:
            difficulty = "easy"
        elif episode < episodes * 0.7:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Reset environment
        obs = requests.post(f"{base_url}/reset?difficulty={difficulty}").json()
        state = encoder.encode(obs)

        # Take action
        action_idx = agent.select_action(state)
        priority = action_idx + 1  # Convert 0,1,2 → 1,2,3

        # Allocations based on priority
        allocations = {1: "icu", 2: "emergency", 3: "waiting_room"}

        result = requests.post(f"{base_url}/step", json={
            "priority_level": priority,
            "allocation": allocations[priority],
            "reasoning": f"DQN agent decision — priority {priority}"
        }).json()

        reward = result["reward"]
        next_state = state  # Single-step episode

        # Store and learn
        agent.remember(state, action_idx, reward, next_state, True)
        loss = agent.learn()

        rewards_history.append(reward)
        if loss > 0:
            losses_history.append(loss)

        # Progress report every 20 episodes
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_loss = np.mean(losses_history[-20:]) if losses_history else 0
            print(f"Episode {episode+1:3d}/{episodes} | "
                  f"Difficulty: {difficulty:6s} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    # Save the trained model
    agent.save("dqn_model.pth")

    print("\n✅ Training Complete!")
    print(f"Final avg reward (last 20): {np.mean(rewards_history[-20:]):.3f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")

    return agent


if __name__ == "__main__":
    trained_agent = train_dqn(episodes=200)