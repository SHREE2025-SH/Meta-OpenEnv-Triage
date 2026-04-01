---
title: MediTriage OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---


# 🏥 MediTriage-Env — Medical Triage RL Environment

> An open reinforcement learning environment for training AI agents to perform medical triage — built for the **Meta Open Environment Challenge (Scaler Hackathon 2025)**.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🧠 Overview

**MediTriage-Env** is a custom OpenAI Gym-compatible reinforcement learning environment that simulates real-world medical triage scenarios. An RL agent must assess patients based on their symptoms and assign a triage priority — mimicking how emergency medical staff operate under pressure.

The environment exposes a REST API (FastAPI), making it easy to plug in any RL framework (Stable-Baselines3, RLlib, custom agents, etc.) over HTTP.

---

## 🎯 Key Features

- ✅ **3 Difficulty Levels** — Easy, Medium, Hard — with increasing symptom complexity
- ✅ **Real Medical Data** — Symptoms sourced from curated medical CSVs (`dia_3.csv`, `symptoms2.csv`, `sym_dis_matrix.csv`)
- ✅ **Reward Signal** — Continuous reward from `0.0` to `1.0` based on triage accuracy
- ✅ **REST API Interface** — Full environment control via HTTP endpoints (reset, step, render)
- ✅ **Docker Ready** — One-command deployment, HuggingFace Spaces compatible
- ✅ **Baseline Agent** — Included inference script for quick benchmarking

---

## 🗂️ Project Structure

```
Meta-OpenEnv-Triage/
├── meditriage_env/
│   ├── app.py              # FastAPI server — exposes environment as REST API
│   ├── environment.py      # Core RL environment logic (step, reset, reward)
│   ├── models.py           # Pydantic models for request/response schemas
│   ├── __init__.py
│   ├── dia_3.csv           # Disease-symptom dataset
│   ├── symptoms2.csv       # Symptom descriptions
│   └── sym_dis_matrix.csv  # Symptom-disease matrix
├── Dockerfile              # Docker config (port 7860 for HuggingFace)
├── requirements.txt
├── openenv.yaml            # Environment metadata
├── baseline_inference.py   # Baseline random/heuristic agent
├── train_rl_agent.py       # RL training script (SB3 compatible)
└── test_all.py             # Integration tests for all difficulties
```

---

## 🚀 Quick Start

### 1. Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Meta-OpenEnv-Triage.git
cd Meta-OpenEnv-Triage

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn meditriage_env.app:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

### 2. Run with Docker

```bash
docker build -t meditriage-env .
docker run -p 8000:7860 meditriage-env
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take an action, returns (obs, reward, done, info) |
| `/render` | GET | Get current state as human-readable text |
| `/info` | GET | Environment metadata (action space, obs space) |

### Example: Reset the environment

```python
import requests

res = requests.post("http://localhost:8000/reset", json={"difficulty": "medium"})
obs = res.json()
print(obs)
```

### Example: Take a step

```python
res = requests.post("http://localhost:8000/step", json={"action": 2})
result = res.json()
print(result["reward"])   # 0.0 – 1.0
print(result["done"])     # True/False
```

---

## 🏋️ Training an RL Agent

```bash
python train_rl_agent.py
```

The training script connects to the running FastAPI server and trains a policy using the reward signal. Modify the script to swap in your preferred RL algorithm.

---

## 📊 Baseline Performance

Run the included baseline agent to benchmark the environment:

```bash
python baseline_inference.py
```

---

## 🧪 Testing

```bash
python test_all.py
```

Tests all three difficulty modes and verifies reward, observation, and done signals.

---

## 🌡️ Environment Details

| Property | Value |
|---|---|
| Observation Space | Symptom vector (multi-hot encoded) |
| Action Space | Discrete — triage priority levels |
| Reward Range | `[0.0, 1.0]` |
| Difficulty Levels | `easy`, `medium`, `hard` |
| Data Source | Real symptom-disease datasets |

---

## 🤗 HuggingFace Spaces Deployment

This environment is deployed on HuggingFace Spaces (Docker SDK, port 7860).

👉 **[Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/Meta-OpenEnv-Triage)**

---

## 👤 Author

**Spandan Das**
ML Student | Pimpri-Chinchwad, Pune
Built for the Meta Open Environment Challenge — Scaler Hackathon


---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.