from fastapi import FastAPI, HTTPException
from .environment import MedicalTriageEnv
from .models import TriageAction, MedicalObservation

# This is the 'app' attribute Uvicorn was looking for!
app = FastAPI(title="MediTriage OpenEnv")
env = MedicalTriageEnv()

@app.get("/")
def health_check():
    """Confirms the environment is alive"""
    return {"status": "active", "environment": "MediTriage-v1"}

@app.post("/reset", response_model=MedicalObservation)
def reset(difficulty: str = "easy"):
    """Starts a new triage session"""
    try:
        return env.reset(difficulty=difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset Error: {str(e)}")

@app.post("/step")
def step(action: TriageAction):
    """Processes a doctor's decision and returns a reward"""
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step Error: {str(e)}")

@app.get("/state")
def get_state():
    """Returns the current internal state of the simulation"""
    if not hasattr(env, 'state') or env.state is None:
        raise HTTPException(status_code=400, detail="Environment not reset yet")
    return env.state