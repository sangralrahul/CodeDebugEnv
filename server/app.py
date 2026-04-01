"""
FastAPI server exposing the OpenEnv HTTP API for CodeDebugEnv.
"""

import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from environment import CodeDebugEnv, Action, Observation, State, TASKS

app = FastAPI(
    title="CodeDebugEnv",
    description="OpenEnv environment: AI agent fixes buggy Python code.",
    version="1.0.0",
)

_envs: dict[str, CodeDebugEnv] = {}


# ── Request / Response schemas ────────────────────────────────────────────────

class StepRequest(BaseModel):
    session_id: str
    fixed_code: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CodeDebugEnv</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 60px auto; padding: 20px; background: #0f0f0f; color: #e0e0e0; }
            h1 { color: #4fc3f7; font-size: 2.5em; }
            h2 { color: #81d4fa; margin-top: 40px; }
            .badge { background: #1e88e5; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; margin-right: 8px; }
            .badge.green { background: #43a047; }
            .badge.orange { background: #fb8c00; }
            .badge.red { background: #e53935; }
            table { width: 100%; border-collapse: collapse; margin-top: 16px; }
            th { background: #1e1e1e; color: #4fc3f7; padding: 10px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #333; }
            code { background: #1e1e1e; padding: 2px 8px; border-radius: 4px; color: #a5d6a7; }
            .status { color: #66bb6a; font-weight: bold; font-size: 1.1em; }
        </style>
    </head>
    <body>
        <h1>🐛 CodeDebugEnv</h1>
        <p class="status">✅ Server is running</p>
        <p>An <strong>OpenEnv</strong> environment where AI agents fix buggy Python code.<br>
        Built for the <strong>Meta × Scaler OpenEnv Hackathon</strong>.</p>
        <h2>Tasks</h2>
        <table>
            <tr><th>Task ID</th><th>Difficulty</th><th>Description</th></tr>
            <tr><td><code>easy_syntax</code></td><td><span class="badge green">Easy</span></td><td>Fix a syntax error in a sum function</td></tr>
            <tr><td><code>medium_logic</code></td><td><span class="badge orange">Medium</span></td><td>Fix a logic bug in a prime checker</td></tr>
            <tr><td><code>hard_algorithm</code></td><td><span class="badge red">Hard</span></td><td>Fix an off-by-one error in binary search</td></tr>
        </table>
        <h2>API Endpoints</h2>
        <table>
            <tr><th>Method</th><th>Path</th><th>Description</th></tr>
            <tr><td><code>GET</code></td><td><code>/health</code></td><td>Health check</td></tr>
            <tr><td><code>GET</code></td><td><code>/tasks</code></td><td>List all tasks</td></tr>
            <tr><td><code>POST</code></td><td><code>/reset</code></td><td>Start a new episode</td></tr>
            <tr><td><code>POST</code></td><td><code>/step</code></td><td>Submit fixed code</td></tr>
            <tr><td><code>GET</code></td><td><code>/state/{session_id}</code></td><td>Get episode state</td></tr>
            <tr><td><code>GET</code></td><td><code>/docs</code></td><td>Interactive API docs</td></tr>
        </table>
        <h2>Quick Test</h2>
        <p>Try the interactive API docs: <a href="/docs" style="color:#4fc3f7">/docs</a></p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "env": "CodeDebugEnv"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": tid, "description": t["description"]}
            for tid, t in TASKS.items()
        ]
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = (body or {}).get("task_id", "easy_syntax")
    if task_id not in TASKS:
        task_id = "easy_syntax"
    session_id = str(uuid.uuid4())
    env = CodeDebugEnv(task_id=task_id)
    obs = env.reset()
    _envs[session_id] = env
    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _envs.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    if env.state().done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset to start a new one.")
    action = Action(fixed_code=req.fixed_code)
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state/{session_id}", response_model=State)
def get_state(session_id: str):
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
