"""
CodeDebugEnv — FastAPI HTTP server
Exposes the standard OpenEnv step()/reset()/state() interface over HTTP.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import CodeDebugEnv

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CodeDebugEnv",
    description=(
        "An OpenEnv environment where an AI agent fixes buggy Python code. "
        "Implements the standard step() / reset() / state() interface."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (session-based, thread-safe dict)
_env = CodeDebugEnv()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str | None = Field(
        default=None,
        description="Task to start. One of: easy_syntax | medium_logic | hard_algorithm",
    )


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /reset")
    fixed_code: str = Field(..., description="Corrected Python source code")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness probe — returns 200 when the server is ready."""
    return {"status": "ok", "env": "CodeDebugEnv", "version": "1.0.0"}


@app.get("/tasks", tags=["meta"])
def list_tasks() -> dict[str, Any]:
    """List all available tasks with difficulty and description."""
    return {"tasks": _env.list_tasks()}


@app.post("/reset", tags=["env"])
async def reset(request: Request) -> dict[str, Any]:
    """
    Start a new episode. Body is optional — if empty or missing, defaults to easy_syntax.
    """
    task_id = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id", None)
    except Exception:
        # Empty body or non-JSON — that's fine, use default task
        task_id = None

    try:
        result = _env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/step", tags=["env"])
def step(req: StepRequest) -> dict[str, Any]:
    """
    Submit a fixed version of the buggy code.

    Returns:
    - `reward` — fraction of test cases that pass (0.0 – 1.0)
    - `done` — True when perfect score OR max_attempts reached
    - `observation` — updated state
    - `info` — per-test pass/fail details and grader errors
    """
    try:
        result = _env.step(req.session_id, req.fixed_code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@app.get("/state/{session_id}", tags=["env"])
def state(session_id: str) -> dict[str, Any]:
    """Return the full internal state of an episode."""
    try:
        result = _env.state(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result
