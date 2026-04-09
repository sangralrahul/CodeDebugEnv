"""
CodeDebugEnv — FastAPI HTTP server
Exposes the standard OpenEnv step()/reset()/state() interface over HTTP.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
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
        examples=["easy_syntax"],
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
def reset(req: ResetRequest) -> dict[str, Any]:
    """
    Start a new episode.

    Returns a session_id and the initial observation containing:
    - `buggy_code` — the broken Python function
    - `task_description` — what the function should do
    - `error_hint` — stderr from running the buggy code
    """
    try:
        result = _env.reset(req.task_id)
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
    - `observation` — updated state (attempt counter, same code prompt)
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
