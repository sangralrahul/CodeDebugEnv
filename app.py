"""
CodeDebugEnv — FastAPI HTTP server
"""
from __future__ import annotations
from typing import Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from environment import CodeDebugEnv

app = FastAPI(title="CodeDebugEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_env = CodeDebugEnv()

class StepRequest(BaseModel):
    session_id: str = Field(...)
    fixed_code: str = Field(...)

@app.get("/health")
def health():
    return {"status": "ok", "env": "CodeDebugEnv", "version": "1.0.0"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": _env.list_tasks()}

@app.post("/reset")
async def reset(request: Request):
    task_id = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id", None)
    except Exception:
        task_id = None
    try:
        return _env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/step")
def step(req: StepRequest):
    try:
        return _env.step(req.session_id, req.fixed_code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.get("/state/{session_id}")
def state(session_id: str):
    try:
        return _env.state(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860, workers=2)

if __name__ == "__main__":
    main()
