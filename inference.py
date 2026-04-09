"""
CodeDebugEnv — Baseline Inference Script
=========================================
Runs a baseline LLM agent against all three CodeDebugEnv tasks and emits
structured stdout logs in the exact [START] / [STEP] / [END] format required
by the hackathon evaluator.

Usage
-----
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    export HF_TOKEN="hf_..."
    export ENV_URL="http://localhost:7860"   # or your HF Space URL
    python inference.py

Environment variables
---------------------
API_BASE_URL  : OpenAI-compatible chat completions endpoint
MODEL_NAME    : Model identifier
HF_TOKEN      : HuggingFace / API key
ENV_URL       : Base URL of the running CodeDebugEnv server
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
ENV_URL:      str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

MAX_ATTEMPTS: int = 5   # must match environment's max_attempts


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python debugger. "
    "You will be given a buggy Python function, a description of what it should do, "
    "and an error hint. "
    "Your task: return ONLY the corrected Python source code with no extra text, "
    "no markdown fences, no explanations. "
    "Output only the raw Python function definition."
)


def ask_llm(buggy_code: str, task_description: str, error_hint: str, attempt: int) -> str:
    """Call the LLM and return the raw Python code it suggests."""
    user_msg = (
        f"Task description:\n{task_description}\n\n"
        f"Error hint:\n{error_hint}\n\n"
        f"Buggy code (attempt {attempt}):\n```python\n{buggy_code}\n```\n\n"
        "Return ONLY the corrected Python function. No explanation, no markdown."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    # Strip accidental markdown fences if the model adds them
    raw = re.sub(r"```(?:python)?", "", raw).strip("`").strip()
    return raw


# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------

def env_get(path: str) -> dict:
    resp = requests.get(f"{ENV_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_post(path: str, payload: dict) -> dict:
    resp = requests.post(
        f"{ENV_URL}{path}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Per-task agent loop
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    """Run one full episode; return best reward achieved."""
    # ── reset ──────────────────────────────────────────────────────────────
    reset_resp = env_post("/reset", {"task_id": task_id})
    session_id: str = reset_resp["session_id"]
    obs: dict     = reset_resp["observation"]

    print(json.dumps({
        "event": "[START]",
        "task_id": task_id,
        "session_id": session_id,
        "difficulty": obs.get("difficulty"),
        "max_attempts": obs.get("max_attempts"),
    }), flush=True)

    best_reward: float = 0.0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # ── agent decides ──────────────────────────────────────────────────
        fixed_code = ask_llm(
            buggy_code=obs["buggy_code"],
            task_description=obs["task_description"],
            error_hint=obs["error_hint"],
            attempt=attempt,
        )

        # ── step ───────────────────────────────────────────────────────────
        step_resp = env_post("/step", {
            "session_id": session_id,
            "fixed_code": fixed_code,
        })

        reward: float  = step_resp["reward"]
        done: bool     = step_resp["done"]
        info: dict     = step_resp.get("info", {})

        if reward > best_reward:
            best_reward = reward

        print(json.dumps({
            "event": "[STEP]",
            "task_id": task_id,
            "session_id": session_id,
            "attempt": attempt,
            "reward": reward,
            "done": done,
            "grader_error": info.get("grader_error"),
            "tests_passed": sum(
                1 for t in info.get("test_results", []) if t.get("passed")
            ),
            "tests_total": len(info.get("test_results", [])),
        }), flush=True)

        if done:
            break

    print(json.dumps({
        "event": "[END]",
        "task_id": task_id,
        "session_id": session_id,
        "best_reward": best_reward,
        "total_attempts": attempt,
    }), flush=True)

    return best_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Health check
    try:
        health = env_get("/health")
        print(json.dumps({"event": "health_check", "response": health}), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "health_check_failed", "error": str(exc)}), flush=True)
        sys.exit(1)

    # Discover tasks
    tasks_resp = env_get("/tasks")
    task_ids = [t["task_id"] for t in tasks_resp.get("tasks", [])]

    scores: dict[str, float] = {}
    for task_id in task_ids:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({
                "event": "task_error",
                "task_id": task_id,
                "error": str(exc),
            }), flush=True)
            scores[task_id] = 0.0

    avg = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

    print(json.dumps({
        "event": "summary",
        "scores": scores,
        "average": avg,
    }), flush=True)


if __name__ == "__main__":
    main()
