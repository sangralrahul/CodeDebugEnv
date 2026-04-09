"""
CodeDebugEnv — Fixed Inference Script
=======================================
Emits [START] / [STEP] / [END] structured stdout logs in the EXACT format
required by the hackathon evaluator.
"""

from __future__ import annotations

import json
import os
import re
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
ENV_URL:      str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

MAX_ATTEMPTS: int = 5

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
    # reset
    reset_resp = env_post("/reset", {"task_id": task_id})
    session_id: str = reset_resp["session_id"]
    obs: dict       = reset_resp["observation"]

    # EXACT FORMAT the evaluator requires
    print(f"[START] task={task_id}", flush=True)

    best_reward: float = 0.0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        fixed_code = ask_llm(
            buggy_code=obs["buggy_code"],
            task_description=obs["task_description"],
            error_hint=obs["error_hint"],
            attempt=attempt,
        )

        step_resp = env_post("/step", {
            "session_id": session_id,
            "fixed_code": fixed_code,
        })

        reward: float = step_resp["reward"]
        done: bool    = step_resp["done"]

        if reward > best_reward:
            best_reward = reward

        # EXACT FORMAT the evaluator requires
        print(f"[STEP] step={attempt} reward={reward}", flush=True)

        if done:
            break

    # EXACT FORMAT the evaluator requires
    print(f"[END] task={task_id} score={best_reward} steps={attempt}", flush=True)

    return best_reward

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Health check
    try:
        env_get("/health")
    except Exception as exc:
        print(f"[ERROR] health_check_failed: {exc}", flush=True)
        sys.exit(1)

    # Discover tasks
    tasks_resp = env_get("/tasks")
    task_ids = [t["task_id"] for t in tasks_resp.get("tasks", [])]

    scores: dict[str, float] = {}
    for task_id in task_ids:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as exc:
            print(f"[ERROR] task={task_id} error={exc}", flush=True)
            scores[task_id] = 0.0

    avg = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
    print(f"[SUMMARY] average={avg} scores={json.dumps(scores)}", flush=True)


if __name__ == "__main__":
    main()
