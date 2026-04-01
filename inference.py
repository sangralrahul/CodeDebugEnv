"""
inference.py — Baseline inference script for CodeDebugEnv.

Mandatory stdout format:
  [START] task=<task_id> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
  export API_BASE_URL="https://api-inference.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
  export HF_TOKEN="your_hf_token"
  export ENV_URL="http://localhost:7860"
  python inference.py
"""

import os
import requests
from typing import Optional, List
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       "http://localhost:7860")
BENCHMARK    = "code_debug_env"
MAX_STEPS    = 5
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
TASKS = ["easy_syntax", "medium_logic", "hard_algorithm"]


# ── Mandatory log helpers ─────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_oneline = action.replace("\n", "\\n")[:120]
    print(f"[STEP] step={step} action={action_oneline} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Agent ─────────────────────────────────────────────────────────────────────

def agent_fix_code(buggy_code: str, task_description: str, error_hint: str) -> str:
    prompt = (
        f"You are an expert Python debugger.\n\n"
        f"Task: {task_description}\n\n"
        f"Buggy code:\n{buggy_code}\n\n"
        f"Error hint: {error_hint}\n\n"
        f"Return ONLY the fixed Python code. No explanation, no markdown fences."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()
        return raw
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return buggy_code


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    # Reset environment
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    data       = r.json()
    session_id = data["session_id"]
    obs        = data["observation"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        fixed_code = agent_fix_code(
            obs["buggy_code"],
            obs["task_description"],
            obs["error_hint"],
        )

        r = requests.post(f"{ENV_URL}/step", json={
            "session_id": session_id,
            "fixed_code": fixed_code,
        })
        r.raise_for_status()
        step_data = r.json()

        reward = step_data["reward"]
        done   = step_data["done"]
        error  = step_data["info"].get("error") if step_data.get("info") else None
        obs    = step_data["observation"]

        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=fixed_code, reward=reward, done=done, error=error)

        if done:
            break

    # Compute normalized score (max reward per step = 1.0)
    max_possible = MAX_STEPS * 1.0
    score   = sum(rewards) / max_possible if max_possible > 0 else 0.0
    score   = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}
    for task_id in TASKS:
        score = run_task(task_id)
        results[task_id] = score

    avg = sum(results.values()) / len(results)
    print(f"\n[SUMMARY] avg_score={avg:.3f} tasks={','.join(results.keys())}", flush=True)


if __name__ == "__main__":
    main()
