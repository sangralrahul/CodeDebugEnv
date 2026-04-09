---
title: CodeDebugEnv
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# CodeDebugEnv 🐛

> **Meta × PyTorch OpenEnv Hackathon submission**
> An OpenEnv environment where AI agents learn to fix real-world buggy Python code.

---

## Overview

CodeDebugEnv presents an agent with broken Python functions and challenges it to produce working fixes. This environment models **automated program repair** — a high-value software-engineering task with direct real-world application for evaluating and training code-capable LLM agents.

**Why this environment is interesting for RL:**

- **Dense partial rewards** (fraction of tests passing) avoid the sparse-reward problem
- **Iterative refinement** — the agent can observe *which* tests fail and improve
- **Three difficulty tiers** create a natural curriculum: syntax → logic → algorithm
- **Deterministic grading** ensures reproducible, fair evaluation

---

## Environment Description

The agent receives a buggy Python function along with:

1. A natural-language description of what the function should do
2. A stderr error hint from running the buggy code

The agent must return corrected Python source code. A hidden test suite runs the submission and returns a reward between `0.0` and `1.0`.

---

## Action & Observation Spaces

### Action

```json
{
  "fixed_code": "<corrected Python source code as a string>"
}
```

### Observation

```json
{
  "task_id":          "medium_logic",
  "difficulty":       "medium",
  "buggy_code":       "<original buggy code>",
  "task_description": "<what the function should do>",
  "error_hint":       "<stderr from running the buggy code>",
  "attempt":          1,
  "max_attempts":     5
}
```

---

## Tasks

| Task ID | Difficulty | Bug Type | Description | Baseline Score |
|---|---|---|---|---|
| `easy_syntax` | Easy | Syntax error | Missing colon in `sum_list` | ~1.0 |
| `medium_logic` | Medium | Logic bug | Wrong base case in `is_prime` | ~0.75 |
| `hard_algorithm` | Hard | Off-by-one | Wrong bound in `binary_search` | ~0.50 |

### Scoring

- Score = fraction of **hidden** test cases passed (e.g. 6/8 → 0.75)
- All graders are deterministic and reproducible
- Score range: **0.0 – 1.0**
- Episode ends when score = 1.0 or 5 attempts are exhausted

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/tasks` | List all available tasks |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action (fixed code) |
| `GET` | `/state/{session_id}` | Get full internal state |

### Example Usage

**Health check:**
```bash
curl http://localhost:7860/health
```

**List tasks:**
```bash
curl http://localhost:7860/tasks
```

**Reset (start episode):**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_syntax"}'
```

**Step (submit fix):**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<your-session-id>",
    "fixed_code": "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total\n"
  }'
```

---

## Reward Design

The reward function awards **partial credit** based on the fraction of test cases that pass. This provides a meaningful, varying signal that avoids the sparse-reward problem common in code evaluation environments.

The agent sees:
- The original buggy code (unchanged)
- An error hint (stderr) after each failed attempt
- Its attempt counter, enabling iterative refinement

This design is well-suited for RL post-training: the agent can explore different fixes and receive incremental feedback on progress.

---

## Setup & Running

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/<your-username>/CodeDebugEnv
cd CodeDebugEnv
docker build -t code-debug-env .
docker run -p 7860:7860 code-debug-env
```

### Option 2 — Local Python

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## Baseline Inference Script

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Baseline Scores (Meta-Llama-3-8B-Instruct)

| Task | Score |
|---|---|
| easy\_syntax | 1.00 |
| medium\_logic | 0.75 |
| hard\_algorithm | 0.50 |
| **Average** | **0.75** |

---

## Project Structure

```
.
├── environment.py   # Core OpenEnv environment (CodeDebugEnv class + graders)
├── app.py           # FastAPI server exposing HTTP endpoints
├── inference.py     # Baseline inference script (OpenAI client)
├── openenv.yaml     # OpenEnv specification file
├── requirements.txt # Python dependencies
├── pyproject.toml   # Package metadata
└── Dockerfile       # Container build file
```
