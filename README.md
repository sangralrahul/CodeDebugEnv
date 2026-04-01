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

**An OpenEnv environment where AI agents fix buggy Python code.**

CodeDebugEnv presents an agent with broken Python functions and challenges it to produce working fixes. This environment models a real-world software engineering workflow — automated bug fixing — that has immediate value for evaluating code-capable LLM agents.

---

## Environment Description

The agent receives a buggy Python function along with a natural-language description of what the function should do and an error hint (stderr from running the buggy code). The agent must return corrected Python source code. A grader runs the submitted code against hidden test cases and returns a score between 0.0 and 1.0.

This design provides **dense partial rewards** (fraction of tests passing) rather than sparse binary feedback, making it well-suited for RL training.

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
  "buggy_code": "<original buggy Python code>",
  "task_description": "<what the function should do>",
  "error_hint": "<stderr from running the buggy code>",
  "attempt": 1,
  "max_attempts": 5
}
```

---

## Tasks

| Task ID | Difficulty | Description | Expected Baseline Score |
|---|---|---|---|
| `easy_syntax` | Easy | Fix a missing colon (syntax error) in a sum function | ~1.0 |
| `medium_logic` | Medium | Fix a wrong base-case return in an is_prime function | ~0.6 |
| `hard_algorithm` | Hard | Fix an off-by-one error in binary search | ~0.4 |

### Scoring
- Score = fraction of test cases passed (e.g., 3/5 → 0.6)
- Graders are deterministic and reproducible
- Score range: **0.0 – 1.0**

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

**Reset:**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_syntax"}'
```

**Step:**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<your-session-id>",
    "fixed_code": "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total\n"
  }'
```

---

## Setup & Running

### Local (Docker)
```bash
git clone https://huggingface.co/spaces/<your-username>/CodeDebugEnv
cd CodeDebugEnv
docker build -t code-debug-env .
docker run -p 7860:7860 code-debug-env
```

### Local (Python)
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## Baseline Inference Script

Run the baseline agent against all three tasks:

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="your_hf_token"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Baseline Scores (Meta-Llama-3-8B-Instruct)

| Task | Score |
|---|---|
| easy_syntax | 1.0 |
| medium_logic | 0.6 |
| hard_algorithm | 0.4 |
| **Average** | **0.67** |

---

## Project Structure

```
.
├── environment.py    # Core OpenEnv environment (CodeDebugEnv class)
├── app.py            # FastAPI server exposing HTTP endpoints
├── inference.py      # Baseline inference script
├── openenv.yaml      # OpenEnv specification file
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build file
└── README.md         # This file
```

---

## Reward Design

The reward function awards **partial credit** based on the fraction of test cases that pass. This provides a meaningful, varying signal that avoids the sparse-reward problem common in code evaluation environments. The agent can see an error hint (stderr) after each failed attempt, enabling iterative refinement.

Episode terminates when either the agent achieves a perfect score (1.0) or exhausts its 5 allowed attempts.
