"""
CodeDebugEnv - OpenEnv environment for code debugging/bug fixing agents.
An AI agent is given buggy Python code and must produce fixed code.
"""

import subprocess
import sys
import tempfile
import os
import textwrap
from typing import Any
from pydantic import BaseModel


# ── Pydantic models ──────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent submits fixed Python source code."""
    fixed_code: str


class Observation(BaseModel):
    """What the agent sees each step."""
    buggy_code: str
    task_description: str
    error_hint: str          # stderr from running the buggy code (partial signal)
    attempt: int
    max_attempts: int


class State(BaseModel):
    """Full internal state (superset of observation)."""
    task_id: str
    buggy_code: str
    task_description: str
    test_cases: list[dict]   # [{"input": ..., "expected": ...}]
    error_hint: str
    attempt: int
    max_attempts: int
    done: bool
    last_score: float


# ── Task definitions ─────────────────────────────────────────────────────────

TASKS = {
    "easy_syntax": {
        "id": "easy_syntax",
        "description": (
            "Fix the syntax error in this Python function. "
            "The function should return the sum of a list of numbers."
        ),
        "buggy_code": textwrap.dedent("""\
            def sum_list(numbers)
                total = 0
                for n in numbers:
                    total += n
                return total
        """),
        "test_cases": [
            {"call": "sum_list([1, 2, 3])", "expected": 6},
            {"call": "sum_list([10, -5, 0])", "expected": 5},
            {"call": "sum_list([])", "expected": 0},
        ],
    },

    "medium_logic": {
        "id": "medium_logic",
        "description": (
            "Fix the logic bug in this function. "
            "It should return True if a number is prime, False otherwise."
        ),
        "buggy_code": textwrap.dedent("""\
            def is_prime(n):
                if n < 2:
                    return True   # bug: should be False
                for i in range(2, n):
                    if n % i == 0:
                        return False
                return True
        """),
        "test_cases": [
            {"call": "is_prime(1)", "expected": False},
            {"call": "is_prime(2)", "expected": True},
            {"call": "is_prime(9)", "expected": False},
            {"call": "is_prime(13)", "expected": True},
            {"call": "is_prime(0)", "expected": False},
        ],
    },

    "hard_algorithm": {
        "id": "hard_algorithm",
        "description": (
            "Fix the algorithmic bug in this binary search implementation. "
            "It should return the index of the target in a sorted list, or -1 if not found."
        ),
        "buggy_code": textwrap.dedent("""\
            def binary_search(arr, target):
                left, right = 0, len(arr)   # bug: should be len(arr) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1
        """),
        "test_cases": [
            {"call": "binary_search([1,3,5,7,9], 5)", "expected": 2},
            {"call": "binary_search([1,3,5,7,9], 1)", "expected": 0},
            {"call": "binary_search([1,3,5,7,9], 9)", "expected": 4},
            {"call": "binary_search([1,3,5,7,9], 4)", "expected": -1},
            {"call": "binary_search([], 1)", "expected": -1},
        ],
    },
}


# ── Grader ───────────────────────────────────────────────────────────────────

def _run_code(source: str, call: str) -> tuple[Any, str]:
    """Execute source + call in a subprocess; return (result, stderr)."""
    script = source + f"\n__result__ = {call}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=5
        )
        if proc.returncode != 0:
            return None, proc.stderr.strip()
        # extract __result__
        read_script = (
            source
            + f"\n__result__ = {call}\nimport json, sys\n"
            "try:\n    print(json.dumps(__result__))\n"
            "except Exception as e:\n    print(repr(__result__))\n"
        )
        proc2 = subprocess.run(
            [sys.executable, "-c", read_script],
            capture_output=True, text=True, timeout=5
        )
        import json
        val = json.loads(proc2.stdout.strip())
        return val, ""
    except Exception as e:
        return None, str(e)
    finally:
        os.unlink(fname)


def grade(task_id: str, fixed_code: str) -> float:
    """Return score 0.0–1.0 based on test cases passed."""
    task = TASKS[task_id]
    tests = task["test_cases"]
    if not tests:
        return 0.0
    passed = 0
    for tc in tests:
        result, err = _run_code(fixed_code, tc["call"])
        if err == "" and result == tc["expected"]:
            passed += 1
    return round(passed / len(tests), 4)


def _get_error_hint(buggy_code: str) -> str:
    """Run the buggy code to get its error output as a hint."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(buggy_code)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=5
        )
        return (proc.stderr or "No immediate runtime error — check logic.").strip()
    except Exception as e:
        return str(e)
    finally:
        os.unlink(fname)


# ── Environment ──────────────────────────────────────────────────────────────

class CodeDebugEnv:
    """
    OpenEnv-compliant environment for code debugging tasks.

    Tasks (difficulty):
        easy_syntax    — fix a syntax error         (easy)
        medium_logic   — fix a logic/semantic bug    (medium)
        hard_algorithm — fix an algorithmic bug      (hard)
    """

    MAX_ATTEMPTS = 5

    def __init__(self, task_id: str = "easy_syntax"):
        assert task_id in TASKS, f"Unknown task: {task_id}. Choose from {list(TASKS)}"
        self._task_id = task_id
        self._state: State | None = None

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        task = TASKS[self._task_id]
        hint = _get_error_hint(task["buggy_code"])
        self._state = State(
            task_id=self._task_id,
            buggy_code=task["buggy_code"],
            task_description=task["description"],
            test_cases=task["test_cases"],
            error_hint=hint,
            attempt=0,
            max_attempts=self.MAX_ATTEMPTS,
            done=False,
            last_score=0.0,
        )
        return self._obs()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        assert self._state is not None, "Call reset() first."
        assert not self._state.done, "Episode is done. Call reset()."

        self._state.attempt += 1
        score = grade(self._state.task_id, action.fixed_code)
        self._state.last_score = score

        done = score == 1.0 or self._state.attempt >= self._state.max_attempts
        self._state.done = done

        # Partial reward signal: update hint to stderr of submitted code
        if score < 1.0:
            _, err = _run_code(action.fixed_code, "None")
            if err:
                self._state.error_hint = err[:500]

        info = {
            "attempt": self._state.attempt,
            "score": score,
            "tests_passed": f"{int(score * len(self._state.test_cases))}/{len(self._state.test_cases)}",
        }
        return self._obs(), score, done, info

    def state(self) -> State:
        assert self._state is not None, "Call reset() first."
        return self._state

    # ── helpers ──────────────────────────────────────────────────────────────

    def _obs(self) -> Observation:
        s = self._state
        return Observation(
            buggy_code=s.buggy_code,
            task_description=s.task_description,
            error_hint=s.error_hint,
            attempt=s.attempt,
            max_attempts=s.max_attempts,
        )
