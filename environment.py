"""
CodeDebugEnv — Core Environment
Scores are always strictly between 0 and 1 (never 0.0 or 1.0).
"""
import ast
import sys
import textwrap
import uuid
from io import StringIO
from typing import Any

TASKS: dict[str, dict[str, Any]] = {
    "easy_syntax": {
        "difficulty": "easy",
        "description": (
            "Fix the Python function so it correctly returns the sum of all numbers in a list. "
            "The function has a syntax error preventing it from running."
        ),
        "buggy_code": textwrap.dedent("""\
            def sum_list(numbers)
                total = 0
                for n in numbers:
                    total += n
                return total
        """),
        "error_hint": "SyntaxError: expected ':' after function definition",
        "tests": [
            {"input": [],           "expected": 0},
            {"input": [1, 2, 3],    "expected": 6},
            {"input": [-1, -2, -3], "expected": -6},
            {"input": [0, 0, 0],    "expected": 0},
            {"input": [100],        "expected": 100},
            {"input": [1,2,3,4,5],  "expected": 15},
            {"input": [10, -10],    "expected": 0},
        ],
        "test_fn": "sum_list",
    },
    "medium_logic": {
        "difficulty": "medium",
        "description": (
            "Fix the is_prime function. It should return True if n is prime, False otherwise. "
            "The base-case logic is wrong causing misclassification of small numbers."
        ),
        "buggy_code": textwrap.dedent("""\
            def is_prime(n):
                if n < 2:
                    return True
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
        """),
        "error_hint": "is_prime(0) returns True but should return False",
        "tests": [
            {"input": 0,   "expected": False},
            {"input": 1,   "expected": False},
            {"input": 2,   "expected": True},
            {"input": 3,   "expected": True},
            {"input": 4,   "expected": False},
            {"input": 17,  "expected": True},
            {"input": 18,  "expected": False},
            {"input": 97,  "expected": True},
            {"input": 100, "expected": False},
            {"input": 101, "expected": True},
            {"input": 7,   "expected": True},
        ],
        "test_fn": "is_prime",
    },
    "hard_algorithm": {
        "difficulty": "hard",
        "description": (
            "Fix the binary_search function. It should return the index of target in sorted arr, "
            "or -1 if not found. There is an off-by-one error causing wrong answers."
        ),
        "buggy_code": textwrap.dedent("""\
            def binary_search(arr, target):
                left, right = 0, len(arr)
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
        "error_hint": "IndexError: list index out of range when right = len(arr)",
        "tests": [
            {"input": ([1,3,5,7,9], 5),          "expected": 2},
            {"input": ([1,3,5,7,9], 1),          "expected": 0},
            {"input": ([1,3,5,7,9], 9),          "expected": 4},
            {"input": ([1,3,5,7,9], 4),          "expected": -1},
            {"input": ([42], 42),                 "expected": 0},
            {"input": ([42], 0),                  "expected": -1},
            {"input": (list(range(0,100,2)), 50), "expected": 25},
            {"input": (list(range(0,100,2)), 51), "expected": -1},
            {"input": ([2,4,6,8,10], 6),         "expected": 2},
            {"input": ([2,4,6,8,10], 11),        "expected": -1},
            {"input": ([1,2,3,4,5], 3),          "expected": 2},
        ],
        "test_fn": "binary_search",
    },
}


def _safe_score(passed: int, total: int) -> float:
    """Return score strictly between 0 and 1. Never 0.0 or 1.0."""
    if total == 0:
        return 0.5
    raw = passed / total
    # Map [0,1] to (0.05, 0.95) strictly
    score = 0.05 + raw * 0.90
    return round(score, 4)


def _exec_code(source: str, fn_name: str):
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc}"
    namespace = {}
    try:
        exec(compile(source, "<submitted>", "exec"), namespace)
    except Exception as exc:
        return None, f"RuntimeError: {exc}"
    fn = namespace.get(fn_name)
    if fn is None or not callable(fn):
        return None, f"NameError: '{fn_name}' not found"
    return fn, None


def _run_tests(fn, task):
    results = []
    passed = 0
    for t in task["tests"]:
        inp, expected = t["input"], t["expected"]
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            got = fn(*inp) if isinstance(inp, tuple) else fn(inp)
            sys.stdout = old_stdout
            ok = got == expected
        except Exception as exc:
            sys.stdout = old_stdout
            ok = False
            got = f"EXCEPTION: {exc}"
        if ok:
            passed += 1
        results.append({"input": str(inp), "expected": expected, "got": str(got), "passed": ok})
    score = _safe_score(passed, len(task["tests"]))
    return score, results


def grade(task_id: str, fixed_code: str) -> dict:
    task = TASKS.get(task_id)
    if task is None:
        return {"score": 0.05, "error": f"Unknown task '{task_id}'", "test_results": []}
    fn, err = _exec_code(fixed_code, task["test_fn"])
    if fn is None:
        return {"score": 0.05, "error": err, "test_results": []}
    score, details = _run_tests(fn, task)
    return {"score": score, "error": None, "test_results": details}


class Episode:
    MAX_ATTEMPTS = 5

    def __init__(self, task_id: str):
        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.attempt = 0
        self.done = False
        self.best_score = 0.05
        self.history = []

    def observation(self):
        task = TASKS[self.task_id]
        return {
            "task_id": self.task_id,
            "difficulty": task["difficulty"],
            "buggy_code": task["buggy_code"],
            "task_description": task["description"],
            "error_hint": task["error_hint"],
            "attempt": self.attempt,
            "max_attempts": self.MAX_ATTEMPTS,
        }

    def step(self, fixed_code: str):
        if self.done:
            return {"observation": self.observation(), "reward": 0.05, "done": True, "info": {}}
        self.attempt += 1
        result = grade(self.task_id, fixed_code)
        score = result["score"]
        if score > self.best_score:
            self.best_score = score
        self.done = score >= 0.94 or self.attempt >= self.MAX_ATTEMPTS
        self.history.append({"attempt": self.attempt, "score": score})
        return {
            "observation": self.observation(),
            "reward": score,
            "done": self.done,
            "info": {
                "grader_error": result["error"],
                "test_results": result["test_results"],
                "best_score": self.best_score,
            },
        }

    def state(self):
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "max_attempts": self.MAX_ATTEMPTS,
            "done": self.done,
            "best_score": self.best_score,
            "history": self.history,
        }


class CodeDebugEnv:
    def __init__(self):
        self._sessions = {}

    def reset(self, task_id=None):
        if task_id is None:
            task_id = "easy_syntax"
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}")
        ep = Episode(task_id)
        self._sessions[ep.episode_id] = ep
        return {"session_id": ep.episode_id, "observation": ep.observation()}

    def step(self, session_id: str, fixed_code: str):
        ep = self._sessions.get(session_id)
        if ep is None:
            raise KeyError(f"session_id '{session_id}' not found.")
        return ep.step(fixed_code)

    def state(self, session_id: str):
        ep = self._sessions.get(session_id)
        if ep is None:
            raise KeyError(f"session_id '{session_id}' not found.")
        return ep.state()

    @staticmethod
    def list_tasks():
        return [
            {"task_id": tid, "difficulty": t["difficulty"], "description": t["description"]}
            for tid, t in TASKS.items()
        ]
