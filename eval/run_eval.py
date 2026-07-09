"""Local eval harness: run the dev set through the router and report
accuracy, offload rate, and remote token count — the three numbers that
decide the leaderboard.

Modes:
  --mock            offline run with deterministic fake models (no GPU/key)
  --local-only      force every task local (measures raw local accuracy)
  --remote-only     force every task remote (upper bound + token cost)

Usage:
  python eval/run_eval.py --mock
  python eval/run_eval.py                # real endpoints from env
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.clients import MockClient, OpenAICompatClient
from frugal_router.config import config_from_env
from frugal_router.pipeline import run_batch
from frugal_router.schemas import Category, Task
from graders import grade

DEV_TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks", "dev_tasks.json")


def load_dev_tasks(path: str = DEV_TASKS_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_clients(args, dev_tasks):
    if args.mock:
        answer_book = {t["prompt"]: _gold_as_text(t) for t in dev_tasks}
        local = MockClient(answer_book, accuracy=args.mock_local_acc)
        remote = MockClient(answer_book, accuracy=args.mock_remote_acc, seed=99)
        return local, remote
    config = config_from_env()
    return (
        OpenAICompatClient(config.local_base_url, extra_body=config.local_extra_body),
        OpenAICompatClient(
            config.fireworks_base_url,
            config.fireworks_api_key,
            timeout=config.remote_timeout_seconds,
            max_retries=0,
        ),
    )


def _gold_as_text(task: dict) -> str:
    gold = task["gold"]
    if "_reference" in gold:
        return gold["_reference"]
    if "answer" in gold:
        return str(gold["answer"])
    if "entities" in gold:
        return ", ".join(gold["entities"])
    if "keywords" in gold:
        return " ".join(gold["keywords"]) + " (summary)"
    if "function" in gold:
        return _reference_code(task["task_id"], gold["function"])
    return ""


def _reference_code(task_id: str, function: str) -> str:
    """Correct reference solutions so the mock can 'answer' code tasks."""
    solutions = {
        "average": "def average(numbers):\n    return sum(numbers) / len(numbers)",
        "find_max": "def find_max(items):\n    return max(items)",
        "sum_to_n": "def sum_to_n(n):\n    return n * (n + 1) // 2",
        "reverse_string": "def reverse_string(s):\n    return s[::-1]",
        "is_prime": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "fizzbuzz": "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
        "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "merge_sorted": "def merge_sorted(a, b):\n    return sorted(a + b)",
        "is_palindrome": "def is_palindrome(s):\n    t = ''.join(s.lower().split())\n    return t == t[::-1]",
        "second_largest": "def second_largest(xs):\n    return sorted(xs)[-2]",
    }
    return solutions[function]


def apply_mode(config, args):
    if args.local_only:
        thresholds = {cat: 0.0 for cat in Category}
        return _replace_thresholds(config, thresholds)
    if args.remote_only:
        thresholds = {cat: 1.01 for cat in Category}
        return _replace_thresholds(config, thresholds)
    return config


def _replace_thresholds(config, thresholds):
    from dataclasses import replace
    return replace(config, thresholds=thresholds)


def evaluate(config, local, remote, dev_tasks) -> dict:
    tasks = [Task(task_id=t["task_id"], prompt=t["prompt"]) for t in dev_tasks]
    gold_by_id = {t["task_id"]: t for t in dev_tasks}

    results = run_batch(config, local, remote, tasks)

    per_category = defaultdict(lambda: {"correct": 0, "total": 0, "local": 0})
    correct = 0
    remote_tokens = 0
    dump = []
    for r in results:
        dev = gold_by_id[r.task_id]
        ok = grade(dev["category"], r.answer, dev["gold"])
        stats = per_category[dev["category"]]
        stats["total"] += 1
        stats["correct"] += int(ok)
        stats["local"] += int(r.remote_tokens == 0)
        correct += int(ok)
        remote_tokens += r.remote_tokens
        dump.append({
            "task_id": r.task_id, "category": dev["category"],
            "route": r.route.value, "model": r.model, "correct": ok,
            "remote_tokens": r.remote_tokens, "reason": r.reason,
            "answer": r.answer,  # FULL answer — truncation broke offline regrading
        })

    total = len(results)
    local_count = sum(1 for r in results if r.remote_tokens == 0)
    return {
        "accuracy": correct / total,
        "offload_rate": local_count / total,
        "remote_tokens": remote_tokens,
        "per_category": {k: dict(v) for k, v in sorted(per_category.items())},
        "_dump": dump,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-local-acc", type=float, default=0.75)
    parser.add_argument("--mock-remote-acc", type=float, default=0.95)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--remote-only", action="store_true")
    parser.add_argument("--tasks", default=DEV_TASKS_PATH)
    parser.add_argument("--dump", default=None,
                        help="write per-task records (route, correctness, answer) to this JSON file")
    args = parser.parse_args()

    dev_tasks = load_dev_tasks(args.tasks)
    config = apply_mode(config_from_env(), args)
    local, remote = build_clients(args, dev_tasks)

    summary = evaluate(config, local, remote, dev_tasks)
    dump = summary.pop("_dump")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
