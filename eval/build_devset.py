"""Build the full dev set from public benchmarks + authored tasks.

Sources:
- GSM8K test split (MIT, openai/grade-school-math)      -> math_reasoning
- HumanEval (MIT, openai/human-eval)                    -> code_generation
- Bug-injected known-good functions (authored here)     -> code_debugging
- eval/tasks/dev_tasks.json (authored smoke set)        -> all categories
- eval/tasks/authored_extra.json (authored expansion)   -> 5 categories

Outputs:
- eval/tasks/all_tasks.json      everything, with gold specs
- eval/tasks/train_tasks.json    ~80% — the ONLY file tuning may touch
- eval/tasks/heldout_tasks.json  ~20% — validation only, max 2 evaluations

Deterministic: fixed seed, hash-based split. Downloads cached in eval/cache/.

Usage: python eval/build_devset.py
"""

import gzip
import hashlib
import json
import os
import random
import re
import urllib.request

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache")
TASKS_DIR = os.path.join(HERE, "tasks")

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

GSM8K_COUNT = 50
HUMANEVAL_COUNT = 50
SEED = 42
HELDOUT_FRACTION = 0.2


def fetch(url: str, filename: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, filename)
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def build_gsm8k() -> list:
    path = fetch(GSM8K_URL, "gsm8k_test.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.Random(SEED).shuffle(rows)
    tasks = []
    for i, row in enumerate(rows[:GSM8K_COUNT]):
        marker = row["answer"].rfind("####")
        gold_raw = row["answer"][marker + 4:].strip().replace(",", "")
        tasks.append({
            "task_id": f"gsm8k-{i + 1}",
            "category": "math_reasoning",
            "prompt": row["question"],
            "gold": {"answer": float(gold_raw)},
        })
    return tasks


def build_humaneval() -> list:
    path = fetch(HUMANEVAL_URL, "humaneval.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.Random(SEED).shuffle(rows)
    tasks = []
    for i, row in enumerate(rows[:HUMANEVAL_COUNT]):
        prompt_code = row["prompt"]
        tasks.append({
            "task_id": f"humaneval-{i + 1}",
            "category": "code_generation",
            "prompt": ("Complete the following Python function. "
                       "Output the complete function definition.\n\n"
                       f"```python\n{prompt_code}\n```"),
            "gold": {
                "function": row["entry_point"],
                "check_code": row["test"],
                "context": prompt_code,
                "_reference": prompt_code + row["canonical_solution"],
            },
        })
    return tasks


# (correct_source, buggy_source, function, tests) — bugs are realistic slips:
# off-by-one, wrong comparison, bad initialization, inverted condition.
_DEBUG_SPECS = [
    ("factorial",
     "def factorial(n):\n    result = 1\n    for i in range(2, n):\n        result *= i\n    return result",
     [{"args": [5], "expected": 120}, {"args": [1], "expected": 1}]),
    ("count_evens",
     "def count_evens(nums):\n    count = 0\n    for n in nums:\n        if n % 2 == 1:\n            count += 1\n    return count",
     [{"args": [[2, 4, 5]], "expected": 2}, {"args": [[1, 3]], "expected": 0}]),
    ("min_value",
     "def min_value(nums):\n    best = 0\n    for n in nums:\n        if n < best:\n            best = n\n    return best",
     [{"args": [[5, 2, 8]], "expected": 2}, {"args": [[-1, -5]], "expected": -5}]),
    ("join_words",
     "def join_words(words, sep):\n    result = ''\n    for w in words:\n        result += w + sep\n    return result",
     [{"args": [["a", "b"], "-"], "expected": "a-b"}, {"args": [["x"], ","], "expected": "x"}]),
    ("clamp",
     "def clamp(x, low, high):\n    if x < low:\n        return high\n    if x > high:\n        return low\n    return x",
     [{"args": [5, 0, 10], "expected": 5}, {"args": [-3, 0, 10], "expected": 0}, {"args": [42, 0, 10], "expected": 10}]),
    ("running_total",
     "def running_total(nums):\n    totals = []\n    total = 0\n    for n in nums:\n        totals.append(total)\n        total += n\n    return totals",
     [{"args": [[1, 2, 3]], "expected": [1, 3, 6]}, {"args": [[]], "expected": []}]),
    ("char_frequency",
     "def char_frequency(s):\n    freq = {}\n    for ch in s:\n        freq[ch] = 1\n    return freq",
     [{"args": ["aab"], "expected": {"a": 2, "b": 1}}, {"args": [""], "expected": {}}]),
    ("last_index_of",
     "def last_index_of(items, target):\n    for i in range(len(items)):\n        if items[i] == target:\n            return i\n    return -1",
     [{"args": [[1, 2, 1], 1], "expected": 2}, {"args": [[3], 9], "expected": -1}]),
    ("is_sorted",
     "def is_sorted(nums):\n    for i in range(len(nums) - 1):\n        if nums[i] > nums[i + 1]:\n            return True\n    return False",
     [{"args": [[1, 2, 3]], "expected": True}, {"args": [[2, 1]], "expected": False}]),
    ("celsius_to_fahrenheit",
     "def celsius_to_fahrenheit(c):\n    return c * 5 / 9 + 32",
     [{"args": [0], "expected": 32.0}, {"args": [100], "expected": 212.0}]),
    ("remove_duplicates_keep_order",
     "def remove_duplicates_keep_order(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x in seen:\n            result.append(x)\n            seen.add(x)\n    return result",
     [{"args": [[1, 2, 1, 3]], "expected": [1, 2, 3]}, {"args": [[]], "expected": []}]),
    ("word_lengths",
     "def word_lengths(sentence):\n    return [len(w) for w in sentence]",
     [{"args": ["hi there"], "expected": [2, 5]}, {"args": [""], "expected": []}]),
]


_DEBUG_FIXES = {
    "factorial": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
    "count_evens": "def count_evens(nums):\n    return sum(1 for n in nums if n % 2 == 0)",
    "min_value": "def min_value(nums):\n    return min(nums)",
    "join_words": "def join_words(words, sep):\n    return sep.join(words)",
    "clamp": "def clamp(x, low, high):\n    return max(low, min(high, x))",
    "running_total": "def running_total(nums):\n    totals = []\n    total = 0\n    for n in nums:\n        total += n\n        totals.append(total)\n    return totals",
    "char_frequency": "def char_frequency(s):\n    freq = {}\n    for ch in s:\n        freq[ch] = freq.get(ch, 0) + 1\n    return freq",
    "last_index_of": "def last_index_of(items, target):\n    for i in range(len(items) - 1, -1, -1):\n        if items[i] == target:\n            return i\n    return -1",
    "is_sorted": "def is_sorted(nums):\n    for i in range(len(nums) - 1):\n        if nums[i] > nums[i + 1]:\n            return False\n    return True",
    "celsius_to_fahrenheit": "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
    "remove_duplicates_keep_order": "def remove_duplicates_keep_order(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x not in seen:\n            result.append(x)\n            seen.add(x)\n    return result",
    "word_lengths": "def word_lengths(sentence):\n    return [len(w) for w in sentence.split()]",
}


def build_debug_tasks() -> list:
    tasks = []
    for i, (name, buggy, tests) in enumerate(_DEBUG_SPECS):
        tasks.append({
            "task_id": f"xdebug-{i + 1}",
            "category": "code_debugging",
            "prompt": ("This Python function has a bug. Fix it and output the "
                       f"corrected function:\n```python\n{buggy}\n```"),
            "gold": {"function": name, "tests": tests,
                     "_reference": _DEBUG_FIXES[name]},
        })
    return tasks


def load_authored() -> list:
    tasks = []
    for filename in ("dev_tasks.json", "authored_extra.json"):
        path = os.path.join(TASKS_DIR, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                tasks.extend(json.load(f))
        else:
            print(f"note: {filename} not found, skipping")
    return tasks


def split(tasks: list) -> tuple:
    """Deterministic hash split, stable as the set grows."""
    train, heldout = [], []
    for task in tasks:
        digest = int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16)
        (heldout if (digest % 100) < HELDOUT_FRACTION * 100 else train).append(task)
    return train, heldout


def main() -> int:
    all_tasks = load_authored() + build_gsm8k() + build_humaneval() + build_debug_tasks()

    ids = [t["task_id"] for t in all_tasks]
    assert len(ids) == len(set(ids)), "duplicate task_id"

    train, heldout = split(all_tasks)
    for name, subset in (("all_tasks", all_tasks), ("train_tasks", train),
                         ("heldout_tasks", heldout)):
        path = os.path.join(TASKS_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=1)
        print(f"{name}: {len(subset)} tasks -> {path}")

    by_category = {}
    for t in all_tasks:
        by_category[t["category"]] = by_category.get(t["category"], 0) + 1
    print(json.dumps(by_category, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
