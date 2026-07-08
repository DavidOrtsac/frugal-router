"""Grade a container's /output/results.json against a gold task file.

This is the dress-rehearsal scorer: feed the container an unseen tasks.json,
then grade what it wrote, exactly as an external judge would.

Usage: python eval/grade_results.py results.json eval/tasks/heldout_tasks.json
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from graders import grade


def main() -> int:
    results_path, gold_path = sys.argv[1], sys.argv[2]
    with open(results_path, encoding="utf-8") as f:
        answers = {r["task_id"]: r.get("answer", "") for r in json.load(f)}
    with open(gold_path, encoding="utf-8") as f:
        gold_tasks = json.load(f)

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    missing = 0
    for task in gold_tasks:
        answer = answers.get(task["task_id"])
        if answer is None:
            missing += 1
            answer = ""
        ok = grade(task["category"], answer, task["gold"])
        stats = per_category[task["category"]]
        stats["total"] += 1
        stats["correct"] += int(ok)

    total = sum(v["total"] for v in per_category.values())
    correct = sum(v["correct"] for v in per_category.values())
    print(f"ACCURACY: {correct / total:.1%} ({correct}/{total}, {missing} missing)")
    for k, v in sorted(per_category.items()):
        print(f"  {k:<28} {v['correct']}/{v['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
