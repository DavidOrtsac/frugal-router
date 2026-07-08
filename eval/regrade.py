"""Re-grade an existing dump offline after an extraction fix.

The dump stores each task's raw stored answer; we re-run extract_answer and
the graders against the gold specs — no model calls, instant. Writes a
corrected dump alongside and prints the corrected per-category summary.

Usage: python eval/regrade.py <dump.json> [--tasks eval/tasks/train_tasks.json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from frugal_router.prompts import extract_answer
from frugal_router.schemas import Category
from graders import grade


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--tasks", default=os.path.join(
        os.path.dirname(__file__), "tasks", "train_tasks.json"))
    args = parser.parse_args()

    with open(args.tasks, encoding="utf-8") as f:
        gold_by_id = {t["task_id"]: t for t in json.load(f)}
    with open(args.dump, encoding="utf-8") as f:
        records = json.load(f)

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    fixed = []
    flips = 0
    for r in records:
        dev = gold_by_id[r["task_id"]]
        answer = extract_answer(Category(r["category"]), r["answer"])
        ok = grade(dev["category"], answer, dev["gold"])
        flips += int(ok != r["correct"])
        stats = per_category[dev["category"]]
        stats["total"] += 1
        stats["correct"] += int(ok)
        fixed.append({**r, "answer": answer, "correct": ok})

    out_path = args.dump.replace(".json", "_regraded.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=1)

    total = sum(v["total"] for v in per_category.values())
    correct = sum(v["correct"] for v in per_category.values())
    print(f"regraded {total} tasks, {flips} verdicts changed -> {out_path}")
    print(f"ACCURACY: {correct / total:.1%}")
    for k, v in sorted(per_category.items()):
        print(f"  {k:<28} {v['correct']}/{v['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
