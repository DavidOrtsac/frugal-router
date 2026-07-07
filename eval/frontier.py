"""Offline threshold frontier from recorded runs — the tuning core.

Inputs (per-task dumps produced by run_eval.py --dump):
  --local  dump of a --local-only run (has consistency score + local correctness)
  --remote NAME=dumpfile   one per candidate expert (remote-only run)

For every category x expert x threshold, replays the routing decision offline:
  consistency >= threshold -> local answer (its recorded correctness, 0 tokens)
  consistency <  threshold -> expert answer (recorded correctness + tokens)

Then picks, per category: the expert and threshold that maximize accuracy,
tie-broken by fewest tokens. Prints the recommended config and the projected
overall score. No model calls, instant.

Usage:
  python eval/frontier.py --local dump_local.json \
      --remote kimi=dump_kimi.json --remote minimax=dump_minimax.json
"""

import argparse
import json
import re
from collections import defaultdict

THRESHOLD_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
_CONSISTENCY = re.compile(r"consistency (\d\.\d+)")


def load_local(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    out = {}
    for r in records:
        m = _CONSISTENCY.search(r["reason"])
        if not m:
            continue
        out[r["task_id"]] = {
            "category": r["category"],
            "consistency": float(m.group(1)),
            "correct": bool(r["correct"]),
        }
    return out


def load_remote(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {r["task_id"]: {"correct": bool(r["correct"]),
                           "tokens": int(r["remote_tokens"])} for r in records}


def evaluate_combo(tasks: list, local: dict, remote: dict, threshold: float) -> dict:
    correct = tokens = escalated = 0
    for tid in tasks:
        line = local[tid]
        if threshold <= 1.0 and line["consistency"] >= threshold:
            correct += int(line["correct"])
        else:
            r = remote.get(tid)
            if r is None:  # expert has no recording for this task; treat as local
                correct += int(line["correct"])
                continue
            correct += int(r["correct"])
            tokens += r["tokens"]
            escalated += 1
    n = len(tasks)
    return {"accuracy": correct / n, "tokens": tokens,
            "escalated": escalated, "n": n}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", action="append", default=[],
                        help="NAME=dumpfile, repeatable")
    args = parser.parse_args()

    local = load_local(args.local)
    experts = {}
    for spec in args.remote:
        name, _, path = spec.partition("=")
        experts[name] = load_remote(path)

    by_category = defaultdict(list)
    for tid, line in local.items():
        by_category[line["category"]].append(tid)

    plan = {}
    print(f"{'category':<26} {'expert':<10} {'thresh':>6} {'acc':>7} {'tokens':>7} {'esc':>4}")
    print("-" * 66)
    for category, tasks in sorted(by_category.items()):
        best = None
        for name, remote in experts.items():
            for threshold in THRESHOLD_GRID:
                result = evaluate_combo(tasks, local, remote, threshold)
                key = (result["accuracy"], -result["tokens"])
                if best is None or key > best[0]:
                    best = (key, name, threshold, result)
        _, name, threshold, result = best
        plan[category] = {"expert": name, "threshold": threshold, **result}
        print(f"{category:<26} {name:<10} {threshold:>6.2f} "
              f"{result['accuracy']:>7.2%} {result['tokens']:>7} {result['escalated']:>4}")

    total = sum(p["n"] for p in plan.values())
    weighted_acc = sum(p["accuracy"] * p["n"] for p in plan.values()) / total
    total_tokens = sum(p["tokens"] for p in plan.values())
    total_escalated = sum(p["escalated"] for p in plan.values())
    print("-" * 66)
    print(f"PROJECTED: accuracy {weighted_acc:.2%}, remote tokens {total_tokens}, "
          f"escalated {total_escalated}/{total} ({1 - total_escalated / total:.1%} free)")
    print(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
