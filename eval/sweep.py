"""Threshold sweep: the tuning loop that wins or loses the leaderboard.

Runs the eval at a range of global consistency thresholds and prints the
accuracy / offload-rate / remote-token frontier so we can pick the operating
point that stays comfortably above the (undisclosed) accuracy bar while
maximizing free local answers.

Usage:
  python eval/sweep.py --mock
  python eval/sweep.py                  # real endpoints from env
"""

import argparse
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(__file__))

from run_eval import DEV_TASKS_PATH, build_clients, evaluate, load_dev_tasks
from frugal_router.config import config_from_env
from frugal_router.schemas import Category

FREEFORM = {Category.SENTIMENT, Category.SUMMARIZATION}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-local-acc", type=float, default=0.75)
    parser.add_argument("--mock-remote-acc", type=float, default=0.95)
    parser.add_argument("--local-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tasks", default=DEV_TASKS_PATH)
    parser.add_argument("--steps", type=float, nargs="*",
                        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.01])
    args = parser.parse_args()

    dev_tasks = load_dev_tasks(args.tasks)
    base_config = config_from_env()
    local, remote = build_clients(args, dev_tasks)

    print(f"{'threshold':>9} | {'accuracy':>8} | {'offload':>7} | {'remote_tokens':>13}")
    print("-" * 48)
    rows = []
    for threshold in args.steps:
        thresholds = {
            cat: (0.0 if cat in FREEFORM else threshold) for cat in Category
        }
        config = replace(base_config, thresholds=thresholds)
        summary = evaluate(config, local, remote, dev_tasks)
        summary.pop("_dump", None)
        rows.append({"threshold": threshold, **summary})
        print(f"{threshold:>9.2f} | {summary['accuracy']:>8.2%} | "
              f"{summary['offload_rate']:>7.2%} | {summary['remote_tokens']:>13}")

    out_path = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
