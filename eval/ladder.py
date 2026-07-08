"""The submission ladder: the accuracy/token curve as executable configs.

Scoring is a hidden accuracy gate, then fewest tokens wins among passers.
The optimal play is therefore a LADDER of configurations from safest
(max accuracy, most tokens) to stingiest (pure local, zero tokens):
submit down the ladder until ACCURACY_GATE_FAILED, ship the last survivor.

Greedy construction: start pure-local, then repeatedly buy the single
category-threshold upgrade with the best marginal accuracy-per-token until
maxed. Each rung prints its projected accuracy, token cost, and the exact
THRESHOLDS_JSON to ship it.

Usage:
  python eval/ladder.py --local dump_local.json --remote kimi=dump_kimi.json
"""

import argparse
import json
from collections import defaultdict

from frontier import THRESHOLD_GRID, evaluate_combo, load_local, load_remote


def category_options(tasks, local, remote):
    """All (threshold, correct, tokens) options for one category, deduped to
    the efficient frontier (no option dominated by a cheaper, better one)."""
    options = []
    for threshold in THRESHOLD_GRID:
        result = evaluate_combo(tasks, local, remote, threshold)
        correct = round(result["accuracy"] * result["n"])
        options.append({"threshold": threshold, "correct": correct,
                        "tokens": result["tokens"]})
    best = {}
    for opt in options:
        key = (opt["correct"], opt["tokens"])
        best[key] = opt
    frontier = []
    for opt in sorted(best.values(), key=lambda o: (o["tokens"], -o["correct"])):
        if not frontier or opt["correct"] > frontier[-1]["correct"]:
            frontier.append(opt)
    return frontier


def build_ladder(by_category, local, remote):
    options = {cat: category_options(tasks, local, remote)
               for cat, tasks in by_category.items()}
    state = {cat: 0 for cat in options}  # index into each category's frontier
    total_n = sum(len(t) for t in by_category.values())

    def totals():
        correct = sum(options[c][state[c]]["correct"] for c in state)
        tokens = sum(options[c][state[c]]["tokens"] for c in state)
        return correct, tokens

    rungs = []
    correct, tokens = totals()
    rungs.append(_rung(state, options, correct, tokens, total_n))
    while True:
        best_move = None
        for cat in state:
            idx = state[cat]
            if idx + 1 >= len(options[cat]):
                continue
            nxt, cur = options[cat][idx + 1], options[cat][idx]
            gain = nxt["correct"] - cur["correct"]
            cost = nxt["tokens"] - cur["tokens"]
            if gain <= 0:
                continue
            ratio = gain / max(cost, 1)
            if best_move is None or ratio > best_move[0]:
                best_move = (ratio, cat)
        if best_move is None:
            break
        state[best_move[1]] += 1
        correct, tokens = totals()
        rungs.append(_rung(state, options, correct, tokens, total_n))
    return rungs


def _rung(state, options, correct, tokens, total_n):
    thresholds = {cat: options[cat][state[cat]]["threshold"] for cat in state}
    return {"accuracy": correct / total_n, "tokens": tokens,
            "thresholds_json": json.dumps(thresholds, sort_keys=True)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True, help="NAME=dumpfile")
    parser.add_argument("--rungs", type=int, default=8,
                        help="print approximately this many evenly spread rungs")
    args = parser.parse_args()

    local = load_local(args.local)
    _, _, path = args.remote.partition("=")
    remote = load_remote(path)

    by_category = defaultdict(list)
    for tid, line in local.items():
        by_category[line["category"]].append(tid)

    rungs = build_ladder(by_category, local, remote)

    # Thin to a readable ladder: always keep first (pure local) and last (max)
    if len(rungs) > args.rungs:
        step = (len(rungs) - 1) / (args.rungs - 1)
        keep = sorted({round(i * step) for i in range(args.rungs)})
        rungs = [rungs[i] for i in keep]

    print(f"{'rung':>4} | {'accuracy':>8} | {'tokens':>7} | thresholds")
    print("-" * 100)
    for i, rung in enumerate(rungs):
        print(f"{i:>4} | {rung['accuracy']:>8.2%} | {rung['tokens']:>7} | {rung['thresholds_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
