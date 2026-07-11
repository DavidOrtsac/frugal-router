"""Grade a results.json against the organizers' PUBLIC validation examples.

These five tasks (T01-T05) are retired scoring examples published in the AMD
Hackathon Judging FAQ. They are the only ground truth we have about the real
judge's expectations: answer content, label vocabulary ("mixed" is a valid
sentiment), and format compliance ("exactly two sentences").

Usage: python eval/grade_official.py results.json eval/tasks/official_validation.json
"""

import json
import re
import sys


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s.\-]", "", str(text).lower()).split())


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def grade(task: dict, answer: str) -> tuple:
    gold = task["gold"]
    norm = _norm(answer)
    problems = []

    if "answer" in gold:  # numeric
        numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
        if not numbers:
            problems.append("no number in answer")
        elif abs(float(numbers[-1]) - float(gold["answer"])) > 1e-6:
            problems.append(f"got {numbers[-1]}, want {gold['answer']}")

    if "accept" in gold:
        # An accepted phrase counts when all of its words appear in the answer:
        # "red, green, and blue (RGB)" satisfies "red green blue".
        hits = [a for a in gold["accept"]
                if all(w in norm.split() for w in _norm(a).split())]
        if not hits:
            problems.append(f"none of {gold['accept']} found")

    if "must_contain" in gold:
        missing = [w for w in gold["must_contain"] if _norm(w) not in norm]
        if missing:
            problems.append(f"missing terms: {missing}")

    if "sentences" in gold:
        count = len(_sentences(answer))
        if count != gold["sentences"]:
            problems.append(f"{count} sentences, want exactly {gold['sentences']}")

    return (not problems), problems


def main() -> int:
    results_path, gold_path = sys.argv[1], sys.argv[2]
    with open(results_path, encoding="utf-8") as f:
        answers = {r["task_id"]: r.get("answer", "") for r in json.load(f)}
    with open(gold_path, encoding="utf-8") as f:
        tasks = json.load(f)

    passed = 0
    for task in tasks:
        answer = answers.get(task["task_id"], "")
        ok, problems = grade(task, answer)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status} {task['task_id']} [{task['category']}]")
        if not ok:
            print(f"     issues: {'; '.join(problems)}")
        print(f"     answer: {answer[:220]!r}")
    print(f"\nOFFICIAL VALIDATION: {passed}/{len(tasks)}")
    return 0 if passed == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
