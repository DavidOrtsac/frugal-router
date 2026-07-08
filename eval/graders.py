"""Per-category graders for the local eval harness.

Each grader returns True/False for one task given the model answer and the
gold spec from the dev set. Code tasks are graded by executing the candidate
against authored test cases (we wrote every test input ourselves)."""

import re


def grade(category: str, answer: str, gold: dict) -> bool:
    grader = _GRADERS[category]
    try:
        return grader(answer, gold)
    except Exception:
        return False


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s.\-]", "", text.lower()).split())


def _grade_contains(answer: str, gold: dict) -> bool:
    normalized = _norm(answer)
    accepted = gold.get("accept", [gold.get("answer", "")])
    return any(_norm(str(a)) in normalized or normalized == _norm(str(a)) for a in accepted)


def _grade_numeric(answer: str, gold: dict) -> bool:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    if not numbers:
        return False
    expected = float(gold["answer"])
    return any(abs(float(n) - expected) < 1e-6 for n in numbers[-1:] or numbers)


def _grade_label(answer: str, gold: dict) -> bool:
    return _norm(answer).startswith(_norm(str(gold["answer"]))) or _norm(str(gold["answer"])) in _norm(answer)


def _grade_keywords(answer: str, gold: dict) -> bool:
    """Summarization: require coverage of the key facts, cap length drift."""
    normalized = _norm(answer)
    keywords = gold["keywords"]
    hits = sum(1 for k in keywords if _norm(k) in normalized)
    max_words = gold.get("max_words", 120)
    return hits / len(keywords) >= gold.get("min_coverage", 0.6) and len(answer.split()) <= max_words


def _grade_entity_set(answer: str, gold: dict) -> bool:
    expected = {_norm(e) for e in gold["entities"]}
    # Strip "(type)" labels — official practice tasks show typed output is valid.
    cleaned = re.sub(r"\([^)]*\)", "", answer)
    found = {_norm(part) for part in re.split(r"[,\n;]", cleaned) if part.strip()}
    if not expected:
        return not found
    tp = len(expected & found)
    precision = tp / len(found) if found else 0.0
    recall = tp / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return f1 >= gold.get("min_f1", 0.7)


def _grade_code(answer: str, gold: dict) -> bool:
    """Execute candidate code in a SANDBOXED subprocess and check test cases.

    Model-generated code can allocate unbounded memory or loop forever, so it
    never runs in the harness process: a disposable child gets 1GB of address
    space, 10s of CPU, and a wall-clock timeout. Any violation = wrong answer.

    Two gold styles:
    - authored: {"function", "tests": [{"args", "expected"}]}
    - HumanEval: {"function", "check_code", "context"} where check_code defines
      check(candidate) that asserts behavior. context is the original prompt
      (imports + signature) for models that answer with a body-only completion.
    """
    import json as json_mod
    import subprocess
    import sys

    payload = json_mod.dumps({
        "answer": answer,
        "context": gold.get("context", ""),
        "function": gold["function"],
        "check_code": gold.get("check_code"),
        "tests": gold.get("tests", []),
    })
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SANDBOX_RUNNER],
            input=payload, capture_output=True, text=True, timeout=15,
        )
        return proc.stdout.strip().endswith("PASS")
    except Exception:
        return False


_SANDBOX_RUNNER = r"""
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
except Exception:
    pass
payload = json.load(sys.stdin)
ok = False
for source in (payload["answer"], payload["context"] + "\n" + payload["answer"]):
    try:
        namespace = {}
        exec(source, namespace)
        func = namespace.get(payload["function"])
        if func is None:
            continue
        if payload.get("check_code"):
            exec(payload["check_code"], namespace)
            namespace["check"](func)
            ok = True
        else:
            ok = all(func(*case["args"]) == case["expected"]
                     for case in payload["tests"])
        break
    except Exception:
        continue
print("PASS" if ok else "FAIL")
"""


_GRADERS = {
    "factual_knowledge": _grade_contains,
    "math_reasoning": _grade_numeric,
    "sentiment_classification": _grade_label,
    "text_summarization": _grade_keywords,
    "ner": _grade_entity_set,
    "code_debugging": _grade_code,
    "logical_reasoning": _grade_contains,
    "code_generation": _grade_code,
}
