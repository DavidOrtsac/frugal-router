"""Calibration via self-consistency sampling on the LOCAL model.

Local tokens are free under Track 1 scoring, so we can afford k samples per
task. Agreement across samples is a far better correctness predictor than the
model's self-reported confidence, and the majority answer doubles as our local
answer, so calibration costs nothing extra.

Adaptive schedule: sample k_initial; if agreement lands in the borderline band
(genuinely uncertain), spend more free samples to sharpen the estimate before
deciding whether to pay for a remote call.
"""

import re
from collections import Counter

from .clients import ChatClient
from .prompts import (_CODE_CATEGORIES, _MARKER_CATEGORIES, extract_answer,
                      has_valid_final_answer, parses_as_python, system_prompt)
from .schemas import Calibration, Category, Task

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# "mixed" must be checked FIRST: a mixed answer usually contains the words
# "positive" and "negative" in its justification.
_SENTIMENT_LABELS = ("mixed", "positive", "negative", "neutral")


def calibrate_local(client: ChatClient, model: str, task: Task, category: Category,
                    k_initial: int, k_max: int, band: tuple,
                    max_tokens: int) -> Calibration:
    """Sample the local model and measure answer agreement, adaptively."""
    if k_initial <= 1:
        completion = client.complete(model, system_prompt(category), task.prompt,
                                     temperature=0.0, max_tokens=max_tokens)
        answer = extract_answer(category, completion.text)
        # Single-sample confidence signal for marker categories: a completion
        # that never reached its explicit final answer (truncated derivation)
        # is exactly the answer worth paying to escalate.
        score = 1.0
        if (category in _MARKER_CATEGORIES
                and not has_valid_final_answer(category, completion.text)):
            score = 0.0
        # Single-sample confidence signal for code categories: an answer that
        # does not even parse as Python (pure compile-check, never executed)
        # is exactly the answer worth paying to escalate.
        elif category in _CODE_CATEGORIES and not parses_as_python(answer):
            score = 0.0
        return Calibration(score=score, majority_answer=answer, samples=(answer,))

    answers = _sample(client, model, task, category, k_initial, max_tokens, greedy_first=True)
    score, majority = _vote(category, answers)

    band_low, band_high = band
    if band_low <= score <= band_high and k_max > k_initial:
        answers = answers + _sample(client, model, task, category,
                                    k_max - k_initial, max_tokens, greedy_first=False)
        score, majority = _vote(category, answers)

    return Calibration(score=score, majority_answer=majority, samples=tuple(answers))


def _sample(client: ChatClient, model: str, task: Task, category: Category,
            n: int, max_tokens: int, greedy_first: bool) -> list:
    answers = []
    if greedy_first:
        completion = client.complete(model, system_prompt(category), task.prompt,
                                     temperature=0.0, max_tokens=max_tokens)
        answers.append(extract_answer(category, completion.text))
        n -= 1
    if n > 0:
        completions = client.complete_many(model, system_prompt(category), task.prompt,
                                           temperature=0.7, n=n, max_tokens=max_tokens)
        answers.extend(extract_answer(category, c.text) for c in completions)
    return answers


def _vote(category: Category, answers: list) -> tuple:
    votes = Counter(normalize(category, a) for a in answers)
    normalized_majority, count = votes.most_common(1)[0]
    majority = next(a for a in answers if normalize(category, a) == normalized_majority)
    return count / len(answers), majority


def normalize(category: Category, answer: str) -> str:
    """Category-aware canonical form, so votes measure semantic agreement:
    '6.0', '6', and 'The answer is 6.' must all count as the same vote."""
    text = " ".join(answer.lower().split())
    if category == Category.MATH:
        numbers = _NUMBER.findall(text.replace(",", ""))
        if numbers:
            return _canonical_number(numbers[-1])
        return text
    if category == Category.SENTIMENT:
        for label in _SENTIMENT_LABELS:
            if label in text:
                return label
        return text
    if category == Category.NER:
        parts = sorted(p.strip() for p in re.split(r"[,\n;]", text) if p.strip())
        return "; ".join(parts)
    if category in (Category.FACTUAL, Category.LOGICAL):
        return text.strip(" .!\"'")
    return text


def _canonical_number(raw: str) -> str:
    value = float(raw)
    if value == int(value):
        return str(int(value))
    return repr(value)
