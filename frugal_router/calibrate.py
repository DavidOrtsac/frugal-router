"""Calibration via self-consistency sampling on the LOCAL model.

Local tokens are free under Track 1 scoring, so we can afford k samples per
task. Agreement across samples is a far better correctness predictor than the
model's self-reported confidence. The majority answer doubles as our local
answer, so calibration costs nothing extra."""

from collections import Counter

from .clients import ChatClient
from .prompts import extract_answer, system_prompt
from .schemas import Calibration, Category, Task


def calibrate_local(client: ChatClient, model: str, task: Task,
                    category: Category, k: int, max_tokens: int) -> Calibration:
    """Sample the local model k times and measure answer agreement."""
    if k <= 1:
        completion = client.complete(model, system_prompt(category), task.prompt,
                                     temperature=0.0, max_tokens=max_tokens)
        answer = extract_answer(category, completion.text)
        return Calibration(score=1.0, majority_answer=answer, samples=(answer,))

    answers = []
    for i in range(k):
        temp = 0.0 if i == 0 else 0.7  # first sample greedy, rest exploratory
        completion = client.complete(model, system_prompt(category), task.prompt,
                                     temperature=temp, max_tokens=max_tokens)
        answers.append(extract_answer(category, completion.text))

    votes = Counter(_normalize(a) for a in answers)
    normalized_majority, count = votes.most_common(1)[0]
    # Return the original-cased variant of the winning normalized answer.
    majority = next(a for a in answers if _normalize(a) == normalized_majority)
    return Calibration(score=count / len(answers), majority_answer=majority,
                       samples=tuple(answers))


def _normalize(answer: str) -> str:
    return " ".join(answer.lower().split())
