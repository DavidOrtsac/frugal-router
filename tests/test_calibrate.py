import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.calibrate import calibrate_local, normalize
from frugal_router.schemas import Category, Completion, Task


class ScriptedClient:
    """Returns queued responses in order; counts calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        self.calls += 1
        text = self._responses.pop(0)
        return Completion(text=text, prompt_tokens=1, completion_tokens=1)


def test_normalize_math_equates_number_formats():
    assert normalize(Category.MATH, "ANSWER: 6.0") == normalize(Category.MATH, "6")
    assert normalize(Category.MATH, "The total is 1,200 pesos") == normalize(Category.MATH, "1200")


def test_normalize_sentiment_extracts_label():
    assert normalize(Category.SENTIMENT, "The sentiment is Positive.") == "positive"


def test_normalize_ner_is_order_insensitive():
    a = normalize(Category.NER, "Apple, Tim Cook")
    b = normalize(Category.NER, "Tim Cook, Apple")
    assert a == b


def test_unanimous_agreement_skips_extension():
    client = ScriptedClient(["6", "6.0", "ANSWER: 6", "six... ANSWER: 6", "6"])
    task = Task(task_id="t", prompt="what is 2+4? Calculate.")
    calibration = calibrate_local(client, "m", task, Category.MATH,
                                  k_initial=5, k_max=10, band=(0.3, 0.9), max_tokens=64)
    assert calibration.score == 1.0
    assert client.calls == 5  # no adaptive extension needed


def test_borderline_agreement_triggers_extension():
    responses = ["6", "7", "6", "8", "7"] + ["6", "6", "6", "6", "6"]
    client = ScriptedClient(responses)
    task = Task(task_id="t", prompt="what is 2+4? Calculate.")
    calibration = calibrate_local(client, "m", task, Category.MATH,
                                  k_initial=5, k_max=10, band=(0.3, 0.9), max_tokens=64)
    assert client.calls == 10  # extended
    assert calibration.score == 0.7  # 7 of 10 votes for 6
    assert normalize(Category.MATH, calibration.majority_answer) == "6"
