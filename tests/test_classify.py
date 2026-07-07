import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.classify import classify
from frugal_router.schemas import Category, Task


def _cat(prompt: str) -> Category:
    return classify(Task(task_id="t", prompt=prompt)).category


def test_summarization():
    assert _cat("Summarize the following text: The canal ...") == Category.SUMMARIZATION


def test_sentiment():
    assert _cat('Classify the sentiment of this review as positive, negative, or neutral: "Great!"') == Category.SENTIMENT


def test_ner():
    assert _cat('Extract all named entities from this sentence: "Tim Cook leads Apple."') == Category.NER


def test_math():
    assert _cat("Calculate: what is 17 * 24?") == Category.MATH
    assert _cat("A train travels 180 km in 2.5 hours. What is its average speed in km/h?") == Category.MATH


def test_code_debug():
    prompt = "Fix the bug in this Python function:\n```python\ndef f(x):\n    return x + 1\n```"
    assert _cat(prompt) == Category.CODE_DEBUG


def test_code_gen():
    assert _cat("Write a Python function named fizzbuzz that takes an integer n.") == Category.CODE_GEN


def test_logical():
    assert _cat("All roses are flowers. Some flowers fade quickly. Therefore, do some roses fade quickly?") == Category.LOGICAL


def test_factual_default():
    assert _cat("What is the capital city of Australia?") == Category.FACTUAL


def test_dev_set_classification_accuracy():
    """The classifier must nail the authored dev set — it is zero-cost routing input."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "tasks", "dev_tasks.json")
    with open(path) as f:
        dev = json.load(f)
    hits = sum(1 for t in dev if _cat(t["prompt"]).value == t["category"])
    assert hits / len(dev) >= 0.9, f"classifier only got {hits}/{len(dev)}"
