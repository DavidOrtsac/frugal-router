import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from graders import grade


def test_factual_contains():
    assert grade("factual_knowledge", "The capital is Canberra.", {"answer": "Canberra"})
    assert not grade("factual_knowledge", "Sydney", {"answer": "Canberra"})


def test_math_numeric():
    assert grade("math_reasoning", "The cost is 6.00 dollars", {"answer": 6.0})
    assert not grade("math_reasoning", "roughly 7", {"answer": 6.0})
    assert not grade("math_reasoning", "no idea", {"answer": 6.0})


def test_sentiment_label():
    assert grade("sentiment_classification", "positive", {"answer": "positive"})
    assert not grade("sentiment_classification", "negative", {"answer": "positive"})


def test_summary_keywords():
    gold = {"keywords": ["Panama Canal", "Atlantic", "Pacific", "shipping"], "min_coverage": 0.5}
    assert grade("text_summarization", "The Panama Canal links the Atlantic and Pacific.", gold)
    assert not grade("text_summarization", "A big ditch somewhere.", gold)


def test_ner_f1():
    gold = {"entities": ["Tim Cook", "Apple", "OpenAI"]}
    assert grade("ner", "Tim Cook, Apple, OpenAI", gold)
    assert not grade("ner", "Microsoft, Google", gold)


def test_code_generation_exec():
    gold = {"function": "fizzbuzz", "tests": [{"args": [15], "expected": "FizzBuzz"}, {"args": [7], "expected": "7"}]}
    good = "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)"
    bad = "def fizzbuzz(n):\n    return 'Fizz'"
    assert grade("code_generation", good, gold)
    assert not grade("code_generation", bad, gold)


def test_code_grader_survives_broken_code():
    gold = {"function": "f", "tests": [{"args": [1], "expected": 1}]}
    assert not grade("code_generation", "this is not python ][", gold)


def test_code_grader_survives_memory_bomb():
    """Model-generated code that allocates without bound must grade False
    without harming the harness process (the failure mode that OOM-killed a
    full recording run on 2026-07-08)."""
    bomb = "def f(x):\n    data = []\n    while True:\n        data.append('x' * 10_000_000)\n    return x"
    gold = {"function": "f", "tests": [{"args": [1], "expected": 1}]}
    assert not grade("code_generation", bomb, gold)


def test_code_grader_survives_infinite_loop():
    loop = "def f(x):\n    while True:\n        pass"
    gold = {"function": "f", "tests": [{"args": [1], "expected": 1}]}
    assert not grade("code_generation", loop, gold)
