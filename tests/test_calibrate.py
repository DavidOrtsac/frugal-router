import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.clients import MockClient
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

    def complete_many(self, model, system, user, temperature=0.7, n=4, max_tokens=512):
        return [self.complete(model, system, user, temperature, max_tokens)
                for _ in range(n)]


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


def test_mock_client_is_deterministic_under_concurrency():
    prompts = [f"prompt-{i}" for i in range(20)]
    book = {p: f"answer-{i}" for i, p in enumerate(prompts)}

    def run_once():
        client = MockClient(book, accuracy=0.5, seed=123)
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(
                lambda p: client.complete("m", "s", p, temperature=0.7).text,
                prompts,
            ))

    assert run_once() == run_once()


def test_single_sample_math_without_marker_scores_zero():
    from frugal_router.calibrate import calibrate_local
    from frugal_router.schemas import Category, Task

    class Truncated:
        def complete(self, model, system, user, temperature=0.0, max_tokens=512):
            from frugal_router.schemas import Completion
            return Completion(text="Step 1: 15 * 4 = 60. Step 2: we then",
                              prompt_tokens=1, completion_tokens=1)

    cal = calibrate_local(Truncated(), "m", Task("t", "p"), Category.MATH,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 0.0  # escalation-worthy


def test_single_sample_math_with_marker_scores_one():
    from frugal_router.calibrate import calibrate_local
    from frugal_router.schemas import Category, Task

    class Marked:
        def complete(self, model, system, user, temperature=0.0, max_tokens=512):
            from frugal_router.schemas import Completion
            return Completion(text="60 - 3 = 57\nANSWER: 57",
                              prompt_tokens=1, completion_tokens=1)

    cal = calibrate_local(Marked(), "m", Task("t", "p"), Category.MATH,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 1.0
    assert cal.majority_answer == "57"


def test_single_sample_non_marker_category_unaffected():
    from frugal_router.calibrate import calibrate_local
    from frugal_router.schemas import Category, Task

    class Plain:
        def complete(self, model, system, user, temperature=0.0, max_tokens=512):
            from frugal_router.schemas import Completion
            return Completion(text="Canberra is the capital.",
                              prompt_tokens=1, completion_tokens=1)

    cal = calibrate_local(Plain(), "m", Task("t", "p"), Category.FACTUAL,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 1.0


def test_single_sample_code_gen_valid_python_scores_one():
    client = ScriptedClient(["```python\ndef add(a, b):\n    return a + b\n```"])
    cal = calibrate_local(client, "m", Task("t", "p"), Category.CODE_GEN,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 1.0
    assert cal.majority_answer == "def add(a, b):\n    return a + b"


def test_single_sample_code_debug_syntax_error_scores_zero():
    client = ScriptedClient(["```python\ndef add(a, b:\n    return a + b\n```"])
    cal = calibrate_local(client, "m", Task("t", "p"), Category.CODE_DEBUG,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 0.0  # escalation-worthy


def test_single_sample_code_gen_empty_answer_scores_zero():
    client = ScriptedClient([""])
    cal = calibrate_local(client, "m", Task("t", "p"), Category.CODE_GEN,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 0.0


def test_single_sample_non_code_category_skips_parse_check():
    # Prose is not valid Python, but non-code categories must keep score 1.0.
    client = ScriptedClient(["Paris is the capital of France, not Lyon."])
    cal = calibrate_local(client, "m", Task("t", "p"), Category.FACTUAL,
                          k_initial=1, k_max=1, band=(0.3, 0.9), max_tokens=64)
    assert cal.score == 1.0


def test_code_multi_sample_voting_path_ignores_parse_check():
    # k > 1 uses agreement voting: unanimous non-parsing answers still score
    # 1.0, proving the compile-check applies only to the single-sample branch.
    broken = "```python\ndef add(a, b:\n    return a + b\n```"
    client = ScriptedClient([broken, broken, broken])
    cal = calibrate_local(client, "m", Task("t", "p"), Category.CODE_GEN,
                          k_initial=3, k_max=3, band=(0.3, 0.9), max_tokens=64)
    assert client.calls == 3
    assert cal.score == 1.0
