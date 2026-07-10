import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.prompts import extract_answer
from frugal_router.schemas import Category


def test_extract_answer_marker_is_case_insensitive():
    assert extract_answer(Category.MATH, "work\nAnswer: 42") == "42"
    assert extract_answer(Category.LOGICAL, "therefore\nanswer: yes") == "yes"


def test_extract_answer_salvages_unterminated_think_block_math():
    from frugal_router.prompts import extract_answer
    from frugal_router.schemas import Category
    # Budget truncation: whole completion is one unterminated think block.
    text = "<think>15 * 4 = 60, minus 3 gives 57"
    assert extract_answer(Category.MATH, text) == "57"


def test_extract_answer_math_marker_still_wins():
    from frugal_router.prompts import extract_answer
    from frugal_router.schemas import Category
    text = "some steps 12 and 99\nANSWER: 42"
    assert extract_answer(Category.MATH, text) == "42"


def test_extract_answer_logical_keeps_full_text_without_marker():
    # A normal marker-less answer keeps its FULL text: contains-style graders
    # (and LLM judges) can find the verdict anywhere in it. Mining down to
    # the last sentence is reserved for think-block salvage only.
    text = "Yes, the conclusion follows. Note that this assumes transitivity."
    assert extract_answer(Category.LOGICAL, text) == text


def test_extract_answer_logical_mines_last_sentence_from_think_salvage():
    text = "<think>Premise one holds. Therefore Alice is taller than Bob."
    assert extract_answer(Category.LOGICAL, text) == \
        "Therefore Alice is taller than Bob."


def test_extract_answer_code_unterminated_fence():
    from frugal_router.prompts import extract_answer
    from frugal_router.schemas import Category
    text = "Here is the fix:\n```python\ndef add(a, b):\n    return a + b"
    assert extract_answer(Category.CODE_GEN, text) == "def add(a, b):\n    return a + b"


def test_extract_answer_never_empty_on_pure_think_text():
    from frugal_router.prompts import extract_answer
    from frugal_router.schemas import Category
    text = "<think>the sentiment seems clearly positive here"
    out = extract_answer(Category.SENTIMENT, text)
    assert out != ""
    assert "positive" in out
