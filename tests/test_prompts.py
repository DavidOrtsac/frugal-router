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


def test_extract_answer_math_boxed_survives_truncation():
    # Budget cut the completion mid-\boxed{}: the boxed value IS the answer.
    text = "Long derivation with 12 and 99 steps... so \\boxed{36"
    assert extract_answer(Category.MATH, text) == "36"


def test_extract_answer_math_truncated_derivation_mines_last_number():
    # No marker, no boxed, ends mid-sentence -> truncated: mine last number.
    text = "Step 1: 15 * 4 = 60. Step 2: subtract 3 to get 57 which means"
    assert extract_answer(Category.MATH, text) == "57"


def test_extract_answer_math_complete_text_stays_whole():
    # Complete sentence, no marker: full text is preserved (proven behavior).
    text = "The total is 57, found in step 4 of 12."
    assert extract_answer(Category.MATH, text) == text


def test_enforce_summary_format_one_sentence():
    from frugal_router.prompts import enforce_summary_format
    prompt = "Summarize the following article in exactly one sentence."
    answer = "The rover landed. It sent photos. NASA celebrated. More later."
    assert enforce_summary_format(prompt, answer) == "The rover landed."


def test_enforce_summary_format_word_cap():
    from frugal_router.prompts import enforce_summary_format
    prompt = "Summarize in no more than 5 words."
    answer = "The quick brown fox jumps over the lazy dog"
    out = enforce_summary_format(prompt, answer)
    assert len(out.rstrip(".").split()) == 5


def test_enforce_summary_format_trims_truncated_tail():
    from frugal_router.prompts import enforce_summary_format
    prompt = "Summarize the passage."
    answer = "The treaty was signed in 1848. It ended the war. The final part was abou"
    assert enforce_summary_format(prompt, answer) == \
        "The treaty was signed in 1848. It ended the war."


def test_enforce_summary_format_leaves_compliant_answers_alone():
    from frugal_router.prompts import enforce_summary_format
    prompt = "Summarize the passage."
    answer = "The treaty was signed in 1848 and ended the war."
    assert enforce_summary_format(prompt, answer) == answer
