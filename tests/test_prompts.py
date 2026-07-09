import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.prompts import extract_answer
from frugal_router.schemas import Category


def test_extract_answer_marker_is_case_insensitive():
    assert extract_answer(Category.MATH, "work\nAnswer: 42") == "42"
    assert extract_answer(Category.LOGICAL, "therefore\nanswer: yes") == "yes"
