"""Per-category prompting. Two goals:
1. Force terse, gradable answers (remote completion tokens are scored).
2. Keep escalation prompts minimal — no few-shot padding on the remote path.
"""

import re

from .schemas import Category

SYSTEM_PROMPTS = {
    Category.FACTUAL: "Answer the question with only the answer itself. No explanation, no punctuation beyond the answer.",
    Category.MATH: "Solve the problem. Think step by step briefly, then give the final line as: ANSWER: <number>",
    Category.SENTIMENT: "Classify the sentiment. Reply with exactly one word: positive, negative, or neutral.",
    Category.SUMMARIZATION: "Summarize the given text faithfully in 1-3 sentences. Output only the summary.",
    Category.NER: "Extract the named entities requested. Output them as a comma-separated list, nothing else.",
    Category.CODE_DEBUG: "Fix the bug in the given code. Output only the corrected code, no commentary.",
    Category.LOGICAL: "Reason through the problem briefly, then give the final line as: ANSWER: <answer>",
    Category.CODE_GEN: "Write the requested code. Output only the code, no commentary.",
}

# Categories whose answers end with an "ANSWER:" marker we should extract.
_MARKER_CATEGORIES = frozenset({Category.MATH, Category.LOGICAL})


def system_prompt(category: Category) -> str:
    return SYSTEM_PROMPTS[category]


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def extract_answer(category: Category, text: str) -> str:
    """Pull the gradable answer out of a completion."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if category in _MARKER_CATEGORIES:
        marker = cleaned.rfind("ANSWER:")
        if marker != -1:
            return cleaned[marker + len("ANSWER:"):].strip()
    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        return _strip_code_fences(cleaned)
    return cleaned


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
