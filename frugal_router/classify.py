"""Zero-cost task classifier. Pure heuristics — no model call, no tokens.

The task set has 8 fixed capability categories, so a rule-based classifier is
reliable and free. Order of checks matters: the most distinctive signals
(explicit instructions like "summarize", code fences) are tested first.
"""

import re

from .schemas import Category, Classification, Task

_CODE_BLOCK = re.compile(r"```|def |function |class |;\s*$|\breturn\b", re.MULTILINE)
_DEBUG_WORDS = re.compile(r"\b(bug|fix|debug|error|incorrect|broken|wrong output|fails?)\b", re.I)
_CODEGEN_WORDS = re.compile(
    r"\b(write|implement|create|generate)\b.{0,40}\b(function|class|method|script|program|code)\b", re.I
)
_SUMMARY_WORDS = re.compile(r"\b(summariz|summary|tl;?dr|condense|shorten)\w*\b", re.I)
_SENTIMENT_WORDS = re.compile(r"\b(sentiment|positive|negative|neutral)\b.*\b(review|text|sentence|statement)\b|classify the sentiment|sentiment of", re.I)
_NER_WORDS = re.compile(r"\b(named entit|entities|NER)\b|extract (all )?(the )?(person|people|organization|location|date)s?\b", re.I)
_MATH_WORDS = re.compile(r"\b(calculate|compute|solve|sum|product|how many|how much|what is \d|percent|average|remainder|equation)\b", re.I)
_MATH_SYMBOLS = re.compile(r"\d+\s*[\+\-\*/\^=]\s*\d+")
_LOGIC_WORDS = re.compile(r"\b(if all|therefore|deduce|logical|premise|conclusion|implies|syllogism|who is (lying|telling the truth)|riddle|puzzle)\b", re.I)


def classify(task: Task) -> Classification:
    p = task.prompt
    has_code = bool(_CODE_BLOCK.search(p))

    category = _detect_category(p, has_code)
    return Classification(category=category, prompt_chars=len(p), has_code_block=has_code)


def _detect_category(p: str, has_code: bool) -> Category:
    if _SUMMARY_WORDS.search(p):
        return Category.SUMMARIZATION
    if _NER_WORDS.search(p):
        return Category.NER
    if _SENTIMENT_WORDS.search(p):
        return Category.SENTIMENT
    if has_code and _DEBUG_WORDS.search(p):
        return Category.CODE_DEBUG
    if _CODEGEN_WORDS.search(p):
        return Category.CODE_GEN
    if has_code:
        return Category.CODE_DEBUG
    if _MATH_SYMBOLS.search(p) or _MATH_WORDS.search(p):
        return Category.MATH
    if _LOGIC_WORDS.search(p):
        return Category.LOGICAL
    return Category.FACTUAL
