"""Per-category prompting. Two goals:
1. Force terse, gradable answers (remote completion tokens are scored).
2. Keep escalation prompts minimal — no few-shot padding on the remote path.
"""

import re

from .schemas import Category

SYSTEM_PROMPTS = {
    Category.FACTUAL: "Answer the question concisely and completely, covering every part asked. No filler, no restating the question.",
    Category.MATH: "Solve the problem. Think step by step briefly, then give the final line as: ANSWER: <number>",
    Category.SENTIMENT: "Classify the sentiment. Reply with one word (positive, negative, or neutral) unless the task asks for justification — then add one short sentence.",
    Category.SUMMARIZATION: "Summarize the given text faithfully. Follow the task's format and length instructions exactly. Output only the summary.",
    Category.NER: "Extract the named entities requested. If types are requested, label each entity with its type in parentheses. Output a comma-separated list, nothing else.",
    Category.CODE_DEBUG: "Fix the bug in the given code. Output only the corrected code, no commentary.",
    Category.LOGICAL: "Reason through the problem briefly, then give the final line as: ANSWER: <answer>",
    Category.CODE_GEN: "Write the requested code. Output only the code, no commentary.",
}

# Categories whose answers end with an "ANSWER:" marker we should extract.
_MARKER_CATEGORIES = frozenset({Category.MATH, Category.LOGICAL})


def system_prompt(category: Category) -> str:
    return SYSTEM_PROMPTS[category]


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_ORPHAN_CLOSE = re.compile(r"^\s*</think>\s*", re.IGNORECASE)
_LITERAL_THINK_TAGS = re.compile(r"</?think>", re.IGNORECASE)
_ANSWER_MARKER = re.compile(r"\banswer\s*:\s*", re.IGNORECASE)
_FENCED_CODE = re.compile(r"```(?:[a-zA-Z0-9]*)\n(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_answer(category: Category, text: str) -> str:
    """Pull the gradable answer out of a completion.

    Handles reasoning-model debris: full <think>...</think> blocks AND the
    orphan leading </think> that llama.cpp emits when reasoning is disabled.
    Salvage rules guarantee a non-empty best effort: an unterminated think
    block that swallowed the whole completion is re-mined with only the
    literal tags removed, marker categories fall back to the last number /
    last sentence, and code falls back through unterminated fences."""
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _ORPHAN_CLOSE.sub("", cleaned).strip()
    salvaged_from_think = False
    if not cleaned:
        # The whole completion was inside an unterminated think block. The
        # truncated trace still often contains the work — keep it and let the
        # category fallbacks below mine it.
        cleaned = _LITERAL_THINK_TAGS.sub("", text).strip()
        salvaged_from_think = True
    if category in _MARKER_CATEGORIES:
        matches = list(_ANSWER_MARKER.finditer(cleaned))
        if matches:
            marked = cleaned[matches[-1].end():].strip()
            if marked:
                return marked
        if salvaged_from_think:
            # Only a recovered reasoning trace gets mined down to a terse
            # best effort; a normal marker-less answer keeps its full text
            # (which contains everything any grader could want).
            if category is Category.MATH:
                numbers = _NUMBER.findall(cleaned)
                if numbers:
                    return numbers[-1]
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(cleaned)
                         if s.strip()]
            if sentences:
                return sentences[-1]
        return cleaned
    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        fenced = _FENCED_CODE.search(cleaned)
        if fenced:
            return fenced.group(1).strip()
        open_fence = cleaned.rfind("```")
        if open_fence != -1 and cleaned[open_fence:].count("\n"):
            # Unterminated fence: everything after the opening fence line.
            tail = cleaned[open_fence:]
            return tail[tail.find("\n") + 1:].strip()
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
