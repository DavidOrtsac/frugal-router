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

# Categories whose answers are code we can parse-check.
_CODE_CATEGORIES = frozenset({Category.CODE_DEBUG, Category.CODE_GEN})


def system_prompt(category: Category) -> str:
    return SYSTEM_PROMPTS[category]


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_ORPHAN_CLOSE = re.compile(r"^\s*</think>\s*", re.IGNORECASE)
_LITERAL_THINK_TAGS = re.compile(r"</?think>", re.IGNORECASE)
_ANSWER_MARKER = re.compile(r"\banswer\s*:\s*", re.IGNORECASE)
_FENCED_CODE = re.compile(r"```(?:[a-zA-Z0-9]*)\n(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BOXED = re.compile(r"\\boxed\s*\{\s*([^{}]*)")
_ONE_SENTENCE = re.compile(r"\b(?:exactly\s+)?one\s+sentence\b", re.IGNORECASE)
_WORD_CAP = re.compile(
    r"\b(?:in|within|under|at most|no more than|fewer than|maximum of)\s+"
    r"(\d{1,3})\s+words?\b", re.IGNORECASE)
_SENTENCE_CAP = re.compile(
    r"\b(?:in|within|under|at most|no more than|maximum of)\s+"
    r"(\d{1,2})\s+sentences?\b", re.IGNORECASE)


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
        for match in reversed(list(_ANSWER_MARKER.finditer(cleaned))):
            marked = cleaned[match.end():].strip()
            if _valid_marker_payload(category, marked):
                return marked
        # \boxed{...} is a completed final answer even when the marker (or
        # the closing brace) got cut off by the token budget.
        boxed = _BOXED.findall(cleaned)
        if boxed and boxed[-1].strip():
            return boxed[-1].strip()
        truncated = bool(cleaned) and cleaned[-1] not in ".!?\"')]}"
        if salvaged_from_think or truncated:
            # A recovered or budget-cut reasoning trace is a near-certain
            # judge fail as-is: mine a terse best effort. A COMPLETE
            # marker-less answer keeps its full text (proven behavior —
            # the answer sits in context for both grader styles).
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


def has_valid_final_answer(category: Category, text: str) -> bool:
    """True when the raw completion contains a usable explicit final answer:
    an ANSWER: marker followed by real content, or a non-empty \\boxed{}.
    Used as the single-sample confidence signal for marker categories."""
    cleaned = _LITERAL_THINK_TAGS.sub("", text)
    for match in _ANSWER_MARKER.finditer(cleaned):
        if _valid_marker_payload(category, cleaned[match.end():].strip()):
            return True
    boxed = _BOXED.findall(cleaned)
    return bool(boxed and boxed[-1].strip())


def parses_as_python(code: str) -> bool:
    """True when the code parses as Python — a pure syntax check via
    compile(); the code is NEVER executed. Used as the single-sample
    confidence signal for code categories: an empty or non-parsing answer
    (truncated, commentary-wrapped, or non-Python) is exactly the answer
    worth paying to escalate."""
    if not code.strip():
        return False
    try:
        compile(code, "<answer>", "exec")
    except (SyntaxError, ValueError):
        # ValueError covers source with null bytes — also escalation-worthy.
        return False
    return True


def _valid_marker_payload(category: Category, payload: str) -> bool:
    """A marker only counts when real content follows it. Models sometimes
    echo the instruction template itself ('ANSWER: <number>') — that echoed
    marker must not shadow the real answer or fake a confidence signal."""
    if not payload or payload.startswith("<"):
        return False
    if category is Category.MATH:
        return bool(_NUMBER.search(payload[:40]))
    return True


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def enforce_summary_format(task_prompt: str, answer: str) -> str:
    """Post-enforce explicit format constraints on a summary.

    A string-matching grader never penalizes a 4-sentence answer to
    'summarize in exactly one sentence', but an LLM judge fails the
    instruction violation outright. Applied ONLY to summarization answers.
    Also trims any cap-truncated summary back to its last complete sentence."""
    answer = answer.strip()
    if not answer:
        return answer
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
    if _ONE_SENTENCE.search(task_prompt) and sentences:
        return sentences[0]
    cap = _SENTENCE_CAP.search(task_prompt)
    if cap and sentences:
        return " ".join(sentences[: max(1, int(cap.group(1)))])
    words_cap = _WORD_CAP.search(task_prompt)
    if words_cap:
        limit = max(1, int(words_cap.group(1)))
        words = answer.split()
        if len(words) > limit:
            return " ".join(words[:limit]).rstrip(",;:") + "."
    # Token-budget truncation: drop a trailing incomplete sentence when at
    # least one complete sentence exists.
    if answer[-1] not in ".!?\"'" and len(sentences) > 1:
        complete = _SENTENCE_SPLIT.split(answer)[:-1]
        return " ".join(s.strip() for s in complete if s.strip())
    return answer
