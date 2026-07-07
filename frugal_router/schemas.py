"""Immutable data structures shared across the router pipeline."""

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "math_reasoning"
    SENTIMENT = "sentiment_classification"
    SUMMARIZATION = "text_summarization"
    NER = "ner"
    CODE_DEBUG = "code_debugging"
    LOGICAL = "logical_reasoning"
    CODE_GEN = "code_generation"


class Route(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str


@dataclass(frozen=True)
class Classification:
    category: Category
    # Cheap features the policy can use without any model call.
    prompt_chars: int
    has_code_block: bool


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Calibration:
    """Agreement across local self-consistency samples (free tokens)."""

    score: float  # 0..1 majority-vote share; 1.0 means all samples agree
    majority_answer: str
    samples: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    model: str
    reason: str


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    answer: str
    category: Category
    route: Route
    model: str
    remote_tokens: int  # 0 when handled locally — this is the scored quantity
    reason: str
