"""Chat clients. Local vLLM and Fireworks are both OpenAI-compatible, so one
thin wrapper covers both. A deterministic mock client powers the offline eval
harness and threshold sweeps (no GPU, no API key, no cost)."""

import hashlib
import random
from typing import Protocol

from .schemas import Completion


class ChatClient(Protocol):
    def complete(self, model: str, system: str, user: str,
                 temperature: float, max_tokens: int) -> Completion: ...


class OpenAICompatClient:
    """Wraps any OpenAI-compatible endpoint (vLLM serve, Fireworks)."""

    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        from openai import OpenAI  # lazy import so offline tools need no deps
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    def complete(self, model: str, system: str, user: str,
                 temperature: float = 0.0, max_tokens: int = 512) -> Completion:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


class MockClient:
    """Deterministic fake model for offline harness runs.

    Answers from an answer book with a configurable per-sample accuracy.
    Wrong answers are drawn from a small pool of distractors, so
    self-consistency agreement genuinely correlates with correctness —
    the same statistical shape the real router exploits.
    """

    def __init__(self, answer_book: dict, accuracy: float, seed: int = 7):
        self._book = dict(answer_book)
        self._accuracy = accuracy
        self._seed = seed
        self._calls = 0

    def complete(self, model: str, system: str, user: str,
                 temperature: float = 0.0, max_tokens: int = 512) -> Completion:
        self._calls += 1
        gold = self._book.get(user, "UNKNOWN")
        stable = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
        rng = random.Random(self._seed + stable + (self._calls if temperature > 0 else 0))
        if rng.random() < self._accuracy:
            text = gold
        else:
            # Distractors must never contain the gold string, or graders
            # would score wrong answers as correct.
            text = rng.choice(["WRONG_GUESS_A", "WRONG_GUESS_B", "UNKNOWN"])
        approx_prompt_tokens = max(1, (len(system) + len(user)) // 4)
        approx_completion_tokens = max(1, len(text) // 4)
        return Completion(text=text, prompt_tokens=approx_prompt_tokens,
                          completion_tokens=approx_completion_tokens)
