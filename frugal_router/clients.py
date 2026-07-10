"""Chat clients. llama.cpp, vLLM, and Fireworks are OpenAI-compatible enough
for one thin wrapper to cover them. A deterministic mock client powers the
offline eval harness and threshold sweeps (no GPU, no API key, no cost)."""

import hashlib
import random
from collections import defaultdict
from threading import Lock
from typing import Protocol

from .schemas import Completion


class ChatClient(Protocol):
    def complete(self, model: str, system: str, user: str,
                 temperature: float, max_tokens: int) -> Completion: ...

    def complete_many(self, model: str, system: str, user: str,
                      temperature: float, n: int, max_tokens: int) -> list: ...


def _message_text(message) -> str:
    """A reasoning model that exhausts max_tokens mid-think can return an
    empty content with the partial trace in a side field (reasoning_content /
    reasoning). A truncated trace still often contains the answer, so it
    beats returning nothing."""
    text = getattr(message, "content", None)
    if text:
        return text
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if value:
            return value
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        if extra.get(key):
            return extra[key]
    return ""


class OpenAICompatClient:
    """Wraps any OpenAI-compatible endpoint (llama.cpp, vLLM, Fireworks).

    extra_body is forwarded verbatim — used e.g. to disable Qwen3 thinking
    mode via {"chat_template_kwargs": {"enable_thinking": false}}.
    """

    def __init__(self, base_url: str, api_key: str = "EMPTY",
                 extra_body: dict = None, timeout: float = None,
                 max_retries: int = None, default_headers: dict = None,
                 http_client=None):
        from openai import OpenAI  # lazy import so offline tools need no deps
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if default_headers:
            kwargs["default_headers"] = dict(default_headers)
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", **kwargs)
        self._extra_body = dict(extra_body) if extra_body else None

    def _create(self, model: str, system: str, user: str, **kwargs):
        """Some chat templates (notably Gemma's) reject a system role —
        fall back to folding the instruction into the user message."""
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=self._extra_body,
                **kwargs,
            )
        except Exception as exc:
            if "system role" not in str(exc).lower():
                raise
            return self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{system}\n\n{user}"}],
                extra_body=self._extra_body,
                **kwargs,
            )

    def complete(self, model: str, system: str, user: str,
                 temperature: float = 0.0, max_tokens: int = 512) -> Completion:
        resp = self._create(model, system, user,
                            temperature=temperature, max_tokens=max_tokens)
        usage = resp.usage
        return Completion(
            text=_message_text(resp.choices[0].message),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def complete_many(self, model: str, system: str, user: str,
                      temperature: float = 0.7, n: int = 4,
                      max_tokens: int = 512) -> list:
        """n sampled completions, in ONE request where the server supports it.
        Servers that reject or ignore n>1 fall back to sequential calls."""
        try:
            resp = self._create(model, system, user,
                                temperature=temperature, max_tokens=max_tokens, n=n)
            usage = resp.usage
            first = Completion(
                text=_message_text(resp.choices[0].message),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
            completions = [first] + [
                Completion(text=_message_text(c.message), prompt_tokens=0, completion_tokens=0)
                for c in resp.choices[1:]
            ]
        except Exception:
            completions = []
        while len(completions) < n:
            completions.append(self.complete(model, system, user,
                                             temperature=temperature, max_tokens=max_tokens))
        return completions


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
        self._sample_counts = defaultdict(int)
        self._lock = Lock()

    def complete(self, model: str, system: str, user: str,
                 temperature: float = 0.0, max_tokens: int = 512) -> Completion:
        sample_index = 0
        if temperature > 0:
            with self._lock:
                sample_index = self._sample_counts[user] + 1
                self._sample_counts[user] += 1
        gold = self._book.get(user, "UNKNOWN")
        stable = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
        rng = random.Random(self._seed + stable + sample_index)
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

    def complete_many(self, model: str, system: str, user: str,
                      temperature: float = 0.7, n: int = 4,
                      max_tokens: int = 512) -> list:
        return [self.complete(model, system, user, temperature, max_tokens)
                for _ in range(n)]
