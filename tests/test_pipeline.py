import os
import sys
import time
from dataclasses import replace
from threading import Lock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.config import Config
from frugal_router.pipeline import _escalate, run_batch
from frugal_router.schemas import Category, Completion, Task


class ConstantClient:
    def __init__(self, text="ok"):
        self.text = text

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        return Completion(text=self.text, prompt_tokens=1, completion_tokens=1)

    def complete_many(self, model, system, user, temperature=0.7, n=4, max_tokens=512):
        return [self.complete(model, system, user, temperature, max_tokens)
                for _ in range(n)]


class RetryableError(Exception):
    status_code = 429


class FlakyRemote(ConstantClient):
    def __init__(self):
        super().__init__("remote-ok")
        self.calls = 0

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        self.calls += 1
        if self.calls == 1:
            raise RetryableError("rate limited")
        return super().complete(model, system, user, temperature, max_tokens)


class SlowRemote(ConstantClient):
    def __init__(self):
        super().__init__("remote-ok")
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return super().complete(model, system, user, temperature, max_tokens)


def test_remote_escalation_retries_quick_proxy_failures():
    remote = FlakyRemote()
    config = replace(Config(), remote_attempts=2, remote_retry_delay_seconds=0.0)

    result = _escalate(config, remote, Task("t", "What is the capital of Australia?"),
                       Category.FACTUAL, reason="forced")

    assert result.answer == "remote-ok"
    assert result.remote_tokens == 2
    assert remote.calls == 2


def test_run_batch_caps_remote_concurrency():
    tasks = [Task(str(i), "What is the capital of Australia?") for i in range(6)]
    remote = SlowRemote()
    config = replace(
        Config(),
        thresholds={cat: 1.01 for cat in Category},
        workers=6,
        remote_workers=2,
    )

    results = run_batch(config, ConstantClient("local"), remote, tasks)

    assert len(results) == 6
    assert all(r.remote_tokens == 2 for r in results)
    assert remote.max_active <= 2
