import os
import sys
import time
from dataclasses import replace
from threading import Lock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.config import Config
from frugal_router.pipeline import _escalate, run_batch, write_results
from frugal_router.schemas import Category, Completion, Task


class ConstantClient:
    def __init__(self, text="ok"):
        self.text = text

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        return Completion(text=self.text, prompt_tokens=1, completion_tokens=1)

    def complete_many(self, model, system, user, temperature=0.7, n=4, max_tokens=512):
        return [self.complete(model, system, user, temperature, max_tokens)
                for _ in range(n)]


class CountingClient(ConstantClient):
    def __init__(self, text="ok"):
        super().__init__(text)
        self.calls = 0

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        self.calls += 1
        return super().complete(model, system, user, temperature, max_tokens)


class ExplodingRemote(ConstantClient):
    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        raise AssertionError("remote should not be called")


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
    config = replace(
        Config(),
        fireworks_api_key="test-key",
        remote_attempts=2,
        remote_retry_delay_seconds=0.0,
    )

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
        fireworks_api_key="test-key",
        thresholds={cat: 1.01 for cat in Category},
        workers=6,
        remote_workers=2,
    )

    results = run_batch(config, ConstantClient("local"), remote, tasks)

    assert len(results) == 6
    assert all(r.remote_tokens == 2 for r in results)
    assert remote.max_active <= 2


def test_probe_failure_disables_remote_and_uses_local_sampling():
    # remote=None is how main.py signals "startup probe found no channel".
    local = CountingClient("local-ok")
    config = replace(Config(), consistency_samples=4, consistency_samples_max=6)

    result = _escalate(
        config,
        None,
        Task("t", "What is the capital of Australia?"),
        Category.FACTUAL,
        reason="forced",
        local=local,
    )

    assert result.answer == "local-ok"
    assert result.remote_tokens == 0
    assert "probe=ALL_FAILED" in result.reason
    assert local.calls == 4


class StructuralError(Exception):
    status_code = 404


class AlwaysFailingRemote(ConstantClient):
    def __init__(self, exc_factory=StructuralError):
        super().__init__("never")
        self.calls = 0
        self._exc_factory = exc_factory

    def complete(self, model, system, user, temperature=0.0, max_tokens=512):
        self.calls += 1
        raise self._exc_factory("structurally dead")


def test_breaker_opens_on_structural_failures_and_stops_remote_calls():
    from frugal_router.pipeline import RemoteBreaker

    local = ConstantClient("local-ok")
    remote = AlwaysFailingRemote()
    config = replace(Config(), fireworks_api_key="test-key", remote_attempts=1)
    breaker = RemoteBreaker(limit=2, retry_after=300.0)

    for i in range(4):
        result = _escalate(
            config, remote, Task(str(i), "What is the capital of Australia?"),
            Category.FACTUAL, reason="forced", local=local, breaker=breaker,
        )
        assert result.remote_tokens == 0
        assert result.answer == "local-ok"

    # Two failures open the circuit; the last two tasks never touch remote.
    assert remote.calls == 2
    assert breaker.dead


def test_breaker_ignores_timeouts_and_half_opens_for_recovery():
    from frugal_router.pipeline import RemoteBreaker

    class FakeTimeout(Exception):
        pass
    FakeTimeout.__name__ = "ReadTimeoutError"

    breaker = RemoteBreaker(limit=2, retry_after=0.0)
    # Timeouts never open the circuit, no matter how many.
    for _ in range(10):
        breaker.record_failure(FakeTimeout("slow"))
    assert not breaker.dead

    # Structural failures open it; retry_after=0 half-opens immediately.
    breaker = RemoteBreaker(limit=2, retry_after=0.0)
    breaker.record_failure(StructuralError("x"))
    breaker.record_failure(StructuralError("x"))
    assert not breaker.dead  # half-open: a trial call is allowed
    breaker.record_success()
    assert not breaker.dead  # fully closed again

    # With a long retry window it stays latched until the window passes.
    breaker = RemoteBreaker(limit=2, retry_after=300.0)
    breaker.record_failure(StructuralError("x"))
    breaker.record_failure(StructuralError("x"))
    assert breaker.dead


def test_write_results_creates_output_directory(tmp_path):
    out = tmp_path / "nested" / "results.json"
    results = run_batch(
        replace(Config(), workers=1, thresholds={cat: 0.0 for cat in Category}),
        ConstantClient("answer"),
        ConstantClient("remote"),
        [Task("t1", "What is the capital of Australia?")],
    )

    write_results(str(out), results)

    assert out.exists()


def test_execution_order_locals_first_forced_remote_last():
    from frugal_router.pipeline import _execution_order
    from frugal_router.schemas import Category

    config = replace(Config(), thresholds={
        Category.FACTUAL: 0.0, Category.MATH: 0.5, Category.CODE_GEN: 1.01,
        Category.SENTIMENT: 0.0, Category.SUMMARIZATION: 0.0,
        Category.NER: 0.0, Category.CODE_DEBUG: 1.01, Category.LOGICAL: 1.01,
    })
    tasks = [
        Task("code", "Write a python function that adds two numbers."),
        Task("fact", "What is the capital of Australia?"),
        Task("math", "What is 15 * 4 - 3? Show your work."),
    ]
    ordered_ids = [t.task_id for _, t in _execution_order(config, tasks)]
    assert ordered_ids.index("fact") < ordered_ids.index("math") < ordered_ids.index("code")


def test_run_batch_preserves_input_order_despite_reordering():
    tasks = [
        Task("t0", "Write a python function that adds two numbers."),
        Task("t1", "What is the capital of Australia?"),
        Task("t2", "Classify the sentiment: I love this product."),
    ]
    config = replace(
        Config(),
        fireworks_api_key="test-key",
        thresholds={cat: (1.01 if cat in (Category.CODE_GEN, Category.CODE_DEBUG)
                          else 0.0) for cat in Category},
        workers=2,
    )
    results = run_batch(config, ConstantClient("local"), ConstantClient("remote"), tasks)
    assert [r.task_id for r in results] == ["t0", "t1", "t2"]


def test_connection_errors_are_retryable():
    from frugal_router.pipeline import _retryable_remote_error

    class APIConnectionError(Exception):
        pass

    class ReadTimeoutError(Exception):
        pass

    assert _retryable_remote_error(APIConnectionError("boom"))
    assert not _retryable_remote_error(ReadTimeoutError("slow"))


def test_checkpointer_writes_every_task_id_immediately(tmp_path):
    from frugal_router.pipeline import ResultCheckpointer, write_placeholder
    from frugal_router.schemas import Route, TaskResult
    import json as _json

    out = tmp_path / "results.json"
    tasks = [Task("a", "p1"), Task("b", "p2"), Task("c", "p3")]
    write_placeholder(str(out), tasks)
    payload = _json.loads(out.read_text())
    assert [r["task_id"] for r in payload] == ["a", "b", "c"]
    assert all(r["answer"] == "" for r in payload)

    cp = ResultCheckpointer(str(out), tasks)
    cp.record(TaskResult(task_id="b", answer="answer-b",
                         category=Category.FACTUAL, route=Route.LOCAL,
                         model="m", remote_tokens=0, reason="r"))
    payload = _json.loads(out.read_text())
    assert {r["task_id"]: r["answer"] for r in payload} == {
        "a": "", "b": "answer-b", "c": ""}


def test_sampling_plan_never_increases_k_under_pressure():
    from frugal_router.pipeline import _sampling_plan

    config = replace(Config(), consistency_samples=1, consistency_samples_max=1,
                     time_budget_seconds=100.0)
    level = [1]
    k_init, k_max = _sampling_plan(config, started=time.monotonic(),
                                   done=0, total=10, degrade_level=level)
    assert k_init <= 1 and k_max <= 1
