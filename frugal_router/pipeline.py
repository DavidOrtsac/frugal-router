"""Batch pipeline per the Track 1 container contract:
read /input/tasks.json ([{task_id, prompt}]) → write /output/results.json
([{task_id, answer}]).

Flow per task: classify (free) → local self-consistency sampling (free) →
escalation decision → remote call only when calibration is below threshold.

Time budget: an unanswered task is a wrong task, so if the projected runtime
exceeds the budget, later tasks degrade to fewer samples instead of the batch
ever failing to finish.
"""

import json
import os
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock, Semaphore

from .calibrate import calibrate_local
from .clients import ChatClient
from .config import Config
from .classify import classify
from .policy import decide, resolve_remote_model
from .prompts import enforce_summary_format, extract_answer, system_prompt
from .schemas import Category, Route, Task, TaskResult


def _finalize(category: Category, task: Task, answer: str) -> str:
    """Last-mile answer shaping shared by every route."""
    if category is Category.SUMMARIZATION and answer:
        return enforce_summary_format(task.prompt, answer)
    return answer


def run_task(config: Config, local: ChatClient, remote: ChatClient, task: Task,
             k_initial: int, k_max: int, remote_gate=None,
             breaker=None) -> TaskResult:
    plan = plan_task(config, local, task, k_initial, k_max)
    if plan.result is not None:
        return plan.result
    return _escalate(config, remote, task, plan.category, reason=plan.reason,
                     model=plan.model, local=local,
                     fallback_answer=plan.fallback_answer,
                     remote_gate=remote_gate, breaker=breaker)


@dataclass(frozen=True)
class TaskPlan:
    """Phase-1 outcome: either a finished LOCAL result, or an escalation
    request to execute in phase 2 (when the local model is idle — remote
    calls made while llama saturates the CPU quota die mid-handshake)."""

    category: Category
    result: TaskResult = None
    reason: str = ""
    model: str = None
    fallback_answer: str = None


def plan_task(config: Config, local: ChatClient, task: Task,
              k_initial: int, k_max: int) -> TaskPlan:
    """Phase 1: everything that needs only the LOCAL model. Never touches
    the network beyond localhost."""
    classification = classify(task)
    category = classification.category

    threshold = config.thresholds.get(category, 0.6)
    if threshold > 1.0:
        # Forced-remote (threshold unreachable): skip local sampling entirely.
        # Used by --remote-only baselines and per-category forced escalation.
        return TaskPlan(category=category,
                        reason="forced remote (threshold > 1)")
    if threshold == 0.0:
        # Always-local category: voting cannot change the decision, so one
        # greedy sample is enough — a large wall-clock saving on 2 vCPUs.
        k_initial = k_max = 1

    max_tokens = config.local_max_tokens_by_category.get(
        category, config.local_max_tokens)
    calibration = calibrate_local(
        local, config.local_model, task, category,
        k_initial=k_initial, k_max=k_max, band=config.adaptive_band,
        max_tokens=max_tokens,
    )
    decision = decide(config, category, calibration)

    if decision.route == Route.LOCAL:
        return TaskPlan(category=category, result=TaskResult(
            task_id=task.task_id,
            answer=_finalize(category, task, calibration.majority_answer),
            category=category, route=Route.LOCAL, model=decision.model,
            remote_tokens=0, reason=decision.reason,
        ))
    return TaskPlan(category=category, reason=decision.reason,
                    model=decision.model,
                    fallback_answer=calibration.majority_answer)


class RemoteBreaker:
    """Protects the wall clock from a structurally-dead remote WITHOUT ever
    permanently disabling a probe-verified channel.

    Only structural failures (definite HTTP status, excluding 408/429) count
    toward opening: timeouts and rate limits are transient by nature and a
    timeout storm is already survivable (non-retryable, bounded per call).
    An open circuit HALF-OPENS after `retry_after` seconds — one failure
    re-latches it, one success closes it fully. A transient blip can
    therefore cost at most one retry window, never the whole run."""

    def __init__(self, limit: int = 5, retry_after: float = 30.0):
        self._limit = limit
        self._retry_after = retry_after
        self._consecutive = 0
        self._opened_at = None
        self._lock = Lock()

    @property
    def dead(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self._retry_after:
                # Half-open: let calls through; the next outcome decides.
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0
            if self._opened_at is not None:
                print("[frugal-router] remote circuit CLOSED after successful "
                      "trial call", file=sys.stderr)
            self._opened_at = None

    def record_failure(self, exc: Exception = None) -> None:
        if not _is_structural_failure(exc):
            return
        with self._lock:
            self._consecutive += 1
            if self._consecutive >= self._limit:
                if self._opened_at is None:
                    print(f"[frugal-router] remote circuit OPEN after "
                          f"{self._consecutive} consecutive structural "
                          f"failures — retrying in {self._retry_after:.0f}s",
                          file=sys.stderr)
                self._opened_at = time.monotonic()


def _is_structural_failure(exc: Exception) -> bool:
    """A definite HTTP status other than 408/429 means the request itself is
    being rejected (bad model id, bad route, bad auth) — the class of failure
    worth opening the circuit for. Timeouts and transport blips are not."""
    if exc is None:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    if status is None:
        return False
    return isinstance(status, int) and status not in (408, 429)


def _escalate(config: Config, remote: ChatClient, task: Task, category: Category,
              reason: str, model: str = None, local: ChatClient = None,
              fallback_answer: str = None, remote_gate=None,
              breaker=None) -> TaskResult:
    if model is None:
        preferred = config.remote_by_category.get(category, "gemma-4-31b-it")
        model = resolve_remote_model(config, preferred)
    if remote is None:
        # The startup probe found no working channel: local-only mode.
        return _local_fallback(
            config, task, category,
            reason=f"{reason}; remote disabled (probe=ALL_FAILED)",
            local=local,
            fallback_answer=fallback_answer,
            k_initial=config.consistency_samples,
            k_max=config.consistency_samples_max,
        )
    if breaker is not None and breaker.dead:
        return _local_fallback(
            config, task, category,
            reason=f"{reason}; remote circuit open",
            local=local,
            fallback_answer=fallback_answer,
            k_initial=config.consistency_samples,
            k_max=config.consistency_samples_max,
        )
    code_categories = (Category.CODE_DEBUG, Category.CODE_GEN)
    remote_budget = (config.remote_max_tokens_code if category in code_categories
                     else config.remote_max_tokens)
    started = time.monotonic()
    try:
        completion = _complete_remote_with_retry(
            config, remote, remote_gate, model, category, task.prompt,
            remote_budget, breaker=breaker)
        elapsed = time.monotonic() - started
        if breaker is not None:
            breaker.record_success()
        return TaskResult(
            task_id=task.task_id,
            answer=_finalize(category, task,
                             extract_answer(category, completion.text)),
            category=category, route=Route.REMOTE, model=model,
            remote_tokens=completion.total_tokens,
            reason=f"{reason}; remote {elapsed:.1f}s",
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        cause = exc
        chain = []
        while cause is not None and len(chain) < 5:
            chain.append(f"{type(cause).__name__}({cause})")
            cause = cause.__cause__ or cause.__context__
        print(f"[frugal-router] REMOTE CALL FAILED model={model} "
              f"after {elapsed:.1f}s: {' <- '.join(chain)}",
              file=sys.stderr)
        if breaker is not None:
            breaker.record_failure(exc)
        return _local_fallback(
            config, task, category,
            reason=f"remote failed ({exc})",
            local=local,
            fallback_answer=fallback_answer,
            k_initial=3,
            k_max=3,
        )


def _local_fallback(config: Config, task: Task, category: Category, reason: str,
                    local: ChatClient = None, fallback_answer: str = None,
                    k_initial: int = 3, k_max: int = 3) -> TaskResult:
    # A dead or unavailable remote must never produce an empty answer. Prefer
    # an already-computed local majority; otherwise do a local vote.
    answer = fallback_answer
    if answer is None and local is not None:
        from .calibrate import calibrate_local
        max_tokens = config.local_max_tokens_by_category.get(
            category, config.local_max_tokens)
        calibration = calibrate_local(
            local, config.local_model, task, category,
            k_initial=k_initial, k_max=k_max, band=config.adaptive_band,
            max_tokens=max_tokens)
        answer = calibration.majority_answer
    return TaskResult(
        task_id=task.task_id, answer=_finalize(category, task, answer or ""),
        category=category, route=Route.LOCAL, model=config.local_model,
        remote_tokens=0, reason=f"{reason}; local fallback",
    )


class _CircuitOpen(Exception):
    """Raised when the breaker opened while this thread waited at the gate.
    Carries no status_code, so it never counts as a structural failure."""


def _complete_remote_with_retry(config: Config, remote: ChatClient, remote_gate,
                                model: str, category: Category, prompt: str,
                                remote_budget: int, breaker=None):
    attempts = max(1, config.remote_attempts)
    gate = remote_gate if remote_gate is not None else nullcontext()
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            with gate:
                # The breaker may have opened while this thread queued at the
                # semaphore — re-check before spending a real request.
                if breaker is not None and breaker.dead:
                    raise _CircuitOpen("remote circuit open")
                return remote.complete(
                    model, system_prompt(category), prompt,
                    temperature=0.0, max_tokens=remote_budget,
                )
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not _retryable_remote_error(exc):
                raise
            time.sleep(config.remote_retry_delay_seconds * attempt)
    raise last_exc


def _retryable_remote_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return False
    # Connection-class errors carry no HTTP status but are the signature of
    # a flapping proxy — precisely the failures worth retrying.
    if "connect" in name:
        return True
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return status in (408, 409, 429) or (isinstance(status, int) and status >= 500)


def run_batch(config: Config, local: ChatClient, remote: ChatClient, tasks: list) -> list:
    """Process tasks concurrently. The local server batches parallel requests
    efficiently, and the scoring time budget is far too small for sequential
    processing. Results keep input order."""
    # The scoring clock starts at CONTAINER start (model load included), not
    # at batch start — anchor the budget there when the entrypoint tells us.
    container_start = os.environ.get("CONTAINER_START_TS")
    already_spent = (time.time() - float(container_start)) if container_start else 0.0
    started = time.monotonic() - already_spent
    if already_spent:
        print(f"[frugal-router] {already_spent:.0f}s spent before batch start",
              file=sys.stderr)
    done_count = [0]
    degrade_level = [0]  # ratchet: only ever increases
    progress_lock = Lock()
    remote_gate = Semaphore(max(1, config.remote_workers))
    breaker = RemoteBreaker()

    def _cause_chain(exc):
        chain, cause = [], exc
        while cause is not None and len(chain) < 5:
            chain.append(f"{type(cause).__name__}({cause})")
            cause = cause.__cause__ or cause.__context__
        return " <- ".join(chain)

    def _guard(task, fn, phase_label):
        """Run fn with the one-retry guard; a failed task never sinks the
        batch and never silently emits an empty answer without a retry."""
        try:
            return fn()
        except Exception as exc:
            print(f"[frugal-router] task {task.task_id} failed ({phase_label}):"
                  f" {_cause_chain(exc)} — retrying once", file=sys.stderr)
            time.sleep(3.0)
            try:
                return fn()
            except Exception as exc2:
                print(f"[frugal-router] task {task.task_id} failed twice: "
                      f"{_cause_chain(exc2)}", file=sys.stderr)
                return TaskResult(
                    task_id=task.task_id, answer="",
                    category=classify(task).category,
                    route=Route.LOCAL, model=config.local_model,
                    remote_tokens=0, reason=f"error: {exc2}",
                )

    # PHASE 1 — local-only. Remote calls made while llama saturates the
    # 2-CPU cgroup quota die mid-handshake (throttling starves the sockets),
    # so nothing touches the network here beyond localhost.
    def _phase1(index_task):
        i, task = index_task
        with progress_lock:
            k_initial, k_max = _sampling_plan(config, started,
                                              done=done_count[0], total=len(tasks),
                                              degrade_level=degrade_level)
        if (time.monotonic() - started > config.time_budget_seconds * 0.92
                and remote is not None):
            # Clock nearly gone: skip slow local generation, answer remotely
            # in phase 2 — emergency tokens beat an unscored run.
            return TaskPlan(category=classify(task).category,
                            reason="time emergency")
        plan_or_result = _guard(
            task, lambda: plan_task(config, local, task, k_initial, k_max),
            "local phase")
        with progress_lock:
            done_count[0] += 1
        if isinstance(plan_or_result, TaskResult):  # guard fallback
            plan_or_result = TaskPlan(category=plan_or_result.category,
                                      result=plan_or_result)
        state = ("local-done" if plan_or_result.result is not None
                 else "escalate-later")
        print(f"[frugal-router] P1 {done_count[0]}/{len(tasks)} "
              f"{task.task_id} [{plan_or_result.category.value}] -> {state}",
              file=sys.stderr)
        return plan_or_result

    ordered = _execution_order(config, tasks)
    if config.workers <= 1:
        plans = [(i, task, _phase1((i, task))) for i, task in ordered]
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            phase1 = list(pool.map(_phase1, ordered))
        plans = [(pair[0], pair[1], plan) for pair, plan in zip(ordered, phase1)]

    # PHASE 2 — escalations only, with the local model idle. The full CPU
    # quota belongs to networking now; this is why forced-remote categories
    # never died while mid-batch escalations always did.
    pending = [(i, task, plan) for i, task, plan in plans if plan.result is None]
    if pending:
        print(f"[frugal-router] P2: {len(pending)} escalations with local idle",
              file=sys.stderr)

    def _phase2(entry):
        i, task, plan = entry
        result = _guard(
            task, lambda: _escalate(
                config, remote, task, plan.category, reason=plan.reason,
                model=plan.model, local=local,
                fallback_answer=plan.fallback_answer,
                remote_gate=remote_gate, breaker=breaker),
            "remote phase")
        print(f"[frugal-router] P2 {task.task_id} [{result.category.value}] "
              f"-> {result.route.value} (remote_tokens={result.remote_tokens})",
              file=sys.stderr)
        return i, result

    computed = [(i, plan.result) for i, task, plan in plans
                if plan.result is not None]
    if pending:
        if config.workers <= 1:
            computed.extend(_phase2(entry) for entry in pending)
        else:
            with ThreadPoolExecutor(max_workers=max(1, config.remote_workers)) as pool:
                computed.extend(pool.map(_phase2, pending))
    return [result for _, result in sorted(computed, key=lambda item: item[0])]


def _execution_order(config: Config, tasks: list) -> list:
    """(index, task) pairs sorted: always-local categories first, marker-
    confidence categories next, forced-remote last. Stable within a class."""
    def escalation_class(task: Task) -> int:
        threshold = config.thresholds.get(classify(task).category, 0.6)
        if threshold == 0.0:
            return 0
        if threshold > 1.0:
            return 2
        return 1
    return sorted(enumerate(tasks), key=lambda pair: (escalation_class(pair[1]), pair[0]))


def _sampling_plan(config: Config, started: float, done: int, total: int,
                   degrade_level: list) -> tuple:
    """Shrink sampling when the projected runtime would blow the time budget.

    The degrade level is a RATCHET: fast degraded tasks improve the average,
    but recovering to full sampling would oscillate straight back into
    timeout territory, so pressure only ever tightens."""
    if done > 0:
        elapsed = time.monotonic() - started
        projected_total = elapsed / done * total
        if projected_total > config.time_budget_seconds * 0.9:
            new_level = 2
        elif projected_total > config.time_budget_seconds * 0.7:
            new_level = 1
        else:
            new_level = 0
        if new_level > degrade_level[0]:
            degrade_level[0] = new_level
            print(f"[frugal-router] time pressure: degrade level {new_level}",
                  file=sys.stderr)
    if degrade_level[0] >= 2:
        return 1, 1
    if degrade_level[0] == 1:
        return max(3, config.consistency_samples - 2), config.consistency_samples
    return config.consistency_samples, config.consistency_samples_max


def load_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Task(task_id=str(item["task_id"]), prompt=item["prompt"]) for item in raw]


def write_results(path: str, results: list) -> None:
    payload = [{"task_id": r.task_id, "answer": r.answer} for r in results]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def report(results: list) -> dict:
    total = len(results)
    local_count = sum(1 for r in results if r.route == Route.LOCAL)
    remote_tokens = sum(r.remote_tokens for r in results)
    return {
        "tasks": total,
        "offload_rate": (local_count / total) if total else 0.0,
        "remote_tokens": remote_tokens,
    }
