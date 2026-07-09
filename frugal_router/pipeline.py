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
from threading import Lock, Semaphore

from .calibrate import calibrate_local
from .clients import ChatClient
from .config import Config
from .classify import classify
from .policy import decide, resolve_remote_model
from .prompts import extract_answer, system_prompt
from .schemas import Category, Route, Task, TaskResult


def run_task(config: Config, local: ChatClient, remote: ChatClient, task: Task,
             k_initial: int, k_max: int, remote_gate=None) -> TaskResult:
    classification = classify(task)
    category = classification.category

    threshold = config.thresholds.get(category, 0.6)
    if threshold > 1.0:
        # Forced-remote (threshold unreachable): skip local sampling entirely.
        # Used by --remote-only baselines and per-category forced escalation.
        return _escalate(config, remote, task, category,
                         reason="forced remote (threshold > 1)", local=local,
                         remote_gate=remote_gate)
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
        return TaskResult(
            task_id=task.task_id, answer=calibration.majority_answer,
            category=category, route=Route.LOCAL, model=decision.model,
            remote_tokens=0, reason=decision.reason,
        )

    return _escalate(config, remote, task, category, reason=decision.reason,
                     model=decision.model,
                     fallback_answer=calibration.majority_answer,
                     remote_gate=remote_gate)


def _escalate(config: Config, remote: ChatClient, task: Task, category: Category,
              reason: str, model: str = None, local: ChatClient = None,
              fallback_answer: str = None, remote_gate=None) -> TaskResult:
    if model is None:
        preferred = config.remote_by_category.get(category, "gemma-4-31b-it")
        model = resolve_remote_model(config, preferred)
    code_categories = (Category.CODE_DEBUG, Category.CODE_GEN)
    remote_budget = (config.remote_max_tokens_code if category in code_categories
                     else config.remote_max_tokens)
    started = time.monotonic()
    try:
        completion = _complete_remote_with_retry(
            config, remote, remote_gate, model, category, task.prompt, remote_budget)
        elapsed = time.monotonic() - started
        return TaskResult(
            task_id=task.task_id, answer=extract_answer(category, completion.text),
            category=category, route=Route.REMOTE, model=model,
            remote_tokens=completion.total_tokens,
            reason=f"{reason}; remote {elapsed:.1f}s",
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f"[frugal-router] REMOTE CALL FAILED model={model} "
              f"after {elapsed:.1f}s: {exc}",
              file=sys.stderr)
        # A dead remote must never produce an empty answer: fall back to the
        # local majority answer, or a fresh local majority vote.
        answer = fallback_answer
        if answer is None and local is not None:
            from .calibrate import calibrate_local
            max_tokens = config.local_max_tokens_by_category.get(
                category, config.local_max_tokens)
            calibration = calibrate_local(
                local, config.local_model, task, category,
                k_initial=3, k_max=3, band=(2.0, 2.0), max_tokens=max_tokens)
            answer = calibration.majority_answer
        return TaskResult(
            task_id=task.task_id, answer=answer or "",
            category=category, route=Route.LOCAL, model=config.local_model,
            remote_tokens=0, reason=f"remote failed ({exc}); local fallback",
        )


def _complete_remote_with_retry(config: Config, remote: ChatClient, remote_gate,
                                model: str, category: Category, prompt: str,
                                remote_budget: int):
    attempts = max(1, config.remote_attempts)
    gate = remote_gate if remote_gate is not None else nullcontext()
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            with gate:
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

    def _one(index_task):
        i, task = index_task
        with progress_lock:
            k_initial, k_max = _sampling_plan(config, started,
                                              done=done_count[0], total=len(tasks),
                                              degrade_level=degrade_level)
        try:
            result = run_task(config, local, remote, task, k_initial, k_max,
                              remote_gate=remote_gate)
        except Exception as exc:  # a failed task must never sink the batch
            print(f"[frugal-router] task {task.task_id} failed: {exc}", file=sys.stderr)
            result = TaskResult(
                task_id=task.task_id, answer="", category=classify(task).category,
                route=Route.LOCAL, model=config.local_model,
                remote_tokens=0, reason=f"error: {exc}",
            )
        with progress_lock:
            done_count[0] += 1
            done = done_count[0]
        print(
            f"[frugal-router] {done}/{len(tasks)} {task.task_id} "
            f"[{result.category.value}] -> {result.route.value} "
            f"(k={k_initial}..{k_max}, remote_tokens={result.remote_tokens})",
            file=sys.stderr,
        )
        return result

    if config.workers <= 1:
        return [_one(pair) for pair in enumerate(tasks)]
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        return list(pool.map(_one, enumerate(tasks)))


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
