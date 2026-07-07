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
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .calibrate import calibrate_local
from .clients import ChatClient
from .config import Config
from .classify import classify
from .policy import decide
from .prompts import extract_answer, system_prompt
from .schemas import Category, Route, Task, TaskResult


def run_task(config: Config, local: ChatClient, remote: ChatClient, task: Task,
             k_initial: int, k_max: int) -> TaskResult:
    classification = classify(task)
    category = classification.category

    if config.thresholds.get(category, 0.6) > 1.0:
        # Forced-remote (threshold unreachable): skip local sampling entirely.
        # Used by --remote-only baselines and per-category forced escalation.
        return _escalate(config, remote, task, category,
                         reason="forced remote (threshold > 1)")

    calibration = calibrate_local(
        local, config.local_model, task, category,
        k_initial=k_initial, k_max=k_max, band=config.adaptive_band,
        max_tokens=config.local_max_tokens,
    )
    decision = decide(config, category, calibration)

    if decision.route == Route.LOCAL:
        return TaskResult(
            task_id=task.task_id, answer=calibration.majority_answer,
            category=category, route=Route.LOCAL, model=decision.model,
            remote_tokens=0, reason=decision.reason,
        )

    return _escalate(config, remote, task, category, reason=decision.reason,
                     model=decision.model)


def _escalate(config: Config, remote: ChatClient, task: Task, category: Category,
              reason: str, model: str = None) -> TaskResult:
    if model is None:
        short_name = config.remote_by_category.get(category, "gemma-4-31b-it")
        model = config.remote_model_prefix + short_name
    code_categories = (Category.CODE_DEBUG, Category.CODE_GEN)
    remote_budget = (config.remote_max_tokens_code if category in code_categories
                     else config.remote_max_tokens)
    completion = remote.complete(
        model, system_prompt(category), task.prompt,
        temperature=0.0, max_tokens=remote_budget,
    )
    return TaskResult(
        task_id=task.task_id, answer=extract_answer(category, completion.text),
        category=category, route=Route.REMOTE, model=model,
        remote_tokens=completion.total_tokens, reason=reason,
    )


def run_batch(config: Config, local: ChatClient, remote: ChatClient, tasks: list) -> list:
    """Process tasks concurrently. The local server batches parallel requests
    efficiently, and the scoring time budget is far too small for sequential
    processing. Results keep input order."""
    started = time.monotonic()
    done_count = [0]

    def _one(index_task):
        i, task = index_task
        k_initial, k_max = _sampling_plan(config, started,
                                          done=done_count[0], total=len(tasks))
        try:
            result = run_task(config, local, remote, task, k_initial, k_max)
        except Exception as exc:  # a failed task must never sink the batch
            print(f"[frugal-router] task {task.task_id} failed: {exc}", file=sys.stderr)
            result = TaskResult(
                task_id=task.task_id, answer="", category=classify(task).category,
                route=Route.LOCAL, model=config.local_model,
                remote_tokens=0, reason=f"error: {exc}",
            )
        done_count[0] += 1
        print(
            f"[frugal-router] {done_count[0]}/{len(tasks)} {task.task_id} "
            f"[{result.category.value}] -> {result.route.value} "
            f"(k={k_initial}..{k_max}, remote_tokens={result.remote_tokens})",
            file=sys.stderr,
        )
        return result

    if config.workers <= 1:
        return [_one(pair) for pair in enumerate(tasks)]
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        return list(pool.map(_one, enumerate(tasks)))


def _sampling_plan(config: Config, started: float, done: int, total: int) -> tuple:
    """Shrink sampling when the projected runtime would blow the time budget."""
    if done == 0:
        return config.consistency_samples, config.consistency_samples_max
    elapsed = time.monotonic() - started
    projected_total = elapsed / done * total
    if projected_total <= config.time_budget_seconds * 0.85:
        return config.consistency_samples, config.consistency_samples_max
    if projected_total <= config.time_budget_seconds:
        return max(3, config.consistency_samples - 2), config.consistency_samples
    print("[frugal-router] time budget pressure: degrading to single-sample mode",
          file=sys.stderr)
    return 1, 1


def load_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Task(task_id=str(item["task_id"]), prompt=item["prompt"]) for item in raw]


def write_results(path: str, results: list) -> None:
    payload = [{"task_id": r.task_id, "answer": r.answer} for r in results]
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
