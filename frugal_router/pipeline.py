"""Batch pipeline per the Track 1 container contract:
read /input/tasks.json ([{task_id, prompt}]) → write /output/results.json
([{task_id, answer}]).

Flow per task: classify (free) → local self-consistency sampling (free) →
escalation decision → remote call only when calibration is below threshold.
"""

import json
import sys

from .calibrate import calibrate_local
from .clients import ChatClient
from .config import Config
from .classify import classify
from .policy import decide
from .prompts import extract_answer, system_prompt
from .schemas import Route, Task, TaskResult


def run_task(config: Config, local: ChatClient, remote: ChatClient, task: Task) -> TaskResult:
    classification = classify(task)
    category = classification.category

    calibration = calibrate_local(
        local, config.local_model, task, category,
        k=config.consistency_samples, max_tokens=config.local_max_tokens,
    )
    decision = decide(config, category, calibration)

    if decision.route == Route.LOCAL:
        return TaskResult(
            task_id=task.task_id, answer=calibration.majority_answer,
            category=category, route=Route.LOCAL, model=decision.model,
            remote_tokens=0, reason=decision.reason,
        )

    completion = remote.complete(
        decision.model, system_prompt(category), task.prompt,
        temperature=0.0, max_tokens=config.remote_max_tokens,
    )
    return TaskResult(
        task_id=task.task_id, answer=extract_answer(category, completion.text),
        category=category, route=Route.REMOTE, model=decision.model,
        remote_tokens=completion.total_tokens, reason=decision.reason,
    )


def run_batch(config: Config, local: ChatClient, remote: ChatClient, tasks: list) -> list:
    results = []
    for i, task in enumerate(tasks):
        try:
            result = run_task(config, local, remote, task)
        except Exception as exc:  # a failed task must never sink the batch
            print(f"[frugal-router] task {task.task_id} failed: {exc}", file=sys.stderr)
            result = TaskResult(
                task_id=task.task_id, answer="", category=classify(task).category,
                route=Route.LOCAL, model=config.local_model,
                remote_tokens=0, reason=f"error: {exc}",
            )
        results.append(result)
        print(
            f"[frugal-router] {i + 1}/{len(tasks)} {task.task_id} "
            f"[{result.category.value}] -> {result.route.value} "
            f"(remote_tokens={result.remote_tokens})",
            file=sys.stderr,
        )
    return results


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
