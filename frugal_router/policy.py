"""Escalation policy: the single decision that determines the score.

Rule: stay local unless the calibration signal says the local answer is
likely below the accuracy bar. Freeform categories (summarization, sentiment)
have threshold 0.0 — agreement sampling is meaningless there, and the local
model handles them well, so they never escalate on consistency.

Model resolution: the harness publishes ALLOWED_MODELS as exact model IDs at
runtime (short names or full paths — format unknown until scoring). Every
remote model name is resolved against that list; the router can never emit a
model ID that is not on it.
"""

from .config import Config
from .schemas import Calibration, Category, Route, RouteDecision

_FALLBACK_PREFERENCE = ("gemma", "kimi", "minimax")


def resolve_remote_model(config: Config, preferred: str) -> str:
    """Map a preferred model name onto the ALLOWED_MODELS list, safely.

    Handles all list formats: short names ("gemma-4-31b-it") or full IDs
    ("accounts/fireworks/models/gemma-4-31b-it"). Falls back to a preference
    order drawn FROM the allowed list itself, so an out-of-list call is
    impossible by construction.
    """
    preferred_short = preferred.rsplit("/", 1)[-1]
    for allowed in config.allowed_models:
        if allowed.rsplit("/", 1)[-1] == preferred_short:
            return _full_id(config, allowed)
    for pattern in _FALLBACK_PREFERENCE:
        for allowed in config.allowed_models:
            if pattern in allowed.lower():
                return _full_id(config, allowed)
    return _full_id(config, config.allowed_models[0])


def _full_id(config: Config, allowed_entry: str) -> str:
    if "/" in allowed_entry:
        return allowed_entry
    return config.remote_model_prefix + allowed_entry


def decide(config: Config, category: Category, calibration: Calibration) -> RouteDecision:
    threshold = config.thresholds.get(category, 0.6)
    if calibration.score >= threshold or threshold == 0.0:
        return RouteDecision(
            route=Route.LOCAL,
            model=config.local_model,
            reason=f"consistency {calibration.score:.2f} >= threshold {threshold:.2f}",
        )

    preferred = config.remote_by_category.get(category, "gemma-4-31b-it")
    return RouteDecision(
        route=Route.REMOTE,
        model=resolve_remote_model(config, preferred),
        reason=f"consistency {calibration.score:.2f} < threshold {threshold:.2f}",
    )
