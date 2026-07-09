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

_FALLBACK_PREFERENCE = ("kimi", "gemma", "minimax")


def resolve_remote_model(config: Config, preferred: str) -> str:
    """Map a preferred model name onto the ALLOWED_MODELS list, safely.

    The harness publishes EXACT model IDs and the guide says to use them
    verbatim — so the matched allowed-list entry is returned character for
    character, never rewritten, never prefixed. (For local development
    against api.fireworks.ai directly, put full model paths into your own
    ALLOWED_MODELS env value.) Fallbacks are drawn FROM the allowed list
    itself, so an out-of-list call is impossible by construction.
    """
    preferred_short = preferred.rsplit("/", 1)[-1]
    for allowed in config.allowed_models:
        if allowed.rsplit("/", 1)[-1] == preferred_short:
            return allowed
    for pattern in _FALLBACK_PREFERENCE:
        for allowed in config.allowed_models:
            if pattern in allowed.lower():
                return allowed
    return config.allowed_models[0]


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
