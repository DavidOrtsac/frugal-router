"""Escalation policy: the single decision that determines the score.

Rule: stay local unless the calibration signal says the local answer is
likely below the accuracy bar. Freeform categories (summarization, sentiment)
have threshold 0.0 — agreement sampling is meaningless there, and the local
model handles them well, so they never escalate on consistency.

Model resolution: Fireworks direct API needs full
accounts/fireworks/models/... IDs, while a judge proxy may publish short
aliases in ALLOWED_MODELS. Resolve preferred models against ALLOWED_MODELS,
then canonicalize short aliases only when talking to Fireworks directly.
"""

from .config import Config
from .schemas import Calibration, Category, Route, RouteDecision

_FALLBACK_PREFERENCE = ("kimi", "gemma", "minimax")


def resolve_remote_model(config: Config, preferred: str) -> str:
    """Map a preferred model name onto the ALLOWED_MODELS list, safely.

    Fallbacks are drawn FROM the allowed list itself. If the target endpoint is
    direct Fireworks and the allowed entry is a short alias, convert it to the
    canonical Fireworks path; short IDs 404 on the public Fireworks API.
    """
    preferred_short = preferred.rsplit("/", 1)[-1]
    for allowed in config.allowed_models:
        if allowed.rsplit("/", 1)[-1] == preferred_short:
            return _canonical_model_id(config, allowed)
    for pattern in _FALLBACK_PREFERENCE:
        for allowed in config.allowed_models:
            if pattern in allowed.lower():
                return _canonical_model_id(config, allowed)
    return _canonical_model_id(config, config.allowed_models[0])


def _canonical_model_id(config: Config, model: str) -> str:
    if "/" in model:
        return model
    if "api.fireworks.ai" in config.fireworks_base_url:
        return f"accounts/fireworks/models/{model}"
    return model


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
