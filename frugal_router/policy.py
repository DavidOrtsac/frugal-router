"""Escalation policy: the single decision that determines the score.

Rule: stay local unless the calibration signal says the local answer is
likely below the accuracy bar. Freeform categories (summarization, sentiment)
have threshold 0.0 — agreement sampling is meaningless there, and the local
model handles them well, so they never escalate on consistency."""

from .config import Config
from .schemas import Calibration, Category, Route, RouteDecision


def decide(config: Config, category: Category, calibration: Calibration) -> RouteDecision:
    threshold = config.thresholds.get(category, 0.6)
    if calibration.score >= threshold or threshold == 0.0:
        return RouteDecision(
            route=Route.LOCAL,
            model=config.local_model,
            reason=f"consistency {calibration.score:.2f} >= threshold {threshold:.2f}",
        )

    remote_short_name = config.remote_by_category.get(category, "gemma-4-31b-it")
    if remote_short_name not in config.allowed_models:
        # Hard guard: an out-of-list call invalidates the submission.
        fallback = next(m for m in config.allowed_models if "gemma" in m)
        remote_short_name = fallback
    return RouteDecision(
        route=Route.REMOTE,
        model=config.remote_model_prefix + remote_short_name,
        reason=f"consistency {calibration.score:.2f} < threshold {threshold:.2f}",
    )
