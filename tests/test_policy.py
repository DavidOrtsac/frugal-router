import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.config import Config
from frugal_router.policy import decide
from frugal_router.schemas import Calibration, Category, Route


def _calibration(score: float) -> Calibration:
    return Calibration(score=score, majority_answer="x", samples=("x",))


def test_high_consistency_stays_local():
    decision = decide(Config(), Category.MATH, _calibration(0.9))
    assert decision.route == Route.LOCAL


def test_low_consistency_escalates():
    decision = decide(Config(), Category.MATH, _calibration(0.2))
    assert decision.route == Route.REMOTE
    assert decision.model == "accounts/fireworks/models/kimi-k2p7-code"
    assert decision.model in Config().allowed_models


def test_code_escalates_to_code_model():
    decision = decide(Config(), Category.CODE_GEN, _calibration(0.0))
    assert decision.route == Route.REMOTE
    assert decision.model == "accounts/fireworks/models/kimi-k2p7-code"


def test_freeform_never_escalates():
    decision = decide(Config(), Category.SUMMARIZATION, _calibration(0.0))
    assert decision.route == Route.LOCAL


def test_remote_model_always_in_allowed_list():
    config = Config()
    for category in Category:
        decision = decide(config, category, _calibration(0.0))
        if decision.route == Route.REMOTE:
            assert decision.model in config.allowed_models


def test_resolver_handles_full_path_allowed_list():
    from dataclasses import replace
    from frugal_router.policy import resolve_remote_model
    config = replace(Config(), allowed_models=(
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/gemma-4-31b-it",
    ))
    assert resolve_remote_model(config, "gemma-4-31b-it") == \
        "accounts/fireworks/models/gemma-4-31b-it"
    # preferred model absent -> falls back FROM the list, no invention
    assert resolve_remote_model(config, "minimax-m3") == \
        "accounts/fireworks/models/kimi-k2p7-code"


def test_resolver_returns_allowed_entries_verbatim():
    # The startup probe verifies and pins every allowed entry; resolution must
    # return them character for character — no prefixing, no rewriting.
    from dataclasses import replace
    from frugal_router.policy import resolve_remote_model
    config = replace(Config(), allowed_models=("kimi-k2p7-code",))
    assert resolve_remote_model(config, "accounts/fireworks/models/kimi-k2p7-code") == \
        "kimi-k2p7-code"
    config = replace(
        Config(),
        fireworks_base_url="https://judge-proxy.example/v1",
        allowed_models=("kimi-k2p7-code",),
    )
    assert resolve_remote_model(config, "accounts/fireworks/models/kimi-k2p7-code") == \
        "kimi-k2p7-code"


def test_resolver_fallback_prefers_measured_kimi_when_available():
    from dataclasses import replace
    from frugal_router.policy import resolve_remote_model
    config = replace(Config(), allowed_models=(
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/kimi-k2p7-code",
    ))
    assert resolve_remote_model(config, "missing-model") == \
        "accounts/fireworks/models/kimi-k2p7-code"
