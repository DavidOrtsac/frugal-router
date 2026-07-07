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
    assert "gemma" in decision.model


def test_code_escalates_to_code_model():
    decision = decide(Config(), Category.CODE_GEN, _calibration(0.0))
    assert decision.route == Route.REMOTE
    assert "kimi-k2p7-code" in decision.model


def test_freeform_never_escalates():
    decision = decide(Config(), Category.SUMMARIZATION, _calibration(0.0))
    assert decision.route == Route.LOCAL


def test_remote_model_always_in_allowed_list():
    config = Config()
    for category in Category:
        decision = decide(config, category, _calibration(0.0))
        if decision.route == Route.REMOTE:
            short = decision.model.removeprefix(config.remote_model_prefix)
            assert short in config.allowed_models
