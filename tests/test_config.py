import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.config import config_from_env
from frugal_router.schemas import Category


def test_env_defaults_match_submission_runtime(monkeypatch):
    for key in (
        "LOCAL_BASE_URL",
        "LOCAL_MODEL",
        "CONSISTENCY_SAMPLES",
        "CONSISTENCY_SAMPLES_MAX",
        "WORKERS",
        "REMOTE_WORKERS",
        "REMOTE_TIMEOUT_SECONDS",
        "REMOTE_ATTEMPTS",
        "REMOTE_MAX_TOKENS",
        "REMOTE_MAX_TOKENS_CODE",
        "REMOTE_MAP_JSON",
    ):
        monkeypatch.delenv(key, raising=False)

    config = config_from_env()

    assert config.local_base_url == "http://localhost:8901/v1"
    assert config.local_model == "qwen3-1.7b"
    assert config.consistency_samples == 3
    assert config.consistency_samples_max == 5
    assert config.workers == 6
    assert config.remote_workers == 3
    assert config.remote_timeout_seconds == 24
    assert config.remote_attempts == 2
    assert config.remote_max_tokens == 512
    assert config.remote_max_tokens_code == 900


def test_env_json_overrides_are_parsed(monkeypatch):
    monkeypatch.setenv("THRESHOLDS_JSON", '{"math_reasoning": 1.01, "ner": 0.4}')
    monkeypatch.setenv("REMOTE_MAP_JSON", '{"math_reasoning": "gemma-4-31b-it"}')

    config = config_from_env()

    assert config.thresholds[Category.MATH] == 1.01
    assert config.thresholds[Category.NER] == 0.4
    assert config.remote_by_category[Category.MATH] == "gemma-4-31b-it"
