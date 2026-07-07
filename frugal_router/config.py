"""Environment-driven configuration. Every scoring lever lives here so the eval
harness can sweep it without code changes."""

import json
import os
from dataclasses import dataclass, field

from .schemas import Category

DEFAULT_ALLOWED_MODELS = (
    "minimax-m3",
    "kimi-k2p7-code",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
)

# Which remote model to escalate to, per category. Code-shaped tasks go to the
# code-specialized model; everything else defaults to Gemma 4 (bonus-eligible).
DEFAULT_REMOTE_BY_CATEGORY = {
    Category.CODE_DEBUG: "kimi-k2p7-code",
    Category.CODE_GEN: "kimi-k2p7-code",
    Category.FACTUAL: "gemma-4-31b-it",
    Category.MATH: "gemma-4-31b-it",
    Category.SENTIMENT: "gemma-4-31b-it",
    Category.SUMMARIZATION: "gemma-4-31b-it",
    Category.NER: "gemma-4-31b-it",
    Category.LOGICAL: "gemma-4-31b-it",
}

# Per-category consistency thresholds. Below the threshold → escalate.
# 0.0 means "never escalate on consistency" (freeform tasks where sampling
# agreement is meaningless). These are the numbers eval/sweep.py tunes.
DEFAULT_THRESHOLDS = {
    Category.FACTUAL: 0.6,
    Category.MATH: 0.6,
    Category.SENTIMENT: 0.0,
    Category.SUMMARIZATION: 0.0,
    Category.NER: 0.4,
    Category.CODE_DEBUG: 0.4,
    Category.LOGICAL: 0.6,
    Category.CODE_GEN: 0.4,
}


@dataclass(frozen=True)
class Config:
    local_base_url: str = "http://localhost:8000/v1"
    local_model: str = "google/gemma-4-26B-A4B-it"
    local_extra_body: dict = field(default_factory=dict)  # e.g. disable Qwen3 thinking
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_api_key: str = ""
    remote_model_prefix: str = "accounts/fireworks/models/"
    allowed_models: tuple = DEFAULT_ALLOWED_MODELS
    remote_by_category: dict = field(default_factory=lambda: dict(DEFAULT_REMOTE_BY_CATEGORY))
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    consistency_samples: int = 5
    consistency_samples_max: int = 10  # adaptive extension for borderline agreement
    adaptive_band: tuple = (0.3, 0.9)  # extend sampling when score falls inside
    local_max_tokens: int = 1024
    remote_max_tokens: int = 512
    remote_max_tokens_code: int = 1600  # reasoning models eat budget before code
    time_budget_seconds: float = 3300.0  # degrade sampling rather than not finish
    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"


def config_from_env() -> Config:
    allowed = os.environ.get("ALLOWED_MODELS", "")
    allowed_models = (
        tuple(m.strip() for m in allowed.split(",") if m.strip())
        if allowed
        else DEFAULT_ALLOWED_MODELS
    )
    remote_by_category = dict(DEFAULT_REMOTE_BY_CATEGORY)
    remote_default = os.environ.get("REMOTE_DEFAULT_MODEL")
    if remote_default:
        remote_code = os.environ.get("REMOTE_CODE_MODEL", remote_default)
        code_categories = (Category.CODE_DEBUG, Category.CODE_GEN)
        remote_by_category = {
            cat: (remote_code if cat in code_categories else remote_default)
            for cat in Category
        }

    global_threshold = os.environ.get("CONSISTENCY_THRESHOLD")
    thresholds = dict(DEFAULT_THRESHOLDS)
    if global_threshold is not None:
        override = float(global_threshold)
        thresholds = {
            cat: (0.0 if base == 0.0 else override) for cat, base in thresholds.items()
        }
    # Highest priority: full per-category maps as JSON, so tuned values ship
    # via environment without code edits, e.g.
    #   THRESHOLDS_JSON='{"math_reasoning": 0.8, "ner": 0.4}'
    #   REMOTE_MAP_JSON='{"code_generation": "kimi-k2p7-code"}'
    thresholds_json = os.environ.get("THRESHOLDS_JSON")
    if thresholds_json:
        for key, value in json.loads(thresholds_json).items():
            thresholds[Category(key)] = float(value)
    remote_map_json = os.environ.get("REMOTE_MAP_JSON")
    if remote_map_json:
        for key, value in json.loads(remote_map_json).items():
            remote_by_category[Category(key)] = value
    return Config(
        local_base_url=os.environ.get("LOCAL_BASE_URL", Config.local_base_url),
        local_model=os.environ.get("LOCAL_MODEL", Config.local_model),
        local_extra_body=json.loads(os.environ.get("LOCAL_EXTRA_BODY", "{}")),
        fireworks_base_url=os.environ.get("FIREWORKS_BASE_URL", Config.fireworks_base_url),
        fireworks_api_key=os.environ.get("FIREWORKS_API_KEY", ""),
        remote_model_prefix=os.environ.get("REMOTE_MODEL_PREFIX", Config.remote_model_prefix),
        allowed_models=allowed_models,
        remote_by_category=remote_by_category,
        thresholds=thresholds,
        consistency_samples=int(os.environ.get("CONSISTENCY_SAMPLES", "5")),
        consistency_samples_max=int(os.environ.get("CONSISTENCY_SAMPLES_MAX", "10")),
        time_budget_seconds=float(os.environ.get("TIME_BUDGET_SECONDS", "3300")),
        input_path=os.environ.get("INPUT_PATH", Config.input_path),
        output_path=os.environ.get("OUTPUT_PATH", Config.output_path),
    )
