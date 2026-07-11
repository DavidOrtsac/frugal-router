"""Environment-driven configuration. Every scoring lever lives here so the eval
harness can sweep it without code changes.

The three judge-injected variables (FIREWORKS_BASE_URL, FIREWORKS_API_KEY,
ALLOWED_MODELS) arrive in an UNDOCUMENTED serialization. Everything read from
them is sanitized defensively: quotes, brackets, whitespace, JSON arrays, and
alternative separators are all tolerated, because a single wrong assumption
here silently kills every remote call (proven by seven scored submissions).
"""

import json
import os
import re
from dataclasses import dataclass, field

from .schemas import Category

DEFAULT_ALLOWED_MODELS = (
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/gemma-4-31b-it",
    "accounts/fireworks/models/gemma-4-26b-a4b-it",
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
)

# Which remote model to escalate to, per category. The final qualification
# profile pins every escalation to the measured-best allowed expert.
DEFAULT_REMOTE_BY_CATEGORY = {
    Category.CODE_DEBUG: "kimi-k2p7-code",
    Category.CODE_GEN: "kimi-k2p7-code",
    Category.FACTUAL: "kimi-k2p7-code",
    Category.MATH: "kimi-k2p7-code",
    Category.SENTIMENT: "kimi-k2p7-code",
    Category.SUMMARIZATION: "kimi-k2p7-code",
    Category.NER: "kimi-k2p7-code",
    Category.LOGICAL: "kimi-k2p7-code",
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


_MODEL_TOKEN = re.compile(r"^[A-Za-z0-9._/-]+$")
_QUOTE_CHARS = " \t\r\n'\"[](){}"


def _sanitize_env(value: str) -> str:
    return (value or "").strip().strip("'\"").strip()


def sanitize_base_url(raw: str) -> str:
    """Normalize an injected base URL: strip quotes/whitespace, a trailing
    slash, and a mistakenly-included /chat/completions suffix (the OpenAI SDK
    appends that path itself). Returns "" when unset so callers can default."""
    base = _sanitize_env(raw).rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    if base and "://" not in base:
        base = "https://" + base
    return base


def parse_allowed_models(raw: str) -> tuple:
    """Parse ALLOWED_MODELS in whatever serialization the harness uses.

    Accepts: bare comma-separated, comma+space, JSON arrays (single- or
    double-quoted), space/semicolon/newline separated, and quote- or
    bracket-wrapped tokens. Returns (clean_tokens, raw_tokens): clean tokens
    are validated model-id shapes; raw tokens preserve the original split
    pieces as last-resort probe candidates."""
    text = (raw or "").strip()
    if not text:
        return (), ()
    for candidate in (text, text.replace("'", '"')):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            tokens = tuple(str(item).strip() for item in data if str(item).strip())
            return tokens, tokens
        if isinstance(data, str) and data.strip():
            text = data.strip()
            break
    pieces = [p for p in re.split(r"[,;\s]+", text) if p]
    cleaned = []
    for piece in pieces:
        token = piece.strip(_QUOTE_CHARS)
        if token and _MODEL_TOKEN.match(token):
            cleaned.append(token)
    return tuple(dict.fromkeys(cleaned)), tuple(dict.fromkeys(pieces))


@dataclass(frozen=True)
class Config:
    local_base_url: str = "http://127.0.0.1:8901/v1"
    local_model: str = "qwen3-1.7b"
    local_extra_body: dict = field(default_factory=dict)  # e.g. disable Qwen3 thinking
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_api_key: str = ""
    allowed_models: tuple = DEFAULT_ALLOWED_MODELS
    allowed_models_raw: tuple = ()  # unsanitized split pieces, probe fallback
    remote_by_category: dict = field(default_factory=lambda: dict(DEFAULT_REMOTE_BY_CATEGORY))
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    consistency_samples: int = 3
    consistency_samples_max: int = 5  # adaptive extension for borderline agreement
    adaptive_band: tuple = (0.3, 0.9)  # extend sampling when score falls inside
    local_max_tokens: int = 1024
    # Per-category local generation caps: on a 2-vCPU grading box the local
    # token budget is ~10K per 10 min, so every category pays only what its
    # answer shape needs.
    local_max_tokens_by_category: dict = field(default_factory=lambda: {
        # 96 truncated the official T01 answer mid-sentence: multi-part
        # factual questions ("what are X, and why Y?") need room to land.
        Category.FACTUAL: 200,
        Category.MATH: 300,
        Category.SENTIMENT: 48,
        Category.SUMMARIZATION: 256,
        Category.NER: 96,
        Category.CODE_DEBUG: 700,
        Category.LOGICAL: 400,
        Category.CODE_GEN: 700,
    })
    remote_max_tokens: int = 768
    # Code budget vs the judge's 30s/request cap: 1400 tokens needs ~50 tok/s
    # sustained through the proxy — rehearse before changing either number.
    remote_max_tokens_code: int = 1400
    remote_timeout_seconds: float = 28.0
    remote_attempts: int = 2  # retry quick 429/5xx proxy failures, not slow timeouts
    remote_retry_delay_seconds: float = 0.75
    remote_workers: int = 3  # separate cap so the judge proxy is not stampeded
    time_budget_seconds: float = 540.0  # scoring budget ~10 min; leave boot margin
    workers: int = 6  # concurrent tasks; local server batches these efficiently
    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"


def config_from_env() -> Config:
    allowed_clean, allowed_raw = parse_allowed_models(
        os.environ.get("ALLOWED_MODELS", ""))
    allowed_models = allowed_clean or DEFAULT_ALLOWED_MODELS
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
        fireworks_base_url=(sanitize_base_url(os.environ.get("FIREWORKS_BASE_URL", ""))
                            or Config.fireworks_base_url),
        fireworks_api_key=_sanitize_env(os.environ.get("FIREWORKS_API_KEY", "")),
        allowed_models=allowed_models,
        allowed_models_raw=allowed_raw,
        remote_by_category=remote_by_category,
        thresholds=thresholds,
        consistency_samples=int(os.environ.get("CONSISTENCY_SAMPLES", "3")),
        consistency_samples_max=int(os.environ.get("CONSISTENCY_SAMPLES_MAX", "5")),
        remote_max_tokens=int(os.environ.get("REMOTE_MAX_TOKENS", "512")),
        remote_max_tokens_code=int(os.environ.get("REMOTE_MAX_TOKENS_CODE", "900")),
        remote_timeout_seconds=float(os.environ.get("REMOTE_TIMEOUT_SECONDS", "24")),
        remote_attempts=int(os.environ.get("REMOTE_ATTEMPTS", "2")),
        remote_retry_delay_seconds=float(os.environ.get("REMOTE_RETRY_DELAY_SECONDS", "0.75")),
        remote_workers=int(os.environ.get("REMOTE_WORKERS", "3")),
        time_budget_seconds=float(os.environ.get("TIME_BUDGET_SECONDS", "540")),
        workers=int(os.environ.get("WORKERS", "6")),
        input_path=os.environ.get("INPUT_PATH", Config.input_path),
        output_path=os.environ.get("OUTPUT_PATH", Config.output_path),
    )
