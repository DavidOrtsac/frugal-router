import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.config import parse_allowed_models, sanitize_base_url
from frugal_router.probe import (base_url_variants, key_candidates,
                                 model_id_forms, ordered_model_candidates)


def test_parse_allowed_models_comma():
    clean, raw = parse_allowed_models("minimax-m3,kimi-k2p7-code")
    assert clean == ("minimax-m3", "kimi-k2p7-code")


def test_parse_allowed_models_comma_space():
    clean, _ = parse_allowed_models("minimax-m3, kimi-k2p7-code, gemma-4-31b-it")
    assert clean == ("minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it")


def test_parse_allowed_models_json_array():
    clean, _ = parse_allowed_models('["minimax-m3", "kimi-k2p7-code"]')
    assert clean == ("minimax-m3", "kimi-k2p7-code")


def test_parse_allowed_models_single_quoted_json_array():
    clean, _ = parse_allowed_models("['minimax-m3', 'kimi-k2p7-code']")
    assert clean == ("minimax-m3", "kimi-k2p7-code")


def test_parse_allowed_models_space_separated():
    clean, _ = parse_allowed_models("minimax-m3 kimi-k2p7-code")
    assert clean == ("minimax-m3", "kimi-k2p7-code")


def test_parse_allowed_models_semicolon_and_newline():
    clean, _ = parse_allowed_models("minimax-m3;kimi-k2p7-code\ngemma-4-31b-it")
    assert clean == ("minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it")


def test_parse_allowed_models_quoted_tokens():
    clean, _ = parse_allowed_models('"minimax-m3", "kimi-k2p7-code"')
    assert clean == ("minimax-m3", "kimi-k2p7-code")


def test_parse_allowed_models_full_paths():
    clean, _ = parse_allowed_models(
        "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code")
    assert clean == ("accounts/fireworks/models/minimax-m3",
                     "accounts/fireworks/models/kimi-k2p7-code")


def test_parse_allowed_models_empty():
    assert parse_allowed_models("") == ((), ())
    assert parse_allowed_models("   ") == ((), ())


def test_parse_allowed_models_json_quoted_single_string():
    clean, _ = parse_allowed_models('"kimi-k2p7-code"')
    assert clean == ("kimi-k2p7-code",)


def test_sanitize_base_url_strips_quotes_slash_and_completions_suffix():
    assert sanitize_base_url('"https://proxy.judge/v1/" ') == "https://proxy.judge/v1"
    assert sanitize_base_url("https://proxy.judge/v1/chat/completions") == \
        "https://proxy.judge/v1"
    assert sanitize_base_url("proxy.judge/v1") == "https://proxy.judge/v1"
    assert sanitize_base_url("") == ""


def test_base_url_variants_orders_and_dedupes():
    variants = base_url_variants("https://proxy.judge")
    assert variants == ["https://proxy.judge",
                        "https://proxy.judge/v1",
                        "https://proxy.judge/inference/v1"]
    # already ends in /v1 -> no double append, but /inference/v1 still tried
    variants = base_url_variants("https://proxy.judge/v1")
    assert variants[0] == "https://proxy.judge/v1"
    assert "https://proxy.judge/v1/v1" not in variants


def test_model_id_forms_covers_both_namespaces():
    forms = model_id_forms("accounts/fireworks/models/kimi-k2p7-code")
    assert forms == ["accounts/fireworks/models/kimi-k2p7-code",
                     "kimi-k2p7-code"]
    forms_short = model_id_forms("kimi-k2p7-code")
    assert forms_short == ["kimi-k2p7-code",
                           "accounts/fireworks/models/kimi-k2p7-code"]


def test_ordered_model_candidates_prefers_kimi_then_gemma():
    ordered = ordered_model_candidates(
        ("minimax-m3", "gemma-4-31b-it", "kimi-k2p7-code"), ())
    assert ordered[0] == "kimi-k2p7-code"
    assert ordered[1] == "gemma-4-31b-it"
    assert ordered[2] == "minimax-m3"


def test_key_candidates_sanitizes_and_defaults():
    assert key_candidates('  "fw_abc123" ') == ["fw_abc123"]
    keys = key_candidates("")
    assert keys[-1] == "EMPTY" or len(keys) >= 1
