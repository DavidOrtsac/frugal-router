"""Hostile judge-proxy integration tests.

Each scenario boots a real HTTP server that mimics one diagnosed way the
judge's metering proxy could be shaped, injects the corresponding env vars,
and asserts the probe + client stack ends up making a SUCCESSFUL remote
completion. These are the exact failure modes that produced seven identical
63.2% scores; none of them may ever ship unhandled again.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frugal_router.config import config_from_env
from frugal_router.main import _build_remote
from frugal_router.pipeline import _escalate
from frugal_router.schemas import Category, Task

pytest.importorskip("openai")


class MockJudgeProxy:
    """OpenAI-compatible endpoint with configurable strictness."""

    def __init__(self, serve_path="/v1/chat/completions",
                 accepted_models=("kimi-k2p7-code",),
                 auth_header="authorization",
                 expected_key="judge-key"):
        self.serve_path = serve_path
        self.accepted_models = set(accepted_models)
        self.auth_header = auth_header
        self.expected_key = expected_key
        self.requests_seen = []

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                proxy.requests_seen.append(
                    {"path": self.path, "model": body.get("model")})
                if self.path != proxy.serve_path:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "no such route"}')
                    return
                if proxy.auth_header == "authorization":
                    supplied = self.headers.get("Authorization", "")
                    ok_auth = supplied == f"Bearer {proxy.expected_key}"
                else:
                    ok_auth = (self.headers.get(proxy.auth_header, "")
                               == proxy.expected_key)
                if not ok_auth:
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'{"error": "bad auth"}')
                    return
                if body.get("model") not in proxy.accepted_models:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "model not found"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "choices": [{"message": {"role": "assistant",
                                             "content": "REMOTE-ANSWER"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def host(self):
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self):
        self.server.shutdown()


@pytest.fixture
def judge_env(monkeypatch):
    def _apply(base_url, allowed_models, api_key="judge-key"):
        monkeypatch.setenv("FIREWORKS_BASE_URL", base_url)
        monkeypatch.setenv("ALLOWED_MODELS", allowed_models)
        monkeypatch.setenv("FIREWORKS_API_KEY", api_key)
        monkeypatch.delenv("REMOTE_MAP_JSON", raising=False)
        monkeypatch.delenv("THRESHOLDS_JSON", raising=False)
        return config_from_env()
    return _apply


def _assert_remote_works(config, remote):
    assert remote is not None, "probe failed to find the channel"
    result = _escalate(config, remote,
                       Task("t1", "What is the capital of Australia?"),
                       Category.FACTUAL, reason="test")
    assert result.answer == "REMOTE-ANSWER"
    assert result.remote_tokens == 9
    assert result.route.value == "remote"


def test_base_published_without_v1_json_array_short_names(judge_env):
    # Proxy serves /v1/chat/completions but publishes the bare host;
    # ALLOWED_MODELS arrives as a JSON array of short names.
    proxy = MockJudgeProxy(serve_path="/v1/chat/completions",
                           accepted_models=("kimi-k2p7-code",))
    try:
        config = judge_env(proxy.host, '["minimax-m3", "kimi-k2p7-code"]')
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
    finally:
        proxy.shutdown()


def test_short_aliases_published_but_proxy_wants_full_paths(judge_env):
    # The namespace-mismatch scenario: ALLOWED_MODELS says short names, the
    # proxy upstream only accepts canonical full paths.
    proxy = MockJudgeProxy(
        serve_path="/v1/chat/completions",
        accepted_models=("accounts/fireworks/models/kimi-k2p7-code",))
    try:
        config = judge_env(proxy.host + "/v1",
                           "minimax-m3,kimi-k2p7-code,gemma-4-31b-it")
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
    finally:
        proxy.shutdown()


def test_full_paths_published_but_proxy_wants_short_aliases(judge_env):
    # The reverse mismatch, with space-separated serialization on top.
    proxy = MockJudgeProxy(serve_path="/chat/completions",
                           accepted_models=("kimi-k2p7-code",))
    try:
        config = judge_env(
            proxy.host,
            "accounts/fireworks/models/minimax-m3 "
            "accounts/fireworks/models/kimi-k2p7-code")
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
    finally:
        proxy.shutdown()


def test_quote_polluted_key_and_trailing_slash_base(judge_env):
    proxy = MockJudgeProxy(serve_path="/v1/chat/completions",
                           accepted_models=("kimi-k2p7-code",))
    try:
        config = judge_env(f'"{proxy.host}/v1/"', "kimi-k2p7-code",
                           api_key='"judge-key"\n')
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
    finally:
        proxy.shutdown()


def test_proxy_requires_x_api_key_header(judge_env):
    proxy = MockJudgeProxy(serve_path="/v1/chat/completions",
                           accepted_models=("kimi-k2p7-code",),
                           auth_header="x-api-key")
    try:
        config = judge_env(proxy.host + "/v1", "kimi-k2p7-code")
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
    finally:
        proxy.shutdown()


def test_preferred_model_down_probe_demotes_to_working_model(judge_env):
    # kimi is listed but dead on the proxy; gemma answers. The router must
    # discover a channel anyway and route escalations to the live model.
    proxy = MockJudgeProxy(serve_path="/v1/chat/completions",
                           accepted_models=("gemma-4-31b-it",))
    try:
        config = judge_env(proxy.host + "/v1",
                           "kimi-k2p7-code,gemma-4-31b-it")
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
        assert all(m == "gemma-4-31b-it"
                   for m in config.remote_by_category.values())
    finally:
        proxy.shutdown()


def test_everything_dead_returns_none_for_local_only_mode(judge_env):
    proxy = MockJudgeProxy(serve_path="/nowhere",
                           accepted_models=())
    try:
        config = judge_env(proxy.host, "kimi-k2p7-code")
        config, remote = _build_remote(config)
        assert remote is None
    finally:
        proxy.shutdown()


def test_direct_fireworks_shape_still_works(judge_env):
    # Our own rehearsal path: base already ends in /inference/v1 and full
    # model paths are required (the real api.fireworks.ai contract).
    proxy = MockJudgeProxy(
        serve_path="/inference/v1/chat/completions",
        accepted_models=("accounts/fireworks/models/kimi-k2p7-code",))
    try:
        config = judge_env(
            proxy.host + "/inference/v1",
            "accounts/fireworks/models/minimax-m3,"
            "accounts/fireworks/models/kimi-k2p7-code")
        config, remote = _build_remote(config)
        _assert_remote_works(config, remote)
    finally:
        proxy.shutdown()
