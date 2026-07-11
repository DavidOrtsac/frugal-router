"""Container entrypoint: python -m frugal_router.main"""

import json
import sys
from dataclasses import replace

from .clients import OpenAICompatClient
from .config import config_from_env
from .pipeline import (ResultCheckpointer, load_tasks, report, run_batch,
                       write_placeholder, write_results)
from .probe import bootstrap_remote


def _build_remote(config):
    """Probe the judge environment and pin a verified remote channel.

    When NOTHING answers, return a sanitized BEST-GUESS client instead of
    surrendering: per-call failures fall back to local answers and the
    circuit breaker's half-open cycle rediscovers the proxy if it recovers
    mid-run — strictly better than local-flooring the whole batch because
    of a bad minute at boot. Only a missing base URL yields remote=None."""
    runtime = bootstrap_remote(config)
    import httpx
    if not runtime.ok:
        from .probe import base_url_variants
        bases = base_url_variants(config.fireworks_base_url)
        if not bases or not config.allowed_models:
            return config, None
        from .policy import resolve_remote_model
        best_guess_model = resolve_remote_model(
            config, next(iter(config.remote_by_category.values()), "kimi"))
        print(f"[frugal-router] probe found no channel — arming BEST-GUESS "
              f"remote base={bases[0]} model={best_guess_model}; breaker "
              f"half-open will retry as the run proceeds", file=sys.stderr)
        remote = OpenAICompatClient(
            bases[0],
            config.fireworks_api_key or "EMPTY",
            timeout=config.remote_timeout_seconds,
            max_retries=0,
            http_client=httpx.Client(
                trust_env=True,
                timeout=httpx.Timeout(config.remote_timeout_seconds,
                                      connect=20.0)),
        )
        return replace(config, fireworks_base_url=bases[0]), remote
    # Generous connect budget: on a CPU-saturated 2-core box the TLS
    # handshake itself gets starved — do not let the default 5s kill it.
    http_client = httpx.Client(
        verify=runtime.verify,
        trust_env=runtime.trust_env,
        timeout=httpx.Timeout(config.remote_timeout_seconds, connect=20.0),
    )
    remote = OpenAICompatClient(
        runtime.base_url,
        runtime.api_key,
        timeout=config.remote_timeout_seconds,
        max_retries=0,
        default_headers=runtime.headers or None,
        http_client=http_client,
    )
    # Pin probe-verified model IDs so policy resolution can only ever emit
    # strings that have already answered through this exact channel.
    resolved_by_category = {
        category: runtime.model_map.get(model.rsplit("/", 1)[-1], model)
        for category, model in config.remote_by_category.items()
    }
    verified_allowed = tuple(dict.fromkeys(runtime.model_map.values()))
    config = replace(
        config,
        fireworks_base_url=runtime.base_url,
        fireworks_api_key=runtime.api_key,
        remote_by_category=resolved_by_category,
        allowed_models=verified_allowed or config.allowed_models,
    )
    return config, remote


def main() -> int:
    config = config_from_env()
    tasks = load_tasks(config.input_path)
    print(f"[frugal-router] loaded {len(tasks)} tasks from {config.input_path}", file=sys.stderr)

    # A schema-valid file with every task_id exists from second zero: if the
    # judge kills the container at its runtime limit, a partial score beats
    # OUTPUT_MISSING (which scores nothing at all).
    write_placeholder(config.output_path, tasks)
    checkpointer = ResultCheckpointer(config.output_path, tasks)

    import httpx
    local = OpenAICompatClient(
        config.local_base_url,
        extra_body=config.local_extra_body,
        timeout=240.0,
        max_retries=2,  # localhost retries are nearly free; absorb 4GB-box hiccups
        # trust_env=False: judge-set HTTP(S)_PROXY vars must never hijack the
        # localhost hop. Long keepalive + capped pool: reuse warm IPv4
        # connections instead of dialing fresh ones mid-run (fresh dials to
        # "localhost" resolved ::1 first and died with EADDRNOTAVAIL inside
        # IPv6-less Docker — the root cause of three days of task deaths).
        http_client=httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(240.0, connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=8,
                                max_connections=8,
                                keepalive_expiry=600.0),
        ),
    )
    config, remote = _build_remote(config)

    results = run_batch(config, local, remote, tasks, checkpointer=checkpointer)
    write_results(config.output_path, results)

    summary = report(results)
    print(f"[frugal-router] done: {json.dumps(summary)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
