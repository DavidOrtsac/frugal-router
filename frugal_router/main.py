"""Container entrypoint: python -m frugal_router.main"""

import json
import sys
from dataclasses import replace

from .clients import OpenAICompatClient
from .config import config_from_env
from .pipeline import load_tasks, report, run_batch, write_results
from .probe import bootstrap_remote


def _build_remote(config):
    """Probe the judge environment and pin a verified remote channel.

    Returns (possibly rewritten) config plus a remote client, or None when no
    combination answers — the pipeline then runs local-only, which is exactly
    the pre-probe behavior, never worse."""
    runtime = bootstrap_remote(config)
    if not runtime.ok:
        return config, None

    import httpx
    http_client = httpx.Client(
        verify=runtime.verify,
        trust_env=runtime.trust_env,
        timeout=config.remote_timeout_seconds,
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

    import httpx
    local = OpenAICompatClient(
        config.local_base_url,
        extra_body=config.local_extra_body,
        timeout=240.0,
        max_retries=2,  # localhost retries are nearly free; absorb 4GB-box hiccups
        # trust_env=False: judge-set HTTP(S)_PROXY vars must never hijack
        # the localhost hop to the local model server.
        http_client=httpx.Client(trust_env=False, timeout=240.0),
    )
    config, remote = _build_remote(config)

    results = run_batch(config, local, remote, tasks)
    write_results(config.output_path, results)

    summary = report(results)
    print(f"[frugal-router] done: {json.dumps(summary)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
