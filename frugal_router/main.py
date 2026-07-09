"""Container entrypoint: python -m frugal_router.main"""

import json
import sys

from .clients import OpenAICompatClient
from .config import config_from_env
from .pipeline import load_tasks, report, run_batch, write_results


def main() -> int:
    config = config_from_env()
    tasks = load_tasks(config.input_path)
    print(f"[frugal-router] loaded {len(tasks)} tasks from {config.input_path}", file=sys.stderr)

    local = OpenAICompatClient(config.local_base_url, extra_body=config.local_extra_body)
    remote = OpenAICompatClient(
        config.fireworks_base_url,
        config.fireworks_api_key,
        timeout=config.remote_timeout_seconds,
        max_retries=0,
    )

    results = run_batch(config, local, remote, tasks)
    write_results(config.output_path, results)

    summary = report(results)
    print(f"[frugal-router] done: {json.dumps(summary)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
