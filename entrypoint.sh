#!/usr/bin/env bash
set -euo pipefail

# Optionally start the local vLLM server inside the container, then run the
# routing pipeline against /input/tasks.json -> /output/results.json.

if [[ "${START_LOCAL_VLLM:-0}" == "1" ]]; then
  echo "[entrypoint] starting vLLM for ${LOCAL_MODEL}" >&2
  vllm serve "${LOCAL_MODEL}" \
    --port 8000 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    >/tmp/vllm.log 2>&1 &
  VLLM_PID=$!

  echo "[entrypoint] waiting for vLLM to become healthy" >&2
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
      echo "[entrypoint] vLLM ready after ${i} checks" >&2
      break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[entrypoint] vLLM died during startup; last log lines:" >&2
      tail -n 50 /tmp/vllm.log >&2
      exit 1
    fi
    sleep 5
  done
fi

exec python -m frugal_router.main
