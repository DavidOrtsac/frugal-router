#!/bin/bash
# Boot the local model server (GPU via Vulkan when available, CPU otherwise),
# wait for health, then run the router. Must be ready well inside 60s.
set -u

if vulkaninfo --summary >/dev/null 2>&1; then
  LLAMA_BIN=$(find /opt/llama-vulkan -name llama-server | head -1)
  echo "[entrypoint] GPU detected: using Vulkan build" >&2
else
  LLAMA_BIN=$(find /opt/llama-cpu -name llama-server | head -1)
  echo "[entrypoint] no GPU: using CPU build" >&2
fi

"$LLAMA_BIN" -m /models/local.gguf --port 8901 \
  -c "${LLAMA_CTX:-8192}" -np "${LLAMA_SLOTS:-8}" --no-webui \
  --jinja --reasoning-budget 0 >/tmp/llama.log 2>&1 &
LLAMA_PID=$!

for i in $(seq 1 50); do
  if curl -sf http://localhost:8901/health >/dev/null 2>&1; then
    echo "[entrypoint] local model ready after ~$i checks" >&2
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "[entrypoint] llama-server died at startup:" >&2
    tail -n 30 /tmp/llama.log >&2
    # No local model: the router still runs — every task escalates via the
    # threshold guard rather than the container failing outright.
    export THRESHOLDS_JSON='{"factual_knowledge":1.01,"math_reasoning":1.01,"sentiment_classification":1.01,"text_summarization":1.01,"ner":1.01,"code_debugging":1.01,"logical_reasoning":1.01,"code_generation":1.01}'
    break
  fi
  sleep 1
done

exec python3 -m frugal_router.main
