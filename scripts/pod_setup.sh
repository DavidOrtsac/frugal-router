#!/usr/bin/env bash
# Optional vLLM tuning setup for an AMD GPU pod / JupyterLab terminal.
# The final submission image uses the Dockerfile's baked llama.cpp runtime.
# Usage:  bash scripts/pod_setup.sh
# Env:    LOCAL_MODEL (default Qwen/Qwen3-1.7B)
#         HF_TOKEN    (required only for gated alternate weights)
set -euo pipefail

LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen3-1.7B}"

echo "=== GPU check ==="
rocm-smi --showmeminfo vram 2>/dev/null || rocm-smi 2>/dev/null || \
  echo "WARNING: rocm-smi not found - is this an AMD GPU machine?"

echo "=== Python deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM not found - installing (ROCm build expected on AMD images)..."
  python3 -m pip install --quiet vllm
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "NOTE: HF_TOKEN is not set. Public Qwen weights do not need it;"
  echo "      gated alternate models may require: export HF_TOKEN=hf_..."
fi

echo "=== Starting vLLM: ${LOCAL_MODEL} ==="
nohup vllm serve "${LOCAL_MODEL}" \
  --port 8000 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  > vllm.log 2>&1 &
echo "vLLM PID: $! (log: vllm.log - first run downloads weights, can take a while)"

echo "=== Waiting for vLLM health (checks every 15s, up to 45 min) ==="
for i in $(seq 1 180); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "vLLM is READY."
    break
  fi
  sleep 15
  if (( i % 8 == 0 )); then
    echo "still waiting... ($((i * 15 / 60)) min) - tail of vllm.log:"
    tail -n 2 vllm.log || true
  fi
done

echo "=== Smoke: raw local accuracy on the dev set (no remote calls) ==="
python3 eval/run_eval.py --local-only || true

echo ""
echo "Done. Next steps:"
echo "  export FIREWORKS_API_KEY=fw-...   # then:"
echo "  python3 eval/run_eval.py          # full routed eval"
echo "  python3 eval/sweep.py             # threshold frontier"
