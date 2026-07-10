# FrugalRouter — Track 1 submission container.
#
# Contract (Participant Guide): linux/amd64, compressed size < 10GB, ready in
# 60s, /input/tasks.json -> /output/results.json, exit 0 on success. Harness
# injects FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS at runtime.
#
# Local model is baked into the image (no network needed at start). Serving is
# llama.cpp with two builds: Vulkan (uses a GPU when present) and CPU
# (guaranteed fallback) — the entrypoint picks at runtime.

FROM ubuntu:24.04

# Release stamp: forces a fresh image digest so the judge's puller never
# dedupes a resubmission against a previously-scored digest.
ARG RELEASE_STAMP=rung1d-r2
LABEL org.transcendiant.release="${RELEASE_STAMP}"

ARG LLAMA_TAG=b9910
# Local model chosen by bake-off under the 4GB/2vCPU grading constraints:
# Qwen3-4B-Instruct-2507 Q4_K_M — 91.4% local floor on the train set, and
# VM-rehearsed at 100% strict (40/40) with code+logic escalation in 2m09s.
ARG MODEL_URL=https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
ARG HF_TOKEN=""

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    python3 python3-pip curl ca-certificates libgomp1 \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp: CPU build + Vulkan build
RUN mkdir -p /opt/llama-cpu /opt/llama-vulkan \
    && curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" \
       | tar xz -C /opt/llama-cpu --strip-components=1 \
    && curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz" \
       | tar xz -C /opt/llama-vulkan --strip-components=1

# Local model weights, baked in at build time
RUN mkdir -p /models && \
    curl -sL ${HF_TOKEN:+-H "Authorization: Bearer ${HF_TOKEN}"} \
      -o /models/local.gguf "${MODEL_URL}" && \
    test "$(stat -c%s /models/local.gguf)" -gt 500000000

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages openai

COPY frugal_router/ frugal_router/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Routing configuration (tuned values baked at build; env overrides win).
# Rung-1 podium profile: Qwen3-4B answers factual/math/sentiment/summary/NER
# locally at zero cost (its measured-perfect categories); the three
# code/logic categories are forced remote to kimi. VM-rehearsed under
# --cpus=2 --memory=4g before every change to this block.
ENV LOCAL_MODEL=qwen3-4b \
    LOCAL_BASE_URL=http://localhost:8901/v1 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    TIME_BUDGET_SECONDS=540 \
    WORKERS=2 \
    REMOTE_WORKERS=2 \
    REMOTE_TIMEOUT_SECONDS=28 \
    REMOTE_ATTEMPTS=2 \
    REMOTE_MAX_TOKENS=768 \
    REMOTE_MAX_TOKENS_CODE=1400 \
    LLAMA_THREADS=2 \
    LLAMA_CTX=6144 \
    LLAMA_SLOTS=1 \
    CONSISTENCY_SAMPLES=1 \
    CONSISTENCY_SAMPLES_MAX=1 \
    THRESHOLDS_JSON='{"code_debugging": 1.01, "code_generation": 1.01, "factual_knowledge": 0.0, "logical_reasoning": 1.01, "math_reasoning": 0.5, "ner": 0.0, "sentiment_classification": 0.0, "text_summarization": 0.0}' \
    REMOTE_MAP_JSON='{"code_debugging": "kimi-k2p7-code", "code_generation": "kimi-k2p7-code", "factual_knowledge": "kimi-k2p7-code", "logical_reasoning": "kimi-k2p7-code", "math_reasoning": "kimi-k2p7-code", "ner": "kimi-k2p7-code", "sentiment_classification": "kimi-k2p7-code", "text_summarization": "kimi-k2p7-code"}'

ENTRYPOINT ["./entrypoint.sh"]
