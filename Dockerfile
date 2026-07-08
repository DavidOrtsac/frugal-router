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

ARG LLAMA_TAG=b9910
# Ungated default (Qwen3-4B). For the Gemma build pass MODEL_URL and HF_TOKEN.
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
    test "$(stat -c%s /models/local.gguf)" -gt 1000000000

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages openai

COPY frugal_router/ frugal_router/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Routing configuration (tuned values baked at build; env overrides win)
ENV LOCAL_MODEL=local-gguf \
    LOCAL_BASE_URL=http://localhost:8901/v1 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    TIME_BUDGET_SECONDS=540 \
    WORKERS=8 \
    CONSISTENCY_SAMPLES=5 \
    CONSISTENCY_SAMPLES_MAX=10

ENTRYPOINT ["./entrypoint.sh"]
