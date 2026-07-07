# FrugalRouter — Track 1 submission container.
#
# BASE_IMAGE options:
#   rocm/vllm:latest      -> self-contained: serves the local model in-container
#                            on the AMD GPU pod (default for scoring)
#   python:3.12-slim      -> lightweight: expects an external vLLM server at
#                            LOCAL_BASE_URL (useful for development)
ARG BASE_IMAGE=rocm/vllm:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frugal_router/ frugal_router/
COPY eval/ eval/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# START_LOCAL_VLLM=1 launches vllm serve inside the container before routing.
# LOCAL_MODEL chosen by bake-off: Qwen3-14B (93.6% local accuracy on dev set).
ENV START_LOCAL_VLLM=1 \
    LOCAL_MODEL=Qwen/Qwen3-14B \
    LOCAL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
    LOCAL_BASE_URL=http://localhost:8000/v1 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json

ENTRYPOINT ["./entrypoint.sh"]
