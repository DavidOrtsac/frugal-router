# TranscendiantRouter

Token-efficient routing agent for the **AMD Developer Hackathon: ACT II, Track 1**.
Team **Transcendiant**. Built solo by David Alfonso M. Castro.

**Frozen submission artifact (deadline Jul 13, 2026):**
`ghcr.io/davidortsac/frugal-router:a4` —
OCI index digest `sha256:6927f90beab96ac8e50f2ecd03c4a0d6ef347075fe066e813b0bd40415eb0bf6`.

TranscendiantRouter answers most tasks with a free local model that must
PROVE each answer before it ships, escalates only unproven answers to a
remote expert, and adapts its own spending to the speed of the machine it
wakes up on.

## Why this design

Track 1 scoring has two stages: an accuracy gate, then ranking by fewest
Fireworks tokens among the submissions that pass. Local tokens count as zero.
So the router routes by measured weakness: a strong local model answers for
free wherever it can demonstrate correctness, and the remote expert is paid
for exactly the categories and answers where it cannot.

## How it works

```
/input/tasks.json
      |
1. classify     rule-based, zero tokens: 8 task categories
2. answer       Qwen3-4B-Instruct-2507 Q4_K_M (llama.cpp, weights baked in)
                answers locally on 2 CPU cores
3. prove        math/logic need an explicit final answer; code must parse
                (compile-check, never executed); other categories vote by
                self-consistency
4. escalate     logical reasoning and any unproven answer go through the
                Fireworks proxy: kimi-k2p7-code, resolved from ALLOWED_MODELS
      |
/output/results.json   (placeholder at t=0; atomically checkpointed
                        after every task — a hard kill still leaves a
                        valid, scoreable file)
```

## Engineered for the grading environment

- **Image**: ~3.5GB compressed (limit 10GB), model weights baked in, no
  downloads at start, boots in seconds (limit 60s).
- **Self-healing probe**: at boot, sweeps base-URL variants x model-ID forms
  x transports x auth shapes and pins the first combination that answers.
  Nothing about the judge proxy is assumed; off-list calls are impossible
  by construction.
- **Time-fit preemption**: a local generation that cannot finish inside the
  remaining time budget (450s internal, vs the 600s limit) never starts —
  it escalates instead. Timeouts are engineered out, not hoped away.
- **Adaptive cost**: measured on a clone of the grading box across a 2x
  speed envelope (19-task judge-scale rehearsals): 19/19 accuracy in every
  cell, walls 231-380s, tokens 331 (full speed) to ~3,034 (heavily degraded).
- **OOM-tuned local server**: llama.cpp runs with `--cache-ram 0
  --ctx-checkpoints 0 --no-cache-prompt -np 1` — the b9910 defaults
  (8 GiB host prompt cache) OOM-kill the server inside a 4GB container.
- **Robustness**: connection-class errors retry (3 attempts), a half-open
  circuit breaker protects the clock, and a failed remote call falls back
  to a voted local answer; a task can never return blank.
- **Harness contract**: reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
  `ALLOWED_MODELS` from the environment, exactly as injected at evaluation.
  On the organizers' published validation tasks (T01-T05): 5/5 at zero
  Fireworks tokens.


## Measured results

On a 227-task benchmark harness (GSM8K, HumanEval, and authored tasks across
all 8 categories) with a held-out split never used for tuning:

- Local-only floor: **77.0%** at 0 tokens.
- Safe shipped rung: forced-remote for every category except summarization,
  projected around **89%** while finishing inside the 4GB/2vCPU budget.
- Cheaper ladder rungs are encoded in `eval/ladder.py`; probe them only after
  the current image is verified as a pass because the leaderboard keeps the
  latest score, not the best score.

Thresholds come from recorded runs replayed offline (`eval/frontier.py`,
`eval/ladder.py`): record the local model once and each expert once, then
evaluate every threshold combination instantly.

## Run it

```bash
docker pull ghcr.io/davidortsac/frugal-router:latest
mkdir -p io/input io/output
cp eval/tasks/practice_tasks.json io/input/tasks.json   # or your own
docker run --rm --cpus=2 --memory=4g \
  -v $(pwd)/io/input:/input -v $(pwd)/io/output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=kimi-k2p7-code,gemma-4-31b-it \
  ghcr.io/davidortsac/frugal-router:latest
cat io/output/results.json
```

Build from source: `docker buildx build --platform linux/amd64 -t transcendiantrouter .`

## Develop and evaluate

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # 97 tests
python eval/build_devset.py              # assemble the 227-task benchmark
python eval/run_eval.py --mock           # offline smoke run, no GPU or key
python eval/run_eval.py --local-only --tasks eval/tasks/train_tasks.json --dump dump_local.json
REMOTE_DEFAULT_MODEL=kimi-k2p7-code python eval/run_eval.py --remote-only --tasks eval/tasks/train_tasks.json --dump dump_kimi.json
python eval/ladder.py --local dump_local.json --remote kimi=dump_kimi.json
```

`LOCAL_BASE_URL` points at any OpenAI-compatible server (llama.cpp or vLLM).
All routing levers are environment variables; see `.env.example`.

## Remote Experts

The qualification image pins escalations to `kimi-k2p7-code`, the measured
expert available through the allowed list. Gemma remains an allowed-list
candidate and fallback if the evaluator publishes it, but local Gemma was
ruled out honestly: the smallest useful checkpoint cannot fit the 4GB grading
environment alongside the agent.

## License

MIT. Note: the repository slug and image path retain the project's original
working name (`frugal-router`); the project is TranscendiantRouter.
