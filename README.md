# TranscendiantRouter

Token-efficient routing agent for the **AMD Developer Hackathon: ACT II, Track 1**.
Team **Transcendiant**. Built solo by David Alfonso M. Castro.

TranscendiantRouter answers tasks with a free local model and pays for a
remote expert only when the local model cannot agree with itself on an answer.

## Why this design

Track 1 scoring has two stages: an accuracy gate, then ranking by fewest
Fireworks tokens among the submissions that pass. Local tokens count as zero.
The cheapest reliable way to catch a local model's mistakes is the local model
itself: sample the same task several times and measure agreement. Agreement
predicts correctness far better than a model's self-reported confidence, and
under these rules the measurement is free.

## How it works

```
/input/tasks.json
      |
1. classify     rule-based, zero tokens: 8 task categories
2. vote         Qwen3-1.7B (llama.cpp, weights baked into the image)
                answers each task 3-5 times; agreement is measured
3. decide       agreement >= per-category tuned threshold -> ship the
                free majority answer (0 scored tokens)
4. escalate     otherwise, that single task goes through the Fireworks
                proxy: Gemma 4 by default, a code specialist for code
      |
/output/results.json
```

## Engineered for the grading environment

- **Image**: ~3GB compressed (limit 10GB), model weights baked in, no
  downloads at start, boots in seconds (limit 60s).
- **Hardware fit**: sampling counts, per-category token caps, and thread
  settings sized for 4GB RAM and 2 vCPUs.
- **Time**: a ratcheting time guard sheds sampling before the 10-minute limit
  is at risk. Measured 5m58s end-to-end on a replica grading box.
- **Robustness**: a failed remote call falls back to a local answer; a task
  can never return blank; model IDs resolve from `ALLOWED_MODELS` at runtime,
  so off-list calls are impossible by construction.
- **Harness contract**: reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
  `ALLOWED_MODELS` from the environment, exactly as injected at evaluation.

## Measured results

On a 227-task benchmark harness (GSM8K, HumanEval, and authored tasks across
all 8 categories) with a held-out split never used for tuning:

- Local-only floor: **77.0%** at 0 tokens.
- Dress rehearsal on a 4GB/2vCPU replica: **92.5%** held-out accuracy,
  5m58s, most answers free.
- Send-everything-remote baseline: ~95% at roughly 8x the token cost.

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
pytest                                   # 30 tests
python eval/build_devset.py              # assemble the 227-task benchmark
python eval/run_eval.py --mock           # offline smoke run, no GPU or key
python eval/run_eval.py --local-only --tasks eval/tasks/train_tasks.json --dump dump_local.json
python eval/ladder.py --local dump_local.json --remote kimi=dump_kimi.json
```

`LOCAL_BASE_URL` points at any OpenAI-compatible server (llama.cpp or vLLM).
All routing levers are environment variables; see `.env.example`.

## Gemma via Fireworks

Gemma 4 is the default escalation model for six of the eight categories,
resolved from the runtime `ALLOWED_MODELS` list. Local Gemma was evaluated
and ruled out honestly: the smallest Gemma 4 checkpoint cannot fit the 4GB
grading environment alongside an agent. Gemma's role is the remote brain.

## License

MIT. Note: the repository slug and image path retain the project's original
working name (`frugal-router`); the project is TranscendiantRouter.
