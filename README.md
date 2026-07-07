# FrugalRouter

Hybrid token-efficient routing agent for the **AMD Developer Hackathon: ACT II, Track 1**.
Team: **Transcendiant** (solo, David Alfonso M. Castro).

FrugalRouter completes each task with the fewest scored tokens possible. Local
tokens are free under Track 1 scoring, so the router keeps everything on a
self-hosted local model (Gemma 4 26B-A4B via vLLM on AMD ROCm) and escalates to
a remote model on Fireworks AI only when a calibration signal says the local
answer is likely wrong.

## How it works

```
/input/tasks.json
      |
      v
1. classify        rule-based, zero tokens: 8 fixed capability categories
2. calibrate       k self-consistency samples on the LOCAL model (free);
                   answer agreement predicts correctness far better than
                   self-reported confidence
3. decide          agreement >= per-category threshold -> ship the local
                   majority answer (0 scored tokens)
                   agreement <  threshold -> escalate to Fireworks
4. escalate        terse single-shot prompt, low max_tokens, model chosen
                   per category (Gemma 4 by default, Kimi K2p7 for code)
      |
      v
/output/results.json
```

The escalation thresholds are tuned with the included eval harness against a
labeled dev set covering all 8 categories.

## Gemma 4, twice

- **Local:** `google/gemma-4-26B-A4B-it` served by vLLM on the AMD GPU pod
  (MoE, ~4B active params — fast and fits comfortably in 48GB).
- **Remote:** `gemma-4-31b-it` via Fireworks AI as the default escalation model.

## Setup

```bash
git clone https://github.com/DavidOrtsac/frugal-router
cd frugal-router
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your FIREWORKS_API_KEY
```

## Run (Docker, as scored)

```bash
docker build -t frugal-router .
docker run --rm \
  -v /path/to/input:/input -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  --device=/dev/kfd --device=/dev/dri \
  frugal-router
```

The container starts vLLM with the local model, waits for health, reads
`/input/tasks.json` (`[{task_id, prompt}]`), and writes `/output/results.json`
(`[{task_id, answer}]`).

For development against an external vLLM server:

```bash
docker build --build-arg BASE_IMAGE=python:3.12-slim -t frugal-router:dev .
docker run --rm -v ...:/input -v ...:/output \
  -e START_LOCAL_VLLM=0 -e LOCAL_BASE_URL=http://host.docker.internal:8000/v1 \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY frugal-router:dev
```

## Eval harness

```bash
python eval/run_eval.py --mock          # offline smoke run, no GPU or API key
python eval/run_eval.py                 # real endpoints from .env
python eval/run_eval.py --local-only    # raw local accuracy (routing floor)
python eval/run_eval.py --remote-only   # remote accuracy + token cost ceiling
python eval/sweep.py                    # threshold sweep -> frontier table
```

Reported metrics: `accuracy`, `offload_rate` (fraction answered locally, i.e.
free), and `remote_tokens` (the scored quantity).

## Tests

```bash
pytest
```

## Configuration

All routing levers are environment variables — see `.env.example`. The allowed
remote model list is enforced in code: the router can never call a model
outside `ALLOWED_MODELS`.

## License

MIT — see [LICENSE](LICENSE).
