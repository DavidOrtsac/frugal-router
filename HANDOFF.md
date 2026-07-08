# HANDOFF: How To Finish This Alone

Written Jul 8, 2026 for David, in case Claude is gone. Everything here is
copy-paste. Deadline: submit on lablab.ai before Fri Jul 11, 11:00 PM PHT
(aim for 6:00 PM).

## Where things stand

- The router works. Proven numbers on the 187-question practice exam:
  local-only accuracy 93.6% (Qwen3-14B); first routed run 92.5% accuracy,
  96.8% answered free, 3,749 remote tokens.
- Code is on GitHub: github.com/DavidOrtsac/frugal-router (PRIVATE — make it
  public before submitting: repo Settings > General > Danger Zone > Change
  visibility).
- Local model decision: Qwen/Qwen3-14B. Gemma cannot run locally on the free
  pod (its software predates Gemma 4). Gemma's role is REMOTE, once the
  organizers answer the Discord question about the correct model path.

## Step 0: Restart the GPU machine whenever it expires

The free notebook dies after some hours. To revive:
1. Go to the AMD/AnruiCloud page where you claimed it, claim/start again.
2. Open the JupyterLab link. Click the dark "Terminal" tile.
3. Upload the newest `fr.zip` (drag into the file list). Get it from your
   Mac: in Terminal, `cd ~/Desktop/Projects/frugal-router && git archive
   --format=zip --prefix=frugal-router/ -o ~/Desktop/fr.zip HEAD` — or
   download the repo as zip from GitHub (rename the inner folder to
   `frugal-router`).
4. Paste into the Jupyter terminal (replace YOUR keys):

```bash
cd /workspace
python3 -c 'import zipfile; zipfile.ZipFile("/workspace/fr.zip").extractall(".")' 2>/dev/null || \
  python3 -c 'import zipfile,glob; zipfile.ZipFile(glob.glob("fr*.zip")[0]).extractall(".")'
printf 'export HF_TOKEN=%s\nexport FIREWORKS_API_KEY=%s\n' \
  'hf_YOUR_HUGGINGFACE_TOKEN' 'fw_YOUR_FIREWORKS_KEY' > /workspace/secrets.env
chmod 600 /workspace/secrets.env
/opt/venv/bin/pip install -q openai pytest
nohup /opt/venv/bin/vllm serve Qwen/Qwen3-14B --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 > /workspace/vllm.log 2>&1 &
echo "Model starting. Takes 5-30 min first time. Check with:"
echo "  curl -s http://localhost:8000/health && echo HEALTHY"
```

## Step 1: Run the tuning battery (once per code change, ~2 hours)

After the server says HEALTHY, paste:

```bash
cd /workspace/frugal-router
chmod +x /workspace/day2.sh 2>/dev/null
cp ../day2.sh /workspace/day2.sh 2>/dev/null
nohup bash /workspace/day2.sh > /workspace/day2_runner.log 2>&1 &
echo "Battery running. Watch progress with:"
echo "  cat /workspace/day2_status.txt"
```

(If day2.sh is missing, it lives in this repo's history / Claude's messages;
its three runs can also be done by hand — see Step 1b.)

Step 1b, manual equivalent of the battery:

```bash
cd /workspace/frugal-router
export $(grep -o 'FIREWORKS_API_KEY=.*' /workspace/secrets.env)
export LOCAL_MODEL=Qwen/Qwen3-14B
export LOCAL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
/opt/venv/bin/python eval/run_eval.py --local-only  --tasks eval/tasks/train_tasks.json --dump /workspace/dump_local.json   > /workspace/eval_L.json
REMOTE_DEFAULT_MODEL=kimi-k2p7-code /opt/venv/bin/python eval/run_eval.py --remote-only --tasks eval/tasks/train_tasks.json --dump /workspace/dump_kimi.json    > /workspace/eval_RK.json
REMOTE_DEFAULT_MODEL=minimax-m3     /opt/venv/bin/python eval/run_eval.py --remote-only --tasks eval/tasks/train_tasks.json --dump /workspace/dump_minimax.json > /workspace/eval_RM.json
/opt/venv/bin/python eval/frontier.py --local /workspace/dump_local.json \
  --remote kimi=/workspace/dump_kimi.json --remote minimax=/workspace/dump_minimax.json
```

The last command prints a table: for each question category, which expert to
use and how paranoid to be (the threshold), plus PROJECTED accuracy/tokens.

## Step 2: Apply the tuning results (copy-paste, no coding)

Take the table's threshold + expert per category and build two lines like:

```bash
export THRESHOLDS_JSON='{"factual_knowledge": 0.8, "math_reasoning": 0.6, "logical_reasoning": 0.6, "ner": 0.4, "code_debugging": 0.4, "code_generation": 0.6, "sentiment_classification": 0.0, "text_summarization": 0.0}'
export REMOTE_MAP_JSON='{"factual_knowledge": "minimax-m3", "code_generation": "kimi-k2p7-code"}'
```

(Numbers above are examples — use the frontier table's numbers.)

## Step 3: Final validation on the held-out questions (run ONCE)

```bash
cd /workspace/frugal-router
export $(grep -o 'FIREWORKS_API_KEY=.*' /workspace/secrets.env)
export LOCAL_MODEL=Qwen/Qwen3-14B
export LOCAL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
# plus your THRESHOLDS_JSON and REMOTE_MAP_JSON from Step 2
/opt/venv/bin/python eval/run_eval.py --tasks eval/tasks/heldout_tasks.json --dump /workspace/dump_final.json
```

If accuracy lands within ~2 points of the frontier's projection: tuning is
real, freeze everything. If it drops much more: use ONE global threshold of
0.6 for all non-freeform categories and move on. Do not endlessly fiddle.

## Step 4: When the organizers answer about Gemma

If they give a working Gemma path (say accounts/fireworks/models/gemma-4-31b-it):
1. Rerun ONE remote-only recording for it (copy the RK line from Step 1b,
   change REMOTE_DEFAULT_MODEL to the Gemma name, dump to dump_gemma.json).
2. Rerun frontier.py adding: `--remote gemma-4-31b-it=/workspace/dump_gemma.json`
3. If Gemma wins categories, update REMOTE_MAP_JSON accordingly. The Gemma
   bonus likes seeing Gemma in the map — prefer it on ties.
4. If the scoring env's ALLOWED_MODELS uses different full paths, set
   REMOTE_MODEL_PREFIX accordingly (default "accounts/fireworks/models/").

## Step 5: Package and submit (Friday)

1. Bake the tuned values into the Dockerfile: open Dockerfile, in the ENV
   block add lines (backslash-continued) for THRESHOLDS_JSON and
   REMOTE_MAP_JSON with your tuned JSON, and FIREWORKS_API_KEY stays OUT
   (passed at runtime with -e).
2. Sanity: `docker build -t frugal-router .` must succeed from a fresh clone.
3. Make the GitHub repo PUBLIC.
4. lablab.ai > your team > Submit. Form content is prewritten in
   docs/SUBMISSION.md (title, descriptions, tags). Upload cover image,
   video, slides.
5. Screenshot the confirmation. Done. Sleep well, champion.

## If something breaks

| Symptom | Fix |
|---|---|
| curl localhost:8000 refuses | model still loading; `tail /workspace/vllm.log` |
| vLLM crashes on start | `/opt/venv/bin/pip install 'transformers==4.57.6'` then restart (5.x breaks it) |
| NOT_FOUND from Fireworks | model name/path wrong; only kimi-k2p7-code and minimax-m3 confirmed working on personal keys |
| eval crashes mid-run | rerun it; results overwrite cleanly |
| pod gone again | Step 0 |
| grader seems wrong on a task | delete that task from the tasks JSON; never tune against a grader you distrust |

## The one rule

If in doubt, prefer HIGHER accuracy over fewer tokens. The accuracy bar is
secret; being cheap below the bar scores zero.

## Endgame protocol (added Jul 8, after submission)

Submitted. Target to beat: Pahfinder0, 5,121 tokens. Rules of engagement:

1. After each hourly scoring run, check the leaderboard.
2. If TranscendiantRouter QUALIFIED with a token count: we are on the board.
   To lower the count, edit THRESHOLDS_JSON in the Dockerfile toward a
   cheaper ladder rung (see eval/ladder.py output), commit, push, run the
   GitHub Action (Actions tab > build-submission-image > Run workflow).
   The judge re-pulls :latest automatically next cycle.
3. If ACCURACY_GATE_FAILED: move THRESHOLDS_JSON one rung SAFER (more
   categories at 1.01, lower vote thresholds), rebuild, wait a cycle.
4. NEVER leave a failing config as the last push before the deadline.
   When a config passes, keep it until a cheaper one also passes.
5. Stop all changes 3 hours before the Jul 11, 11 PM PHT deadline.
   Last known-passing configuration wins.

Safer rung (if gate bites): all eight categories at 1.01 except
summarization 0.0 and sentiment 1.0. Expensive (~35-40K tokens) but maximum
accuracy short of full remote.

Cheaper rungs (if we pass with margin): raise code_generation from 1.01 to
0.6, then factual_knowledge from 1.01 to 0.7. Each step saves thousands of
tokens; validate one step per scoring cycle.
