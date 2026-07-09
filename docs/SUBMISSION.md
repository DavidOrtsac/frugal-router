# lablab.ai Submission Content (copy-paste into the form)

## Project title
FrugalRouter

## Short description
A hybrid routing agent that uses a free baked local model for calibration, then
pays a Fireworks expert only when the current accuracy-safe rung requires it.

## Long description
FrugalRouter is a Track 1 submission built around one observation: under the
scoring rules, local tokens are free, so the cheapest reliable lie detector
for a local model is the local model itself.

For each task, FrugalRouter:
1. Classifies the task into one of the 8 capability categories with
   zero-cost rules (no model call).
2. Samples the baked local model (Qwen3-1.7B GGUF served by llama.cpp) 3 to 5
   times and measures answer agreement. Calibration costs exactly zero scored
   tokens.
3. Ships the majority answer if agreement clears a per-category threshold
   tuned on a 227-task benchmark harness (GSM8K, HumanEval, and authored
   tasks across all 8 categories, with a held-out validation split).
4. Otherwise escalates that single task to the best-measured remote model on
   Fireworks AI with a terse prompt and a capped token budget.

Safety rails: the router cannot call any model outside the ALLOWED_MODELS
list, a failed task can never sink the batch, Fireworks proxy calls have
explicit timeouts/retries/concurrency caps, and a time-budget guard degrades
sampling instead of ever leaving tasks unanswered.

Measured local floor with the baked Qwen3-1.7B model: 77.0% at zero remote
tokens. The current submission image starts from a qualification-safe rung:
remote for every category except summarization, projected around 89% while
fitting the 4GB/2vCPU grading budget. Cheaper ladder rungs are ready once a
pass is verified.

Built solo in 4 days, developed and tuned with a reproducible eval harness;
the final image uses llama.cpp with CPU fallback and Vulkan acceleration when
available.

## Tags
AMD Developer Cloud, llama.cpp, Vulkan, Fireworks AI, Gemma, Qwen3, AI Agents,
Model Routing, Token Efficiency

## Video script (2-3 min, screen recording + voiceover)

[0:00-0:25] THE PROBLEM
"Track 1 scores two things: accuracy, and how many remote tokens you burn.
Local tokens are free. So the whole game is: how do you know WHEN your free
local model is about to be wrong? Ask it to vote against itself."

[0:25-1:25] DEMO (screen: terminal on the AMD pod)
- Show tasks.json (a few questions visible).
- Run the router; point at the scrolling log: "category, votes, decision -
  local, local, local... and here, the votes scattered, so this one task
  escalates to Fireworks."
- Show results.json appearing, then the summary line: offload rate, tokens.

[1:25-2:10] THE NUMBERS (screen: frontier table / results)
"The baked local Qwen3-1.7B floor is 77% at zero remote tokens. The submitted
safe rung forces remote for most categories to protect the hidden accuracy
gate, then the ladder lets us step down toward fewer Fireworks tokens one
verified scoring cycle at a time."

[2:10-2:40] THE STACK
"Local: Qwen3-1.7B GGUF served by llama.cpp, baked into the image so startup
does not need a model download. Remote: Fireworks AI, with model IDs resolved
from ALLOWED_MODELS exactly as the harness publishes them. Everything is
containerized, reproducible, MIT-licensed, and the eval harness ships in the
repo. Solo build, AMD Developer Hackathon ACT II, team Transcendiant."

## Slide outline (10 slides)
1. FrugalRouter - token-efficient routing agent (team Transcendiant, solo)
2. The rules: accuracy + remote tokens; local = free
3. The insight: free tokens can buy calibration (self-consistency voting)
4. Architecture diagram (classify -> vote -> threshold -> escalate)
5. The harness: 227 benchmark tasks, 8 categories, held-out split
6. The frontier method: record once, replay every threshold offline
7. Results: 77% local floor; safe pass rung first; ladder for token cuts
8. Safety rails: allowed-list guard, time budget, never-fail batch
9. Stack: llama.cpp + Vulkan/CPU fallback + Fireworks (+ Gemma where available)
10. What's next: smarter per-task feature routing, learned calibrator

## Cover image idea (1200x675)
Dark background, big text "FrugalRouter", subtitle "Qualify first. Spend less
next." Simple diagram: task -> [vote] -> safe path / cheaper ladder path with
AMD red and Fireworks orange accents. No stock photos, no clip art, black
SVG-style icons only.
