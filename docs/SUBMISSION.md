# lablab.ai Submission Content (copy-paste into the form)

## Project title
FrugalRouter

## Short description
A hybrid routing agent that answers 97% of tasks with a free local model by
buying confidence with free tokens: it votes the local model against itself
and pays a Fireworks expert only when the votes scatter.

## Long description
FrugalRouter is a Track 1 submission built around one observation: under the
scoring rules, local tokens are free, so the cheapest reliable lie detector
for a local model is the local model itself.

For each task, FrugalRouter:
1. Classifies the task into one of the 8 capability categories with
   zero-cost rules (no model call).
2. Samples the local model (Qwen3-14B served by vLLM on an AMD GPU) 5 to 10
   times and measures answer agreement. Sampling is batched into a single
   vLLM request, so calibration costs almost no wall-clock time and exactly
   zero scored tokens.
3. Ships the majority answer if agreement clears a per-category threshold
   tuned on a 227-task benchmark harness (GSM8K, HumanEval, and authored
   tasks across all 8 categories, with a held-out validation split).
4. Otherwise escalates that single task to the best-measured remote model on
   Fireworks AI with a terse prompt and a capped token budget.

Safety rails: the router cannot call any model outside the ALLOWED_MODELS
list, a failed task can never sink the batch, and a time-budget guard
degrades sampling instead of ever leaving tasks unanswered.

Measured on our 187-task train split: 93.6% accuracy fully local; routed
mode answered 96.8% of tasks locally at 92.5%+ accuracy for under 4K remote
tokens (vs ~50K+ remote tokens for a send-everything baseline).

Built solo in 4 days, developed and tuned on AMD GPUs (ROCm + vLLM) with a
reproducible eval harness included in the repo.

## Tags
AMD Developer Cloud, ROCm, vLLM, Fireworks AI, Gemma, Qwen3, AI Agents,
Model Routing, Token Efficiency

## Video script (2-3 min, screen recording + voiceover)

[0:00-0:25] THE PROBLEM
"Track 1 scores two things: accuracy, and how many remote tokens you burn.
Local tokens are free. So the whole game is: how do you know WHEN your free
local model is about to be wrong? Ask it to vote against itself."

[0:25-1:25] DEMO (screen: terminal on the AMD pod)
- Show tasks.json (a few questions visible).
- Run the router; point at the scrolling log: "category, votes, decision —
  local, local, local... and here, the votes scattered, so this one task
  escalates to Fireworks."
- Show results.json appearing, then the summary line: offload rate, tokens.

[1:25-2:10] THE NUMBERS (screen: frontier table / results)
"On our 187-task benchmark: fully local scores 93.6%. Routed mode keeps 97%
of tasks free and spends under four thousand remote tokens; sending
everything remote costs fifty thousand plus. The thresholds per category
come from an offline frontier sweep: we record one local run and one run per
remote expert, then replay every threshold combination instantly."

[2:10-2:40] THE STACK
"Local: Qwen3-14B on vLLM on an AMD GPU, sampled in batches so 10 votes cost
one request. Remote: Fireworks AI, model chosen per category by measured
accuracy-per-token. Everything is containerized, reproducible, MIT-licensed,
and the eval harness ships in the repo. Solo build, AMD Developer Hackathon
ACT II, team Transcendiant."

## Slide outline (10 slides)
1. FrugalRouter — token-efficient routing agent (team Transcendiant, solo)
2. The rules: accuracy + remote tokens; local = free
3. The insight: free tokens can buy calibration (self-consistency voting)
4. Architecture diagram (classify -> vote -> threshold -> escalate)
5. The harness: 227 benchmark tasks, 8 categories, held-out split
6. The frontier method: record once, replay every threshold offline
7. Results: 93.6% local floor; 97% offload; <4K tokens vs 50K baseline
8. Safety rails: allowed-list guard, time budget, never-fail batch
9. Stack: AMD GPU + ROCm + vLLM + Fireworks (+ Gemma where available)
10. What's next: smarter per-task feature routing, learned calibrator

## Cover image idea (1200x675)
Dark background, big text "FrugalRouter", subtitle "97% free. Every token
earned." Simple diagram: task -> [vote] -> free path (thick) / paid path
(thin) with AMD red and Fireworks orange accents. No stock photos, no clip
art, black SVG-style icons only.
