# FrugalRouter: Master Plan

AMD Developer Hackathon ACT II, Track 1 (Hybrid Token-Efficient Routing Agent).
Team Transcendiant (solo: David). Written Jul 7, 2026 (Tue), PHT.

**Hard internal deadline: Fri Jul 11, 6:00 PM PHT submitted. Absolute cutoff Jul 11, 11:00 PM PHT.**
(Official close: Jul 12, 12:00 AM PHT. We never touch the last hour.)

---

## 1. The objective, restated precisely

Scoring is leaderboard-only: output accuracy plus remote token count. Local
tokens score zero. The accuracy bar is undisclosed, so the strategy is NOT
"skate just above the bar" (we cannot see the bar). The strategy is:

> Maximize accuracy first. Then, among configurations within ~1 point of our
> best achievable accuracy, pick the one with the fewest remote tokens.

A confidently-wrong local answer is the worst outcome (loses accuracy AND
looks like savings). Therefore calibration quality outranks token greed at
every decision point.

Secondary objective: the $1,000 "Best Use of Gemma via Fireworks" bonus.
Locked in by architecture: Gemma 4 26B-A4B local (via vLLM on AMD) plus
gemma-4-31b-it remote (via Fireworks) as default escalation model. The README,
slides, and video must all state this loudly.

## 2. Architecture (final unless data disproves it)

Per task, in order, with cumulative cost in scored tokens:

1. **Classify** into one of 8 categories. Rule-based regex. Cost: 0.
2. **Calibrate**: k self-consistency samples on the local model, majority
   vote. Adaptive: k=5; if agreement is borderline (0.4-0.6), sample 5 more.
   Cost: 0 (local is free).
3. **Decide**: agreement >= category threshold -> ship local majority answer.
   Otherwise escalate. Cost: 0.
4. **Escalate**: single terse call to the per-category remote model, capped
   max_tokens, no few-shot padding. Cost: the only scored tokens we ever spend.

Guardrails already in code:
- Remote model name must be in ALLOWED_MODELS or the call is rewritten to a
  Gemma fallback (out-of-list calls invalidate the submission).
- A task-level exception can never sink the batch; worst case a task ships
  the local answer or empty string.
- Output contract exactly [{task_id, answer}].

To add during Phase R (see below):
- **Time budget guard**: track elapsed wall-clock; if projected runtime
  exceeds budget, reduce k for remaining tasks (5 -> 3 -> 1) rather than risk
  not finishing. Unanswered = wrong.
- **Adaptive k** (5 then +5 on borderline agreement).
- **Answer normalization** hardening per category (numbers, labels, entity
  lists) so majority voting and grading are robust.

## 3. Timeline (PHT), day by day

### Tue Jul 7 (tonight): UNBLOCK + BASELINE
- [ ] DAVID: ADP portal -> Member Perks / My Benefits -> confirm $100 AMD
      Cloud credit state; add payment method to AMD Cloud account.
      If approval not yet triggered: trigger NOW (2-3 day manual queue).
- [ ] DAVID: Fireworks account -> confirm $50 hackathon credit visible;
      create API key; paste into .env (never commit).
- [ ] DAVID: Discord recon (Track 1 channel + pins + FAQ). Answers needed:
      1. Is a sample tasks.json provided anywhere? (paste raw if yes)
      2. Wall-clock limit per scoring run?
      3. Does the scoring env run OUR container with GPU access (vLLM inside
         container), or is a local endpoint provided to us?
      4. Does the scoring env have internet for HF model download at start,
         or must weights be baked into the image?
      5. How is accuracy graded per category (exact match? LLM judge?)
- [ ] CLAUDE: commit initial scaffold (on David's go).
- [ ] CLAUDE: start dev-set expansion (Workstream A).

### Wed Jul 8: REAL MODELS DAY
- [ ] DAVID: launch AMD Developer Cloud pod (MI300X), run vLLM with
      google/gemma-4-26B-A4B-it, share endpoint URL (or SSH access pattern).
- [ ] CLAUDE: Workstream A complete: ~400-task benchmark-sourced dev set with
      held-out split (see section 4).
- [ ] CLAUDE: local model bake-off on the pod: gemma-4-26B-A4B-it vs
      Qwen3-30B-A3B. Measure per-category accuracy floor (--local-only) and
      throughput (tasks/min at k=5). Decision gate D1: pick local model.
      Tie goes to Gemma (bonus narrative).
- [ ] CLAUDE: remote baseline: --remote-only on Fireworks for each allowed
      model on a 50-task slice. Measures accuracy ceiling + tokens per task
      per model. Decision gate D2: remote model per category.

### Thu Jul 9: TUNING DAY (this is where the rank is decided)
- [ ] CLAUDE: implement Phase R router upgrades (adaptive k, time guard,
      normalization hardening).
- [ ] CLAUDE: full sweep on TRAIN split: per-category thresholds x k
      schedule. Produce frontier table (accuracy vs remote tokens).
- [ ] CLAUDE: pick operating point by the rule in section 1. Validate ONCE on
      HELD-OUT split. If held-out accuracy drops >2 points vs train: thresholds
      are overfit -> simplify (fewer distinct thresholds), re-validate.
- [ ] BOTH: checkpoint review: are we confident? If local model is weak in a
      category (e.g. factual recall), consider forced-escalation for that
      category and measure the token price of safety.

### Fri Jul 10: PACKAGING + DRESS REHEARSAL
- [ ] CLAUDE: final Docker image on the pod (ROCm base). Decide weight
      strategy per Discord answer Q4: bake weights into image (safe, huge) vs
      HF download on start (fast to ship, needs network). Default if no
      answer: bake weights.
- [ ] BOTH: full dress rehearsal, twice: fresh container, synthetic
      tasks.json of 100 mixed tasks it has never seen, wall-clock timed,
      /output verified. Must pass clean twice in a row.
- [ ] CLAUDE: README final (setup + run + architecture diagram + Gemma
      story). Repo public on GitHub (DavidOrtsac), MIT, no secrets in
      history.
- [ ] CLAUDE: slides draft (8-10 slides, see section 6) + video script.
- [ ] DAVID: record 2-3 min video (screen capture + voiceover from script).
- [ ] DAVID: cover image (Claude drafts, or Canva).

### Sat Jul 11: SUBMIT (nothing new gets built today)
- [ ] BOTH: morning: final smoke run. Freeze code. Tag release v1.0.0.
- [ ] DAVID: fill lablab submission form (title, short desc, long desc, tags,
      cover, video, slides, GitHub URL, app URL). Submit by 6:00 PM PHT.
- [ ] DAVID: screenshot the submission confirmation.
- [ ] Buffer 6 PM - 11 PM: emergencies only.

## 4. Workstream A: the dev set (our whetstone)

Target ~400 tasks total, sourced from public benchmarks matching each
category, converted into our tasks.json + gold format:

| Category | Source | Train/Held-out |
|---|---|---|
| factual_knowledge | TriviaQA / MMLU (general) | 40/10 |
| math_reasoning | GSM8K | 40/10 |
| sentiment_classification | SST-2 / IMDB | 40/10 |
| text_summarization | XSum or CNN-DM (keyword golds) | 40/10 |
| ner | CoNLL-2003 | 40/10 |
| code_debugging | HumanEval solutions with injected bugs | 40/10 |
| logical_reasoning | LogiQA / bAbI-style | 40/10 |
| code_generation | HumanEval / MBPP | 40/10 |

Rules:
- Held-out split is generated once, never used for tuning, evaluated at most
  twice total (Thu validation, Fri rehearsal).
- Any official sample tasks from Discord become highest-authority items and
  seed the phrasing of the classifier regexes.
- Graders must be verified against 5 hand-checked answers per category before
  trusting harness numbers (bad grader = tuning on noise).

## 5. Risk register (trigger -> action)

| # | Risk | Trigger | Action |
|---|---|---|---|
| 1 | AMD credit approval misses deadline | Not approved by Wed noon | Pay-as-you-go MI300X (~$2/hr, ~$20-40 total, David's call), or rent equivalent GPU elsewhere for dev and only final-verify on AMD |
| 2 | Gemma 4 26B too slow / doesn't fit alongside KV cache | Bake-off throughput < 8 tasks/min at k=5 | Qwen3-30B-A3B; if still slow, gemma-4-12b-it class model; k drops before model quality does |
| 3 | Scoring env can't download weights | Discord Q4 answer or observed | Bake weights into image Friday; test image pull + cold start on pod |
| 4 | Unknown wall-clock limit | No answer by Thu | Assume 60 min for the batch; time budget guard enforces it; measure dress rehearsal at 2x task count for margin |
| 5 | Local model confidently wrong on factual recall | Category accuracy floor < 70% locally | Forced escalation for that category; the token price of safety is measured, not guessed |
| 6 | Overfit thresholds | Held-out drop > 2 pts | Collapse to 2 global thresholds (strict for verifiable, 0 for freeform); re-validate |
| 7 | Fireworks credit exhaustion during tuning | Balance < $10 | Tuning uses cached local samples + small remote slices; remote-only full runs are capped at 2 |
| 8 | lablab platform issues at deadline | Any submission error | We submit Fri 6 PM, 30 hours early; escalate in Discord #ineedhelp immediately |
| 9 | Rules ambiguity (e.g. what counts as "local") | Any doubt | Ask in Discord, screenshot the organizer answer, keep in repo /docs/rulings.md |

## 6. Submission assets checklist

- [ ] GitHub repo public, MIT LICENSE, README with one-command run
- [ ] Dockerfile builds from clean clone (verified on the pod, not just Mac)
- [ ] Cover image (1200x675): FrugalRouter name, one-line pitch, AMD/Gemma/Fireworks logos context
- [ ] Slides (8-10): 1 problem, 2 scoring rules, 3 architecture diagram, 4 self-consistency insight ("local tokens are free, so we buy confidence with them"), 5 adaptive-k + guardrails, 6 eval methodology, 7 frontier chart from sweep, 8 results, 9 Gemma-on-both-sides, 10 what's next
- [ ] Video 2-3 min: 30s problem + rules, 60s live demo (tasks.json in, results.json out, routing log scrolling), 45s frontier chart + numbers, 15s Gemma + AMD stack callout
- [ ] lablab form: title "FrugalRouter", short + long description, tags (AMD Developer Cloud, ROCm, Gemma, Fireworks AI, Qwen3 if used), app URL = repo or demo
- [ ] Build-in-public (optional, low effort): 3 posts with #AMDDevHackathon: (1) architecture sketch, (2) frontier chart, (3) submission ship post

## 7. Decision log

| ID | Decision | Status | Basis |
|---|---|---|---|
| D1 | Local model | OPEN (Wed bake-off) | per-category accuracy floor + throughput; tie -> Gemma |
| D2 | Remote model per category | OPEN (Wed baseline) | accuracy per token on 50-task slice; default Gemma 4 31B, code -> kimi-k2p7-code |
| D3 | k schedule | OPEN (Thu sweep) | accuracy vs runtime on train split |
| D4 | Thresholds / operating point | OPEN (Thu sweep) | max accuracy, then min tokens within 1 pt |
| D5 | Weights baked vs downloaded | OPEN (Discord Q4) | default: baked |
| D6 | Architecture: classify -> self-consistency -> threshold -> escalate | DECIDED | free local sampling is the exploitable asymmetry of the scoring rules |
| D7 | Gemma both paths for bonus | DECIDED | $1,000 bonus, zero architectural cost |

## 8. What David personally must do (complete list)

1. Tonight: credits (ADP + Fireworks) + Discord recon (5 questions above).
2. Wed: launch the GPU pod, share access.
3. Thu: 30-min checkpoint review with Claude.
4. Fri: record the video; approve cover image.
5. Sat: submit on lablab by 6 PM PHT; screenshot confirmation.

Everything else is Claude's.
