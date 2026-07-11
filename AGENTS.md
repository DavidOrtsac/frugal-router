# AGENTS.md — Operational handoff for TranscendiantRouter (AMD Hackathon ACT II, Track 1)

You (the agent reading this) may be picking up mid-competition from a prior AI operator.
Read this WHOLE file before acting. The human is David Castro (team "Transcendiant", solo).
He is direct and profane under pressure; give him outcomes, not hedging. No emojis, no em dashes.

## The competition

- Track 1, lablab.ai "AMD Developer Hackathon: ACT II". Submission: "TranscendiantRouter5"
  at https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/transcendiant/submission
- DEADLINE: Sunday July 12, 2026, 3:00 PM PT (= Monday July 13, ~6:00 AM Philippine time).
- Scoring: hidden set of exactly 19 tasks, graded by a NONDETERMINISTIC LLM judge.
  Accuracy gate = 80% (need 16/19 = 84.2% judged). Below gate = not on leaderboard.
  Among gate-passers, FEWEST Fireworks tokens wins. Local model tokens are free.
- The leaderboard keeps only your LATEST check result. A failed re-check knocks you off.
- Live board: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live?track=1
- Their LLM judge grades ~13 points harsher than our strict grader. Ship only configs
  that rehearse >= 95% strict on the 40-task held-out set.

## Current state (as of Sat Jul 11, ~2:30 PM PHT)

- Track 1 scoring infrastructure was BROKEN from ~Jul 11 02:00 PHT overnight (organizer-
  confirmed on Discord). Our row shows DNQ 63.2% — an OUTAGE ARTIFACT, not a real score.
  We previously passed at 84.2% / 7,185 tokens. Organizers said resubmit-caused DNQs
  "will be reviewed".
- Organizers asked everyone to STOP resubmitting until their fix lands. They promised a
  follow-up announcement SUNDAY MORNING declaring the pipeline fixed.
- THE PLAN: wait for that announcement (David checks Discord), then make ONE clean
  re-save of the submission, let it score, and stop touching it. Do not resubmit
  repeatedly. Do not build new rungs unless a rehearsal proves >=95% strict AND David
  approves the swap.
- The submission form's Docker Image field currently points at
  ghcr.io/davidortsac/frugal-router:r10 (an alias of the verified R-safe image).
  Tags r7-r26 are pre-minted aliases of the same digest (via .github/workflows/retag.yml).

## The router (this repo)

- Qwen3-4B-Instruct-2507 Q4_K_M runs locally in the container (llama.cpp, CPU) and answers
  factual/math/sentiment/summarization/NER for zero tokens; kimi-k2p7-code via Fireworks
  handles code_generation + logical_reasoning; code_debugging and math run local with
  self-escalation (compile-check / ANSWER-marker confidence). See Dockerfile ENV block.
- frugal_router/probe.py: boot-time self-healing probe of the judge's metering proxy
  (base-url variants x model-id forms x transport x auth). NEVER hand-rewrite model IDs.
- Resilience: per-call retries (connection errors retryable, 3 attempts), per-task retry,
  half-open circuit breaker, best-guess remote when probe fails, local-first ordering,
  emergency remote valve at 92% time budget.
- Tests: python3 -m pytest tests/ -q (91 passing, includes hostile-proxy integration).
- Rehearsal lab: GCP VM "frugal-lab" (project wifimapproject-489218, us-central1-a).
  SSH: ssh -i ~/.ssh/google_compute_engine davidcastro@34.30.123.252
  Rehearse: docker build -t test . then
  docker run --rm --cpus=2 --memory=4g -v ~/input:/input -v ~/output:/output \
    -e FIREWORKS_API_KEY=<from repo .env> <image>
  Grade: python3 eval/grade_results.py ~/output/results.json eval/tasks/heldout_tasks.json
  R-safe verified: 97.5% strict clean, ~6,900-7,900 rehearsal tokens (~3,000-3,600 hidden).
  DELETE this VM when the hackathon ends (it bills ~$1.17/hr).
- Builds: push a git tag v* -> GitHub Actions builds ghcr.io/davidortsac/frugal-router:latest
  + :<sha>. WARNING: :latest updates on every tag push. The judge grades whatever tag the
  FORM references — keep the form on an immutable alias, never :latest.
  Retag without rebuild: gh workflow run retag-image -f source_ref=<sha|tag> -f new_tag=<t>.

## Hard-won operational rules (violate these and you will repeat our disasters)

1. NEVER ship an image the judge will see without a same-day dress rehearsal on the VM
   under --cpus=2 --memory=4g. Three identical runs > one lucky run (llama.cpp multi-slot
   batching is nondeterministic; we run LLAMA_SLOTS=1 for determinism).
2. Verify the judge actually PULLED your image (GHCR download counter on the package page)
   — pushing is not delivering. Their runners cache tags; a NEW tag alias forces a pull
   and triggers a re-check within minutes.
3. Watch the organizers' Discord announcements BEFORE escalating anything. Metrics without
   ground truth breed wrong theories. The operator's channel outranks the leaderboard.
4. Respect their queue: no repeated resubmits when they ask for calm. Max ~1/hour even
   when allowed (their cap is 10/hour).
5. Never run heavy compute on David's 8GB MacBook. VM only.
6. Secrets live in this repo's .env (gitignored): FIREWORKS_API_KEY, HF_TOKEN,
   ELEVENLABS_API_KEY. Never commit them.
7. David's Aside browser (CLI: `aside repl` / `aside exec`) holds his logged-in lablab.ai
   session — that is how form edits are automated. Ask David before touching his accounts.

## History worth knowing (why the rules exist)

- Seven identical 63.2% scores came from ONE line that rewrote their exact model IDs;
  the probe replaced all such assumptions with runtime discovery.
- A "verified" image sat unpulled for 3 hours because their pipeline dedupes by digest
  and their nodes cache tags — hence rules 2 and the tag-alias magazine (r7-r26).
- A 9-hour overnight resubmission siege pushed against their explicit request for calm
  (announced on Discord while unwatched) — hence rules 3 and 4.
- Full narrative and endgame protocol: HANDOFF.md. Media/submission text: docs/SUBMISSION.md.
