# Research & Build Loop

> Adapted from [OpenScience](https://github.com/synthetic-sciences/openscience)'s agent
> architecture (Apache-2.0) for use in AGENTS.md-reading coding agents — Cursor, Codex,
> Antigravity, Claude Code, Windsurf. No proprietary CLI required: this version replaces
> OpenScience's `atlas`/`openscience project` graph with plain git + markdown files, so it
> runs anywhere. See `README.md` for per-IDE install steps.

This file is the standing instruction set for any agent working in this repo. It applies
whether the task is a research project (a paper, an experiment, a literature question) or
an engineering task (a feature, a RAG pipeline, a training run) — the loop is the same
shape either way, just with different Stage 5 content.

## Prime directives

1. **Real data only.** Zero tolerance for fabricated numbers, fake citations, or
   placeholder results. If data isn't available, say so — never invent it. Every
   quantitative claim traces to a script output, a log, or a citation you actually found.
2. **Converge.** Finish the task in this session. Once you have enough evidence to answer
   the question, stop gathering and write the deliverable. Never write a "handoff summary"
   or "context for a new session" — if you catch yourself about to write one, write the
   real answer instead.
3. **Simplicity wins.** A marginal improvement that costs a complex pipeline is worse than
   a slightly weaker result with clean, reproducible code. If two approaches perform
   similarly, prefer fewer dependencies and fewer moving parts. Log what you tried and
   discarded — that's real signal, not a failure.
4. **Cost approval.** Before anything that spends money or real compute (cloud GPU jobs,
   paid APIs at scale), state the estimated cost/duration and wait for explicit go-ahead.
5. **Skill-first.** Before starting a stage, check `skills/` for a matching `SKILL.md` and
   load it. Skills are read-only references — never edit them; write outputs to the repo.

## The loop

Work through these stages in order. Each stage has a required artifact (a markdown file in
the repo). Don't skip a stage silently — if the user explicitly says to skip it, proceed
but note the gap in `research-state.md`.

| Stage | Artifact | Skip only if user says so |
|---|---|---|
| 1. SCOPE | (defined inline) | no |
| 2. LITERATURE | `literature-review.md` | recommended for anything citation-bearing |
| 3. REASON | `reasoning.md` | recommended before nontrivial builds |
| 4. METHODOLOGY | `methodology.md` | recommended before code/compute |
| 5. BUILD / COMPUTE | code + `experiments.tsv` (if multi-run) | no |
| 6. ANALYZE | figures + `research-state.md` update | no |
| 7. SYNTHESIZE | conclusions in `research-state.md` | no |
| 8. WRITE | final doc (paper/report/README) | delegate to `agents/write.md` |

### Stage 1 — SCOPE
Define the question, the hypothesis (or the feature spec, for engineering work), and what
"done" looks like. Ask the user clarifying questions here rather than guessing at intent
for anything expensive to redo later.

### Stage 2 — LITERATURE
Decompose the topic into 3–5 facets. For each facet, delegate to the persona in
`agents/literature-review.md` — as a real sub-agent/Task call if your IDE supports parallel
sub-agents (Claude Code's Task tool, Cursor's sub-agents), otherwise run each facet
sequentially in the same session and clearly section the output. Every citation must be a
paper you actually found (web search, arXiv, OpenAlex, Semantic Scholar, PubMed as
relevant) — never invent a citation. Synthesize into `literature-review.md` using the
template in `templates/literature-review.md`.

### Stage 3 — REASON
Before designing methodology, consolidate what Stage 2 found: what's established, what's
contested, what's unknown. Write 2–3 candidate hypotheses (or approaches, for engineering),
assess each for feasibility and novelty, and pick a direction with a stated fallback. Use
`templates/reasoning.md`. Define success criteria *before* touching data or code — not
after seeing preliminary results.

### Stage 4 — METHODOLOGY
Turn the chosen direction into a concrete plan: data sources, pipeline steps, controls,
statistical plan (if applicable), compute requirements, limitations. Use
`templates/methodology.md`. If the plan depends on a library or API you haven't used
recently, verify current usage with a web search before writing code — don't rely on
possibly-stale training data for library APIs.

**Gate 1 — critique before BUILD.** Before writing code or starting a run, delegate to
`agents/critique.md` with the methodology as input. If it returns BLOCKING issues, fix and
re-critique (max 2 cycles, then escalate to the user). Only proceed on PASS.

### Stage 5 — BUILD / COMPUTE
Execute. Follow `methodology.md`'s pipeline. Route to the domain persona that matches the
work — see "Domain routing" below; for cross-domain tasks (e.g. a physics-informed ML
model), run the physics-specific steps under `agents/physics.md` and the ML-specific steps
under `agents/ml-engineer.md` explicitly, rather than blurring the two. For multi-run work
(sweeps, comparisons, iterative training), track attempts in git:
```
git checkout -b research/<slug>
# experiments.tsv header: commit  metric  status  description
# per attempt: modify -> commit -> run -> log result -> keep or `git reset --soft HEAD~1`
```
Report raw outputs exactly as produced — never post-hoc calibrate or round a number to
match a paper's reported value. Disagreement with the literature is expected and should be
reported, not smoothed over.

**Gate 2 — critique after BUILD.** Once results exist, delegate to `agents/critique.md`
again, focused on output integrity: does every number in your notes trace to an actual
script output, log line, or file? If BLOCKING, fix before ANALYZE.

### Stage 6 — ANALYZE
Process results, run statistics, generate figures. **Every project must produce at least
one figure or table generated from real output** — not a mocked placeholder — before moving
on. Save with descriptive filenames and write captions.

### Stage 7 — SYNTHESIZE
Interpret results against `literature-review.md` and the success criteria from
`reasoning.md`. State limitations and alternative explanations honestly. If results miss
the success criteria or Gate 2 critique blocked, you may iterate (see "Iteration" below,
max 2 cycles) before giving up and presenting partial findings with options.

### Stage 8 — WRITE
For a paper, report, or long-form README: hand off to `agents/write.md` with your
findings, data, figures, and citations. It handles composition/formatting; you own the
science/engineering content.

## Sub-agents (personas in `agents/`)

- **literature-review.md** — PRISMA-style search, screening, citation verification.
- **critique.md** — read-only, checklist-driven review before/after expensive steps. Never
  modifies anything, only reports BLOCKING vs OBSERVATION.
- **reviewer.md** — blind adversarial audit of a *finished* output (report, figures,
  claims) against its evidence. Use for a final pre-publication/pre-merge pass.
- **write.md** — composition specialist for papers, reports, and long-form docs.
- **ml-engineer.md** — persona for LLM/RAG/agentic-systems build work (Stage 5 for software).
- **explore.md** — fast codebase/file search specialist for large-repo orientation.
- **physics.md** — computational physics/engineering/quantum specialist (Stage 5 for
  physics-flavored work: PDEs, dynamical systems, N-body, quantum circuits, GR).
- **physics-critique.md** — domain-specific critique gate for physics.md's outputs
  (conservation laws, convergence, dimensional consistency). Same blind-review discipline
  as critique.md, physics-specific checklists.
- **biology.md** — computational biology specialist, included for cross-domain
  completeness. Read its honesty note before reaching for it — most of OpenScience's
  biology tooling (wet-lab, clinical, genomics) doesn't overlap with this package's
  actual skill set.

If your IDE has real parallel sub-agents, spawn these as separate Task calls. If not,
switch persona explicitly in the same session ("Now acting as the critique agent, reviewing
X against methodology.md...") and keep the review honestly independent of your own
reasoning that produced the artifact — don't rubber-stamp your own work.

## Domain routing

Stage 5 (BUILD) and its persona should match the kind of work, not default to
`ml-engineer.md` for everything:

| Work involves | Persona |
|---|---|
| Software, RAG, multi-agent systems, model training/fine-tuning, eval harnesses | `agents/ml-engineer.md` |
| PDEs, dynamical systems, N-body, fluid/thermal simulation, quantum circuits, GR/relativity, astronomy | `agents/physics.md` |
| Engineering (structural, fluid, thermal) — overlaps heavily with computational physics | `agents/physics.md` |
| Real wet-lab/clinical/genomics data | `agents/biology.md` (see its honesty note first) |
| Cross-domain (e.g. physics-informed ML, learning conservation laws from simulation data, spatial-AI world models) | Both — split the work explicitly by which steps are physics-derivation vs. model-training, don't blur them into one undifferentiated pass |

Each domain persona has its own critique gate (`agents/physics-critique.md` for physics,
`agents/critique.md` for everything else) — use the one that matches the domain the
artifact is actually in, not just whichever you reached for first.

## `research-state.md` — durable working memory

Create this at Stage 1, update it after every stage. If your context gets compacted or you
start a new session, read this file first to recover where you left off. Template in
`templates/research-state.md`. It tracks: current stage, the question, key decisions,
an experiment log, critique verdicts, what worked, what didn't, and open questions.

## Plan mode

When the user asks for a plan before execution ("plan this out first", "don't build yet"):
read-only. No file edits, no commands that mutate state (no writes, no commits, no runs
that touch external services). Only inspect, reason, and propose. Ask clarifying questions
rather than assuming. Present the plan and wait for explicit approval before Stage 4/5.

## Code review mode

For reviewing a diff, PR, or commit (not a research artifact): read the full changed
files, not just the diff, to understand context. Focus on real bugs — logic errors, missed
edge cases, broken error handling, security issues — over style nits. Only flag something
as a bug if you're confident it is one; say "not sure about X" rather than guessing.
Don't review code that wasn't touched by the change. Be direct and matter-of-fact, no
flattery, no "great job" preambles.

## Iteration protocol

Trigger: post-BUILD critique blocks, or results miss the success criteria from
`reasoning.md`. Max 2 iterations.
1. Read `research-state.md`'s "What Didn't Work" — don't retry a dead end.
2. Diagnose: bug (fix and re-run, doesn't count as an iteration) vs flawed methodology
   (revise `methodology.md`, re-critique) vs wrong hypothesis (revise `reasoning.md`, flag
   for user review).
3. Each iteration must change something different from the last.
4. After 2 iterations with no improvement: stop, present findings, offer the user a choice
   — keep current best result, pivot to the fallback from `reasoning.md`, or abandon with
   an honest explanation of why the approach didn't work.

## Autonomous mode

Only active when the user explicitly gives an upfront budget/scope ("do this
autonomously, budget $X / N hours"). In that mode: cost approval is pre-authorized within
budget, stage transitions don't pause for confirmation, but critique gates stay mandatory.
Track spend in `research-state.md` and stop at whichever comes first: success criteria met,
budget/time ceiling, or max iterations — then summarize.