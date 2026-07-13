# /lit-review

Run a standalone literature review on the topic below (Stage 2 only — no methodology or
build work). Useful when you just need to know what's already known before deciding
whether a fuller research loop is worth running.

Topic: $ARGUMENTS

Steps:
1. Decompose the topic into 3-5 facets.
2. Delegate each facet to the `agents/literature-review.md` persona (in parallel if your
   IDE supports real sub-agents, sequentially with clear section headers otherwise).
3. Synthesize into `literature-review.md` using `templates/literature-review.md`. Flag
   contradictions between facets explicitly rather than silently resolving them.
4. Do not proceed to methodology or code unless asked.
