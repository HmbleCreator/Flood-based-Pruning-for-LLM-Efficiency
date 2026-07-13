---
name: hypothesis-generation
description: Generate testable hypotheses from observations or literature - formulate competing explanations, define null hypotheses, assess feasibility and novelty, and pick a direction with a stated fallback. Use at Stage 3 (REASON) of a research task, or whenever a claim needs to be turned into something falsifiable before work starts.
category: research
---

# Hypothesis Generation

## Core idea
A hypothesis is only useful if it's falsifiable and you've defined, in advance, what
evidence would confirm or refute it. Define success criteria *before* running any
analysis - post-hoc criteria are how research fools itself.

## Workflow
1. **Consolidate** what's already known (from a literature review if one exists): what's
   established, what's contested, what's genuinely unknown.
2. **Generate 2-4 candidate hypotheses**, not just one. For each:
   - State it precisely and testably.
   - State the null / alternative that would disprove it.
   - What evidence would this take to test - is that evidence obtainable?
   - Feasibility (high/medium/low) and novelty (does it extend beyond existing work?).
   - If confirmed: what does that mean? If refuted: what does that suggest instead?
3. **Compare candidates** on strengths, weaknesses, key uncertainty, and expected
   information gain - a table forces this comparison to be explicit rather than vibes.
4. **Select a direction** with an explicit rationale tied to specific findings, the key
   risks, and a fallback plan if the primary hypothesis doesn't pan out.

## Common failure modes to avoid
- Picking the hypothesis most likely to "work" rather than the one most informative either
  way. A hypothesis with an interesting negative result is often more valuable than a safe
  positive one.
- Vague hypotheses ("X helps with Y") that can't actually be refuted by any result.
- Skipping the null hypothesis - if you can't state what "no effect" looks like, you can't
  tell a real effect from noise.

## Output
`reasoning.md` (template in `templates/reasoning.md`). This becomes the reference point
for success criteria throughout the rest of the loop - don't redefine "success" after
seeing results.
