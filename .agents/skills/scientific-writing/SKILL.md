---
name: scientific-writing
description: Write and structure academic prose - paper skeletons, formal register, honest limitations sections, and the specific conventions (flowing paragraphs, no bullet-heavy body text) that reviewers expect. Use for Stage 8 WRITE, or any paper/report/abstract drafting.
category: writing
---

# Scientific Writing

## Structure first
Before writing prose, lay out the full section skeleton (Abstract, Introduction, Related
Work, Method, Results, Discussion, Limitations, Conclusion - adapt to venue). Filling in a
skeleton produces a more coherent document than writing linearly and hoping it holds
together.

## Prose conventions
- Flowing paragraphs with real transitions in the body - avoid bullet lists inside
  narrative sections (fine in an appendix or a background box, not in Results/Discussion).
- Formal, precise register throughout; avoid hedge-stacking ("might possibly suggest") and
  avoid overclaiming in equal measure.
- State limitations as a real, specific list, not one boilerplate sentence buried at the
  end - and make sure the abstract doesn't quietly contradict what Limitations says.
- Precise numbers, not rounded-for-effect ones: if the metric is 0.669, write 0.669, not
  "approximately 0.7," unless the precision genuinely isn't meaningful.

## Abstract discipline
An abstract usually has a hard word limit (250 words is common). Write it last, after the
paper is stable - and preserve exact metrics precisely when tightening it; don't let
significant figures drift during editing passes. Structure: problem -> approach -> key
result (with the actual number) -> significance, in that order, no throat-clearing.

## Figures
Every paper needs at least one figure that actually shows something (a graphical
abstract/pipeline diagram at minimum for any paper; real result plots for an empirical
one). A figure generated from placeholder data is worse than no figure - it will get
caught, and it undermines trust in the real figures.

## Before calling it done
1. Read the PDF as rendered (not just the source) - check figure placement, caption
   spacing, overlaps, page breaks mid-table.
2. Run a self-review pass with the `agents/reviewer.md` persona in this package: does
   every number trace to something real, does every citation actually say what it's cited
   for.
3. Check the abstract's claims still match the body after any late edits - this is the
   single most common source of "the abstract oversells the result" reviewer comments.
