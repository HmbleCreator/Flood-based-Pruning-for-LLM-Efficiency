---
name: literature-review
description: Run a PRISMA-inspired systematic literature search - multi-source search, screening with explicit inclusion/exclusion criteria, citation verification, contradiction detection, and thematic synthesis. Use when a research task needs grounding in existing work, or the user explicitly asks for a literature review / related-work section.
category: research
---

# Literature Review

Adapted from OpenScience's literature-review workflow. Full method lives in
`agents/literature-review.md` in this package - this file is the loadable reference.

## Core rule
Every citation must come from an actual search. Never invent a paper, DOI, author list, or
result. If you can't verify something, mark it `[CITATION NEEDED]`.

## Workflow
1. **Search** - 3-5 query variants: exact terms, synonyms/abbreviations, broader
   conceptual terms, specific technical terms (algorithm/dataset/benchmark names). Search
   general web + domain sources (arXiv, OpenAlex, Semantic Scholar, PubMed as relevant).
2. **Chain** - forward ("papers citing X") and backward (X's own references) from the
   strongest hits, until new rounds stop surfacing new relevant papers.
3. **Dedupe** - match by DOI/title; prefer the peer-reviewed version over a preprint when
   both exist.
4. **Screen** - state inclusion/exclusion criteria *before* screening; log a one-line
   reason for every exclusion.
5. **Tier for quality** - Tier 1 (peer-reviewed, strong evidence) / Tier 2 (moderate venue
   or highly-cited preprint) / Tier 3 (recent preprint, small-scale) / Tier 4 (blog post,
   unreproduced, single-author preprint).
6. **Detect contradictions** - actively look for disagreement between papers; note the
   likely cause (different data/metric/setup) rather than silently picking a side.
7. **Synthesize by theme**, not paper-by-paper: consensus / emerging / contested.

## Output
`literature-review.md` (template in `templates/literature-review.md`): summary, findings
by theme, contradictions, gaps, references. For a narrow ask ("find papers on X"), a
shorter 5-15-paper version without the full screening table is fine - match effort to the
scope of the question.

## Quality bar
~10+ real, verified papers minimum for a full review. Every claim traces to a specific
paper. Prefer recent work but don't discard a foundational older paper just for being old.
