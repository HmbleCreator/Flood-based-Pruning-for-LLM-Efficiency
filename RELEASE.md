# Release Plan — FLOOD Trilogy

This file drafts the human-gated release steps (Bucket A). The agent prepared
everything below; the **bolded** steps require you (credentials / human judgement).

## 1. Pre-release checklist (agent-done)
- [x] All three papers written, polished through Phase 6 (commit `33f4f83`).
- [x] Hedging tightened, repetition reduced, one-question-per-paper framing added.
- [x] Reproducibility package (`REPRODUCIBILITY.md`, `requirements.txt`) written.
- [x] Git tag `v1.0.0-paper` exists; source compiles on Overleaf.
- [x] 167 cached `.npy` + 116 figures present; scripts regenerate them.

## 2. External review (YOU — Bucket A 6.5)
- [ ] Send the three `.tex` to: (a) one ML PhD student, (b) one professor, (c) one engineer.
- [ ] Ask each: **"Where did you stop understanding?"**
- [ ] Fold responses back; the peer-review report (`paper3_peer_review_report.md`) already
      pre-empts 8 standard objections — verify they still hold after edits.

## 3. arXiv submission (YOU — Bucket A 6.7)
Recommended order (build the arc, then standalone methods):
1. **Paper 1 — Invisible Bridges** (cs.CL / cs.LG). Core discovery; self-contained.
2. **Paper 2 — Conserved Perturbation Geometry** (cs.LG). Pure representation geometry.
3. **Paper 3 — Estimating Routing Importance** (cs.LG). Builds on 1 & 2; has FLOOD.

For each: export PDF from Overleaf, fill arXiv metadata:
- Subjects: Machine Learning (cs.LG), Computation and Language (cs.CL)
- Comments: "25 pages, 10 figures" (Paper 3) — adjust per paper.

## 4. GitHub release (YOU — Bucket A 6.7)
- Repo: `HmbleCreator/Flood-based-Pruning-for-LLM-Efficiency`
- Tag: `v1.0.0-paper` (already exists locally; push it)
- Release body (suggested):
  > Three-paper trilogy on structural routing in transformer attention heads.
  > - **Invisible Bridges**: low-weight heads missed by magnitude pruning cause 213x more damage.
  > - **Conserved Perturbation Geometry**: downstream perturbations collapse onto a shared low-dimensional subspace.
  > - **Estimating Routing Importance**: routing centralities explain up to 80% of causal-head damage; FLOOD pruning preserves perplexity 31x better than magnitude.
  >
  > Reproducibility package: see `REPRODUCIBILITY.md`. Cached matrices in `*.npy`; figures in `paper/figures/`.

## 5. Known limitations to state prominently in all three
- Decoder-only only (no BERT/T5 evaluated).
- Calibration on 3 WikiText-2 prompts.
- Scale-up to 1B/7B not yet done.
- `extract_dependency_graph.py` not run on GPT-2 Small (Paper 3 already has GPT-2 Small graphs from other scripts).
