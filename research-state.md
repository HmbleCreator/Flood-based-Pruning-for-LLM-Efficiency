# Research State: Flood-Based Pruning for LLM Efficiency

## Current Stage
SYNTHESIZE (all three papers complete + Phase 6 polish done; release prep in progress)

## Trilogy (all COMPLETE, polished through Phase 6 — commit `33f4f83`)
- **Paper 1 — Invisible Bridges** (`paper/invisible_bridges.tex`): low-weight "bridge" heads missed by Wanda; 213x damage ratio; confounders ruled out; downstream amplification; domain-conditional scores.
- **Paper 2 — Conserved Perturbation Geometry** (`paper/conserved_perturbation_geometry.tex`): perturbations collapse onto a shared low-dimensional subspace (PR ~1.1–3.2 vs shuffled 2.7–11.5; v1 alignment 0.935–0.976); bridge heads = amplitude injectors.
- **Paper 3 — Estimating Routing Importance** (`paper/estimating_routing_importance.tex`): RIE centralities explain up to 80.2% of causal-damage variance (p<1e-5 beyond layer depth; shuffle R2<1.3%); FLOOD pruning preserves perplexity 31x better than magnitude.

## Phase 6 polish (DONE — commit `33f4f83`)
- Tightened hedging: "prove that"->"provide evidence that"; 3x "confirming that"->"consistent with"; "explain causal damage"->"explain variance in experimentally measured causal damage".
- Reduced latency repetition (Paper 3): removed duplicated 51.26ms/40x figure.
- Added explicit one-question-per-paper "\paragraph{Why this paper?}" to all three.
- Fixed broken sentence (Paper 1 entropy discussion) + tangled null-baseline sentence (Paper 2).

## Naming reconciliation (important)
The `future/*.md` proposal docs use OLD roadmap names that were SUPERSEDED:
- `future/paper3_routing_mechanisms.md` ("Routing Mechanisms") -> its content (dependency graph, PageRank, communities) is now embedded in Paper 3 (`estimating_routing_importance.tex`), which is COMPLETE.
- The only item from that old roadmap NOT yet executed: `src/paper3_routing/extract_dependency_graph.py` on **GPT-2 Small** specifically (run on Pythia-70M/160M, OPT-125M, GPT-2 Medium). Not required for the trilogy — Paper 3 already has GPT-2 Small dependency network + PageRank figures.

## Bucket C (release prep I can do) — DONE
- `REPRODUCIBILITY.md`: run order, script index, cached-data notes, compile instructions.
- `requirements.txt`: real deps (numpy, torch, transformers, matplotlib, scipy, statsmodels, networkx).
- Read-aloud pass: fixed 2 awkward sentences.

## Bucket A (release — needs user)
- 6.5 External review (ML PhD / prof / engineer): "where did you stop understanding?"
- 6.7 arXiv upload + GitHub release + reproducibility package zip.
- Repo already tagged `v1.0.0-paper`. Source verified to compile on Overleaf.

## Bucket B (genuine new research — deferred, not started)
1. **Adaptive RIE** (Paper 3 Future Work, Eq. 7): make alpha_i a function of (Q, lambda2) so FLOOD auto-selects Broadcast-vs-Betweenness per architecture. Uses cached dependency matrices for 5 models — cheap, high value. RECOMMENDED next research step.
2. Scale-up replication (1B/7B) — biggest validity threat; deferred by user.
3. Architecture generalization (BERT/T5 encoder-decoder).
4. MLP-layer routing graphs.

## Key numbers (from walkthrough.md, all traced to logs)
- FLOOD 31x better than magnitude @30% on GPT-2 Medium (51.28 vs 1594.18 PPL).
- OLS R2: GPT-2 Small 23.5%, OPT-125M 65.4%, Pythia-160M 80.2%.
- Shuffle R2 < 1.3%. OOD transfer rho=0.469 (p=0.0008).
- Topology preservation @30%: subspace align FLOOD 0.99 vs Magnitude 0.73; hub overlap 100% vs 20%.
