# Literature Review: Adaptive / Architecture-Conditioned Routing-Aware Pruning

## Summary
The trilogy established that (a) low-weight "bridge" heads are causally critical
(Paper 1), (b) downstream perturbations collapse onto a shared low-dimensional subspace
(Paper 2), and (c) routing centralities (Broadcast/Receiver/Injector/Betweenness) explain
up to 80% of causal-head damage and drive FLOOD, a routing-aware pruning algorithm that
preserves perplexity 31x better than magnitude pruning (Paper 3). The remaining gap is
**Bucket B**: making the RIE combination weights alpha_i a function of a model's global
routing-graph statistics (algebraic connectivity lambda2, modularity Q) so FLOOD needs no
per-model tuning. This review checks whether that specific idea — graph-stat-conditioned
pruning weights — already exists.

Two facets were searched. Facet 1 (adaptive / architecture-conditioned pruning) shows a
rich literature on metrics that adapt to architecture (SNIP connection sensitivity, SNACS
adaptive per-layer connectivities, learning-to-prune agents) but none that derive pruning
weights from a *global routing graph's spectral properties*. Facet 2 (graph-centrality /
routing-based pruning) shows **LLM-Rank (arXiv:2410.13299)** as the only work using a
network-science centrality (modified weighted PageRank) as the transformer pruning criterion
— but on a *local* MLP dependency graph with *fixed* weights. DepGraph (arXiv:2301.12900)
models a dependency graph but uses norm-based importance. Path patching / ACDC / attribution
graphs supply the causal-routing methodology our dependency matrix is built from.

**Gap (the novelty of this work):** No prior method prunes transformers using a *causal
routing dependency graph built from activation patching* AND makes the pruning weights a
function of that graph's algebraic connectivity (lambda2) / modularity (Q). Adaptive RIE
fills exactly this gap.

## Key Findings by Facet

### Facet 1 — Adaptive / architecture-conditioned pruning criteria
- **SNIP** (Lee et al., 2019, arXiv:1810.02340): connection-sensitivity saliency from the
  task-loss gradient at init; noted robust to architecture variation. Conditioned on
  architecture/task, but not on a computed graph's spectral properties.
- **SNACS** (Ganesh et al., 2020, arXiv:2006.12463): Adaptive Conditional Mutual Information
  gives per-layer connectivity scores and *automatically* sets max prune fraction per layer.
  Explicit architecture/connectivity-conditioned criterion, still norm/MI-based not graph-spectral.
- **Learning to Prune Filters** (Huang et al., 2018, arXiv:1801.07365): a learned agent
  discovers per-architecture filter-removal policy. Meta/RL approach, not graph-derived.

### Facet 2 — Graph-theoretic / centrality / routing-based pruning
- **LLM-Rank** (Hoffmann et al., 2024, arXiv:2410.13299, DOI 10.48550/arXiv.2410.13299):
  builds a weighted DAG of an MLP/decoder transformer and applies a *modified weighted
  PageRank* centrality to rank nodes for pruning. The only centrality-based transformer
  pruning method found. Uses a *local* dependency graph and *fixed* weights — does NOT
  condition weights on lambda2/Q. **Closest prior art; our differentiator.**
- **CAHP** (Livertovsky et al., 2026, arXiv:2606.19150, IJCNN 2026): global graph-theoretic
  head selection via clustering + information-theoretic distance; auto-determines prune ratio.
  Clustering, not centrality-weighted scoring.
- **DepGraph** (Fang et al., 2023, arXiv:2301.12900, CVPR 2023): dependency graph modeling
  structural coupling so coupled groups prune together; norm-based importance, not centrality.
- **Path Patching** (Goldowsky-Dill et al., 2023, arXiv:2304.05969): causal path localization
  via activation patching — the methodological basis of our dependency matrix D(u,v).
- **ACDC** (Conmy et al., 2023) and **attribution graphs** (Syed et al., 2024): causal
  circuit discovery; confirm routing graphs are a legitimate interpretability substrate.

## Identified Gaps & Opportunities
1. **No graph-stat-conditioned pruning weights.** LLM-Rank fixes PageRank weights; we propose
   alpha_i = f(lambda2, Q), adapting the mix per architecture.
2. **Causal (patching-based) routing graph for pruning is unexplored as a weighting signal.**
   Existing graph pruning uses observational/correlation graphs; ours is intervention-based.
3. **Per-architecture optimal descriptor is known to vary** (walkthrough: Broadcast wins on
   GPT-2 Medium/Pythia-160M, Betweenness on OPT-125M) but no method exploits lambda2/Q to
   predict it. Adaptive RIE does.

## Complete References
```bibtex
@inproceedings{lee2019snip,
  title={SNIP: Single-shot Network Pruning based on Connection Sensitivity},
  author={Lee, Namhoon and Ajanthan, Theshaan and Torr, Philip H. S.},
  booktitle={ICLR}, year={2019}, eprint={1810.02340}
}
@misc{ganesh2020snacs,
  title={Slimming Neural Networks using Adaptive Connectivity Scores (SNACS)},
  author={Ganesh, Madan R. and Blanchard, David and Corso, Jason J. and Sekeh, S. Yasaei},
  year={2020}, eprint={2006.12463}
}
@misc{huang2018learning,
  title={Learning to Prune Filters in Convolutional Neural Networks},
  author={Huang, Qiangui and Zhou, Ke and You, Suya and Neumann, Ulrich},
  year={2018}, eprint={1801.07365}
}
@article{hoffmann2024llmrank,
  title={LLM-Rank: A Graph Theoretical Approach to Pruning Large Language Models},
  author={Hoffmann, David and Budhathoki, Kailash and Kleindessner, Maya},
  journal={arXiv preprint}, year={2024}, eprint={2410.13299}, doi={10.48550/arXiv.2410.13299}
}
@misc{livertovsky2026cahp,
  title={Complementary Attention Head Pruning for Efficient Transformers (CAHP)},
  author={Livertovsky, Yehonatan and Somin, Shiri and Singer, Gali},
  year={2026}, eprint={2606.19150}
}
@inproceedings{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gong and Ma, Xing and Song, Mingli and Mi, Michael Bi and Wang, Xing},
  booktitle={CVPR}, year={2023}, eprint={2301.12900}
}
@misc{goldowskydill2023pathpatching,
  title={Localizing Model Behavior with Path Patching},
  author={Goldowsky-Dill, Nicholas and MacLeod, Callum and Sato, Lucas and Arora, Abhay},
  year={2023}, eprint={2304.05969}
}
```
