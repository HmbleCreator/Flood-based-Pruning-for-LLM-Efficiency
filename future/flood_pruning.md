# Flood Pruning (Paper 2) - Future Roadmap

This document outlines the planned direction for the next phase of the **Flood** research program: **Bridge-Guided Pruning for LLM Efficiency**.

## Vision

Standard pruning metrics (e.g., Wanda, SparseGPT) evaluate the importance of weights or heads locally. Our first paper, *Invisible Bridges*, demonstrated that this local focus systematically overlooks causally critical attention heads (invisible bridges) whose ablation causes cascading downstream disruption.

The goal of the next phase is to build **Flood**: a structured pruning algorithm that leverages downstream representation sensitivity to guide network compression.

## Key Research Objectives

1. **Bridge-Guided Pruning Algorithm (Flood)**
   - Design a metric that combines local weight magnitude with downstream bridge scores to rank attention heads and layers for removal.
   - Formulate a structured pruning schedule that prioritizes redundant early-layer heads and preserves "bridge" components.

2. **Adaptive Bridge Recovery (Paper 3)**
   - Develop efficient post-pruning fine-tuning or recovery techniques.
   - Investigate how gradient routing can adaptively reconstruct "bridge" pathways when early layers are pruned.

3. **Scaling to Modern LLMs**
   - Scale the computation of downstream representation sensitivity to larger models (e.g., Llama-3, Mistral, Gemma).
   - Optimize the forward-pre-hook ablation scanning methodology to run efficiently on multi-GPU setups.
