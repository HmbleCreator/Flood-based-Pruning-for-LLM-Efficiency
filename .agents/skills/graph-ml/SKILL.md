---
name: graph-ml
description: Build and evaluate graph neural networks and graph-structured pipelines - message-passing architecture choice, train/test splitting without leakage on graphs, and common evaluation mistakes specific to graph data (transductive vs inductive, negative sampling for link prediction). Use for GNN, knowledge-graph, or graph-ML tasks.
category: coding
---

# Graph ML

## Splitting graphs is not like splitting tabular data
The single most common source of leakage in graph ML: splitting nodes/edges randomly
without accounting for the graph structure connecting train and test. Specifically:
- **Transductive vs. inductive** - be explicit about which setting you're in. Transductive
  (the whole graph, including test nodes, is visible during training, only labels are
  held out) is a different, easier problem than inductive (test nodes/edges are entirely
  unseen during training). Reporting a transductive result as if it were inductive
  overstates generalization.
- **Link prediction negative sampling** - the ratio and strategy of negative (non-edge)
  samples materially changes reported metrics; state the sampling strategy explicitly, and
  make sure test-time negatives aren't sampled from edges that exist elsewhere in the graph
  but were held out.
- **Information leakage through graph structure** - even with a "clean" node split, a
  message-passing layer can leak test-label information into train nodes' embeddings if
  test edges aren't masked during the forward pass on the training subgraph.

## Architecture choice
- **GCN** - simple, strong default for homogeneous graphs with informative node features.
- **GraphSAGE** - better when you need inductive generalization to unseen nodes (samples
  and aggregates neighborhoods rather than requiring the full graph at train time).
- **GAT** - adds attention over neighbors; worth it when neighbor importance is
  heterogeneous, adds real compute cost for cases where it isn't.
- **Message-passing depth** - more layers isn't free; oversmoothing (node representations
  converging to indistinguishable values) is a real failure mode past 3-4 layers on many
  graphs. If deeper is needed, consider residual connections or jumping-knowledge
  aggregation rather than just stacking layers.

## Evaluation
- Report metrics appropriate to the task: node classification (accuracy/F1 per class, not
  just micro-averaged if classes are imbalanced), link prediction (AUC/AP with the negative
  sampling strategy stated), graph classification (accuracy/F1 with the train/test split
  strategy - random split, scaffold split, or temporal split, stated explicitly).
- Always run a non-GNN baseline (e.g. a feature-only MLP ignoring graph structure, or a
  simple heuristic like common-neighbors for link prediction) - it's common for a GNN's
  apparent gain to actually come from node features alone, with the graph structure adding
  little.
