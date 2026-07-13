---
name: rag-systems
description: Build and evaluate retrieval-augmented generation pipelines - chunking, embedding, vector store choice (FAISS/Chroma/Qdrant/Pinecone), hybrid retrieval, reranking, and separating retrieval quality from generation quality when reporting results. Use for any RAG pipeline build or debugging task.
category: llm-tools
---

# RAG Systems

## Separate retrieval quality from generation quality
The single most common RAG evaluation mistake: reporting one end-to-end "quality" number
that hides whether failures come from retrieval (right chunks not found) or generation
(right chunks found, but the model didn't use them well). Report both:
- Retrieval: recall@k, MRR, precision@k against a labeled or synthetic query set.
- Generation: faithfulness/groundedness (did the answer stick to retrieved content) and
  answer quality (via rubric or LLM-as-judge), evaluated *given* the actual retrieved
  context, not the ideal context.

## Chunking and embedding choices matter more than most hyperparameters
State explicitly: chunk size/overlap strategy, embedding model, and corpus size - these
change results more than tuning the retriever's internal parameters. Semantic/structural
chunking (by heading, sentence boundary) usually beats fixed-size character chunking for
technical or academic corpora.

## Vector store selection
- **FAISS** - local, no server, GPU-accelerable, best for pure similarity search at scale
  without needing metadata filtering or a hosted service.
- **Chroma** - lightweight, good default for local dev and small-to-medium corpora with
  metadata filtering.
- **Qdrant / Pinecone** - hosted or self-hosted with production features (filtering,
  hybrid search, horizontal scaling) - reach for these once FAISS/Chroma's constraints
  actually bite, not by default.

## Hybrid retrieval and reranking
Before crediting a hybrid (dense + BM25) or reranking pipeline with an improvement, compare
against a single dense retriever baseline. It's common for a reranker to fix a genuine
recall problem, but sometimes the "improvement" is actually recovering ground lost by a
bad chunking choice upstream - check where the gain is actually coming from before adding
pipeline complexity to keep it.

## Sanity checks before trusting a result
- Are the same documents ever double-indexed (duplicate chunks inflating recall)?
- Does the query set include queries with no good answer in the corpus (a well-behaved
  system should say so, not hallucinate one)?
- Did preprocessing (embedding model choice, chunking) get finalized before or after
  looking at eval results - tuning on the eval set the same way you wouldn't tune a
  classifier on its test set.
