# Agent Performance Improvement Plan

This document lists practical improvements to increase answer quality, latency, and reliability for the STM32F446RE agentic RAG system.

## 1) Fix critical response-path bugs first

1. **Unreachable answer-generation code in `AgenticRAG._generate_answer`**
   - The method currently returns a fallback response before context preparation, prompt construction, answer generation, citation extraction, and follow-up generation can execute.
   - Impact: the agent cannot leverage retrieved evidence for real answers.
   - Action: remove the premature return and execute the complete generation flow.

2. **Query classification misses multi-word intent phrases**
   - `QueryAnalyzer.analyze()` splits query by whitespace and compares single tokens against sets that include multi-word phrases (`"set up"`, `"not working"`, `"tell me about"`).
   - Impact: lower routing quality, wrong retrieval filters, weaker answer relevance.
   - Action: move to phrase-aware matching (n-grams or regex over full query string).

3. **High reranking cost due to per-result on-the-fly chunk embeddings**
   - `BiEncoderReranker.rerank()` re-embeds each candidate chunk text every query.
   - Impact: avoidable latency and compute waste.
   - Action: rerank directly with stored chunk embeddings or cache chunk embeddings by `embedding_index`.

## 2) Retrieval quality upgrades

4. **Adopt hybrid retrieval with BM25 + dense vectors + reciprocal rank fusion**
   - Current hybrid retrieval uses simple substring keyword overlap.
   - Impact: misses exact register/bit names and reduces precision on technical queries.
   - Action: add lexical index (e.g., BM25), fuse with dense scores, and tune on STM32 query set.

5. **Improve metadata filtering with component extraction**
   - Component extraction exists but `_build_filters()` does not apply component-level filters.
   - Impact: poor narrowing for peripheral/register-specific questions.
   - Action: index normalized component metadata (peripheral, register, pin, family document type) and apply structured filters.

6. **Use dynamic top-k and threshold calibration**
   - Fixed `k` (5/8) can under/over-retrieve.
   - Impact: either missing context or noisy prompts.
   - Action: adaptive k based on query entropy and score gaps; add minimum relevance threshold with backoff strategy.

## 3) Generation robustness and hallucination control

7. **Two-pass generation: draft then grounded verifier**
   - Current validator flags hallucination via term overlap heuristics only.
   - Impact: false positives/negatives on paraphrases and abbreviations.
   - Action: add a verification pass that checks each claim against source spans and removes unsupported statements.

8. **Cite exact snippet spans, not just section/page metadata**
   - Current citations are metadata-oriented.
   - Impact: lower trust and debuggability.
   - Action: attach answer sentences to exact chunk offsets/snippets and expose evidence highlights in API.

9. **Confidence model should combine retrieval + grounding + generation signals**
   - Confidence currently relies on heuristic scoring.
   - Impact: low calibration.
   - Action: combine top-score margin, evidence coverage, contradiction checks, and abstention rate; calibrate on held-out QA set.

## 4) Latency and scalability

10. **Move vector search from Python loops to ANN index (FAISS/HNSW)**
    - Similarity search currently computes cosine similarity over all embeddings in Python.
    - Impact: latency grows linearly with corpus size.
    - Action: use ANN index with persisted metadata sidecar and batch query support.

11. **Cache frequent queries and retrieval results**
    - Add query embedding cache and retrieval cache keyed by normalized query + filters.
    - Impact: faster repeated interactions and lower API cost.

12. **Async/background ingestion and read/write locking**
    - Ensure ingestion updates do not block query traffic and avoid inconsistent vector store state.
    - Action: reader-writer lock or versioned snapshots for atomic index swaps.

## 5) Data/chunking quality

13. **STM32-aware chunking strategy**
    - Chunk by semantic anchors: peripheral section, register table, bitfield list, and procedure steps.
    - Impact: better retrieval granularity and fewer mixed-topic chunks.

14. **Normalize technical aliases and canonical entities**
    - Build alias map (`USART2`/`UART2`, register short/long names, alternate pin notation).
    - Impact: improved recall and entity matching.

15. **Document-type routing**
    - Route queries to reference manual vs errata vs programming manual using classifier.
    - Impact: better precision and fewer contradictory answers.

## 6) API and product-level behavior

16. **Expose reasoning/evidence diagnostics for observability**
    - Add optional debug payload: selected strategy, filter values, retrieved scores, rejected evidence reasons.

17. **Add response contracts for abstention and uncertainty**
    - Standardize fallback response schema with explicit `abstained: true`, `reason`, `next_best_queries`.

18. **Rate limiting and request budgeting**
    - Protect latency under load; add per-request token/context budget policy.

## 7) Evaluation and continuous improvement

19. **Create a benchmark suite of STM32 tasks**
    - Include factual, procedural, troubleshooting, and comparative questions with gold citations.

20. **Track offline and online metrics**
    - Offline: recall@k, MRR, grounded factuality, citation precision.
    - Online: p95 latency, abstention quality, user feedback acceptance.

21. **Regression gating in CI**
    - Fail builds if retrieval/generation metrics regress beyond threshold.

## Suggested implementation order (highest ROI)

1. Fix unreachable generation path and phrase-aware query classification.
2. Replace reranker re-embedding with cached/stored embeddings.
3. Add ANN index + hybrid lexical retrieval.
4. Introduce grounded verifier and span-level citations.
5. Add benchmark dataset and CI regression gates.
