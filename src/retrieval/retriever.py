"""
Retrieval Module for STM32F407 Documentation
Implements advanced retrieval mechanisms for the RAG system
"""
import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.embedding_storage.embedder import VectorStoreManager, Embedder
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Different strategies for retrieving relevant documents."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    FILTERED_SEMANTIC = "filtered_semantic"


class Retriever:
    """Advanced retrieval system with strategy selection, reranking, and lightweight caching."""

    def __init__(self, vector_store: VectorStoreManager, embedder: Embedder = None):
        self.vector_store = vector_store
        self.embedder = embedder or Embedder()
        self.reranker = BiEncoderReranker(self.embedder)
        self._cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        filters: Dict[str, Any] = None,
        rerank: bool = True,
        alpha: float = 0.5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query using specified strategy."""
        normalized_query = query.strip().lower()
        cache_key = (normalized_query, k, strategy.value, tuple(sorted((filters or {}).items())), rerank, round(alpha, 3), round(min_similarity, 3))
        if cache_key in self._cache:
            return [r.copy() for r in self._cache[cache_key]]

        query_embedding = self.embedder.embed_single_text(query)

        if strategy == RetrievalStrategy.SEMANTIC:
            results = self._semantic_retrieve(query_embedding, max(k * 2, k), filters)
        elif strategy == RetrievalStrategy.FILTERED_SEMANTIC:
            results = self._filtered_semantic_retrieve(query_embedding, max(k * 2, k), filters)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_retrieve(query, max(k * 2, k), filters, alpha)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        if rerank and len(results) > 1:
            results = self.reranker.rerank(query, results)

        if min_similarity > 0:
            results = [r for r in results if r.get('similarity', 0.0) >= min_similarity or r.get('combined_score', 0.0) >= min_similarity]

        results = results[:k]
        self._cache[cache_key] = [r.copy() for r in results]
        return results

    def _semantic_retrieve(self, query_embedding: np.ndarray, k: int,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.vector_store.similarity_search(query_embedding, k=k, filters=filters)

    def _filtered_semantic_retrieve(self, query_embedding: np.ndarray, k: int,
                                    filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not filters:
            return self._semantic_retrieve(query_embedding, k)
        return self.vector_store.similarity_search(query_embedding, k=k, filters=filters)

    def _hybrid_retrieve(self, query: str, k: int, filters: Optional[Dict[str, Any]] = None,
                         alpha: float = 0.5) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_single_text(query)
        semantic_results = self._semantic_retrieve(query_embedding, k * 2, filters)

        query_tokens = self._tokenize(query)
        idf_map = self._idf_map(query_tokens, semantic_results)

        for result in semantic_results:
            result['keyword_score'] = self._bm25_lite_score(query_tokens, result['chunk'], idf_map)

        max_semantic = max((r.get('similarity', 0.0) for r in semantic_results), default=1.0)
        max_keyword = max((r.get('keyword_score', 0.0) for r in semantic_results), default=1.0)

        for result in semantic_results:
            norm_semantic = result.get('similarity', 0.0) / max_semantic if max_semantic > 0 else 0.0
            norm_keyword = result.get('keyword_score', 0.0) / max_keyword if max_keyword > 0 else 0.0
            result['combined_score'] = alpha * norm_semantic + (1 - alpha) * norm_keyword

        semantic_results.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
        return semantic_results[:k]

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if t]

    def _idf_map(self, query_tokens: List[str], docs: List[Dict[str, Any]]) -> Dict[str, float]:
        if not docs:
            return {token: 1.0 for token in query_tokens}

        n_docs = len(docs)
        idf: Dict[str, float] = {}
        for token in set(query_tokens):
            df = sum(1 for d in docs if token in self._tokenize(d.get('chunk', '')))
            idf[token] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return idf

    def _bm25_lite_score(self, query_tokens: List[str], text: str, idf_map: Dict[str, float]) -> float:
        if not query_tokens or not text:
            return 0.0

        doc_tokens = self._tokenize(text)
        if not doc_tokens:
            return 0.0

        token_freq: Dict[str, int] = {}
        for token in doc_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

        doc_len = len(doc_tokens)
        avg_doc_len = 200.0  # lightweight fixed estimate for chunked corpus
        k1 = 1.2
        b = 0.75

        score = 0.0
        for term in query_tokens:
            tf = token_freq.get(term, 0)
            if tf == 0:
                continue
            idf = idf_map.get(term, 1.0)
            denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))

        return score

    def get_relevant_documents(self, query: str, k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        results = self.retrieve(query, k=k, min_similarity=min_similarity)
        logger.info(f"Found {len(results)} relevant documents out of {k} requested")
        return results

    def find_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Dict[str, Any]]:
        reference_chunk = self.vector_store.get_chunk_by_id(chunk_id)
        reference_embedding = reference_chunk['embedding']
        all_results = self.vector_store.similarity_search(reference_embedding, k=k + 1)

        similar_results = []
        for result in all_results:
            if result['metadata']['chunk_id'] != chunk_id:
                similar_results.append(result)
        return similar_results[:k]


class BiEncoderReranker:
    """Bi-encoder based reranker that reorders results based on query-chunk relevance."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        if not results:
            return results

        query_embedding = self.embedder.embed_single_text(query)
        for result in results:
            chunk_embedding = self._resolve_chunk_embedding(result)
            relevance_score = self.embedder.calculate_similarity(query_embedding, chunk_embedding)
            result['reranked_similarity'] = relevance_score

        results.sort(key=lambda x: x['reranked_similarity'], reverse=True)
        if top_k is not None:
            results = results[:top_k]

        logger.info(f"Reranked {len(results)} results based on query relevance")
        return results

    def _resolve_chunk_embedding(self, result: Dict[str, Any]) -> np.ndarray:
        if 'embedding' in result and result['embedding'] is not None:
            return result['embedding']
        return self.embedder.embed_single_text(result['chunk'])


class ContextualRetriever(Retriever):
    """Enhanced retriever that considers context and relationships between chunks."""

    def __init__(self, vector_store: VectorStoreManager, embedder: Embedder = None):
        super().__init__(vector_store, embedder)
        self.context_expander = ContextExpander(vector_store)

    def retrieve_with_context(self, query: str, k: int = 5, expand_context: bool = True) -> List[Dict[str, Any]]:
        results = self.retrieve(query, k=k)
        if expand_context:
            for result in results:
                context = self.context_expander.expand_context(result['metadata']['chunk_id'])
                result['expanded_context'] = context
        return results


class ContextExpander:
    """Expands context around a given chunk by including neighboring chunks."""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def expand_context(self, chunk_id: str, window_size: int = 1) -> Dict[str, Any]:
        target_idx = None
        for i, meta in enumerate(self.vector_store.metadata):
            if meta.get('chunk_id') == chunk_id:
                target_idx = i
                break

        if target_idx is None:
            return {'error': f'Chunk with ID {chunk_id} not found'}

        start_idx = max(0, target_idx - window_size)
        end_idx = min(len(self.vector_store.chunks), target_idx + window_size + 1)

        context_chunks = []
        for i in range(start_idx, end_idx):
            if i != target_idx:
                context_chunks.append({
                    'chunk_id': self.vector_store.metadata[i]['chunk_id'],
                    'content': self.vector_store.chunks[i],
                    'position': 'before' if i < target_idx else 'after',
                    'distance': abs(i - target_idx)
                })

        return {
            'central_chunk_id': chunk_id,
            'context_chunks': context_chunks,
            'total_context_chunks': len(context_chunks)
        }


if __name__ == "__main__":
    pass
