"""
Retrieval Module for STM32F407 Documentation
Implements advanced retrieval mechanisms for the RAG system
"""
import numpy as np
from typing import List, Dict, Any, Optional
from src.embedding_storage.embedder import VectorStoreManager, Embedder
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """
    Different strategies for retrieving relevant documents
    """
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    FILTERED_SEMANTIC = "filtered_semantic"


class Retriever:
    """
    Advanced retrieval system for the STM32F407 documentation
    Supports various retrieval strategies and filtering options
    """
    
    def __init__(self, vector_store: VectorStoreManager, embedder: Embedder = None):
        """
        Initialize the retriever with a vector store and embedder
        
        Args:
            vector_store: Vector store containing embeddings
            embedder: Embedder for generating query embeddings (creates default if None)
        """
        self.vector_store = vector_store
        self.embedder = embedder or Embedder()
        self.reranker = BiEncoderReranker(self.embedder)
    
    def retrieve(self, query: str, k: int = 5, strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
                filters: Dict[str, Any] = None, rerank: bool = True, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query using specified strategy
        
        Args:
            query (str): User query
            k (int): Number of results to return
            strategy (RetrievalStrategy): Retrieval strategy to use
            filters (Dict[str, Any]): Metadata filters to apply
            rerank (bool): Whether to rerank results
            alpha (float): Weight for reranking (0-1)
            
        Returns:
            List[Dict[str, Any]]: Retrieved documents with metadata and scores
        """
        query_embedding = self.embedder.embed_single_text(query)
        
        if strategy == RetrievalStrategy.SEMANTIC:
            results = self._semantic_retrieve(query_embedding, k, filters)
        elif strategy == RetrievalStrategy.FILTERED_SEMANTIC:
            results = self._filtered_semantic_retrieve(query_embedding, k, filters)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_retrieve(query, k, filters, alpha)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        # Optionally rerank results
        if rerank and len(results) > 1:
            results = self.reranker.rerank(query, results)
        
        return results
    
    def _semantic_retrieve(self, query_embedding: np.ndarray, k: int, 
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform pure semantic similarity search
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results to return
            filters (Optional[Dict[str, Any]]): Metadata filters
            
        Returns:
            List[Dict[str, Any]]: Retrieved documents
        """
        return self.vector_store.similarity_search(query_embedding, k=k, filters=filters)
    
    def _filtered_semantic_retrieve(self, query_embedding: np.ndarray, k: int, 
                                  filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform semantic search with metadata filtering
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results to return
            filters (Dict[str, Any]): Metadata filters to apply
            
        Returns:
            List[Dict[str, Any]]: Retrieved documents
        """
        if not filters:
            return self._semantic_retrieve(query_embedding, k)
        
        return self.vector_store.similarity_search(query_embedding, k=k, filters=filters)
    
    def _hybrid_retrieve(self, query: str, k: int, filters: Optional[Dict[str, Any]] = None, 
                        alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining semantic and keyword-based search
        
        Args:
            query (str): User query
            k (int): Number of results to return
            filters (Optional[Dict[str, Any]]): Metadata filters
            alpha (float): Weight for semantic vs keyword scoring (0-1)
            
        Returns:
            List[Dict[str, Any]]: Retrieved documents
        """
        # For now, we'll implement a simple hybrid approach
        # In a production system, this would integrate with a keyword search system
        query_embedding = self.embedder.embed_single_text(query)
        
        # Get semantic results
        semantic_results = self._semantic_retrieve(query_embedding, k*2, filters)
        
        # Score based on keyword matching (simplified approach)
        for result in semantic_results:
            # Simple keyword matching score
            content_lower = result['chunk'].lower()
            query_lower = query.lower().split()
            keyword_score = sum(1 for word in query_lower if word in content_lower) / len(query_lower)
            result['keyword_score'] = keyword_score
        
        # Combine scores: weighted average of semantic and keyword scores
        max_semantic = max((r['similarity'] for r in semantic_results), default=1.0)
        max_keyword = max((r['keyword_score'] for r in semantic_results), default=1.0)
        
        for result in semantic_results:
            norm_semantic = result['similarity'] / max_semantic if max_semantic > 0 else 0
            norm_keyword = result['keyword_score'] / max_keyword if max_keyword > 0 else 0
            combined_score = alpha * norm_semantic + (1 - alpha) * norm_keyword
            result['combined_score'] = combined_score
        
        # Sort by combined score
        semantic_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return semantic_results[:k]
    
    def get_relevant_documents(self, query: str, k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Get relevant documents with minimum similarity threshold
        
        Args:
            query (str): User query
            k (int): Number of results to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            List[Dict[str, Any]]: Relevant documents above threshold
        """
        results = self.retrieve(query, k=k)
        
        # Filter by minimum similarity
        relevant_results = [r for r in results if r['similarity'] >= min_similarity]
        
        logger.info(f"Found {len(relevant_results)} relevant documents out of {len(results)} retrieved")
        return relevant_results
    
    def find_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk
        
        Args:
            chunk_id (str): ID of the reference chunk
            k (int): Number of similar chunks to return
            
        Returns:
            List[Dict[str, Any]]: Similar chunks
        """
        # Get the chunk by ID
        reference_chunk = self.vector_store.get_chunk_by_id(chunk_id)
        reference_embedding = reference_chunk['embedding']
        
        # Find similar chunks (excluding the reference chunk itself)
        all_results = self.vector_store.similarity_search(reference_embedding, k=k+1)
        
        # Remove the reference chunk if it's in the results
        similar_results = []
        for result in all_results:
            if result['metadata']['chunk_id'] != chunk_id:
                similar_results.append(result)
        
        return similar_results[:k]


class BiEncoderReranker:
    """
    Bi-encoder based reranker that reorders results based on query-chunk relevance
    """
    
    def __init__(self, embedder: Embedder):
        """
        Initialize the reranker
        
        Args:
            embedder: Embedder for generating embeddings
        """
        self.embedder = embedder
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank results based on query-chunk relevance
        
        Args:
            query (str): Original query
            results (List[Dict[str, Any]]): Retrieved results to rerank
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        if not results:
            return results
        
        # Generate embeddings for query
        query_embedding = self.embedder.embed_single_text(query)
        
        # Calculate relevance scores for each result
        for result in results:
            chunk_embedding = self.embedder.embed_single_text(result['chunk'])
            relevance_score = self.embedder.calculate_similarity(query_embedding, chunk_embedding)
            result['reranked_similarity'] = relevance_score
        
        # Sort by reranked similarity
        results.sort(key=lambda x: x['reranked_similarity'], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        logger.info(f"Reranked {len(results)} results based on query relevance")
        return results


class ContextualRetriever(Retriever):
    """
    Enhanced retriever that considers context and relationships between chunks
    """
    
    def __init__(self, vector_store: VectorStoreManager, embedder: Embedder = None):
        super().__init__(vector_store, embedder)
        self.context_expander = ContextExpander(vector_store)
    
    def retrieve_with_context(self, query: str, k: int = 5, expand_context: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve documents with expanded context
        
        Args:
            query (str): User query
            k (int): Number of primary results to return
            expand_context (bool): Whether to expand context around results
            
        Returns:
            List[Dict[str, Any]]: Retrieved documents with context
        """
        # Get primary results
        results = self.retrieve(query, k=k)
        
        # Expand context if requested
        if expand_context:
            for result in results:
                context = self.context_expander.expand_context(result['metadata']['chunk_id'])
                result['expanded_context'] = context
        
        return results


class ContextExpander:
    """
    Expands context around a given chunk by including neighboring chunks
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
    
    def expand_context(self, chunk_id: str, window_size: int = 1) -> Dict[str, Any]:
        """
        Expand context around a chunk by including adjacent chunks
        
        Args:
            chunk_id (str): ID of the central chunk
            window_size (int): Number of chunks to include before and after
            
        Returns:
            Dict[str, Any]: Expanded context information
        """
        # Find the index of the target chunk
        target_idx = None
        for i, meta in enumerate(self.vector_store.metadata):
            if meta.get('chunk_id') == chunk_id:
                target_idx = i
                break
        
        if target_idx is None:
            return {'error': f'Chunk with ID {chunk_id} not found'}
        
        # Collect surrounding chunks
        start_idx = max(0, target_idx - window_size)
        end_idx = min(len(self.vector_store.chunks), target_idx + window_size + 1)
        
        context_chunks = []
        for i in range(start_idx, end_idx):
            if i != target_idx:  # Don't include the target chunk itself
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


# Example usage
if __name__ == "__main__":
    # Example usage
    # vector_store = VectorStoreManager()
    # embedder = Embedder()
    # retriever = Retriever(vector_store, embedder)
    #
    # print("Retriever initialized")
    pass