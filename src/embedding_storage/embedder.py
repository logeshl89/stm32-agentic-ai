"""
Embedding Generation Module for STM32F407 Documentation
Handles generation and management of text embeddings
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Union
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Embedder:
    """
    Class responsible for generating embeddings for text chunks
    Uses sentence transformers for high-quality semantic embeddings
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a pre-trained model
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedder with model: {model_name}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text chunks to embed
            batch_size (int): Batch size for embedding generation
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} text chunks")
        
        # Generate embeddings in batches for efficiency
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        logger.info(f"Completed embedding generation for {len(embeddings)} chunks")
        return embeddings
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            emb1 (np.ndarray): First embedding vector
            emb2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


class VectorStoreManager:
    """
    Manages the storage and retrieval of embeddings using a vector database
    """
    
    def __init__(self, storage_path: str = "./vector_store.pkl"):
        """
        Initialize the vector store manager
        
        Args:
            storage_path (str): Path to store the vector database
        """
        self.storage_path = Path(storage_path)
        self.chunks = []  # Stores the text chunks
        self.embeddings = []  # Stores the embedding vectors
        self.metadata = []  # Stores chunk metadata
        self.is_loaded = False
    
    def add_texts_and_embeddings(self, texts: List[str], embeddings: List[np.ndarray], 
                               metadatas: List[Dict[str, Any]] = None):
        """
        Add texts, embeddings, and metadata to the vector store
        
        Args:
            texts (List[str]): List of text chunks
            embeddings (List[np.ndarray]): Corresponding embeddings
            metadatas (List[Dict[str, Any]]): Optional metadata for each chunk
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("Length of texts, embeddings, and metadatas must be equal")
        
        self.chunks.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadatas)
        
        logger.info(f"Added {len(texts)} items to vector store. Total: {len(self.chunks)}")
    
    def save_to_disk(self):
        """
        Save the vector store to disk
        """
        data = {
            'chunks': self.chunks,
            'embeddings': [emb.tolist() for emb in self.embeddings],  # Convert to list for serialization
            'metadata': self.metadata
        }
        
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector store saved to {self.storage_path}")
    
    def load_from_disk(self):
        """
        Load the vector store from disk
        """
        if not self.storage_path.exists():
            raise FileNotFoundError(f"Vector store file not found: {self.storage_path}")
        
        with open(self.storage_path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = [np.array(emb) for emb in data['embeddings']]  # Convert back to arrays
        self.metadata = data['metadata']
        self.is_loaded = True
        
        logger.info(f"Vector store loaded from {self.storage_path}. Contains {len(self.chunks)} items")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5, 
                         filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector store
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of results to return
            filters (Dict[str, Any]): Optional filters for metadata
            
        Returns:
            List[Dict[str, Any]]: List of matching results with similarity scores
        """
        if not self.is_loaded and not self.chunks:
            raise ValueError("Vector store is empty. Load data first.")
        
        # Calculate similarities
        similarities = []
        for i, stored_embedding in enumerate(self.embeddings):
            similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for idx, similarity in similarities:
                meta = self.metadata[idx]
                match = True
                for key, value in filters.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append((idx, similarity))
            similarities = filtered_results
        
        # Return top k results
        results = []
        for idx, similarity in similarities[:k]:
            results.append({
                'chunk': self.chunks[idx],
                'similarity': similarity,
                'metadata': self.metadata[idx],
                'embedding_index': idx,
                'embedding': self.embeddings[idx]
            })
        
        logger.info(f"Similarity search returned {len(results)} results")
        return results
    
    def _calculate_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            emb1 (np.ndarray): First embedding vector
            emb2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific chunk by its ID
        
        Args:
            chunk_id (str): ID of the chunk to retrieve
            
        Returns:
            Dict[str, Any]: Chunk data including content and metadata
        """
        for i, meta in enumerate(self.metadata):
            if meta.get('chunk_id') == chunk_id:
                return {
                    'chunk': self.chunks[i],
                    'embedding': self.embeddings[i],
                    'metadata': meta
                }
        
        raise ValueError(f"Chunk with ID {chunk_id} not found")


class EmbeddingPipeline:
    """
    Complete pipeline for embedding generation and storage
    """
    
    def __init__(self, embedder: Embedder = None, vector_store: VectorStoreManager = None):
        """
        Initialize the embedding pipeline
        
        Args:
            embedder: Embedder instance (creates default if None)
            vector_store: Vector store manager (creates default if None)
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStoreManager()
    
    def process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32):
        """
        Process a list of chunks through the embedding pipeline
        
        Args:
            chunks (List[Dict[str, Any]]): List of chunks with content and metadata
            batch_size (int): Batch size for embedding generation
        """
        if not chunks:
            logger.warning("No chunks to process")
            return
        
        # Extract texts and metadata
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(texts, batch_size)
        
        # Add to vector store
        self.vector_store.add_texts_and_embeddings(texts, embeddings, metadatas)
        
        logger.info(f"Processed {len(chunks)} chunks through embedding pipeline")
    
    def save_pipeline_state(self):
        """
        Save the current state of the pipeline
        """
        self.vector_store.save_to_disk()
    
    def load_pipeline_state(self):
        """
        Load the pipeline state from disk
        """
        self.vector_store.load_from_disk()


# Example usage
if __name__ == "__main__":
    # Example usage
    embedder = Embedder()
    vector_store = VectorStoreManager()
    pipeline = EmbeddingPipeline(embedder, vector_store)
    
    print("Embedding pipeline initialized")