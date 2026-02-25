"""
Dataset Processor for STM32F446 Documentation
Handles loading and processing multiple PDF files from the dataset folder
"""
import os
from typing import List, Dict, Any
from src.document_processing.loader import DocumentLoader
from src.document_processing.preprocessor import DocumentPreprocessor
from src.document_processing.chunker import SemanticChunker
from src.embedding_storage.embedder import EmbeddingPipeline
import logging

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """
    Processes all PDF files in the dataset folder for the STM32F446 documentation
    """
    
    def __init__(self, dataset_path: str, pipeline: EmbeddingPipeline):
        """
        Initialize the dataset processor
        
        Args:
            dataset_path: Path to the dataset folder containing PDF files
            pipeline: Embedding pipeline for processing chunks
        """
        self.dataset_path = dataset_path
        self.pipeline = pipeline
        self.preprocessor = DocumentPreprocessor()
        self.chunker = SemanticChunker()
        self.processed_files = []
        self.total_pages = 0
        
    def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all PDF files in the dataset folder
        
        Returns:
            Dict containing processing results and statistics
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_path}")
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError("No PDF files found in dataset folder")
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            'processed_files': [],
            'failed_files': [],
            'total_pages': 0,
            'total_chunks': 0,
            'processing_time': 0
        }
        
        import time
        start_time = time.time()
        
        # Process each PDF file
        for pdf_file in pdf_files:
            try:
                result = self._process_single_document(pdf_file)
                results['processed_files'].append(result)
                results['total_pages'] += result['pages_count']
                results['total_chunks'] += result['chunks_count']
                self.processed_files.append(pdf_file)
                logger.info(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                error_result = {
                    'filename': pdf_file,
                    'error': str(e)
                }
                results['failed_files'].append(error_result)
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
        
        results['processing_time'] = time.time() - start_time
        
        # Save the final vector store
        self.pipeline.save_pipeline_state()
        
        logger.info(f"Dataset processing completed. Processed {len(results['processed_files'])} files, "
                   f"{results['total_pages']} pages, {results['total_chunks']} chunks")
        
        return results
    
    def _process_single_document(self, filename: str) -> Dict[str, Any]:
        """
        Process a single PDF document
        
        Args:
            filename: Name of the PDF file to process
            
        Returns:
            Dict containing processing results for this document
        """
        file_path = os.path.join(self.dataset_path, filename)
        
        # Load document
        loader = DocumentLoader(file_path)
        if not loader.load_document():
            raise ValueError(f"Failed to load document: {filename}")
        
        # Extract text with metadata
        pages_data = loader.extract_text_with_metadata()
        pages_count = len(pages_data)
        
        # Preprocess pages
        processed_pages = self.preprocessor.preprocess_pages(pages_data)
        
        # Create chunks
        chunks = self.chunker.chunk_document(processed_pages)
        chunks_count = len(chunks)
        
        # Add file metadata to chunks
        for chunk in chunks:
            chunk['metadata']['source_file'] = filename
            chunk['metadata']['file_path'] = file_path
        
        # Process through embedding pipeline
        self.pipeline.process_chunks(chunks)
        
        return {
            'filename': filename,
            'pages_count': pages_count,
            'chunks_count': chunks_count,
            'file_path': file_path
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed dataset
        
        Returns:
            Dict containing dataset statistics
        """
        if not hasattr(self.pipeline.vector_store, 'chunks') or not self.pipeline.vector_store.chunks:
            return {'error': 'No data processed yet'}
        
        stats = {
            'total_chunks': len(self.pipeline.vector_store.chunks),
            'total_embeddings': len(self.pipeline.vector_store.embeddings),
            'processed_files': len(self.processed_files),
            'total_pages': self.total_pages,
            'files_by_type': self._analyze_file_types(),
            'chunk_size_stats': self._analyze_chunk_sizes()
        }
        
        return stats
    
    def _analyze_file_types(self) -> Dict[str, int]:
        """
        Analyze the types of files in the dataset based on filenames
        
        Returns:
            Dict with file type counts
        """
        file_types = {}
        
        for filename in self.processed_files:
            # Extract file type from filename patterns
            if 'rm0390' in filename.lower():
                file_type = 'reference_manual'
            elif 'pm0214' in filename.lower():
                file_type = 'programming_manual'
            elif 'es0298' in filename.lower():
                file_type = 'errata_sheet'
            elif 'stm32f446' in filename.lower() and 'mc' in filename.lower():
                file_type = 'datasheet'
            elif 'um1724' in filename.lower():
                file_type = 'user_manual'
            else:
                file_type = 'other'
            
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return file_types
    
    def _analyze_chunk_sizes(self) -> Dict[str, float]:
        """
        Analyze chunk size statistics
        
        Returns:
            Dict with chunk size statistics
        """
        if not hasattr(self.pipeline.vector_store, 'chunks') or not self.pipeline.vector_store.chunks:
            return {}
        
        chunk_sizes = [len(chunk) for chunk in self.pipeline.vector_store.chunks]
        
        if not chunk_sizes:
            return {}
        
        return {
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'avg_size': sum(chunk_sizes) / len(chunk_sizes),
            'total_chars': sum(chunk_sizes)
        }

# Example usage
if __name__ == "__main__":
    # This would be used in the main application
    pass