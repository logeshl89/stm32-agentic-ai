"""
FastAPI Application for STM32F446RE Agentic RAG System
Provides RESTful endpoints for querying the documentation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import asyncio
from src.document_processing.loader import DocumentLoader
from src.document_processing.preprocessor import DocumentPreprocessor
from src.document_processing.chunker import SemanticChunker
from src.document_processing.dataset_processor import DatasetProcessor
from src.embedding_storage.embedder import Embedder, VectorStoreManager, EmbeddingPipeline
from src.retrieval.retriever import Retriever
from src.agentic_layer.agent import AgenticRAG
from src.agentic_layer.answer_generator import AnswerGenerator
from config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the application
app = FastAPI(
    title="STM32F446RE Agentic RAG API",
    description="API for querying STM32F446RE microcontroller documentation using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the system components
document_loader = None
preprocessor = None
chunker = None
embedder = None
vector_store = None
retriever = None
agent = None
answer_generator = None
pipeline = None

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query about STM32F446RE")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_citations: bool = Field(default=True, description="Include citations in response")
    include_followup: bool = Field(default=True, description="Include follow-up questions")
    include_debug: bool = Field(default=False, description="Include retrieval diagnostics")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    retrieved_docs_count: int
    followup_questions: List[str]
    processing_time: float
    abstained: bool = False
    abstain_reason: str = ""
    next_best_queries: List[str] = Field(default_factory=list)
    debug: Optional[Dict[str, Any]] = None


class DocumentLoadRequest(BaseModel):
    """Request model for document loading"""
    pdf_path: str = Field(..., description="Path to the STM32F446RE PDF documentation")


class DocumentLoadResponse(BaseModel):
    """Response model for document loading"""
    success: bool
    message: str
    pages_loaded: int
    status: str


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    components_initialized: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global document_loader, preprocessor, chunker, embedder, vector_store, retriever, agent, answer_generator, pipeline
    
    logger.info("Initializing STM32F446RE Agentic RAG system components...")
    
    try:
        # Initialize components
        embedder = Embedder()
        vector_store = VectorStoreManager(CONFIG['paths']['vector_store_path'])
        retriever = Retriever(vector_store, embedder)
        answer_generator = AnswerGenerator()
        agent = AgenticRAG(retriever, answer_generator)
        
        # Initialize document processing components
        preprocessor = DocumentPreprocessor()
        chunker = SemanticChunker()
        pipeline = EmbeddingPipeline(embedder, vector_store)
        
        # Try to load existing vector store
        try:
            vector_store.load_from_disk()
            logger.info("Loaded existing vector store from disk")
        except FileNotFoundError:
            logger.info("No existing vector store found, will create new one when documents are loaded")
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    components = []
    if preprocessor: components.append("preprocessor")
    if chunker: components.append("chunker")
    if embedder: components.append("embedder")
    if vector_store: components.append("vector_store")
    if retriever: components.append("retriever")
    if agent: components.append("agent")
    if answer_generator: components.append("answer_generator")
    if pipeline: components.append("pipeline")
    
    return HealthCheckResponse(
        status="healthy" if len(components) >= 8 else "degraded",
        components_initialized=components
    )


@app.post("/load_dataset", response_model=DocumentLoadResponse)
async def load_dataset():
    """Load and process all STM32F446 documentation files from dataset"""
    global pipeline
    
    try:
        # Create dataset processor
        dataset_processor = DatasetProcessor(CONFIG['paths']['dataset_path'], pipeline)
        
        # Process all documents
        results = dataset_processor.process_all_documents()
        
        success = len(results['failed_files']) == 0
        message = f"Processed {len(results['processed_files'])} files with {results['total_pages']} pages"
        
        if results['failed_files']:
            message += f". Failed to process {len(results['failed_files'])} files."
            
        return DocumentLoadResponse(
            success=success,
            message=message,
            pages_loaded=results['total_pages'],
            status="completed"
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documentation(request: QueryRequest):
    """Query the STM32F446RE documentation"""
    global agent, answer_generator
    
    if not agent:
        raise HTTPException(status_code=500, detail="System not initialized properly")
    
    import time
    start_time = time.time()
    
    try:
        # Process the query through the agentic system
        response = agent.process_query(request.query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Check if confidence meets threshold
        debug_payload = None
        if request.include_debug:
            debug_payload = {
                "query_type": str(response.get('query_type')),
                "retrieval_strategy": response.get('retrieval_strategy'),
                "retrieval_filters": response.get('retrieval_filters', {}),
                "reasoning_trace": response.get('reasoning_trace', []),
                "top_similarities": [round(d.get('similarity', 0.0), 4) for d in response.get('retrieved_docs', [])[:5]]
            }

        if response['confidence'] < request.min_confidence:
            # Return a response indicating low confidence
            return QueryResponse(
                query=request.query,
                answer=(
                    f"I found information about your query but with low confidence ({response['confidence']:.2f}). "
                    f"The system requires minimum confidence of {request.min_confidence:.2f}. "
                    f"Please try rephrasing your question or consult the official documentation directly."
                ),
                confidence=response['confidence'],
                citations=[] if not request.include_citations else response['citations'][:3],
                retrieved_docs_count=len(response['retrieved_docs']),
                followup_questions=[] if not request.include_followup else response['followup_questions'][:3],
                processing_time=processing_time,
                abstained=True,
                abstain_reason=response.get('abstain_reason', 'low_confidence'),
                next_best_queries=response.get('next_best_queries', []),
                debug=debug_payload
            )
        
        # Return successful response
        return QueryResponse(
            query=request.query,
            answer=response['answer'],
            confidence=response['confidence'],
            citations=[] if not request.include_citations else response['citations'],
            retrieved_docs_count=len(response['retrieved_docs']),
            followup_questions=[] if not request.include_followup else response['followup_questions'],
            processing_time=processing_time,
            abstained=response.get('abstained', False),
            abstain_reason=response.get('abstain_reason', ''),
            next_best_queries=response.get('next_best_queries', []),
            debug=debug_payload
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get statistics about the loaded documentation"""
    global vector_store
    
    if not vector_store or not vector_store.is_loaded:
        return {"message": "No vector store loaded", "stats": {}}
    
    stats = {
        "total_chunks": len(vector_store.chunks),
        "total_embeddings": len(vector_store.embeddings),
        "metadata_fields": list(set().union(*(d.keys() for d in vector_store.metadata))) if vector_store.metadata else [],
        "section_types": {},
        "avg_chunk_size": 0
    }
    
    # Calculate section type distribution
    for meta in vector_store.metadata:
        sec_type = meta.get('section_type', 'unknown')
        stats['section_types'][sec_type] = stats['section_types'].get(sec_type, 0) + 1
    
    # Calculate average chunk size
    if vector_store.chunks:
        total_chars = sum(len(chunk) for chunk in vector_store.chunks)
        stats['avg_chunk_size'] = total_chars / len(vector_store.chunks)
    
    return {"message": "Statistics retrieved successfully", "stats": stats}


@app.post("/feedback")
async def submit_feedback(feedback: Dict[str, Any]):
    """Submit feedback on query responses"""
    logger.info(f"Received feedback: {feedback}")
    # In a real implementation, this would store feedback for system improvement
    return {"message": "Feedback received", "received": True}


# Additional utility endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "STM32F446RE Agentic RAG API",
        "endpoints": [
            "/health - Health check",
            "/load_dataset - Process all STM32F446RE documentation files",
            "/query - Query the documentation",
            "/stats - Get statistics about loaded documentation",
            "/feedback - Submit feedback"
        ]
    }


def create_app():
    """Factory function to create the FastAPI app"""
    return app


# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)