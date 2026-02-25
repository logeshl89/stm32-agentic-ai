"""
STM32F446RE Agentic RAG System
Main entry point for the application
"""
import os
from dotenv import load_dotenv
import logging
from config import CONFIG

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, CONFIG['system']['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("STM32F407 Agentic RAG System")
    print("Initializing system...")
    
    # Test model loading first
    print("=== Testing STM32F446RE Model Loading ===\n")
    
    try:
        # Check if vector store file exists
        vector_store_path = "vector_store_f446.pkl"
        if os.path.exists(vector_store_path):
            size_mb = os.path.getsize(vector_store_path) / (1024 * 1024)
            print(f"✓ Vector store file found: {vector_store_path}")
            print(f"  Size: {size_mb:.2f} MB")
        else:
            print(f"✗ Vector store file not found: {vector_store_path}")
            print("Model loading failed. Please check your vector store and configuration.")
            return
        
        # Test imports
        print("\n2. Testing imports...")
        from src.embedding_storage.embedder import Embedder, VectorStoreManager
        from src.retrieval.retriever import Retriever
        print("   ✓ All modules imported successfully")
        
        # Load vector store
        print("\n3. Loading vector store...")
        vector_store = VectorStoreManager(vector_store_path)
        vector_store.load_from_disk()
        print(f"   ✓ Vector store loaded successfully")
        print(f"   Chunks: {len(vector_store.chunks)}")
        print(f"   Embeddings: {len(vector_store.embeddings)}")
        print(f"   Metadata entries: {len(vector_store.metadata)}")
        
        # Initialize retriever
        print("\n4. Initializing retriever...")
        embedder = Embedder()
        retriever = Retriever(vector_store, embedder)
        print("   ✓ Retriever initialized successfully")
        
        # Test a simple query
        print("\n5. Testing query processing...")
        test_query = "STM32F446RE GPIO configuration"
        results = retriever.retrieve(test_query, k=3)
        print(f"   ✓ Query processed successfully")
        print(f"   Retrieved {len(results)} results")
        
        if results:
            print(f"   Sample result similarity: {results[0]['similarity']:.3f}")
            print(f"   Sample result length: {len(results[0]['chunk'])} characters")
        
        print("\n=== STM32F446RE Model Loading Test PASSED ===")
        print("✓ Existing vector store loaded successfully")
        print("✓ Model components are working correctly")
        print("✓ Ready for STM32F446RE documentation queries")
        
    except Exception as e:
        print(f"\n✗ Error during model loading: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Model loading failed. Please check your vector store and configuration.")
        return
    
    print("System initialized successfully!")
    print("Starting API server...")
    
    # Start API server
    from src.api.app import create_app
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=CONFIG['api']['host'], port=CONFIG['api']['port'])
    
    print("System initialized successfully!")
    print("Starting API server...")
    
    # Create and run the FastAPI application
    app = create_app()
    
    import uvicorn
    uvicorn.run(
        app, 
        host=CONFIG['api']['host'], 
        port=CONFIG['api']['port']
    )

if __name__ == "__main__":
    main()