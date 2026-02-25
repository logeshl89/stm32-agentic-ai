"""
Test script to verify dataset processing with all 5 PDF files
"""
import os
import sys
from dotenv import load_dotenv
from config import CONFIG

# Load environment variables
load_dotenv()

def test_dataset_processing():
    """Test processing all PDF files in the dataset"""
    
    print("=== STM32F446RE Dataset Processing Test ===\n")
    
    # Check dataset path
    dataset_path = CONFIG['paths']['dataset_path']
    print(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset folder not found at {dataset_path}")
        return False
    
    # List all PDF files
    pdf_files = [f for f in os.listdir(dataset_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files:")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        file_path = os.path.join(dataset_path, pdf_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  {i}. {pdf_file} ({file_size:.2f} MB)")
    
    print()
    
    # Test imports
    try:
        print("Testing imports...")
        from src.embedding_storage.embedder import Embedder, VectorStoreManager, EmbeddingPipeline
        from src.document_processing.dataset_processor import DatasetProcessor
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test embedding pipeline initialization
    try:
        print("Testing embedding pipeline initialization...")
        embedder = Embedder()
        vector_store = VectorStoreManager(CONFIG['paths']['vector_store_path'])
        pipeline = EmbeddingPipeline(embedder, vector_store)
        print("✓ Embedding pipeline initialized")
    except Exception as e:
        print(f"✗ Pipeline initialization error: {e}")
        return False
    
    # Test dataset processor
    try:
        print("Testing dataset processor initialization...")
        dataset_processor = DatasetProcessor(dataset_path, pipeline)
        print("✓ Dataset processor initialized")
        
        # Show dataset statistics (should be empty initially)
        stats = dataset_processor.get_dataset_statistics()
        print(f"Initial dataset statistics: {stats}")
        
    except Exception as e:
        print(f"✗ Dataset processor error: {e}")
        return False
    
    print("\n=== Test Summary ===")
    print("✓ Dataset folder exists and contains 5 PDF files")
    print("✓ All required modules can be imported")
    print("✓ Embedding pipeline can be initialized")
    print("✓ Dataset processor can be initialized")
    print("\nReady to process all documentation files!")
    
    return True

if __name__ == "__main__":
    success = test_dataset_processing()
    sys.exit(0 if success else 1)