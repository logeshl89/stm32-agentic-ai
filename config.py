"""
Configuration file for STM32F446RE Agentic RAG System
Contains all configuration parameters and settings
"""

import os
from typing import Dict, Any

# Paths and file locations
class Paths:
    DATASET_PATH = "src/dataset/"
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store_f446re.pkl")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/app.log")

# Model and embedding settings
class ModelConfig:
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "800"))
    CHUNK_OVERLAP_SIZE = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# API settings
class APISettings:
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Confidence and retrieval settings
class ConfidenceConfig:
    HIGH_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
    MEDIUM_THRESHOLD = float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "0.5"))
    LOW_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))
    DEFAULT_MIN_CONFIDENCE = float(os.getenv("DEFAULT_MIN_CONFIDENCE", "0.3"))

class RetrievalConfig:
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "True").lower() == "true"
    ALPHA_BLEND = float(os.getenv("ALPHA_BLEND", "0.5"))  # For hybrid search

# System configuration
class SystemConfig:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))

# Default configuration dictionary
CONFIG: Dict[str, Any] = {
    "paths": {
        "dataset_path": Paths.DATASET_PATH,
        "vector_store_path": Paths.VECTOR_STORE_PATH,
        "log_file_path": Paths.LOG_FILE_PATH
    },
    "model": {
        "embedding_model_name": ModelConfig.EMBEDDING_MODEL_NAME,
        "chunk_max_size": ModelConfig.CHUNK_MAX_SIZE,
        "chunk_overlap_size": ModelConfig.CHUNK_OVERLAP_SIZE,
        "batch_size": ModelConfig.BATCH_SIZE
    },
    "api": {
        "host": APISettings.HOST,
        "port": APISettings.PORT,
        "debug": APISettings.DEBUG,
        "cors_origins": APISettings.CORS_ORIGINS
    },
    "confidence": {
        "high_threshold": ConfidenceConfig.HIGH_THRESHOLD,
        "medium_threshold": ConfidenceConfig.MEDIUM_THRESHOLD,
        "low_threshold": ConfidenceConfig.LOW_THRESHOLD,
        "default_min_confidence": ConfidenceConfig.DEFAULT_MIN_CONFIDENCE
    },
    "retrieval": {
        "default_top_k": RetrievalConfig.DEFAULT_TOP_K,
        "max_top_k": RetrievalConfig.MAX_TOP_K,
        "rerank_enabled": RetrievalConfig.RERANK_ENABLED,
        "alpha_blend": RetrievalConfig.ALPHA_BLEND
    },
    "system": {
        "log_level": SystemConfig.LOG_LEVEL,
        "max_workers": SystemConfig.MAX_WORKERS,
        "enable_cache": SystemConfig.ENABLE_CACHE,
        "cache_size": SystemConfig.CACHE_SIZE
    }
}


def get_config_value(config_path: str, default=None):
    """
    Get a configuration value using dot notation (e.g., 'model.embedding_model_name')
    
    Args:
        config_path: Path to the config value using dot notation
        default: Default value if config path doesn't exist
        
    Returns:
        Configured value or default
    """
    keys = config_path.split('.')
    value = CONFIG
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def update_config_value(config_path: str, new_value):
    """
    Update a configuration value using dot notation
    
    Args:
        config_path: Path to the config value using dot notation
        new_value: New value to set
    """
    keys = config_path.split('.')
    config_ref = CONFIG
    
    for key in keys[:-1]:
        if key in config_ref and isinstance(config_ref[key], dict):
            config_ref = config_ref[key]
        else:
            raise KeyError(f"Configuration path {config_path} not found")
    
    final_key = keys[-1]
    if final_key in config_ref:
        config_ref[final_key] = new_value
    else:
        raise KeyError(f"Configuration key {final_key} not found in path {config_path}")


# Print configuration summary if run as main
if __name__ == "__main__":
    print("STM32F446RE Agentic RAG System Configuration:")
    print("=" * 50)
    
    for category, settings in CONFIG.items():
        print(f"\n{category.upper()}:")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {settings}")