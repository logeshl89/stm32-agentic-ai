from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    def generate(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM
        
        Args:
            prompt: Input prompt
            context: Additional context/retrieved documents
            **kwargs: Additional parameters
            
        Returns:
            Dict containing 'answer' and 'raw_response' keys
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of model names
        """
        pass