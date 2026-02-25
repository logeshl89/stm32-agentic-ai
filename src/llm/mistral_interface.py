import os
from typing import List, Dict, Any, Optional
from mistralai import Mistral
from .base_interface import LLMInterface

class MistralLLMInterface(LLMInterface):
    """Mistral AI LLM interface implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        """
        Initialize Mistral LLM interface
        
        Args:
            api_key: Mistral API key (if None, will use MISTRAL_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not provided. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = Mistral(api_key=self.api_key)
    
    def generate(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate response using Mistral AI
        
        Args:
            prompt: The input prompt
            context: Additional context/retrieved documents
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Generated response text
        """
        try:
            # Build the complete prompt with context
            full_prompt = self._build_prompt(prompt, context)
            
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in STM32F446RE microcontroller documentation. Provide accurate, detailed technical information with specific addresses, register names, and technical specifications when available. Format your responses clearly with bullet points and technical details."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            
            # Generate response
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            
            return {
                'answer': response.choices[0].message.content,
                'raw_response': response.choices[0].message.content
            }
            
        except Exception as e:
            print(f"Error generating response with Mistral: {e}")
            return self._get_fallback_response(prompt)
    
    def _build_prompt(self, query: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Build a comprehensive prompt with context
        
        Args:
            query: User query
            context: Retrieved documents context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"User Query: {query}\n\n"
        
        if context:
            prompt += "Relevant Documentation Context:\n"
            for i, doc in enumerate(context[:3], 1):  # Limit to top 3 documents
                prompt += f"\nDocument {i}:\n"
                prompt += f"Content: {doc.get('chunk', 'N/A')}\n"
                prompt += f"Source: {doc.get('source', 'N/A')}\n"
                prompt += f"Page: {doc.get('page', 'N/A')}\n"
                prompt += "-" * 50 + "\n"
        
        prompt += "\nPlease provide a detailed technical answer based on the documentation above. Include specific technical details, addresses, register names, and configuration parameters when relevant. Format your response clearly with bullet points and technical specifications."
        
        return prompt
    
    def _get_fallback_response(self, query: str) -> Dict[str, Any]:
        """
        Provide fallback response when API fails
        
        Args:
            query: User query
            
        Returns:
            Fallback response
        """
        fallback_response = f"I apologize, but I'm currently unable to access the Mistral AI service to provide a detailed response to your query: '{query}'. Please check your API key configuration or try again later. In the meantime, you can refer to the STM32F446RE reference manual for detailed technical information."
        return {
            'answer': fallback_response,
            'raw_response': fallback_response
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Mistral models
        
        Returns:
            List of model names
        """
        return [
            "mistral-large-latest",
            "mistral-small-latest",
            "mistral-medium-latest",
            "open-mistral-nemo",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b"
        ]