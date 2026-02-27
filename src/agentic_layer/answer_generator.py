"""
Answer Generation Module with Citation System
Generates precise answers based on retrieved context with proper citations
"""
from typing import List, Dict, Any, Optional
import logging
import re
import os
from src.utils.citation_formatter import CitationFormatter
from src.utils.confidence_scorer import ConfidenceScorer
from src.llm.mistral_interface import MistralLLMInterface

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates accurate answers from retrieved context with proper citations
    Ensures no hallucination and maintains technical precision
    """
    
    def __init__(self, llm_interface=None):
        """
        Initialize the answer generator
        
        Args:
            llm_interface: Interface to the language model
        """
        if llm_interface is None:
            # Try to use Mistral if API key is available
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if mistral_api_key:
                try:
                    self.llm_interface = MistralLLMInterface(api_key=mistral_api_key)
                    logger.info("Using Mistral AI for answer generation")
                except Exception as e:
                    logger.warning(f"Failed to initialize Mistral: {e}. Using mock LLM.")
                    self.llm_interface = MockLLMInterface()
            else:
                self.llm_interface = MockLLMInterface()
        else:
            self.llm_interface = llm_interface
        self.citation_formatter = CitationFormatter()
        self.confidence_scorer = ConfidenceScorer()
        self.response_validator = ResponseValidator()
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                       query_type: str = "general") -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved documents
        
        Args:
            query (str): User query
            retrieved_docs (List[Dict[str, Any]]): Retrieved documents
            query_type (str): Type of query for tailored response
            
        Returns:
            Dict[str, Any]: Generated answer with citations and metadata
        """
        if not retrieved_docs:
            return self._generate_no_result_answer(query)
        
        # Validate retrieved documents
        valid_docs = self._validate_documents(retrieved_docs)
        
        if not valid_docs:
            return self._generate_no_result_answer(query)
        
        # Prepare context
        context = self._prepare_context(valid_docs)
        
        # Create prompt
        prompt = self._create_prompt(query, context, query_type)
        
        # Generate response
        raw_response = self.llm_interface.generate(prompt)
        
        # Process and validate the response
        processed_response = self.response_validator.validate_response(
            raw_response, valid_docs, query
        )
        
        # Extract answer, citations, and compute confidence
        answer = processed_response.get('answer', '')
        
        # Enhance formatting of the answer using Mistral for final presentation
        formatted_answer = self._enhance_answer_formatting(answer, query, valid_docs)
        
        confidence = self.confidence_scorer.compute_confidence(
            formatted_answer, query, valid_docs
        )

        confidence = self._calibrate_confidence(
            confidence,
            valid_docs,
            processed_response.get('validation_passed', True)
        )

        citations = self.citation_formatter.format_citations(
            valid_docs, query, formatted_answer
        )
        
        # Generate follow-up questions
        followup_questions = self._generate_followup_questions(
            query, formatted_answer, valid_docs
        )
        
        return {
            'answer': formatted_answer,
            'confidence': confidence,
            'citations': citations,
            'retrieved_docs_used': len([d for d in valid_docs if d['used_in_response']]),
            'followup_questions': followup_questions,
            'validation_passed': processed_response.get('validation_passed', True)
        }
    
    def _validate_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate retrieved documents for relevance and quality
        
        Args:
            docs: Retrieved documents to validate
            
        Returns:
            List[Dict[str, Any]]: Validated documents with relevance flags
        """
        validated_docs = []
        
        for doc in docs:
            # Check if document has sufficient content
            content = doc.get('chunk', '')
            if len(content.strip()) < 50:  # Minimum content threshold
                continue
            
            # Check if document has relevant metadata
            metadata = doc.get('metadata', {})
            if not metadata:
                continue
            
            # Mark as initially unused
            validated_doc = doc.copy()
            validated_doc['used_in_response'] = False
            validated_docs.append(validated_doc)
        
        logger.info(f"Validated {len(validated_docs)} out of {len(docs)} documents")
        return validated_docs
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from validated documents
        
        Args:
            docs: Validated documents
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            chunk_content = doc['chunk']
            metadata = doc['metadata']
            
            # Create document header with metadata
            doc_header = f"\n--- Document [{i}] ---\n"
            doc_header += f"Section: {metadata.get('section_header', 'N/A')}\n"
            doc_header += f"Page(s): {', '.join(map(str, metadata.get('pages', [])))}\n"
            doc_header += f"Type: {metadata.get('section_type', 'N/A')}\n"
            doc_header += f"Token Count: {metadata.get('token_count', 'N/A')}\n"
            
            # Truncate very long content to prevent prompt overflow
            max_content_len = 2000
            if len(chunk_content) > max_content_len:
                chunk_content = chunk_content[:max_content_len] + "... [truncated]"
            
            context_part = f"{doc_header}\nContent:\n{chunk_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, query_type: str) -> str:
        """
        Create a prompt for the LLM based on query and context
        
        Args:
            query: User query
            context: Retrieved context
            query_type: Type of query
            
        Returns:
            str: Formatted prompt
        """
        prompt_template = """You are an expert STM32F446RE microcontroller assistant. Answer the user query based ONLY on the provided documentation context. Do not hallucinate or invent information.

STRICT RULES:
1. Answer ONLY from provided context
2. If answer not in context, clearly state this
3. Maintain technical precision
4. Preserve register names, addresses, and specifications exactly as written
5. Do not make assumptions beyond provided information

Documentation Context:
{context}

User Query: {query}

Query Type: {query_type}

Format your response with detailed technical precision:
- Use clear headings and bullet points
- Include specific register names, addresses (in 0x format), bit fields, and configuration values
- Format code examples with proper syntax
- Provide page numbers and section references when available
- Structure response logically with technical accuracy
- Focus on STM32F446RE specific features and capabilities

Response:"""
        
        return prompt_template.format(
            context=context,
            query=query,
            query_type=query_type
        )
    
    def _calibrate_confidence(self, base_confidence: float, docs: List[Dict[str, Any]], validation_passed: bool) -> float:
        """Calibrate confidence using retrieval margin and validation signal."""
        if not docs:
            return 0.0

        sorted_scores = sorted([d.get('similarity', 0.0) for d in docs], reverse=True)
        top_score = sorted_scores[0] if sorted_scores else 0.0
        margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else top_score

        calibrated = 0.7 * base_confidence + 0.2 * top_score + 0.1 * max(0.0, margin)
        if not validation_passed:
            calibrated *= 0.7

        return max(0.0, min(1.0, calibrated))

    def _generate_no_result_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer when no relevant documents are found
        
        Args:
            query: User query
            
        Returns:
            Dict[str, Any]: Answer indicating no results found
        """
        return {
            'answer': (
                "I couldn't find relevant information in the STM32F446RE documentation "
                "to answer your query: '{}'. The documentation may not contain this "
                "specific information, or the query terms might not match the "
                "documentation terminology. Please try rephrasing your question or "
                "consult the official STMicroelectronics documentation directly."
            ).format(query),
            'confidence': 0.0,
            'citations': [],
            'retrieved_docs_used': 0,
            'followup_questions': [
                "Could you rephrase your question?",
                "Would you like me to search for related topics?",
                "Do you have a specific section of the documentation in mind?"
            ],
            'validation_passed': True
        }
    
    def _enhance_answer_formatting(self, answer: str, query: str, 
                                 docs: List[Dict[str, Any]]) -> str:
        """
        Enhance the answer formatting using Mistral AI for final presentation
        
        Args:
            answer: Raw answer from LLM
            query: Original query
            docs: Retrieved documents
            
        Returns:
            str: Formatted answer with improved structure and presentation
        """
        # Create a formatting prompt for Mistral to enhance the answer
        formatting_prompt = f"""Please rewrite the following answer to be short, clean, and developer-friendly for STM32F446RE.

Rules:
- Use short bullet points or short paragraph
- Remove headings
- Remove technical report style
- Do NOT add new information
- Keep all values intact

Original Answer:
{answer}

Rewritten Answer:
"""
        
        try:
            # Use Mistral to format the answer if available
            if hasattr(self.llm_interface, 'generate'):
                formatted_response = self.llm_interface.generate(formatting_prompt)
                return formatted_response.get('answer', answer)
            else:
                return answer
        except Exception as e:
            logger.warning(f"Failed to enhance answer formatting: {e}")
            return answer

    def _generate_followup_questions(self, query: str, answer: str, 
                                   docs: List[Dict[str, Any]]) -> List[str]:
        """
        Generate relevant follow-up questions
        
        Args:
            query: Original query
            answer: Generated answer
            docs: Used documents
            
        Returns:
            List[str]: Generated follow-up questions
        """
        followup_generator = FollowUpQuestionGenerator()
        return followup_generator.generate(query, answer, docs)


class ResponseValidator:
    """
    Validates responses to ensure they adhere to constraints and don't hallucinate
    """
    
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
    
    def validate_response(self, raw_response: Dict[str, Any], 
                        docs: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Validate a raw response for accuracy and adherence to constraints
        
        Args:
            raw_response: Raw response from LLM
            docs: Retrieved documents
            query: Original query
            
        Returns:
            Dict[str, Any]: Validated response
        """
        answer = raw_response.get('answer', '')
        
        # Check for hallucination
        hallucination_detected = self.hallucination_detector.detect(answer, docs)
        
        # Check if response acknowledges when information is not available
        no_info_acknowledged = self._check_no_info_acknowledgment(answer, query)
        
        # Determine if validation passed
        validation_passed = not hallucination_detected or no_info_acknowledged
        
        return {
            'answer': answer,
            'validation_passed': validation_passed,
            'hallucination_detected': hallucination_detected,
            'no_info_acknowledged': no_info_acknowledged
        }
    
    def _check_no_info_acknowledgment(self, answer: str, query: str) -> bool:
        """
        Check if the response appropriately acknowledges lack of information
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            bool: True if lack of info is acknowledged
        """
        no_info_phrases = [
            "not found in the documentation",
            "could not find information",
            "documentation does not contain",
            "not mentioned in the provided context",
            "unable to find relevant information"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in no_info_phrases)


class HallucinationDetector:
    """
    Detects hallucinations in generated responses
    """
    
    def detect(self, answer: str, docs: List[Dict[str, Any]]) -> bool:
        """
        Detect if the answer contains hallucinated information
        
        Args:
            answer: Generated answer
            docs: Retrieved documents
            
        Returns:
            bool: True if hallucination detected
        """
        # Check for technical terms that appear in answer but not in docs
        answer_terms = self._extract_technical_terms(answer)
        
        doc_content = " ".join(doc['chunk'] for doc in docs)
        doc_terms = self._extract_technical_terms(doc_content)
        
        # Look for terms in answer that aren't in docs
        hallucinated_terms = []
        for term in answer_terms:
            if term not in doc_terms and self._is_significant_term(term):
                hallucinated_terms.append(term)
        
        # If we find significant terms not in docs, likely hallucination
        return len(hallucinated_terms) > 2  # Threshold for hallucination detection
    
    def _extract_technical_terms(self, text: str) -> set:
        """
        Extract technical terms from text
        
        Args:
            text: Input text
            
        Returns:
            set: Set of technical terms
        """
        # Patterns for technical terms in STM32F446 documentation
        patterns = [
            r'\b[A-Z0-9_]{2,}\b',  # Uppercase register/peripheral names
            r'\b0x[0-9A-Fa-f]{2,8}\b',  # Hex addresses
            r'\b\d+\.\d+\.\d+\b',  # Version numbers
            r'\bP[A-Z][0-9]+\b',  # Pin names
            r'\bF446\b',  # STM32F446 specific
            r'\bRE\b',  # Package type
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        
        return terms
    
    def _is_significant_term(self, term: str) -> bool:
        """
        Determine if a term is significant enough to indicate hallucination
        
        Args:
            term: Term to evaluate
            
        Returns:
            bool: True if term is significant
        """
        # Filter out common words that aren't technical terms
        common_words = {
            'THE', 'AND', 'OR', 'NOT', 'FOR', 'ARE', 'BUT', 'CAN', 'GET',
            'USE', 'SET', 'IF', 'ON', 'AT', 'TO', 'OF', 'IN', 'IS', 'IT'
        }
        
        return len(term) > 2 and term not in common_words


class FollowUpQuestionGenerator:
    """
    Generates relevant follow-up questions based on query and response
    """
    
    def generate(self, query: str, answer: str, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Generate follow-up questions
        
        Args:
            query: Original query
            answer: Generated answer
            docs: Used documents
            
        Returns:
            List[str]: Generated follow-up questions
        """
        followups = []
        
        # Generate based on query patterns
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['configure', 'setup', 'enable', 'initialize']):
            followups.extend([
                "What are the different configuration options available?",
                "Are there any prerequisites for this configuration?",
                "How do I verify that the configuration is working correctly?"
            ])
        elif any(word in query_lower for word in ['difference', 'compare', 'versus']):
            followups.extend([
                "What are the practical implications of these differences?",
                "In what scenarios would I choose one over the other?",
                "Are there performance differences between these options?"
            ])
        elif any(word in query_lower for word in ['how', 'procedure', 'steps']):
            followups.extend([
                "What error conditions should I watch for during this process?",
                "Are there alternative methods to achieve the same result?",
                "What are the common pitfalls when implementing this?"
            ])
        
        # Limit to top 3 follow-ups
        return followups[:3]


class MockLLMInterface:
    """
    Mock interface for the LLM - to be replaced with actual implementation
    """
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the LLM (mock implementation)
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict[str, Any]: Generated response
        """
        # In a real implementation, this would call an actual LLM API
        # For now, we return a mock response
        
        # Extract the actual query from the prompt
        lines = prompt.split('\n')
        query = ""
        for line in lines:
            if line.startswith('User Query:'):
                query = line.replace('User Query:', '').strip()
                break
        
        # Check if this is a formatting request
        if "Please reformat the following technical answer about STM32F445re microcontroller for better presentation:" in prompt:
            # Extract original answer from the prompt
            start_marker = "Original Answer: "
            end_marker = "\n\nRequirements:"
            start_idx = prompt.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = prompt.find(end_marker, start_idx)
                if end_idx != -1:
                    original_answer = prompt[start_idx:end_idx]
                    # Return the original answer with better formatting
                    formatted_answer = original_answer.replace("**", "\n**").replace("•", "\n•")
                    return {
                        'answer': formatted_answer,
                        'raw_response': formatted_answer
                    }
        
        # When no relevant documents are found, return appropriate response
        answer = f"I couldn't find relevant information in the STM32F446RE documentation to answer your query: '{query}'. The documentation may not contain this specific information, or the query terms might not match the documentation terminology. Please try rephrasing your question or consult the official STMicroelectronics documentation directly."
        
        return {
            'answer': answer,
            'raw_response': answer
        }


# Example usage
if __name__ == "__main__":
    generator = AnswerGenerator()
    print("Answer generator initialized")