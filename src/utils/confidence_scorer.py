"""
Confidence Scorer Utility
Computes confidence scores for generated answers based on various factors
"""
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """
    Computes confidence scores for generated answers based on multiple factors
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the confidence scorer
        
        Args:
            weights: Weights for different confidence factors
        """
        if weights is None:
            # Default weights for different confidence factors
            self.weights = {
                'similarity': 0.3,        # Average similarity of retrieved docs
                'coverage': 0.2,          # How much of query is addressed
                'consistency': 0.2,       # Consistency among retrieved docs
                'source_quality': 0.15,   # Quality of source documents
                'answer_precision': 0.15  # Precision of the generated answer
            }
        else:
            self.weights = weights
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def compute_confidence(self, answer: str, query: str, 
                          retrieved_docs: List[Dict[str, Any]]) -> float:
        """
        Compute overall confidence score for an answer
        
        Args:
            answer: Generated answer
            query: Original query
            retrieved_docs: Retrieved documents used to generate answer
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        # Calculate individual confidence factors
        similarity_score = self._calculate_similarity_confidence(retrieved_docs)
        coverage_score = self._calculate_coverage_confidence(query, retrieved_docs)
        consistency_score = self._calculate_consistency_confidence(retrieved_docs)
        source_quality_score = self._calculate_source_quality_confidence(retrieved_docs)
        answer_precision_score = self._calculate_answer_precision_confidence(
            answer, query, retrieved_docs
        )
        
        # Weighted combination of all factors
        overall_confidence = (
            self.weights['similarity'] * similarity_score +
            self.weights['coverage'] * coverage_score +
            self.weights['consistency'] * consistency_score +
            self.weights['source_quality'] * source_quality_score +
            self.weights['answer_precision'] * answer_precision_score
        )
        
        # Ensure confidence is between 0 and 1
        final_confidence = max(0.0, min(1.0, overall_confidence))
        
        logger.debug(
            f"Confidence breakdown - Similarity: {similarity_score:.2f}, "
            f"Coverage: {coverage_score:.2f}, "
            f"Consistency: {consistency_score:.2f}, "
            f"Source Quality: {source_quality_score:.2f}, "
            f"Answer Precision: {answer_precision_score:.2f}, "
            f"Overall: {final_confidence:.2f}"
        )
        
        return final_confidence
    
    def _calculate_similarity_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on similarity scores of retrieved documents
        
        Args:
            docs: Retrieved documents
            
        Returns:
            float: Similarity-based confidence score
        """
        if not docs:
            return 0.0
        
        # Calculate average similarity
        similarities = [doc.get('similarity', 0.0) for doc in docs]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Apply sigmoid transformation to map to 0-1 range more meaningfully
        # High similarities should yield high confidence
        confidence = self._sigmoid(avg_similarity * 5 - 2)  # Center around 0.4 similarity
        
        return confidence
    
    def _calculate_coverage_confidence(self, query: str, docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on how well the documents cover the query
        
        Args:
            query: Original query
            docs: Retrieved documents
            
        Returns:
            float: Coverage-based confidence score
        """
        if not docs or not query:
            return 0.0
        
        query_terms = set(query.lower().split())
        
        # Count how many query terms appear in the retrieved documents
        doc_content = " ".join(doc.get('chunk', '') for doc in docs).lower()
        doc_terms = set(doc_content.split())
        
        covered_terms = query_terms.intersection(doc_terms)
        coverage_ratio = len(covered_terms) / max(len(query_terms), 1)
        
        # Apply transformation to emphasize good coverage
        confidence = min(1.0, coverage_ratio * 1.5)  # Allow some boost for good coverage
        
        return confidence
    
    def _calculate_consistency_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on consistency among retrieved documents
        
        Args:
            docs: Retrieved documents
            
        Returns:
            float: Consistency-based confidence score
        """
        if len(docs) < 2:
            # Can't assess consistency with only one document
            return 0.7  # Default medium-high confidence
        
        # For now, we'll use a simple metric based on similarity variance
        # In a more complex system, we might check content consistency
        similarities = [doc.get('similarity', 0.0) for doc in docs]
        
        # Lower variance in similarities suggests more consistent retrieval
        similarity_variance = np.var(similarities)
        
        # Transform variance to confidence (lower variance = higher confidence)
        # Using exponential decay: lower variance gives higher confidence
        consistency_confidence = np.exp(-similarity_variance * 10)
        
        return float(consistency_confidence)
    
    def _calculate_source_quality_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on quality of source documents
        
        Args:
            docs: Retrieved documents
            
        Returns:
            float: Source quality-based confidence score
        """
        if not docs:
            return 0.0
        
        # Evaluate source quality based on metadata
        quality_scores = []
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            score = 0.5  # Base score
            
            # Boost for important section types
            section_type = metadata.get('section_type', '')
            if section_type in ['register_description', 'peripheral_description']:
                score += 0.3
            elif section_type in ['memory_map', 'clock_tree']:
                score += 0.2
            elif section_type in ['interrupt_description']:
                score += 0.25
            
            # Boost for reasonable chunk size
            token_count = metadata.get('token_count', 0)
            if 200 <= token_count <= 1000:  # Good chunk size
                score += 0.1
            elif token_count > 1000:  # Large chunk, might contain relevant info
                score += 0.05
            
            # Boost for presence of technical terms
            content = doc.get('chunk', '')
            if self._has_technical_terms(content):
                score += 0.15
            
            quality_scores.append(min(score, 1.0))  # Cap at 1.0
        
        # Average quality score
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        return avg_quality
    
    def _calculate_answer_precision_confidence(self, answer: str, query: str, 
                                            docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on precision of the generated answer
        
        Args:
            answer: Generated answer
            query: Original query
            docs: Retrieved documents
            
        Returns:
            float: Answer precision-based confidence score
        """
        if not answer or not query:
            return 0.0
        
        # Check if answer directly addresses the query
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Calculate how much of the query intent is addressed
        query_entities = self._extract_entities(query_lower)
        answer_entities = self._extract_entities(answer_lower)
        
        if query_entities:
            entity_coverage = len(query_entities.intersection(answer_entities)) / len(query_entities)
        else:
            entity_coverage = 0.5  # Default if no entities found
        
        # Check for definitive language vs tentative language
        definitive_indicators = ['according to', 'the documentation states', 'specifically mentions']
        tentative_indicators = ['might', 'possibly', 'perhaps', 'could be']
        
        definitive_score = any(indicator in answer_lower for indicator in definitive_indicators)
        tentative_score = any(indicator in answer_lower for indicator in tentative_indicators)
        
        # Adjust based on language used
        if tentative_score and not definitive_score:
            entity_coverage *= 0.7  # Reduce confidence for tentative answers
        elif definitive_score and not tentative_score:
            entity_coverage *= 1.2  # Increase confidence for definitive answers
        
        return min(1.0, max(0.0, entity_coverage))
    
    def _has_technical_terms(self, text: str) -> bool:
        """
        Check if text contains technical terms typical of STM32 documentation
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if technical terms are present
        """
        import re
        
        # Patterns for technical terms in STM32 documentation
        tech_patterns = [
            r'\b[A-Z0-9_]{3,}\b',  # Register names
            r'\b0x[0-9A-Fa-f]{2,8}\b',  # Addresses
            r'\bP[A-Z][0-9]+\b',  # Pin names
            r'\b\d+MHz\b',  # Frequencies
            r'\b\d+kb\b',  # Memory sizes
        ]
        
        return any(re.search(pattern, text) for pattern in tech_patterns)
    
    def _extract_entities(self, text: str) -> set:
        """
        Extract entities from text (simplified approach)
        
        Args:
            text: Text to extract entities from
            
        Returns:
            set: Set of extracted entities
        """
        import re
        
        # Simple entity extraction based on patterns
        entities = set()
        
        # Technical terms
        tech_pattern = r'\b[A-Z][A-Z0-9_]+\b'
        tech_matches = re.findall(tech_pattern, text)
        entities.update(tech_matches)
        
        # Numbers and measurements
        number_pattern = r'\b\d+[\.]?\d*\s*(?:MHz|kHz|kb|mb|s|ms|us|ns|V|mA|Ω)?\b'
        number_matches = re.findall(number_pattern, text)
        entities.update(number_matches)
        
        # Pin names
        pin_pattern = r'\bP[A-Z][0-9]+\b'
        pin_matches = re.findall(pin_pattern, text)
        entities.update(pin_matches)
        
        return entities
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function
        
        Args:
            x: Input value
            
        Returns:
            float: Sigmoid-transformed value
        """
        return 1 / (1 + np.exp(-x))


class ConfidenceThreshold:
    """
    Manages confidence thresholds and decision-making based on confidence scores
    """
    
    def __init__(self, high_threshold: float = 0.8, 
                 medium_threshold: float = 0.5, low_threshold: float = 0.3):
        """
        Initialize confidence thresholds
        
        Args:
            high_threshold: Threshold for high confidence
            medium_threshold: Threshold for medium confidence
            low_threshold: Threshold for low confidence
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
    
    def categorize_confidence(self, score: float) -> str:
        """
        Categorize confidence score into qualitative level
        
        Args:
            score: Confidence score (0-1)
            
        Returns:
            str: Confidence category
        """
        if score >= self.high_threshold:
            return 'high'
        elif score >= self.medium_threshold:
            return 'medium'
        elif score >= self.low_threshold:
            return 'low'
        else:
            return 'very_low'
    
    def should_respond(self, score: float, required_level: str = 'medium') -> bool:
        """
        Determine if response should be provided based on confidence
        
        Args:
            score: Confidence score
            required_level: Minimum required confidence level
            
        Returns:
            bool: True if response should be provided
        """
        categories = {
            'very_low': 0,
            'low': 1, 
            'medium': 2,
            'high': 3
        }
        
        score_category = self.categorize_confidence(score)
        required_category = categories.get(required_level, categories['medium'])
        actual_category = categories[score_category]
        
        return actual_category >= required_category


# Example usage
if __name__ == "__main__":
    scorer = ConfidenceScorer()
    threshold = ConfidenceThreshold()
    
    print("Confidence scorer initialized")
    print(f"Default thresholds - High: {threshold.high_threshold}, Medium: {threshold.medium_threshold}, Low: {threshold.low_threshold}")