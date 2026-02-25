"""
Citation Formatter Utility
Formats citations for retrieved documents in a standardized way
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CitationFormatter:
    """
    Formats citations for retrieved documents according to academic and technical standards
    """
    
    def __init__(self):
        self.citation_styles = {
            'technical': self._format_technical_citation,
            'academic': self._format_academic_citation,
            'minimal': self._format_minimal_citation
        }
    
    def format_citations(self, docs: List[Dict[str, Any]], query: str = "", 
                        answer: str = "", style: str = 'technical') -> List[Dict[str, Any]]:
        """
        Format citations for a list of documents
        
        Args:
            docs: Retrieved documents
            query: Original query (for relevance assessment)
            answer: Generated answer (for context)
            style: Citation style to use
            
        Returns:
            List[Dict[str, Any]]: Formatted citations
        """
        if style not in self.citation_styles:
            raise ValueError(f"Unknown citation style: {style}")
        
        formatter = self.citation_styles[style]
        citations = []
        
        for i, doc in enumerate(docs):
            citation = formatter(doc, i+1)
            citation['relevance_score'] = self._calculate_relevance_score(doc, query, answer)
            citations.append(citation)
        
        # Sort by relevance score (descending)
        citations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Formatted {len(citations)} citations in '{style}' style")
        return citations
    
    def _format_technical_citation(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Format citation in technical documentation style
        
        Args:
            doc: Document to cite
            index: Index of the document
            
        Returns:
            Dict[str, Any]: Formatted citation
        """
        metadata = doc.get('metadata', {})
        
        return {
            'id': index,
            'formatted': (
                f"STM32F407 Reference Manual, Section: {metadata.get('section_header', 'N/A')}, "
                f"Pages: {', '.join(map(str, metadata.get('pages', [])))}, "
                f"Type: {metadata.get('section_type', 'N/A')}"
            ),
            'section_header': metadata.get('section_header', 'N/A'),
            'pages': metadata.get('pages', []),
            'section_type': metadata.get('section_type', 'N/A'),
            'token_count': metadata.get('token_count', 0),
            'chunk_id': metadata.get('chunk_id', ''),
            'similarity': doc.get('similarity', 0.0)
        }
    
    def _format_academic_citation(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Format citation in academic style
        
        Args:
            doc: Document to cite
            index: Index of the document
            
        Returns:
            Dict[str, Any]: Formatted citation
        """
        metadata = doc.get('metadata', {})
        
        return {
            'id': index,
            'formatted': (
                f"[{index}] STMicroelectronics. STM32F407 Reference Manual. "
                f"Section {metadata.get('section_header', 'N/A')}. "
                f"Pages {', '.join(map(str, metadata.get('pages', [])))}."
            ),
            'author': 'STMicroelectronics',
            'title': 'STM32F407 Reference Manual',
            'section': metadata.get('section_header', 'N/A'),
            'pages': metadata.get('pages', []),
            'year': '2023'  # Assuming current year for demo
        }
    
    def _format_minimal_citation(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Format minimal citation with essential information
        
        Args:
            doc: Document to cite
            index: Index of the document
            
        Returns:
            Dict[str, Any]: Formatted citation
        """
        metadata = doc.get('metadata', {})
        
        return {
            'id': index,
            'formatted': f"Section: {metadata.get('section_header', 'N/A')}, Pages: {', '.join(map(str, metadata.get('pages', [])))}",
            'section': metadata.get('section_header', 'N/A'),
            'pages': metadata.get('pages', [])
        }
    
    def _calculate_relevance_score(self, doc: Dict[str, Any], query: str, answer: str) -> float:
        """
        Calculate relevance score for a document based on query and answer
        
        Args:
            doc: Document to score
            query: Original query
            answer: Generated answer
            
        Returns:
            float: Relevance score between 0 and 1
        """
        # Use the similarity score from the retrieval as base
        base_score = doc.get('similarity', 0.0)
        
        # Boost score if document content appears in the answer
        content = doc.get('chunk', '').lower()
        answer_lower = answer.lower()
        
        # Count overlapping terms between document and answer
        content_words = set(content.split())
        answer_words = set(answer_lower.split())
        overlap = len(content_words.intersection(answer_words))
        
        # Normalize overlap score
        overlap_score = min(overlap / max(len(content_words), 1), 1.0)
        
        # Weighted combination of base similarity and overlap
        final_score = 0.7 * base_score + 0.3 * overlap_score
        
        return min(final_score, 1.0)  # Ensure score is between 0 and 1


class CitationManager:
    """
    Manages collections of citations and provides utilities for citation handling
    """
    
    def __init__(self):
        self.formatter = CitationFormatter()
        self.citation_history = []
    
    def add_citations(self, citations: List[Dict[str, Any]]):
        """
        Add citations to the history
        
        Args:
            citations: List of citations to add
        """
        self.citation_history.extend(citations)
        logger.info(f"Added {len(citations)} citations to history")
    
    def get_top_citations(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N most relevant citations from history
        
        Args:
            n: Number of citations to return
            
        Returns:
            List[Dict[str, Any]]: Top N citations
        """
        # Sort by relevance score and return top N
        sorted_citations = sorted(
            self.citation_history, 
            key=lambda x: x.get('relevance_score', 0), 
            reverse=True
        )
        return sorted_citations[:n]
    
    def export_citations(self, filename: str, style: str = 'technical'):
        """
        Export citations to a file
        
        Args:
            filename: Output filename
            style: Citation style to use
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Citations for STM32F407 Documentation Query\n")
            f.write(f"Generated on: {__import__('datetime').datetime.now()}\n\n")
            
            for citation in self.citation_history:
                f.write(f"[{citation['id']}] {citation['formatted']}\n")
                f.write(f"  Relevance Score: {citation.get('relevance_score', 0):.2f}\n\n")
        
        logger.info(f"Exported {len(self.citation_history)} citations to {filename}")


# Example usage
if __name__ == "__main__":
    formatter = CitationFormatter()
    print("Citation formatter initialized")