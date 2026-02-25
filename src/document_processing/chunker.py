"""
Semantic Chunking Module for STM32F407 Datasheet
Implements header-aware and token-based chunking with overlap
"""
import re
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Class responsible for splitting the STM32F407 documentation into semantically meaningful chunks
    Uses a hybrid approach: header-based + token-based with overlap
    """
    
    def __init__(self, max_chunk_size: int = 800, overlap_size: int = 100):
        """
        Initialize the semantic chunker
        
        Args:
            max_chunk_size (int): Maximum number of tokens per chunk
            overlap_size (int): Number of overlapping tokens between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Define separators for the text splitter
        self.separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation marks
            "? ",    # Question marks
            ";",     # Semicolons
            ":",     # Colons
            " ",     # Spaces
            "",      # Characters
        ]
        
        # Header patterns for STM32F407 documentation
        self.header_patterns = [
            r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)$',  # Section headers like "6.4.1 GPIOx_MODER"
            r'^(\d+\.\d+)\s+(.+)$',           # Section headers like "6.4 Alternate function"
            r'^(\d+)\s+(.+)$',                # Chapter headers like "6 GPIO"
            r'^([A-Z0-9_]+_R[A-Z0-9_]*)$',    # Register names like "GPIO_MODER"
        ]
    
    def chunk_document(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk the entire document based on semantic boundaries
        
        Args:
            pages_data (List[Dict[str, Any]]): Processed pages data
            
        Returns:
            List[Dict[str, Any]]: List of semantic chunks with metadata
        """
        all_chunks = []
        
        # Combine all pages into a single text while preserving page information
        full_text_parts = []
        page_mapping = []  # Maps text position to page metadata
        
        for page_data in pages_data:
            start_pos = len(''.join(full_text_parts))
            text_to_add = page_data.get('cleaned_text', page_data.get('text', ''))
            full_text_parts.append(text_to_add)
            
            end_pos = start_pos + len(text_to_add)
            page_mapping.append({
                'start': start_pos,
                'end': end_pos,
                'page_number': page_data['page_number'],
                'has_registers': page_data.get('has_registers', False),
                'has_memory_maps': page_data.get('has_memory_maps', False),
            })
        
        full_text = '\n\n'.join(full_text_parts)
        
        # First, split by major headers to respect document structure
        sections = self._split_by_headers(full_text)
        
        # Then chunk each section using token-based approach
        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(section['content'])
            
            for chunk_idx, chunk_text in enumerate(section_chunks):
                # Find which page(s) this chunk belongs to
                page_nums = self._find_pages_for_chunk(
                    chunk_text, full_text, page_mapping
                )
                
                chunk_metadata = {
                    'chunk_id': f"{section_idx:03d}_{chunk_idx:03d}",
                    'section_header': section['header'],
                    'pages': page_nums,
                    'token_count': len(self.enc.encode(chunk_text)),
                    'char_count': len(chunk_text),
                    'section_type': self._classify_section_type(section['header'], chunk_text),
                }
                
                all_chunks.append({
                    'content': chunk_text,
                    'metadata': chunk_metadata
                })
        
        logger.info(f"Created {len(all_chunks)} semantic chunks from the document")
        return all_chunks
    
    def _split_by_headers(self, text: str) -> List[Dict[str, str]]:
        """
        Split text by major headers to preserve document structure
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[Dict[str, str]]: List of sections with headers and content
        """
        lines = text.split('\n')
        sections = []
        current_section = {'header': 'Introduction', 'content': ''}
        
        for line in lines:
            # Check if this line is a header
            header_found = False
            for pattern in self.header_patterns:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    # Save the current section if it exists
                    if current_section['content'].strip():
                        sections.append(current_section.copy())
                    
                    # Start a new section
                    current_section = {
                        'header': match.group(0).strip(),
                        'content': ''
                    }
                    header_found = True
                    break
            
            if not header_found:
                current_section['content'] += line + '\n'
        
        # Add the last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section_content: str) -> List[str]:
        """
        Chunk a section using token-based approach with overlap
        
        Args:
            section_content (str): Content of a section to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # Use RecursiveCharacterTextSplitter for robust chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=self._token_length,
            separators=self.separators,
        )
        
        chunks = text_splitter.split_text(section_content)
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def _token_length(self, text: str) -> int:
        """
        Calculate the number of tokens in a text string
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of tokens
        """
        return len(self.enc.encode(text))
    
    def _find_pages_for_chunk(self, chunk_text: str, full_text: str, page_mapping: List[Dict]) -> List[int]:
        """
        Find which pages a chunk belongs to based on position mapping
        
        Args:
            chunk_text (str): The chunk text
            full_text (str): Full document text
            page_mapping (List[Dict]): Mapping of text positions to pages
            
        Returns:
            List[int]: List of page numbers that the chunk belongs to
        """
        try:
            start_pos = full_text.index(chunk_text)
            end_pos = start_pos + len(chunk_text)
            
            pages = []
            for mapping in page_mapping:
                if (start_pos >= mapping['start'] and start_pos <= mapping['end']) or \
                   (end_pos >= mapping['start'] and end_pos <= mapping['end']) or \
                   (mapping['start'] >= start_pos and mapping['start'] <= end_pos):
                    pages.append(mapping['page_number'])
            
            return sorted(list(set(pages))) if pages else [1]  # Default to page 1 if not found
        except ValueError:
            # If the chunk text is not found in full text (shouldn't happen)
            return [1]
    
    def _classify_section_type(self, header: str, content: str) -> str:
        """
        Classify the type of section based on header and content
        
        Args:
            header (str): Section header
            content (str): Section content
            
        Returns:
            str: Section type classification
        """
        header_lower = header.lower()
        content_lower = content.lower()
        
        # Check for specific section types
        if any(keyword in header_lower or keyword in content_lower for keyword in 
               ['register', 'offset', 'bit', 'field', 'configuration']):
            return 'register_description'
        elif any(keyword in header_lower or keyword in content_lower for keyword in 
                 ['memory', 'map', 'address', 'sram', 'flash', 'rom']):
            return 'memory_map'
        elif any(keyword in header_lower or keyword in content_lower for keyword in 
                 ['interrupt', 'nvic', 'irq', 'vector']):
            return 'interrupt_description'
        elif any(keyword in header_lower or keyword in content_lower for keyword in 
                 ['clock', 'rcc', 'pll', 'frequency']):
            return 'clock_tree'
        elif any(keyword in header_lower or keyword in content_lower for keyword in 
                 ['peripheral', 'gpio', 'timer', 'uart', 'spi', 'i2c']):
            return 'peripheral_description'
        elif any(keyword in header_lower for keyword in 
                 ['chapter', 'section', 'overview', 'introduction']):
            return 'section_header'
        else:
            return 'general_content'


class OverlappingChunker:
    """
    Enhanced chunker that creates overlapping chunks to preserve context
    """
    
    def __init__(self, chunker: SemanticChunker):
        self.chunker = chunker
    
    def create_overlapping_chunks(self, pages_data: List[Dict[str, Any]], 
                                pre_overlap_ratio: float = 0.2, 
                                post_overlap_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks to preserve context across boundaries
        
        Args:
            pages_data (List[Dict[str, Any]]): Processed pages data
            pre_overlap_ratio (float): Ratio of chunk to use as preceding overlap
            post_overlap_ratio (float): Ratio of chunk to use as following overlap
            
        Returns:
            List[Dict[str, Any]]: List of chunks with overlaps
        """
        base_chunks = self.chunker.chunk_document(pages_data)
        
        enhanced_chunks = []
        for i, chunk in enumerate(base_chunks):
            # Calculate overlap sizes
            content_tokens = self.chunker._token_length(chunk['content'])
            pre_overlap_tokens = int(content_tokens * pre_overlap_ratio)
            post_overlap_tokens = int(content_tokens * post_overlap_ratio)
            
            # Get preceding context if available
            pre_context = ""
            if i > 0:
                prev_chunk = base_chunks[i-1]['content']
                prev_tokens = self.chunker.enc.encode(prev_chunk)
                pre_tokens = prev_tokens[-pre_overlap_tokens:] if len(prev_tokens) > pre_overlap_tokens else prev_tokens
                pre_context = self.chunker.enc.decode(pre_tokens)
            
            # Get following context if available
            post_context = ""
            if i < len(base_chunks) - 1:
                next_chunk = base_chunks[i+1]['content']
                next_tokens = self.chunker.enc.encode(next_chunk)
                post_tokens = next_tokens[:post_overlap_tokens] if len(next_tokens) > post_overlap_tokens else next_tokens
                post_context = self.chunker.enc.decode(post_tokens)
            
            # Create enhanced chunk with context
            enhanced_chunk = {
                'content': chunk['content'],
                'pre_context': pre_context,
                'post_context': post_context,
                'full_context': pre_context + chunk['content'] + post_context,
                'metadata': {
                    **chunk['metadata'],
                    'has_pre_context': bool(pre_context),
                    'has_post_context': bool(post_context)
                }
            }
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks


# Example usage
if __name__ == "__main__":
    chunker = SemanticChunker(max_chunk_size=500, overlap_size=50)
    print("Semantic chunker initialized")