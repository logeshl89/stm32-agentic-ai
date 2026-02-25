"""
Document Preprocessing Module for STM32F407 Datasheet
Handles cleaning and preprocessing of extracted text
"""
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """
    Class responsible for cleaning and preprocessing the extracted text
    from the STM32F407 datasheet while preserving technical formatting
    """
    
    def __init__(self):
        """Initialize the preprocessor with cleaning rules"""
        self.cleaning_rules = [
            self._remove_excessive_whitespace,
            self._fix_common_pdf_artifacts,
            self._preserve_technical_formatting,
            self._normalize_headers,
        ]
    
    def preprocess_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess all pages of the document
        
        Args:
            pages_data (List[Dict[str, Any]]): Raw pages data from the loader
            
        Returns:
            List[Dict[str, Any]]: Cleaned and preprocessed pages data
        """
        processed_pages = []
        
        for page_data in pages_data:
            processed_page = self.preprocess_single_page(page_data)
            processed_pages.append(processed_page)
            
        logger.info(f"Preprocessed {len(processed_pages)} pages")
        return processed_pages
    
    def preprocess_single_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single page of the document
        
        Args:
            page_data (Dict[str, Any]): Raw page data
            
        Returns:
            Dict[str, Any]: Cleaned and preprocessed page data
        """
        # Make a copy to avoid modifying original data
        processed_data = page_data.copy()
        
        # Apply cleaning rules
        original_text = processed_data['text']
        cleaned_text = original_text
        
        for rule in self.cleaning_rules:
            cleaned_text = rule(cleaned_text)
        
        processed_data['cleaned_text'] = cleaned_text
        processed_data['original_text'] = original_text
        
        return processed_data
    
    def _remove_excessive_whitespace(self, text: str) -> str:
        """
        Remove excessive whitespace while preserving intentional formatting
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with excessive whitespace removed
        """
        # Replace multiple spaces with single space, but preserve newlines
        # This preserves paragraph structure while removing excessive spacing
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize newlines - replace multiple consecutive newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace but preserve paragraph structure
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # Remove leading/trailing whitespace from the entire text
        text = text.strip()
        
        return text
    
    def _fix_common_pdf_artifacts(self, text: str) -> str:
        """
        Fix common artifacts from PDF extraction
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with PDF artifacts fixed
        """
        # Fix hyphenated words that were split across lines
        text = re.sub(r'-\n([a-zA-Z])', r'\1', text)
        
        # Fix spaces that were incorrectly added around hyphens
        text = re.sub(r'\s*-\s*', '-', text)
        
        # Remove page numbers at the beginning or end of pages
        text = re.sub(r'^\s*\d+\s*\n', '', text)  # Page number at beginning
        text = re.sub(r'\n\s*\d+\s*$', '', text)  # Page number at end
        
        # Remove header/footer artifacts (simple heuristics)
        # This removes lines that look like headers/footers
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip lines that appear to be headers/footers
            if (len(line) < 50 and 
                ('www.st.com' in line.lower() or 
                 'stm32' in line.lower() or 
                 'reference manual' in line.lower() or
                 'docid' in line.lower() or
                 'page' in line.lower())):
                continue
            filtered_lines.append(line)
        
        text = '\n'.join(filtered_lines)
        
        return text
    
    def _preserve_technical_formatting(self, text: str) -> str:
        """
        Preserve important technical formatting in STM32F407 documentation
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with technical formatting preserved
        """
        # Preserve register names and addresses
        # Common STM32 register patterns
        text = re.sub(r'([A-Z0-9_]+_R[A-Z0-9_]*)', r'\1', text)  # Register names
        
        # Preserve hex values
        text = re.sub(r'(0x[0-9A-Fa-f]+)', r'\1', text)
        
        # Preserve bit field notations
        text = re.sub(r'(\[[0-9:]+\])', r'\1', text)
        
        # Preserve pin names
        text = re.sub(r'([PA-Z][0-9]+)', r'\1', text)
        
        # Preserve timing values and units
        text = re.sub(r'([0-9.]+(?:ns|us|ms|s|Hz|kHz|MHz|kΩ|Ω))', r'\1', text)
        
        return text
    
    def _normalize_headers(self, text: str) -> str:
        """
        Normalize headers in the document
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized headers
        """
        # Identify and normalize headers
        # STM32F407 RM typically has headers like "6.4.1 GPIOx_MODER"
        lines = text.split('\n')
        normalized_lines = []
        
        for i, line in enumerate(lines):
            # Check if this looks like a section header
            if re.match(r'^\d+(\.\d+)*\s+.+', line):
                # This is likely a section header
                # Normalize the spacing
                normalized_header = re.sub(r'\s+', ' ', line).strip()
                normalized_lines.append(normalized_header)
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract sections from the text based on header patterns
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, str]]: List of sections with headers and content
        """
        sections = []
        lines = text.split('\n')
        
        current_section = {'header': '', 'content': ''}
        
        for line in lines:
            # Check if this line is a header
            if re.match(r'^\d+(\.\d+)*\s+.+', line):
                # Save the previous section if it exists
                if current_section['header']:
                    sections.append(current_section.copy())
                
                # Start a new section
                current_section['header'] = line.strip()
                current_section['content'] = ''
            else:
                # Add to current section content
                current_section['content'] += line + '\n'
        
        # Add the last section
        if current_section['header']:
            sections.append(current_section)
        
        return sections


# Example usage
if __name__ == "__main__":
    preprocessor = DocumentPreprocessor()
    print("Document preprocessor initialized")