"""
Document Loader Module for STM32F407 Datasheet
Handles loading and initial parsing of the PDF document using pdfplumber
"""
import pdfplumber
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Class responsible for loading and extracting text and tables from the STM32F407 datasheet
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the document loader with the path to the STM32F407 datasheet
        
        Args:
            pdf_path (str): Path to the STM32F407 PDF datasheet
        """
        self.pdf_path = pdf_path
        self.pdf = None
        
    def load_document(self) -> bool:
        """
        Load the PDF document using pdfplumber
        
        Returns:
            bool: True if document loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.pdf_path):
                logger.error(f"PDF file not found: {self.pdf_path}")
                return False
            
            self.pdf = pdfplumber.open(self.pdf_path)
            logger.info(f"Successfully loaded document with pdfplumber: {self.pdf_path}")
            logger.info(f"Number of pages: {len(self.pdf.pages)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load document {self.pdf_path}: {str(e)}")
            return False
    
    def extract_text_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Extract text and tables from all pages along with metadata
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing text, tables and metadata
        """
        if self.pdf is None:
            raise ValueError("Document not loaded. Call load_document() first.")
            
        pages_data = []
        
        for page in self.pdf.pages:
            # Extract high-fidelity text
            text = page.extract_text() or ""
            
            # Extract tables with high-fidelity
            tables = []
            try:
                extracted_tables = page.extract_tables()
                if extracted_tables:
                    for table in extracted_tables:
                        # Clean table data
                        clean_table = [[(cell or "").strip() for cell in row] for row in table]
                        tables.append({
                            'content': clean_table,
                            'rows': len(clean_table),
                            'cols': len(clean_table[0]) if clean_table else 0
                        })
            except Exception as e:
                logger.warning(f"Failed to extract tables on page {page.page_number}: {e}")
            
            page_data = {
                'page_number': page.page_number,
                'text': text,
                'tables_count': len(tables),
                'table_details': tables,
                'has_registers': self._detect_register_formatting(text),
                'has_memory_maps': self._detect_memory_map_formatting(text)
            }
            
            pages_data.append(page_data)
            
        return pages_data
    
    def _detect_register_formatting(self, text: str) -> bool:
        """Detect if the text contains register definitions"""
        import re
        patterns = [
            r'[A-Z0-9_]+_R[A-Z0-9_]*', 
            r'0x[0-9A-Fa-f]{2,8}',      
            r'[0-9]+\s*bit',            
            r'RW|RO|WO',                
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _detect_memory_map_formatting(self, text: str) -> bool:
        """Detect if the text contains memory map information"""
        import re
        patterns = [
            r'memory\s+map',
            r'address\s+map',
            r'0x[0-9A-Fa-f]{8}\s*-\s*0x[0-9A-Fa-f]{8}',
            r'(SRAM|Flash|ROM|RAM).*0x[0-9A-Fa-f]{8}'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def close_document(self):
        """Close the document and free resources"""
        if self.pdf is not None:
            self.pdf.close()
            self.pdf = None
            logger.info("Document closed successfully")

if __name__ == "__main__":
    loader = DocumentLoader("dm00037051.pdf")
    if loader.load_document():
        pages = loader.extract_text_with_metadata()
        print(f"Extracted {len(pages)} pages")
        loader.close_document()
