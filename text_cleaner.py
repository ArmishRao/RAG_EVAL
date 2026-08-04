"""
Text cleaning utilities for legal documents
"""

import re
import unicodedata

class TextCleaner:
    """Clean and normalize text from various sources"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Main cleaning function"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal symbols
        text = re.sub(r'[^\w\s\.\,\-\'\”\“\—\:\;\(\)\&\@\#\$\%\*\?\!\+\=\/\°\©\®\™]', '', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Fix common issues
        text = text.replace('�', '')  # Replace unknown characters
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_sections(text: str) -> list:
        """Extract numbered sections from text"""
        # Pattern for Section numbers (e.g., "Section 379", "Sec. 379", "S. 379")
        patterns = [
            r'(?i)section\s+(\d+[A-Za-z]?)',
            r'(?i)sec\.?\s+(\d+[A-Za-z]?)',
            r'(?i)s\.\s+(\d+[A-Za-z]?)',
        ]
        
        sections = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            sections.extend(matches)
        
        return list(set(sections))  # Remove duplicates
    
    @staticmethod
    def extract_legal_terms(text: str) -> dict:
        """Extract common legal terms"""
        terms = {
            'acts': [],
            'sections': [],
            'punishments': [],
            'definitions': [],
        }
        
        # Extract definitions
        def_patterns = [
            r'(?i)definition:?\s*([^\.]+\.)',
            r'(?i)"([^"]+)"\s*means',
            r'(?i)means\s*([^\.]+\.)',
        ]
        
        for pattern in def_patterns:
            matches = re.findall(pattern, text)
            terms['definitions'].extend(matches)
        
        # Extract punishments
        pun_patterns = [
            r'(?i)punishment:?\s*([^\.]+\.)',
            r'(?i)penalty:?\s*([^\.]+\.)',
            r'(?i)imprisonment\s*([^\.]+\.)',
        ]
        
        for pattern in pun_patterns:
            matches = re.findall(pattern, text)
            terms['punishments'].extend(matches)
        
        return terms
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks