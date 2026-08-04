"""
Data Loader for Pakistan Legal Advisor
Supports: CSV, JSON, JSONL, TXT, PDF, Excel, DOCX
"""

import os
import json
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handle loading data from multiple formats"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.supported_formats = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.jsonl': self._load_jsonl,
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.docx': self._load_docx,
        }
    
    def load_all(self) -> List[Document]:
        """Load all documents from all supported formats"""
        documents = []
        
        # Check if data folder exists
        if not os.path.exists(self.data_folder):
            logger.warning(f"Data folder '{self.data_folder}' not found")
            return documents
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in self.supported_formats:
                    logger.info(f"📄 Loading: {file_path}")
                    try:
                        docs = self.supported_formats[ext](file_path)
                        documents.extend(docs)
                        logger.info(f"   ✅ Added {len(docs)} documents from {file}")
                    except Exception as e:
                        logger.error(f"   ❌ Error loading {file}: {e}")
        
        return documents
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV files"""
        documents = []
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read CSV with any encoding")
            
            # Check for required columns (existing format)
            if all(col in df.columns for col in ["Book", "Section", "Heading", "Defination"]):
                for _, row in df.iterrows():
                    if pd.isna(row["Defination"]):
                        continue
                    
                    text = f"""
Book: {row['Book']}
Chapter: {row.get('Chapter Number', 'N/A')}
Section: {row['Section']}
Heading: {row['Heading']}
Definition: {row['Defination']}
"""
                    metadata = {
                        'book': str(row.get('Book', '')),
                        'section': str(row.get('Section', '')),
                        'heading': str(row.get('Heading', '')),
                        'source_file': os.path.basename(file_path),
                        'source_type': 'csv'
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
            else:
                # Generic CSV handling
                for _, row in df.iterrows():
                    # Combine all columns as text
                    content = " ".join([str(v) for v in row.values if pd.notna(v)])
                    if len(content.strip()) > 50:
                        metadata = {
                            'source_file': os.path.basename(file_path),
                            'source_type': 'csv',
                            **{col: str(row[col]) for col in df.columns}
                        }
                        documents.append(Document(page_content=content, metadata=metadata))
        
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
        
        return documents
    
    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON files"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    doc = self._process_json_item(item, file_path)
                    if doc:
                        documents.append(doc)
            elif isinstance(data, dict):
                # Try to find nested data
                for key in ['data', 'laws', 'documents', 'content']:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            doc = self._process_json_item(item, file_path)
                            if doc:
                                documents.append(doc)
                        break
                else:
                    # Single object
                    doc = self._process_json_item(data, file_path)
                    if doc:
                        documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
        
        return documents
    
    def _load_jsonl(self, file_path: str) -> List[Document]:
        """Load JSONL (JSON Lines) files"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            doc = self._process_json_item(item, file_path)
                            if doc:
                                documents.append(doc)
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Error loading JSONL {file_path}: {e}")
        
        return documents
    
    def _process_json_item(self, item: Dict, file_path: str) -> Optional[Document]:
        """Process a JSON item into a Document"""
        # Try to find content
        content = None
        content_keys = ['content', 'text', 'definition', 'defination', 'description', 'body']
        
        for key in content_keys:
            if key in item and item[key]:
                content = str(item[key])
                break
        
        if not content or len(content.strip()) < 50:
            return None
        
        # Build metadata
        metadata = {
            'source_file': os.path.basename(file_path),
            'source_type': 'json'
        }
        
        # Add all other fields as metadata
        for key, value in item.items():
            if key not in content_keys and value is not None:
                metadata[key] = str(value)[:500]  # Limit length
        
        return Document(page_content=content, metadata=metadata)
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load plain text files"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content.strip()) < 50:
                return documents
            
            # Try to split by sections
            sections = content.split('\n\n')  # Double newline
            
            if len(sections) > 3:
                # Multiple sections
                for i, section in enumerate(sections):
                    if len(section.strip()) > 100:
                        metadata = {
                            'source_file': os.path.basename(file_path),
                            'section_number': i + 1,
                            'source_type': 'text'
                        }
                        documents.append(Document(page_content=section, metadata=metadata))
            else:
                # Single document
                metadata = {
                    'source_file': os.path.basename(file_path),
                    'source_type': 'text'
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        except Exception as e:
            logger.error(f"Error loading text {file_path}: {e}")
        
        return documents
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF files"""
        documents = []
        
        try:
            # Try pdfplumber first (better)
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            metadata = {
                                'source_file': os.path.basename(file_path),
                                'page_number': page_num + 1,
                                'total_pages': len(pdf.pages),
                                'source_type': 'pdf'
                            }
                            documents.append(Document(page_content=text, metadata=metadata))
            except ImportError:
                # Fallback to PyPDF2
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            metadata = {
                                'source_file': os.path.basename(file_path),
                                'page_number': page_num + 1,
                                'total_pages': len(reader.pages),
                                'source_type': 'pdf'
                            }
                            documents.append(Document(page_content=text, metadata=metadata))
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
        
        return documents
    
    def _load_excel(self, file_path: str) -> List[Document]:
        """Load Excel files"""
        documents = []
        
        try:
            xl = pd.ExcelFile(file_path)
            
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Try to find text columns
                text_cols = [col for col in df.columns if any(
                    kw in col.lower() 
                    for kw in ['text', 'content', 'definition', 'description', 'details']
                )]
                
                if text_cols:
                    # Use identified text column
                    for _, row in df.iterrows():
                        content = str(row[text_cols[0]]) if pd.notna(row[text_cols[0]]) else ""
                        if len(content.strip()) > 50:
                            metadata = {
                                'source_file': os.path.basename(file_path),
                                'sheet': sheet_name,
                                'source_type': 'excel'
                            }
                            # Add other columns as metadata
                            for col in df.columns:
                                if col != text_cols[0] and pd.notna(row[col]):
                                    metadata[col] = str(row[col])[:200]
                            documents.append(Document(page_content=content, metadata=metadata))
                else:
                    # Combine all columns as content
                    for _, row in df.iterrows():
                        content = " ".join([str(v) for v in row.values if pd.notna(v)])
                        if len(content.strip()) > 50:
                            metadata = {
                                'source_file': os.path.basename(file_path),
                                'sheet': sheet_name,
                                'source_type': 'excel'
                            }
                            documents.append(Document(page_content=content, metadata=metadata))
        
        except Exception as e:
            logger.error(f"Error loading Excel {file_path}: {e}")
        
        return documents
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load Word documents"""
        documents = []
        
        try:
            from docx import Document as DocxDocument
            
            doc = DocxDocument(file_path)
            
            # Extract paragraphs
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs)
            
            if len(full_text.strip()) < 50:
                return documents
            
            # Try to split by headings (if any)
            headings = []
            sections = []
            current_section = []
            
            for para in doc.paragraphs:
                if para.text.strip() and para.style.name.startswith('Heading'):
                    if current_section:
                        sections.append({
                            'heading': headings[-1] if headings else "Section",
                            'content': "\n".join(current_section)
                        })
                        current_section = []
                    headings.append(para.text.strip())
                elif para.text.strip():
                    current_section.append(para.text.strip())
            
            if current_section:
                sections.append({
                    'heading': headings[-1] if headings else "Section",
                    'content': "\n".join(current_section)
                })
            
            if sections:
                for section in sections:
                    content = f"**{section['heading']}**\n\n{section['content']}"
                    metadata = {
                        'source_file': os.path.basename(file_path),
                        'heading': section['heading'],
                        'source_type': 'docx'
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            else:
                # Single document
                metadata = {
                    'source_file': os.path.basename(file_path),
                    'source_type': 'docx'
                }
                documents.append(Document(page_content=full_text, metadata=metadata))
        
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
        
        return documents