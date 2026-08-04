# retriever.py (Simplified - Removed Hybrid Retrieval)
"""
Retriever module with simple FAISS retrieval.
"""

import logging
from typing import List
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global retriever instance
_retriever = None

def get_retriever():
    """
    Singleton pattern for retriever.
    Ensures we only load the model once.
    """
    global _retriever
    if _retriever is None:
        logger.info(" Initializing retriever...")
        try:
            _retriever = create_simple_retriever()
            logger.info(" Retriever initialized successfully!")
        except Exception as e:
            logger.error(f" Failed to initialize retriever: {e}")
            _retriever = None
    return _retriever

def create_simple_retriever():
    """
    Simple FAISS retriever.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,  # Reduced from 8 to save tokens
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )

def retrieve(question: str) -> List[Document]:
    """
    Main retrieval function.
    """
    retriever = get_retriever()
    
    if retriever is None:
        logger.error(" Retriever not available")
        return []
    
    try:
        docs = retriever.invoke(question)
        logger.info(f" Retrieved {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f" Retrieval error: {e}")
        return []