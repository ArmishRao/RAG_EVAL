# reranker.py
"""
Cross-Encoder Reranker for improved retrieval quality.
"""

import logging
from typing import List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Rerank retrieved documents using a cross-encoder model.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.enabled = True
            logger.info(f" Cross-encoder loaded: {model_name}")
        except ImportError:
            logger.warning(" sentence-transformers not installed. Reranking disabled.")
            self.enabled = False
        except Exception as e:
            logger.warning(f" Could not load reranker: {e}")
            self.enabled = False
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: User question
            documents: List of retrieved documents
            top_k: Number of documents to return
        
        Returns:
            Reranked list of documents
        """
        if not self.enabled or not documents:
            return documents[:top_k] if documents else []
        
        try:
            logger.debug(f" Reranking {len(documents)} documents...")
            
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Sort documents by score
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            return [doc for doc, score in doc_scores[:top_k]]
            
        except Exception as e:
            logger.error(f" Reranking error: {e}")
            return documents[:top_k]
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores.
        """
        if not self.enabled or not documents:
            return [(doc, 0.5) for doc in documents[:10]]
        
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.predict(pairs)
            
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores
            
        except Exception as e:
            logger.error(f" Reranking error: {e}")
            return [(doc, 0.5) for doc in documents[:10]]


def create_reranker() -> CrossEncoderReranker:
    """
    Factory function for reranker.
    """
    return CrossEncoderReranker()