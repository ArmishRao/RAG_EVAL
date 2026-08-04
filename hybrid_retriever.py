"""
Enhanced Hybrid Retriever with Vector + BM25 + Reranking
Supports: Dense Vector Search (FAISS), Sparse Keyword Search (BM25), 
          Reciprocal Rank Fusion (RRF), Optional Cross-Encoder Reranking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import re
import logging
from collections import defaultdict
import os
import pickle
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retriever combining multiple search strategies.
    
    Features:
    1. Dense Vector Search (FAISS) - Semantic understanding
    2. Sparse Keyword Search (BM25) - Exact phrase matching
    3. Reciprocal Rank Fusion (RRF) - Combine results intelligently
    4. Optional Cross-Encoder Reranking - Improve relevance ordering
    """
    
    def __init__(
        self,
        vectorstore_path: str = "vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_reranker: bool = False,
        top_k: int = 10,
        rrf_k: int = 60,
        documents_pickle: str = "documents.pkl"
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vectorstore_path: Path to FAISS vector store
            embedding_model: Model name for embeddings
            use_reranker: Enable cross-encoder reranking
            top_k: Number of final documents to return
            rrf_k: RRF constant (higher = more emphasis on top results)
            documents_pickle: Path to pickle file with documents for BM25
        """
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.documents_pickle = documents_pickle
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        self.bm25_index = None
        self.reranker = None
        
        # Load all components
        self._initialize()
    
    def _initialize(self):
        """Initialize all retriever components."""
        self._load_embeddings()
        self._load_vectorstore()
        self._load_documents_for_bm25()
        self._build_bm25_index()
        self._init_reranker()
        self._log_status()
    
    def _load_embeddings(self):
        """Load the embedding model."""
        logger.info("Loading embeddings model...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embeddings model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def _load_vectorstore(self):
        """Load the FAISS vector store."""
        logger.info("Loading FAISS vector store...")
        try:
            self.vectorstore = FAISS.load_local(
                "vectorstore",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully.")
        except FileNotFoundError:
            logger.error("Vector store not found. Run ingest.py first.")
            raise
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def _load_documents_for_bm25(self):
        """Load documents from pickle file for BM25 indexing."""
        logger.info("Loading documents for BM25 indexing...")
        try:
            if os.path.exists(self.documents_pickle):
                with open(self.documents_pickle, "rb") as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from {self.documents_pickle}")
            else:
                logger.warning(f"Documents pickle file not found: {self.documents_pickle}")
                logger.warning("Run ingest.py to create documents.pkl")
                self.documents = []
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            self.documents = []
    
    def _build_bm25_index(self):
        """Build BM25 index from documents."""
        if not self.documents:
            logger.warning("No documents available for BM25 indexing.")
            self.bm25_index = None
            return
        
        logger.info("Building BM25 index...")
        try:
            tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built with {len(self.documents)} documents.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _init_reranker(self):
        """Initialize cross-encoder reranker if enabled."""
        if not self.use_reranker:
            self.reranker = None
            return
        
        logger.info("Loading cross-encoder reranker...")
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Reranker loaded successfully.")
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
            self.reranker = None
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            self.reranker = None
    
    def _log_status(self):
        """Log the current status of all components."""
        status = {
            "Vector Store": "Loaded" if self.vectorstore else "Not Loaded",
            "BM25 Index": "Built" if self.bm25_index else "Not Built",
            "Documents": len(self.documents),
            "Reranker": "Enabled" if self.reranker else "Disabled"
        }
        logger.info(f"Hybrid Retriever Status: {status}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]
    
    def vector_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """
        Perform dense vector search using FAISS.
        
        Args:
            query: User question
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        logger.debug(f"Vector search for: {query[:50]}...")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, distance in results:
                similarity = 1.0 / (1.0 + distance)
                formatted_results.append((doc, similarity))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def keyword_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """
        Perform sparse keyword search using BM25.
        
        Args:
            query: User question
            k: Number of results to return
            
        Returns:
            List of (document, bm25_score) tuples
        """
        if not self.bm25_index or not self.documents:
            return []
        
        logger.debug(f"Keyword search for: {query[:50]}...")
        
        try:
            tokenized_query = self._tokenize(query)
            
            if not tokenized_query:
                return []
            
            scores = self.bm25_index.get_scores(tokenized_query)
            doc_scores = list(zip(self.documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:k]
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def reciprocal_rank_fusion(
        self,
        results_lists: List[List[Tuple[Document, float]]],
        k: int = None
    ) -> List[Document]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings by giving each document a score:
        score = sum(1 / (k + rank))
        where rank is the position in each result list.
        
        Args:
            results_lists: List of result lists to combine
            k: RRF constant (default: self.rrf_k)
            
        Returns:
            List of documents sorted by RRF score
        """
        if k is None:
            k = self.rrf_k
        
        if not results_lists or not any(results_lists):
            return []
        
        doc_ranks = defaultdict(list)
        
        for result_list in results_lists:
            for rank, (doc, score) in enumerate(result_list, start=1):
                doc_id = doc.page_content[:100]
                doc_ranks[doc_id].append({
                    'document': doc,
                    'rank': rank,
                    'score': score
                })
        
        rrf_scores = {}
        for doc_id, entries in doc_ranks.items():
            score = sum(1.0 / (k + entry['rank']) for entry in entries)
            rrf_scores[doc_id] = {
                'document': entries[0]['document'],
                'score': score
            }
        
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['document'] for item in sorted_docs]
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: User question
            documents: List of documents to rerank
            top_k: Number of documents to return
            
        Returns:
            Reranked list of documents
        """
        if not self.reranker or not documents:
            return documents[:top_k]
        
        logger.debug(f"Reranking {len(documents)} documents...")
        
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)
            
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in doc_scores[:top_k]]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return documents[:top_k]
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Main retrieval method - hybrid search with optional reranking.
        
        Args:
            query: User question
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Hybrid search for: {query[:50]}...")
        
        try:
            vector_results = self.vector_search(query, k=20)
            logger.debug(f"Vector search: {len(vector_results)} results")
            
            if self.bm25_index:
                keyword_results = self.keyword_search(query, k=20)
                logger.debug(f"Keyword search: {len(keyword_results)} results")
            else:
                keyword_results = []
            
            if vector_results and keyword_results:
                combined = self.reciprocal_rank_fusion(
                    [vector_results, keyword_results],
                    k=self.rrf_k
                )
                logger.debug(f"Combined: {len(combined)} results")
            elif vector_results:
                combined = [doc for doc, _ in vector_results]
                logger.debug(f"Using vector results only: {len(combined)} results")
            else:
                combined = []
            
            if self.use_reranker and self.reranker and combined:
                final_results = self.rerank_documents(query, combined, k=self.top_k)
                logger.debug(f"Final: {len(final_results)} results")
                return final_results
            
            return combined[:self.top_k]
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with confidence scores.
        
        Args:
            query: User question
            
        Returns:
            List of (document, confidence_score) tuples
        """
        documents = self.retrieve(query)
        return [(doc, 1.0) for doc in documents]


def create_hybrid_retriever(
    vectorstore_path: str = "vectorstore",
    use_reranker: bool = False,
    top_k: int = 10
) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever.
    
    Args:
        vectorstore_path: Path to FAISS vector store
        use_reranker: Enable cross-encoder reranking
        top_k: Number of documents to return
        
    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(
        vectorstore_path=vectorstore_path,
        use_reranker=use_reranker,
        top_k=top_k,
        rrf_k=60
    )


def create_simple_retriever(vectorstore_path: str = "vectorstore"):
    """
    Create a simple retriever as fallback.
    
    Args:
        vectorstore_path: Path to FAISS vector store
        
    Returns:
        FAISS retriever instance
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hybrid Retriever")
    print("=" * 60)
    
    try:
        retriever = create_hybrid_retriever()
        
        test_queries = [
            "What is the punishment for theft under Pakistani law?",
            "Explain divorce procedure in Pakistan",
            "What are the penalties for murder?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 40)
            
            results = retriever.retrieve(query)
            
            print(f"Found {len(results)} results:")
            for i, doc in enumerate(results[:3], 1):
                section = doc.metadata.get('section', 'Unknown')
                heading = doc.metadata.get('heading', 'No heading')
                preview = doc.page_content[:150].replace('\n', ' ')
                
                print(f"\n{i}. Section {section} - {heading}")
                print(f"   {preview}...")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTip: Make sure you've run 'python ingest.py' first.")