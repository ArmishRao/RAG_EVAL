from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from typing import List
import re

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Primary retriever with MMR for diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.5  # Balance between relevance and diversity
    }
)

def keyword_matching(query: str, docs: List) -> List:
    """
    Additional keyword-based retrieval for better coverage.
    """
    query_words = set(query.lower().split())
    # Remove common stopwords
    stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'for', 'on', 'at', 'by', 'in', 'to', 'with', 'without', 'and', 'or', 'but', 'so', 'for', 'nor', 'yet', 'as', 'than'}
    query_words = {w for w in query_words if w not in stopwords}
    
    # Score each document
    scored_docs = []
    for doc in docs:
        content = doc.page_content.lower()
        # Count keyword matches
        matches = sum(1 for w in query_words if w in content)
        # Check for section matches
        section = str(doc.metadata.get('section', ''))
        section_match = 2 if section.lower() in query.lower() else 0
        
        score = matches + section_match
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by score and return
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:4]]

def hybrid_retrieve(question: str) -> List:
    """
    Hybrid retrieval combining vector search and keyword matching.
    """
    # Get vector search results
    vector_docs = retriever.invoke(question)
    
    # Get keyword matches from the vector store's documents
    keyword_docs = keyword_matching(question, vector_docs)
    
    # Combine and deduplicate
    combined = vector_docs + keyword_docs
    
    # Remove duplicates based on content
    seen_content = set()
    unique_docs = []
    for doc in combined:
        content_key = doc.page_content[:100]  # Use first 100 chars as key
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_docs.append(doc)
    
    return unique_docs[:8]  # Return up to 8 documents

def retrieve(question: str) -> List:
    """Legacy retrieve function for compatibility."""
    return retriever.invoke(question)