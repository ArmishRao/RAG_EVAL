"""
Retriever module for programming documentation.
Supports FAISS vector search with documentation-specific optimizations.
"""

import logging
from typing import List
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_retriever = None
_vectorstore = None


def get_retriever():
    """Singleton pattern for retriever."""
    global _retriever
    if _retriever is None:
        logger.info("Initializing documentation retriever...")
        try:
            _retriever = create_documentation_retriever()
            logger.info("Retriever initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            _retriever = None
    return _retriever


def get_vectorstore():
    """Singleton accessor for the raw vectorstore (used for score-based retrieval)."""
    global _vectorstore
    if _vectorstore is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        from langchain_mistralai import MistralAIEmbeddings
        from langchain_community.vectorstores import FAISS

        # Must match the embedding model used in ingest.py when the
        # vectorstore was built — otherwise similarity search breaks.
        embeddings = MistralAIEmbeddings(model="mistral-embed")

        _vectorstore = FAISS.load_local(
            "D:\legal-advisor-vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore


def create_documentation_retriever():
    """
    Create a retriever optimized for programming documentation.

    Tuned for precision over diversity: narrow technical questions usually
    have one or two "right" chunks, so a large k / balanced MMR just pulls
    in more marginally-relevant noise. k/fetch_k are kept small and
    lambda_mult is weighted toward relevance (0.7) rather than diversity.
    """
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 15,
            "lambda_mult": 0.7   # closer to 1.0 = pure relevance, 0.0 = pure diversity
        }
    )


def retrieve(question: str, use_score_threshold: bool = False, score_threshold: float = 0.35) -> List[Document]:
    """
    Main retrieval function.

    Args:
        question: The user's question.
        use_score_threshold: If True, uses similarity_search_with_relevance_scores
            instead of the MMR retriever, and drops any chunk below score_threshold.
            Useful for an apples-to-apples precision test against plain MMR retrieval —
            toggle this on to see if score-filtering beats MMR for your eval set.
        score_threshold: Minimum relevance score (0-1) to keep a chunk when
            use_score_threshold=True. Raise this to be stricter about precision;
            lower it if you start losing recall.
    """
    if use_score_threshold:
        try:
            vectorstore = get_vectorstore()
            results = vectorstore.similarity_search_with_relevance_scores(question, k=6)
            docs = [doc for doc, score in results if score >= score_threshold]
            logger.info(f"Retrieved {len(docs)} chunks above score threshold {score_threshold} "
                        f"(out of {len(results)} candidates)")
            return docs
        except Exception as e:
            logger.error(f"Score-threshold retrieval error: {e}")
            return []

    retriever = get_retriever()

    if retriever is None:
        logger.error("Retriever not available")
        return []

    try:
        docs = retriever.invoke(question)
        

        for i, doc in enumerate(docs):
            print("\n==============================")
            print("CHUNK:", i)
            print("VECTOR ID:", getattr(doc, "id", None))
            print("CHUNK ID:", doc.metadata.get("chunk_id"))
            print("SOURCE FILE:", doc.metadata.get("source_file"))
            print("HEADING:", doc.metadata.get("heading"))
            print("CONTENT:")
            print(doc.page_content[:500])
        logger.info(f"Retrieved {len(docs)} documentation chunks")
        return docs
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []