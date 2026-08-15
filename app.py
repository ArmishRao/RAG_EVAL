"""
Programming Documentation Assistant
Supports: Python, LangChain, FastAPI, PyTorch documentation
"""

import os
import re
import logging
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document
from retriever import retrieve
from prompt import prompt
import tiktoken

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# LLM INITIALIZATION
# ============================================================

# Set your Mistral key here if it's not already in your .env as MISTRAL_API_KEY
# os.environ["MISTRAL_API_KEY"] = "your-key-here"

# Try Mistral API first
try:
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=2
    )
    logger.info("✅ Using Mistral API (mistral-small-latest)")
except Exception as e:
    logger.warning(f"⚠️ Mistral not available: {e}")
    # Fallback to Groq if available
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )
        logger.info(" Using Groq (llama-3.1-8b-instant)")
    except Exception as e2:
        logger.error(f" No LLM available: {e2}")
        llm = None


# ============================================================
# TOKEN MANAGEMENT
# ============================================================

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        # Fallback: approximate 4 chars per token
        return len(text) // 4


def truncate_context(context: str, max_tokens: int = 4000) -> str:
    """
    Truncate context to stay within token limits while preserving important content.
    """
    if count_tokens(context) <= max_tokens:
        return context
    
    # Split into paragraphs
    chunks = context.split('\n\n')
    
    # Score each chunk by relevance to programming topics
    programming_keywords = [
        'function', 'class', 'method', 'parameter', 'return',
        'example', 'import', 'def', 'async', 'await', 'yield',
        'exception', 'raise', 'try', 'except', 'finally',
        'with', 'as', 'global', 'nonlocal', 'lambda',
        'list', 'dict', 'tuple', 'set', 'str', 'int', 'float',
        'bool', 'None', 'True', 'False', 'if', 'else', 'elif',
        'for', 'while', 'break', 'continue', 'pass',
        'tensor', 'module', 'sequential', 'nn', 'fastapi',
        'dependency', 'injection', 'runnable', 'chain', 'langchain'
    ]
    
    scored_chunks = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        
        # Score based on programming keywords
        score = sum(1 for kw in programming_keywords if kw in chunk.lower())
        
        # Boost chunks with code-like patterns
        if re.search(r'def\s+\w+\s*\(', chunk):
            score += 5
        if re.search(r'class\s+\w+', chunk):
            score += 4
        if re.search(r'```', chunk):
            score += 3  # Code blocks
        if re.search(r'import\s+\w+', chunk):
            score += 2
        if re.search(r'@\w+', chunk):  # Decorators
            score += 2
        
        # Boost chunks with specific library names
        libraries = ['python', 'langchain', 'fastapi', 'pytorch', 'tensorflow', 'django', 'flask']
        for lib in libraries:
            if lib in chunk.lower():
                score += 3
        
        # Penalize very long chunks
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens > 1000:
            score *= 0.5
        
        scored_chunks.append((score, chunk))
    
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Build truncated context
    truncated = []
    current_tokens = 0
    for score, chunk in scored_chunks:
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens <= max_tokens:
            truncated.append(chunk)
            current_tokens += chunk_tokens
        else:
            # Try to truncate the chunk itself
            remaining = max_tokens - current_tokens
            if remaining > 100:
                words = chunk.split()
                truncated_chunk = ' '.join(words[:remaining * 4])
                truncated.append(truncated_chunk + "...")
            break
    
    return '\n\n'.join(truncated)


# ============================================================
# WEB SEARCH FALLBACK
# ============================================================

def web_search_fallback(question: str) -> list:
    """Search the web for programming documentation when local results are insufficient."""
    try:
        from web_search_fixed import search_with_fallback
        
        logger.info(f"🌐 Searching web for: {question[:50]}...")
        
        # Clean question for search
        clean_question = question.replace('?', '').strip()
        
        # Detect which library the question is about
        library_keywords = {
            'python': ['python', 'py', 'pip'],
            'langchain': ['langchain', 'chain', 'agent', 'runnable'],
            'fastapi': ['fastapi', 'api', 'endpoint', 'dependency'],
            'pytorch': ['pytorch', 'torch', 'tensor', 'nn'],
            'tensorflow': ['tensorflow', 'tf'],
            'django': ['django'],
            'flask': ['flask']
        }
        
        detected_library = 'python'  # default
        for lib, keywords in library_keywords.items():
            if any(kw in question.lower() for kw in keywords):
                detected_library = lib
                break
        
        # Try different search queries
        search_queries = [
            f"{clean_question} {detected_library} documentation",
            f"{clean_question} {detected_library} example",
            f"{clean_question} {detected_library}",
            clean_question
        ]
        
        all_results = []
        used_queries = set()
        
        for query in search_queries:
            if query in used_queries:
                continue
            used_queries.add(query)
            
            results = search_with_fallback(query)
            if results:
                all_results.extend(results)
                logger.info(f"🌐 Found {len(results)} results for: {query}")
                break
        
        if not all_results:
            return []
        
        # Convert to Document objects
        docs = []
        for result in all_results[:3]:
            content = result.get('body', result.get('snippet', result.get('title', '')))
            if content and len(content.strip()) > 50:
                doc = Document(
                    page_content=f"{content}\n\nSource: {result.get('url', 'Web')}",
                    metadata={
                        'source_url': result.get('url', 'Web'),
                        'title': result.get('title', 'Web Result'),
                        'source': 'web',
                        'doc_type': 'web'
                    }
                )
                docs.append(doc)
        
        return docs
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


# ============================================================
# DOCUMENT RELEVANCE CHECK
# ============================================================

def is_document_relevant(doc: Document, question: str) -> bool:
    """
    Check if a document is relevant to the programming question.
    More lenient than before.
    """
    content = doc.page_content.lower()
    question_lower = question.lower()
    
    # Check if document is too short
    if len(content.strip()) < 50:
        return False
    
    # If document has code, it's likely relevant
    if re.search(r'```', content) or re.search(r'def\s+\w+\s*\(', content) or re.search(r'class\s+\w+', content):
        return True
    
    # Extract key terms from question (3+ letter words)
    question_words = set(re.findall(r'\b[a-z][a-z]{2,}\b', question_lower))
    
    # Extract content terms
    content_terms = set(re.findall(r'\b[a-z][a-z]{2,}\b', content))
    
    # Check for library matches
    libraries = ['python', 'langchain', 'fastapi', 'pytorch', 'tensorflow', 'django', 'flask',
                 'torch', 'numpy', 'pandas', 'scikit', 'matplotlib']
    for lib in libraries:
        if lib in question_lower and lib in content:
            return True
    
    # Check for programming term overlap
    programming_terms = {
        'function', 'class', 'method', 'api', 'module', 'package',
        'library', 'framework', 'syntax', 'parameter', 'argument',
        'return', 'async', 'await', 'decorator', 'generator',
        'iterator', 'context', 'manager', 'exception', 'error',
        'debug', 'test', 'unittest', 'pytest', 'mock', 'patch',
        'tensor', 'nn', 'runnable', 'chain', 'dependency', 'injection'
    }
    
    question_programming_terms = question_words.intersection(programming_terms)
    content_programming_terms = content_terms.intersection(programming_terms)
    
    # If both have programming terms, likely relevant
    if question_programming_terms and content_programming_terms:
        return True
    
    # Check for general keyword overlap (at least 2 common words)
    common_words = question_words.intersection(content_terms)
    if len(common_words) >= 2:
        return True
    
    return False


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask_rag(question: str, enable_web_search: bool = True) -> dict:
    """
    Main RAG function for programming documentation with web fallback.
    """
    logger.info(f" Searching documentation for: {question[:50]}...")
    
    # Step 1: Retrieve documents from local vector store
    docs = retrieve(question)
    
    # Step 2: Filter relevant documents
    relevant_docs = []
    if docs:
        for doc in docs:
            if is_document_relevant(doc, question):
                relevant_docs.append(doc)
        
        if relevant_docs:
            docs = relevant_docs
            source_type = "local"
            logger.info(f" Found {len(docs)} relevant local documentation chunks")
        else:
            logger.info(" No relevant local documentation found")
            docs = []
    
    # Step 3: Try web search if no local results or too few
    if len(docs) < 2 and enable_web_search:
        logger.info(" Trying web search fallback...")
        web_docs = web_search_fallback(question)
        if web_docs:
            docs = web_docs
            source_type = "web"
            logger.info(f" Found {len(docs)} results from web")
        else:
            docs = []
    
    # Step 4: Handle no results
    if not docs:
        return {
            "Response": """I couldn't find specific documentation about this topic.

**Recommendations:**
1. **Try rephrasing** your question to be more specific
2. **Check the official documentation:**
   - Python: https://docs.python.org/3/
   - LangChain: https://python.langchain.com/docs/
   - FastAPI: https://fastapi.tiangolo.com/
   - PyTorch: https://pytorch.org/docs/stable/
3. **Search Stack Overflow:** https://stackoverflow.com/

**Example better questions:**
- "How do I use the map() function in Python?"
- "What is the Runnable interface in LangChain?"
- "How do I create a dependency in FastAPI?"
- "How do I create a tensor in PyTorch?""",
            "context": "",
            "sources": [],
            "source_type": "none"
        }
    
    # Step 5: Prepare and optimize context
    if len(docs) > 4:
        docs = docs[:4]
        logger.info(f" Limited to 4 most relevant documents")
    
    # Build context with source attribution
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source_url', doc.metadata.get('source_file', 'Unknown'))
        doc_type = doc.metadata.get('doc_type', 'unknown')
        context_parts.append(f"[Source {i} - {doc_type} from {source}]\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    context = truncate_context(context, max_tokens=3500)
    logger.info(f" Context size: {count_tokens(context)} tokens")
    
    # Step 6: Generate response
    if llm is None:
        return {
            "Response": " No LLM is available. Please configure MISTRAL_API_KEY or Groq API.",
            "context": context,
            "sources": docs,
            "source_type": "error"
        }
    
    try:
        messages = prompt.format_messages(
            context=context,
            question=question
        )
        
        response = llm.invoke(messages)
        
        return {
            "Response": response.content,
            "context": context,
            "sources": docs,
            "source_type": source_type
        }
        
    except Exception as e:
        logger.error(f" Error generating response: {e}")
        return {
            "Response": f"I encountered an error while generating the response: {str(e)}\n\nPlease try rephrasing your question or check if the LLM is running.",
            "context": context,
            "sources": docs,
            "source_type": "error"
        }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """Command line interface for the documentation assistant."""
    print("=" * 70)
    print("   Programming Documentation Assistant")
    print("=" * 70)
    print("\n This assistant can answer questions about:")
    print("   Python, LangChain, FastAPI, PyTorch, and more!")
    print("\n Features:")
    print("   ✅ Local vector search")
    print("   ✅ Web search fallback")
    print("   ✅ Semantic understanding")
    print("   ✅ Code examples")
    print("   ✅ Source citations")
    print("   ✅ Privacy-first (all local)")
    print("\n" + "=" * 70)
    
    while True:
        try:
            question = input("\n❓ Ask a programming question (or 'exit' to quit): ")
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not question.strip():
                print("⚠️ Please enter a question.")
                continue
            
            print("\n🔍 Searching documentation...")
            result= ask_rag(question, enable_web_search=False)
           # result = ask_rag(question)
            
            print("\n" + "=" * 70)
            print(" 💡 Answer")
            print("=" * 70)
            print("\n" + result["Response"])
            
            print("\n" + "=" * 70)
            print(" 📚 Sources")
            print("=" * 70)
            
            if not result["sources"]:
                print("No sources found.")
            else:
                for i, doc in enumerate(result["sources"], 1):
                    print(f"\n Source {i}")
                    
                    if doc.metadata.get('source') == 'web':
                        print(f"   🌐 Type: Web Search")
                        print(f"   📌 Title: {doc.metadata.get('title', 'N/A')}")
                        print(f"   🔗 URL: {doc.metadata.get('source_url', 'N/A')}")
                    else:
                        print(f"   📄 File: {doc.metadata.get('source_file', 'N/A')}")
                        print(f"   📂 Type: {doc.metadata.get('doc_type', 'N/A')}")
                        if doc.metadata.get('heading'):
                            print(f"   📌 Heading: {doc.metadata.get('heading')}")
                        if doc.metadata.get('title'):
                            print(f"   📌 Title: {doc.metadata.get('title')}")
                    
                    # Show preview
                    preview = doc.page_content[:200].replace('\n', ' ')
                    print(f"   📝 Preview: {preview}...")
            
            print("\n" + "=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")


# ============================================================
# API FUNCTION (for Flask/Streamlit)
# ============================================================

def get_response_only(question: str) -> dict:
    """
    Simplified version of ask_rag that returns just the essential data.
    Useful for the API backend.
    """
    return ask_rag(question)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()