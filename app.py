from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from retriever import retrieve
from prompt import prompt
import requests
from bs4 import BeautifulSoup
import logging
import re
import time
from urllib.parse import quote_plus
from web_search_fixed import search_with_fallback
from legal_fallback import get_fallback_answer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

def is_document_relevant(doc, question):
    """Check if a document is actually relevant to the question."""
    content = doc.page_content.lower()
    question_lower = question.lower()
    
    # Extract key topics from question
    topic_keywords = {
        'domestic violence': ['domestic', 'violence', 'abuse', 'cruelty', 'battered', 'spouse', 'husband', 'wife', 'protection'],
        'theft': ['theft', 'steal', 'robbery', 'stolen', 'property'],
        'murder': ['murder', 'kill', 'homicide', 'death', 'qatl'],
        'divorce': ['divorce', 'talaq', 'dissolution', 'marriage', 'separation'],
        'marriage': ['marriage', 'nikah', 'wedding', 'spouse'],
        'minimum wage': ['wage', 'salary', 'pay', 'minimum', 'earning'],
        'constitution': ['constitution', 'amendment', 'article', 'fundamental right'],
        'inheritance': ['inheritance', 'heir', 'will', 'succession', 'property'],
        'property': ['property', 'transfer', 'sale', 'purchase', 'mortgage', 'lease', 'rent', 'ownership', 'possession'],
        'defamation': ['defamation', 'libel', 'slander', 'reputation', 'imputation'],
        'drug': ['drug', 'narcotic', 'trafficking', 'controlled substance', 'cannabis', 'opium'],
        'labor': ['labor', 'labour', 'employment', 'worker', 'employee', 'wage', 'salary', 'termination'],
        'civil': ['civil', 'suit', 'plaint', 'decree', 'injunction', 'damages', 'specific performance'],
    }
    
    # Find which topic the question is about
    question_topic = None
    for topic, keywords in topic_keywords.items():
        if any(kw in question_lower for kw in keywords):
            question_topic = topic
            break
    
    # If we know the topic, check if document is about it
    if question_topic:
        topic_keywords_list = topic_keywords[question_topic]
        matches = sum(1 for kw in topic_keywords_list if kw in content)
        if matches < 2:
            return False
    
    # Check for specific section numbers that are clearly irrelevant
    irrelevant_sections = ['310A', '455', '495', '220', '494', '493', '496', '223', '123', '230', '10', '67']
    section = doc.metadata.get('section', '')
    if section in irrelevant_sections:
        return False
    
    # Check if document contains "could not find" or similar phrases
    if "could not find" in content or "not available" in content:
        return False
    
    # Check if document is too short
    if len(doc.page_content.strip()) < 50:
        return False
    
    return True

def web_search_fallback(question: str):
    """Enhanced web search with multiple fallback methods."""
    try:
        logger.info(f" Searching web for: {question}")
        
        # Clean question for search
        clean_question = question.replace('?', '').strip()
        
        # Try different search query variations
        search_queries = [
            f"{clean_question} Pakistan law",
            f"{clean_question} Pakistani legal",
            f"Pakistan law {clean_question}",
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
                logger.info(f"Found {len(results)} results for: {query}")
                break
            
            time.sleep(0.5)
        
        if not all_results:
            logger.warning(" No web results found")
            return []
        
        # Convert to Document objects
        docs = []
        for result in all_results[:5]:
            content = result.get('body', result.get('snippet', result.get('title', '')))
            if content and len(content.strip()) > 30:
                doc = Document(
                    page_content=content[:2000],
                    metadata={
                        'source_url': result.get('url', 'N/A'),
                        'title': result.get('title', 'Web Result'),
                        'source': 'web',
                        'snippet': content[:300] + "..." if len(content) > 300 else content
                    }
                )
                docs.append(doc)
                logger.info(f" Retrieved: {result.get('title', 'Unknown')[:50]}")
        
        return docs
        
    except Exception as e:
        logger.error(f" Web search failed: {e}")
        return []

def ask_rag(question: str):
    """Main RAG function with fallback knowledge base."""
    
    # Check for fallback answer first
    fallback_answer = get_fallback_answer(question)
    
    # Step 1: Try local retrieval
    logger.info(f" Searching local vector store for: {question}")
    docs = retrieve(question)
    
    relevant_docs = []
    if docs:
        for doc in docs:
            if is_document_relevant(doc, question):
                relevant_docs.append(doc)
        
        if relevant_docs:
            docs = relevant_docs
            source_type = "local"
            logger.info(f" Found {len(docs)} relevant local documents")
        else:
            logger.info(f" Found {len(docs)} documents but none are relevant")
            docs = []
    
    # Step 2: If no relevant documents, try web search
    if not docs:
        logger.info(" No relevant local results, trying web search...")
        web_docs = web_search_fallback(question)
        
        if web_docs:
            docs = web_docs
            source_type = "web"
        elif fallback_answer:
            # Use fallback answer
            return {
                "Response": fallback_answer,
                "context": "Fallback knowledge base",
                "sources": [],
                "source_type": "fallback"
            }
        else:
            return {
                "Response": """I could not find sufficient information about this in my legal database or on the web.

**Recommendations:**
1. Try rephrasing your question
2. Be more specific about the area of law
3. Consult a qualified lawyer for accurate legal advice

**Helpful resources:**
- Pakistan Law Commission: www.lawcommission.gov.pk
- Pakistan Bar Council: www.pakistanbarcouncil.org""",
                "context": "",
                "sources": [],
                "source_type": "none"
            }
    else:
        source_type = "local"
    
    # Step 3: Prepare context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 4: Generate response
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
def get_response_only(question: str) -> dict:
    """
    Simplified version of ask_rag that returns just the essential data
    Useful for the API backend
    """
    return ask_rag(question)
def main():
    print("=" * 70)
    print("  Pakistan Legal Advisor (Multi-Source Search)")
    print("=" * 70)
    print("\n Features:")
    print("   Answers from local legal documents")
    print("   Smart relevance detection")
    print("   Web search from multiple sources (DuckDuckGo, Google, Wikipedia)")
    print("  Fallback knowledge base for common questions")
    print("   No API keys required")
    print("\n" + "=" * 70)
    
    while True:
        try:
            question = input(" Ask Question (or 'exit' to quit): ")
            
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not question.strip():
                print(" Please enter a question.")
                continue
            
            print("\n Processing your question...")
            result = ask_rag(question)
            
            print("\n" + "=" * 70)
            print(" Answer")
            print("=" * 70)
            
            if result["source_type"] == "web":
                print(" Answer from: Web Search")
            elif result["source_type"] == "local":
                print(" Answer from: Legal Database")
            elif result["source_type"] == "fallback":
                print(" Answer from: Knowledge Base")
            else:
                print(" No sources found")
            
            print("\n" + result["Response"])
            
            print("\n" + "=" * 70)
            print(" Sources")
            print("=" * 70)
            
            if len(result["sources"]) == 0:
                if result["source_type"] == "fallback":
                    print("Source: Fallback Knowledge Base")
                else:
                    print("No sources found.")
            else:
                for i, doc in enumerate(result["sources"], 1):
                    print(f"\n Source {i}")
                    
                    if doc.metadata.get('source') == 'web':
                        print(" Type: Web Search")
                        print("Title:", doc.metadata.get('title', 'N/A'))
                        print("URL  :", doc.metadata.get('source_url', 'N/A'))
                        print("Snippet:", doc.metadata.get('snippet', 'N/A')[:150] + "..." if len(doc.metadata.get('snippet', '')) > 150 else doc.metadata.get('snippet', 'N/A'))
                    else:
                        print(" Type: Legal Database")
                        print("Book   :", doc.metadata.get('book', 'N/A'))
                        print("Section:", doc.metadata.get('section', 'N/A'))
                        print("Heading:", doc.metadata.get('heading', 'N/A'))
                        print("Dataset:", doc.metadata.get('source_file', 'N/A'))
                        
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()