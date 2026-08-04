"""
Test script for hybrid retriever.
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_retrieval():
    """Test the hybrid retriever performance."""
    print("=" * 60)
    print("Testing Hybrid Retriever")
    print("=" * 60)
    
    try:
        from retriever import get_retriever
        
        print("\nInitializing retriever...")
        retriever = get_retriever()
        
        test_queries = [
            "What is the punishment for theft under Pakistani law?",
            "Explain the procedure for divorce in Pakistan",
            "What are the penalties for murder?",
            "How to file a civil suit in Pakistan?",
        ]
        
        results_summary = []
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 40)
            
            start = time.time()
            
            if hasattr(retriever, 'retrieve'):
                results = retriever.retrieve(query)
            else:
                results = retriever.invoke(query)
            
            elapsed = time.time() - start
            
            print(f"Time: {elapsed:.3f} seconds")
            print(f"Retrieved: {len(results)} documents")
            
            print("\nTop Results:")
            if results:
                for i, doc in enumerate(results[:3], 1):
                    section = doc.metadata.get('section', 'Unknown')
                    heading = doc.metadata.get('heading', 'No heading')
                    preview = doc.page_content[:100].replace('\n', ' ')
                    
                    print(f"\n  {i}. Section {section} - {heading}")
                    print(f"     {preview}...")
            else:
                print("  No results found!")
            
            results_summary.append({
                'query': query[:50],
                'time': elapsed,
                'num_results': len(results)
            })
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        
        if results_summary:
            avg_time = sum(r['time'] for r in results_summary) / len(results_summary)
            avg_results = sum(r['num_results'] for r in results_summary) / len(results_summary)
            
            print(f"Average retrieval time: {avg_time:.3f} seconds")
            print(f"Average results: {avg_results:.1f} documents")
        else:
            print("No results to summarize.")
            
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nMake sure you have installed required dependencies:")
        print("  pip install rank-bm25")
        print("  pip install sentence-transformers")
        
    except FileNotFoundError as e:
        print(f"\nFile Not Found: {e}")
        print("\nMake sure you have run:")
        print("  python ingest.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry running with simple retriever:")
        print("  In retriever.py, set _use_hybrid = False")


if __name__ == "__main__":
    test_retrieval()