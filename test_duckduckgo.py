#!/usr/bin/env python
"""
Test DuckDuckGo search functionality
"""

from duckduckgo_search import DDGS

def test_duckduckgo():
    print("=" * 60)
    print("Testing DuckDuckGo Search")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "punishment for domestic violence Pakistan",
        "minimum wage Pakistan",
        "Pakistan constitution amendments",
        "talaq law Pakistan"
    ]
    
    for query in test_queries:
        print(f"\n Searching: {query}")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{query}", max_results=3, region='pk'))
                
                if results:
                    print(f" Found {len(results)} results")
                    for i, result in enumerate(results[:2], 1):
                        print(f"  {i}. {result.get('title', 'No title')[:60]}")
                        print(f"     URL: {result.get('href', 'No URL')[:60]}")
                        print(f"     Preview: {result.get('body', '')[:100]}...")
                        print()
                else:
                    print(" No results found")
                    
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    test_duckduckgo()