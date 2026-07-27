#!/usr/bin/env python
"""
Direct web search using requests and BeautifulSoup
"""

import requests
from bs4 import BeautifulSoup
import time

def search_via_bing(query):
    """Search using Bing (no API key needed)"""
    try:
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find search results
        results = []
        for result in soup.select('.b_algo'):
            title_elem = result.select_one('h2 a')
            snippet_elem = result.select_one('.b_caption p')
            
            if title_elem:
                results.append({
                    'title': title_elem.text,
                    'url': title_elem.get('href', ''),
                    'snippet': snippet_elem.text if snippet_elem else ''
                })
        
        return results[:3]
    except Exception as e:
        print(f"Bing search error: {e}")
        return []

def search_web(question):
    """Main search function"""
    print(f"\n Searching for: {question}")
    
    # Clean query
    query = f"{question} Pakistan law"
    
    # Try Bing
    results = search_via_bing(query)
    
    if results:
        print(f" Found {len(results)} results from Bing")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title']}")
            print(f"   URL: {r['url']}")
            print(f"   Snippet: {r['snippet'][:150]}...")
        return results
    else:
        print(" No results found")
        return []

if __name__ == "__main__":
    # Test queries
    queries = [
        "punishment for domestic violence in Pakistan",
        "minimum wage Pakistan",
        "Pakistan constitution amendments"
    ]
    
    for q in queries:
        search_web(q)
        time.sleep(1) 