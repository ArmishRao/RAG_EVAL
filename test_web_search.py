#!/usr/bin/env python
"""
Test web search functionality
"""

import requests
from bs4 import BeautifulSoup
from googlesearch import search
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_web_search():
    print("=" * 60)
    print("Testing Web Search")
    print("=" * 60)
    
    question = "how can a man transefer his property to his friend?" 
    search_query = f"{question} Pakistan law"
    
    print(f"\n Searching: {search_query}")
    
    try:
        # Try to get URLs
        urls = list(search(search_query, num_results=3, lang='en'))
        
        print(f"\n Found {len(urls)} URLs:")
        for url in urls:
            print(f"   - {url}")
        
        # Try to fetch content
        if urls:
            print("\n Fetching content from first URL...")
            url = urls[0]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "header", "footer"]):
                    element.decompose()
                
                text = soup.get_text()
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = "\n".join(lines[:50])
                
                print(f"   Content length: {len(content)} characters")
                print(f"   Preview: {content[:200]}...")
            
        else:
            print("\n No URLs found!")
            print("This could be due to:")
            print("   - Rate limiting by Google")
            print("   - Network issues")
            print("   - Need to install googlesearch-python properly")
    
    except Exception as e:
        print(f"\n Error: {e}")
        print("Possible solutions:")
        print("  1. Try: pip install --upgrade googlesearch-python")
        print("  2. Try: pip install google")
        print("  3. Check internet connection")

if __name__ == "__main__":
    test_web_search()