#!/usr/bin/env python
"""
Test each search method individually to find what works
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import json

def test_bing():
    print("\n" + "=" * 60)
    print("Testing Bing Search")
    print("=" * 60)
    
    query = "punishment for domestic violence Pakistan"
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = soup.select('.b_algo')
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            title_elem = result.select_one('h2 a')
            snippet_elem = result.select_one('.b_caption p')
            
            if title_elem:
                print(f"\n{i}. Title: {title_elem.text[:60]}")
                print(f"   URL: {title_elem.get('href', '')[:60]}")
                if snippet_elem:
                    print(f"   Snippet: {snippet_elem.text[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_duckduckgo():
    print("\n" + "=" * 60)
    print("Testing DuckDuckGo HTML Search")
    print("=" * 60)
    
    query = "punishment for domestic violence Pakistan"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = soup.select('.result')
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            title_elem = result.select_one('.result__a')
            snippet_elem = result.select_one('.result__snippet')
            
            if title_elem:
                print(f"\n{i}. Title: {title_elem.text[:60]}")
                print(f"   URL: {title_elem.get('href', '')[:60]}")
                if snippet_elem:
                    print(f"   Snippet: {snippet_elem.text[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_google():
    print("\n" + "=" * 60)
    print("Testing Google Custom Search (without API)")
    print("=" * 60)
    
    query = "punishment for domestic violence Pakistan"
    
    try:
        # Try to use Google's HTML interface
        url = f"https://www.google.com/search?q={quote_plus(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = soup.select('.g')
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            title_elem = result.select_one('h3')
            link_elem = result.select_one('a')
            snippet_elem = result.select_one('.VwiC3b')
            
            if title_elem and link_elem:
                print(f"\n{i}. Title: {title_elem.text[:60]}")
                print(f"   URL: {link_elem.get('href', '')[:60]}")
                if snippet_elem:
                    print(f"   Snippet: {snippet_elem.text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_wikipedia():
    print("\n" + "=" * 60)
    print("Testing Wikipedia Search")
    print("=" * 60)
    
    query = "domestic violence in Pakistan"
    
    try:
        # Search for articles
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}&format=json"
        response = requests.get(search_url, timeout=10)
        data = response.json()
        
        results = data.get('query', {}).get('search', [])
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            title = result.get('title', '')
            print(f"\n{i}. Title: {title}")
            
            # Get page extract
            page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={quote_plus(title)}&format=json"
            page_response = requests.get(page_url, timeout=10)
            page_data = page_response.json()
            
            pages = page_data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if page_id != '-1':
                    extract = page_info.get('extract', '')[:100]
                    if extract:
                        print(f"   Preview: {extract}...")
                    break
        
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing All Search Methods")
    print("=" * 60)
    
    methods = [
        ("Bing", test_bing),
        ("DuckDuckGo", test_duckduckgo),
        ("Google", test_google),
        ("Wikipedia", test_wikipedia)
    ]
    
    working_methods = []
    
    for name, method in methods:
        if method():
            working_methods.append(name)
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if working_methods:
        print(f"✅ Working methods: {', '.join(working_methods)}")
    else:
        print("❌ No methods working!")
        print("\nPossible issues:")
        print("1. Internet connection")
        print("2. Website blocking (try using a VPN)")
        print("3. Need to install certifi: pip install certifi")

if __name__ == "__main__":
    main()