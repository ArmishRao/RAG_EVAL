"""
Reliable web search using multiple fallback methods
"""
import requests
from bs4 import BeautifulSoup
import logging
import time
import re
import json
from urllib.parse import quote_plus
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Use a session for better performance
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
})

def search_duckduckgo_lite(query: str) -> List[Dict]:
    """
    Search using DuckDuckGo Lite (lighter version that's less likely to be blocked)
    """
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        
        # Find result tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                link = row.find('a')
                if link and link.get('href'):
                    text = row.get_text(strip=True)
                    if len(text) > 20:
                        results.append({
                            'title': link.text.strip() or text[:50],
                            'url': link.get('href'),
                            'body': text[:500]
                        })
                        if len(results) >= 5:
                            break
            if len(results) >= 5:
                break
                
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo Lite error: {e}")
        return []

def search_google_web(query: str) -> List[Dict]:
    """
    Search using Google's web interface
    """
    try:
        url = f"https://www.google.com/search?q={quote_plus(query)}&num=5"
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        
        for result in soup.select('.g'):
            title_elem = result.select_one('h3')
            link_elem = result.select_one('a')
            snippet_elem = result.select_one('.VwiC3b')
            
            if title_elem and link_elem:
                title = title_elem.text.strip()
                url_link = link_elem.get('href', '')
                snippet = snippet_elem.text.strip() if snippet_elem else title
                
                # Filter for legal content
                legal_keywords = ['law', 'punishment', 'section', 'act', 'penal', 'code', 'legal', 'court', 'judgment', 'constitution', 'ordinance']
                if any(kw in snippet.lower() for kw in legal_keywords) or any(kw in title.lower() for kw in legal_keywords):
                    results.append({
                        'title': title,
                        'url': url_link,
                        'body': snippet
                    })
                
                if len(results) >= 5:
                    break
                    
        return results
    except Exception as e:
        logger.warning(f"Google search error: {e}")
        return []

def search_wikipedia_legal(query: str) -> List[Dict]:
    """
    Search Wikipedia specifically for legal topics in Pakistan
    """
    try:
        # Clean query
        query_clean = query.replace('?', '').strip()
        
        # First search for the query
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query_clean)}%20Pakistan%20law&format=json"
        response = session.get(search_url, timeout=10)
        data = response.json()
        
        results = []
        for result in data.get('query', {}).get('search', [])[:3]:
            title = result.get('title', '')
            
            # Get page extract
            page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={quote_plus(title)}&format=json"
            page_response = session.get(page_url, timeout=10)
            page_data = page_response.json()
            
            pages = page_data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if page_id != '-1':
                    extract = page_info.get('extract', '')[:1000]
                    if extract and len(extract) > 50:
                        results.append({
                            'title': title,
                            'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            'body': extract
                        })
                    break
        
        # If no results, try a broader search
        if not results:
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query_clean)}%20law&format=json"
            response = session.get(search_url, timeout=10)
            data = response.json()
            
            for result in data.get('query', {}).get('search', [])[:2]:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                # Remove HTML tags from snippet
                snippet = re.sub(r'<[^>]+>', '', snippet)
                
                if len(snippet) > 50:
                    results.append({
                        'title': title,
                        'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        'body': snippet
                    })
        
        return results
    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
        return []

def search_with_fallback(query: str) -> List[Dict]:
    """
    Try multiple search methods until one works
    """
    search_methods = [
        ("DuckDuckGo Lite", search_duckduckgo_lite),
        ("Google Web", search_google_web),
        ("Wikipedia", search_wikipedia_legal),
    ]
    
    all_results = []
    
    for method_name, method_func in search_methods:
        try:
            logger.info(f"Trying {method_name}...")
            results = method_func(query)
            if results:
                logger.info(f" {method_name} found {len(results)} results")
                all_results.extend(results)
                break
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"{method_name} failed: {e}")
    
    # If still no results, try a simpler search
    if not all_results:
        logger.info("Trying simplified search...")
        simplified_query = query.replace('?', '').strip().split()[0] if query.split() else query
        for method_name, method_func in search_methods:
            results = method_func(simplified_query)
            if results:
                all_results.extend(results)
                break
    
    # Remove duplicates
    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    return unique_results