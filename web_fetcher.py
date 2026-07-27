"""
Enhanced web content fetching with proper parsing.
"""
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# Known legal websites in Pakistan
LEGAL_SITES = {
    "pakistanlaw.com": "https://pakistanlaw.com",
    "lawcommission.gov.pk": "https://lawcommission.gov.pk",
    "pakistanbarcouncil.org": "https://pakistanbarcouncil.org",
    "supremecourt.gov.pk": "https://supremecourt.gov.pk",
    "lhc.gov.pk": "https://lhc.gov.pk",
    "shc.gov.pk": "https://shc.gov.pk",
    "peshawarhighcourt.gov.pk": "https://peshawarhighcourt.gov.pk",
    "bchighcourt.gov.pk": "https://bchighcourt.gov.pk",
}

def get_random_user_agent() -> str:
    """Get a random user agent."""
    import random
    return random.choice(USER_AGENTS)

def fetch_webpage(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch and clean webpage content.
    """
    try:
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
            element.decompose()
        
        # Get main content
        content = soup.get_text(separator='\n', strip=True)
        
        # Clean up extra whitespace
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content = '\n'.join(lines)
        
        return content
        
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

def extract_relevant_content(content: str, question: str, max_length: int = 2000) -> str:
    """
    Extract the most relevant portion of content for the question.
    """
    if not content:
        return ""
    
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    question_lower = question.lower()
    
    # Score each paragraph for relevance
    scored_paragraphs = []
    for para in paragraphs:
        if len(para) < 30:  # Skip very short paragraphs
            continue
        
        para_lower = para.lower()
        # Check for question keywords
        score = 0
        for word in question_lower.split():
            if len(word) > 3 and word in para_lower:
                score += 1
        
        # Boost score for paragraphs with legal terms
        legal_terms = ['section', 'act', 'law', 'court', 'judgment', 'punishment', 'penalty']
        for term in legal_terms:
            if term in para_lower:
                score += 2
        
        scored_paragraphs.append((score, para))
    
    # Sort by score and take top paragraphs
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Build result
    result = []
    current_length = 0
    for score, para in scored_paragraphs[:10]:
        if current_length + len(para) > max_length:
            break
        result.append(para)
        current_length += len(para)
    
    return '\n\n'.join(result) if result else content[:max_length]

def search_legal_sites(query: str) -> List[Dict]:
    """
    Search known legal sites directly.
    """
    results = []
    query_terms = query.replace(' ', '+')
    
    for site_name, base_url in LEGAL_SITES.items():
        try:
            search_url = f"{base_url}/search?q={query_terms}"
            headers = {'User-Agent': get_random_user_agent()}
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try to find search results (site-specific)
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    text = link.text.strip()
                    
                    if text and len(text) > 10 and any(kw in text.lower() for kw in ['section', 'act', 'law', 'punishment']):
                        if href.startswith('/'):
                            href = urljoin(base_url, href)
                        results.append({
                            'title': text[:100],
                            'url': href,
                            'body': text,
                            'priority': 1  # Higher priority for legal sites
                        })
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Error searching {site_name}: {e}")
    
    return results[:5]