"""
Advanced query processing for legal questions.
"""
import re
from typing import Dict, List, Tuple

# Legal entity recognition patterns
LEGAL_PATTERNS = {
    "section": r"(?:section|s\.|sec\.)\s*(\d+[A-Za-z]?)",
    "chapter": r"(?:chapter|ch\.)\s*(\d+[A-Za-z]?)",
    "article": r"(?:article|art\.)\s*(\d+[A-Za-z]?)",
    "ordinance": r"(?:ordinance|ord\.)\s*(\d+[A-Za-z]?)",
    "act": r"(?:act)\s*(\d+[A-Za-z]?)",
}

# Legal keywords by category
LEGAL_CATEGORIES = {
    "criminal_procedure": ["bail", "warrant", "summons", "arrest", "search", "seizure", "trial", "appeal"],
    "civil_procedure": ["plaint", "written statement", "decree", "injunction", "specific performance", "damages"],
    "evidence": ["admissible", "relevance", "burden of proof", "presumption", "estoppel", "confession"],
    "contract": ["contract", "agreement", "breach", "remedy", "damages", "specific performance", "rescission"],
}

def process_question(question: str) -> Dict:
    """
    Process a legal question and extract key information.
    """
    question_lower = question.lower()
    result = {
        "original": question,
        "entities": {},
        "categories": [],
        "keywords": [],
        "sections": [],
        "is_legal_question": False,
    }
    
    # Extract legal entities
    for entity_type, pattern in LEGAL_PATTERNS.items():
        matches = re.findall(pattern, question_lower)
        if matches:
            result["entities"][entity_type] = matches
    
    # Check for section numbers specifically
    section_matches = re.findall(r"\b(\d{3}[A-Za-z]?)\b", question_lower)
    if section_matches:
        result["sections"] = section_matches
    
    # Categorize the question
    for category, keywords in LEGAL_CATEGORIES.items():
        if any(kw in question_lower for kw in keywords):
            result["categories"].append(category)
    
    # Check if it's a legal question
    legal_indicators = ["law", "legal", "section", "act", "ordinance", "constitution", "penal", "code", "court", "judge", "trial"]
    if any(indicator in question_lower for indicator in legal_indicators):
        result["is_legal_question"] = True
    
    # Extract key phrases
    # Remove question words
    cleaned = re.sub(r'what|which|who|whom|whose|when|where|why|how|is|are|was|were|does|did|has|have|been', '', question_lower)
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    # Extract significant words (length > 3)
    result["keywords"] = [w for w in cleaned.split() if len(w) > 3]
    
    return result

def generate_search_queries(processed: Dict) -> List[str]:
    """
    Generate multiple search queries based on processed question.
    """
    original = processed["original"]
    queries = [original]
    
    # Add Pakistan context if missing
    if "pakistan" not in original.lower():
        queries.append(f"{original} Pakistan")
        queries.append(f"{original} Pakistani law")
    
    # Add section-specific queries
    if processed["sections"]:
        for section in processed["sections"]:
            queries.append(f"Section {section} {original}")
            queries.append(f"Section {section} Pakistan law")
    
    # Add category-specific queries
    for category in processed["categories"]:
        queries.append(f"{original} {category} law")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    
    return unique_queries