# ragas_evaluator.py (Fixed Version)

"""
RAGAS Evaluator with fixed imports.
Supports RAGAS metrics for evaluating RAG systems.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from datasets import Dataset
import pandas as pd
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the judge LLM
judge_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def evaluate_with_ragas(dataset_dict):
    """
    Evaluate RAG performance using RAGAS metrics.
    
    Args:
        dataset_dict: Dictionary with keys:
            - question: list of questions
            - answer: list of generated answers
            - contexts: list of lists of context strings
            - ground_truth: list of reference answers (optional)
    
    Returns:
        DataFrame with evaluation results
    """
    try:
        # Try to use RAGAS if available
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=judge_llm,
        )
        
        return result
        
    except ImportError as e:
        logger.warning(f"RAGAS import error: {e}")
        logger.info("Falling back to custom evaluation...")
        return evaluate_with_custom(dataset_dict)
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {e}")
        return evaluate_with_custom(dataset_dict)


def evaluate_with_custom(dataset_dict):
    """
    Custom evaluation when RAGAS is not available.
    Uses simple heuristics and LLM-based scoring.
    """
    logger.info("Using custom evaluation...")
    
    questions = dataset_dict.get('question', [])
    answers = dataset_dict.get('answer', [])
    contexts = dataset_dict.get('contexts', [])
    
    results = []
    
    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i] if i < len(answers) else ""
        context_list = contexts[i] if i < len(contexts) else []
        
        # Calculate simple metrics
        metrics = {
            'question': question,
            'answer': answer,
            'faithfulness_score': calculate_faithfulness(question, answer, context_list),
            'answer_relevancy_score': calculate_answer_relevancy(question, answer),
            'context_recall_score': calculate_context_recall(question, context_list),
            'context_precision_score': calculate_context_precision(question, context_list),
        }
        results.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create a RAGAS-like result object
    class ResultWrapper:
        def __init__(self, df):
            self.df = df
        
        def to_pandas(self):
            return self.df
        
        def __getattr__(self, name):
            if name in self.df.columns:
                return self.df[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    return ResultWrapper(df)


def calculate_faithfulness(question: str, answer: str, contexts: list) -> float:
    """
    Calculate faithfulness score.
    Higher score = more faithful to context.
    """
    if not answer or not contexts:
        return 0.0
    
    combined_context = " ".join(contexts[:3])
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator. Determine if the answer is factually consistent with the context.
        
        Context: {context}
        Answer: {answer}
        
        Score from 0 to 1:
        0 = Completely unfaithful (contradicts context or makes up information)
        1 = Completely faithful (fully supported by context)
        
        Return ONLY a number between 0 and 1.
        """)
        
        messages = prompt.format_messages(
            context=combined_context[:2000],
            answer=answer[:2000]
        )
        
        response = judge_llm.invoke(messages)
        
        # Extract score from response
        import re
        score_match = re.search(r'(\d+\.?\d*)', response.content)
        if score_match:
            score = float(score_match.group(1))
            return min(1.0, max(0.0, score))
        
        return 0.5
        
    except Exception as e:
        logger.error(f"Faithfulness calculation error: {e}")
        return 0.5


def calculate_answer_relevancy(question: str, answer: str) -> float:
    """
    Calculate answer relevancy score.
    Higher score = answer directly addresses the question.
    """
    if not answer:
        return 0.0
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator. Determine how relevant the answer is to the question.
        
        Question: {question}
        Answer: {answer}
        
        Score from 0 to 1:
        0 = Completely irrelevant (doesn't address the question)
        1 = Perfectly relevant (directly answers the question)
        
        Return ONLY a number between 0 and 1.
        """)
        
        messages = prompt.format_messages(
            question=question,
            answer=answer[:2000]
        )
        
        response = judge_llm.invoke(messages)
        
        import re
        score_match = re.search(r'(\d+\.?\d*)', response.content)
        if score_match:
            score = float(score_match.group(1))
            return min(1.0, max(0.0, score))
        
        return 0.5
        
    except Exception as e:
        logger.error(f"Answer relevancy calculation error: {e}")
        return 0.5


def calculate_context_recall(question: str, contexts: list) -> float:
    """
    Calculate context recall score.
    Higher score = more relevant information retrieved.
    """
    if not contexts:
        return 0.0
    
    # Simple heuristic: check if question keywords appear in contexts
    question_words = set(question.lower().split())
    important_words = [w for w in question_words if len(w) > 3]
    
    if not important_words:
        return 0.5
    
    combined_context = " ".join(contexts).lower()
    matches = sum(1 for w in important_words if w in combined_context)
    
    return min(1.0, matches / len(important_words))


def calculate_context_precision(question: str, contexts: list) -> float:
    """
    Calculate context precision score.
    Higher score = retrieved contexts are more relevant.
    """
    if not contexts:
        return 0.0
    
    # Simple heuristic: check if contexts contain question keywords
    question_words = set(question.lower().split())
    important_words = [w for w in question_words if len(w) > 3]
    
    if not important_words:
        return 0.5
    
    scores = []
    for context in contexts:
        context_lower = context.lower()
        matches = sum(1 for w in important_words if w in context_lower)
        score = matches / len(important_words) if important_words else 0.5
        scores.append(score)
    
    return sum(scores) / len(scores)


def evaluate_individual_sample(question, answer, contexts, ground_truth=None):
    """
    Evaluate a single RAG sample.
    
    Args:
        question: The user question
        answer: The generated answer
        contexts: List of context strings
        ground_truth: Optional reference answer
    
    Returns:
        Dictionary with evaluation scores
    """
    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    }
    
    if ground_truth:
        dataset_dict["ground_truth"] = [ground_truth]
    
    result = evaluate_with_ragas(dataset_dict)
    
    return result


def evaluate_from_csv(csv_path):
    """
    Evaluate multiple samples from a CSV file.
    
    Expected columns:
    - Query: The question
    - Response: The generated answer
    - contexts: List of context strings (can be JSON string)
    - ground_truth: Reference answer (optional)
    """
    df = pd.read_csv(csv_path)
    
    # Parse contexts if stored as JSON string
    if "contexts" in df.columns and isinstance(df["contexts"].iloc[0], str):
        import json
        df["contexts"] = df["contexts"].apply(json.loads)
    
    dataset_dict = {
        "question": df["Query"].tolist(),
        "answer": df["Response"].tolist(),
        "contexts": df["contexts"].tolist(),
    }
    
    if "ground_truth" in df.columns:
        dataset_dict["ground_truth"] = df["ground_truth"].tolist()
    
    return evaluate_with_ragas(dataset_dict)


if __name__ == "__main__":
    # Test the evaluator
    print("=" * 60)
    print("Testing RAGAS Evaluator")
    print("=" * 60)
    
    test_data = {
        "question": ["What is the punishment for theft?"],
        "answer": ["The punishment for theft is imprisonment up to 3 years."],
        "contexts": [["Section 379: Theft - imprisonment up to 3 years."]]
    }
    
    result = evaluate_with_ragas(test_data)
    print("\nEvaluation Results:")
    print(result.to_pandas())