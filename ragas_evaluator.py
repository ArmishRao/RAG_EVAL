import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from datasets import Dataset
import pandas as pd

load_dotenv()

# Initialize the judge LLM
judge_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(judge_llm)

# Define the metrics you want to use
metrics = [
    faithfulness,           # Factual consistency with context
    answer_relevancy,       # How relevant the answer is to the question
    context_recall,         # Can the model retrieve all relevant info?
    context_precision,      # How precise is the retrieved context?
    answer_correctness,     # Accuracy compared to reference answer
    answer_similarity,      # Semantic similarity to reference
]


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
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
    )
    
    return result


def evaluate_individual_sample(question, answer, contexts, ground_truth=None):
    """
    Evaluate a single RAG sample.
    """
    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]  # contexts should be a list of strings
    }
    
    if ground_truth:
        dataset_dict["ground_truth"] = [ground_truth]
    
    return evaluate_with_ragas(dataset_dict)


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