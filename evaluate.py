from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from app import ask_rag
from evaluators import (
    correctness,
    faithfulness,
    relevance,
    citation_quality,
    answer_not_empty
)
from ragas_evaluator import evaluate_individual_sample
import pandas as pd

load_dotenv()

client = Client()


def target(inputs):
    question = inputs["Query"]
    result = ask_rag(question)
    
    # Extract contexts for RAGAS evaluation
    contexts = []
    if result.get("sources"):
        contexts = [doc.page_content for doc in result["sources"]]
    
    return {
        "Response": result["Response"],
        "context": result["context"],
        "contexts": contexts,  # For RAGAS
        "ground_truth": inputs.get("Response", ""),  # If available
    }


def evaluate_with_ragas_on_dataset(test_data_path):
    """
    Run RAGAS evaluation on a test dataset.
    """
    # Load test data
    df = pd.read_csv(test_data_path)
    
    results = []
    
    for _, row in df.iterrows():
        question = row["Query"]
        ground_truth = row.get("Response", "")
        
        # Get RAG response
        rag_result = ask_rag(question)
        
        # Prepare contexts
        contexts = [doc.page_content for doc in rag_result["sources"]]
        
        # Evaluate with RAGAS
        ragas_result = evaluate_individual_sample(
            question=question,
            answer=rag_result["Response"],
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        # Store results
        results.append({
            "question": question,
            "response": rag_result["Response"],
            "ragas_score": ragas_result,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
     results = evaluate(
        target,
        data="legal_evaluation_quarter",
        evaluators=[
            correctness,
            faithfulness,
            relevance,
            citation_quality,
            answer_not_empty
        ],
        experiment_prefix="Pakistan-Legal-RAG",
        max_concurrency=1,
        blocking=True,
    )
    
    print(results)
    
    # Option 2: Run RAGAS evaluation separately
    # Uncomment if you have a test dataset
    # ragas_results = evaluate_with_ragas_on_dataset("test_data.csv")
    # print("\nRAGAS Evaluation Results:")
    # print(ragas_results)