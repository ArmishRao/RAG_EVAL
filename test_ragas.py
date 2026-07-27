from app import ask_rag
from ragas_evaluator import evaluate_individual_sample

def test_ragas():
    # Test a single query
    question = "What are the penalties for theft under Pakistani law?"
    result = ask_rag(question)
    
    contexts = [doc.page_content for doc in result["sources"]]
    
    # Get RAGAS scores
    scores = evaluate_individual_sample(
        question=question,
        answer=result["Response"],
        contexts=contexts,
        ground_truth="Reference answer would go here if available"  # Optional
    )
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['Response'][:200]}...")
    print("\nRAGAS Scores:")
    for metric, score in scores.items():
        if metric.endswith("_score"):
            print(f"  {metric}: {score:.4f}")

if __name__ == "__main__":
    test_ragas()