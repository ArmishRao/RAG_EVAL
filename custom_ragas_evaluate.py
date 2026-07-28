import os
import pandas as pd
from dotenv import load_dotenv
from app import ask_rag
from custom_ragas_evaluator import evaluate_with_custom_ragas

load_dotenv()


def evaluate_questions(questions_list):
    """
    Process questions and evaluate RAG responses.
    """
    results = []
    
    for i, question in enumerate(questions_list):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(questions_list)}")
        print(f"Question: {question}")
        print('='*60)
        
        # Get RAG response
        rag_result = ask_rag(question)
        
        # Extract contexts
        contexts = [doc.page_content for doc in rag_result["sources"]]
        
        print(f"Retrieved {len(contexts)} context documents")
        print(f"Generated answer length: {len(rag_result['Response'])} characters")
        
        # Evaluate with custom RAGAS
        scores = evaluate_with_custom_ragas(
            question=question,
            answer=rag_result["Response"],
            contexts=contexts,
            ground_truth=None  # Add if you have ground truth
        )
        
        # Store results
        results.append({
            "question": question,
            "answer": rag_result["Response"],
            "faithfulness": scores.get("faithfulness_score", 0),
            "answer_relevancy": scores.get("answer_relevancy_score", 0),
            "context_relevancy": scores.get("context_relevancy_score", 0),
            "answer_correctness": scores.get("answer_correctness_score", None),
            "num_contexts": len(contexts),
            "answer_length": len(rag_result["Response"]),
        })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("Pakistan Legal Advisor - RAGAS Style Evaluation")
    print("=" * 70)
    
    #  test questions
    test_questions = [
        "What are the penalties for theft under Pakistani law?",
        "Explain the process of filing a civil suit in Pakistan.",
        "What are the rights of women under the Pakistani constitution?",
        "What is the punishment for murder in Pakistan?",
        "How does the legal system handle property disputes in Pakistan?",
        "What is the procedure for divorce in Pakistan?",
        "What are the laws regarding inheritance in Pakistan?",
        "What is the minimum age for marriage in Pakistan?",
        "What are the penalties for drug trafficking in Pakistan?",
        "How are labor disputes handled in Pakistan?",
        "What is the law regarding defamation in Pakistan?",
        "What are the rights of accused persons in Pakistan?",
    ]
    
    # Run evaluation
    print(f"\nEvaluating {len(test_questions)} questions...")
    results_df = evaluate_questions(test_questions)
    
    # Display results
    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("=" * 70)
    
    avg_faithfulness = results_df["faithfulness"].mean()
    avg_answer_relevancy = results_df["answer_relevancy"].mean()
    avg_context_relevancy = results_df["context_relevancy"].mean()
    
    print(f"\n Average Scores:")
    print(f"  Faithfulness:        {avg_faithfulness:.4f}")
    print(f"  Answer Relevancy:    {avg_answer_relevancy:.4f}")
    print(f"  Context Relevancy:   {avg_context_relevancy:.4f}")
    
    if "answer_correctness" in results_df.columns:
        avg_correctness = results_df["answer_correctness"].dropna().mean()
        print(f"  Answer Correctness:  {avg_correctness:.4f} (based on {len(results_df['answer_correctness'].dropna())} samples)")
    
    print(f"\n Additional Statistics:")
    print(f"  Average contexts retrieved: {results_df['num_contexts'].mean():.1f}")
    print(f"  Average answer length: {results_df['answer_length'].mean():.0f} characters")
    
    # Display detailed results
    print("\n" + "=" * 70)
    print("Detailed Results")
    print("=" * 70)
    
    for idx, row in results_df.iterrows():
        print(f"\n{idx+1}. {row['question']}")
        print(f"   Faithfulness:     {row['faithfulness']:.3f}")
        print(f"   Answer Relevancy: {row['answer_relevancy']:.3f}")
        print(f"   Context Relevancy:{row['context_relevancy']:.3f}")
        if pd.notna(row.get('answer_correctness', None)):
            print(f"   Answer Correctness:{row['answer_correctness']:.3f}")
        print(f"   Answer Preview:   {row['answer'][:150]}...")
        print("-" * 70)
    
    # Save results
    results_df.to_csv("custom_ragas_results.csv", index=False)
    print(f"\n Results saved to custom_ragas_results.csv")
    
    # Create a summary report
    summary = {
        "total_questions": len(results_df),
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_relevancy": avg_answer_relevancy,
        "avg_context_relevancy": avg_context_relevancy,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("custom_ragas_summary.csv", index=False)
    print(f" Summary saved to custom_ragas_summary.csv")


if __name__ == "__main__":
    main()