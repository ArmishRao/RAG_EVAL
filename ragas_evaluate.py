import os
import pandas as pd
from dotenv import load_dotenv
from app import ask_rag
from ragas_evaluator import evaluate_with_ragas
import json

load_dotenv()


def prepare_dataset_for_ragas(questions_file, output_file=None):
    """
    Process questions and prepare dataset for RAGAS evaluation.
    """
    # Load questions
    df = pd.read_csv(questions_file)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for _, row in df.iterrows():
        question = row["Query"]
        ground_truth = row.get("Response", "")
        
        # Get RAG response
        rag_result = ask_rag(question)
        
        questions.append(question)
        answers.append(rag_result["Response"])
        contexts.append([doc.page_content for doc in rag_result["sources"]])
        ground_truths.append(ground_truth)
    
    # Create dataset dictionary
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    # Save if output file specified
    if output_file:
        output_df = pd.DataFrame({
            "Query": questions,
            "Response": answers,
            "contexts": [json.dumps(c) for c in contexts],
            "ground_truth": ground_truths
        })
        output_df.to_csv(output_file, index=False)
    
    return dataset_dict


def main():
    # Example: Evaluate with RAGAS
    test_questions = [
        "What are the penalties for theft under Pakistani law?",
        "Explain the process of filing a civil suit in Pakistan.",
        "What are the rights of women under the Pakistani constitution?",
        "What is the punishment for murder in Pakistan?",
        "How does the legal system handle property disputes in Pakistan?"
    ]
    
    # Prepare dataset
    dataset_dict = {
        "question": test_questions,
        "answer": [],
        "contexts": [],
        "ground_truth": []  # You can add reference answers if available
    }
    
    # Get RAG responses
    for q in test_questions:
        result = ask_rag(q)
        dataset_dict["answer"].append(result["Response"])
        dataset_dict["contexts"].append([doc.page_content for doc in result["sources"]])
        dataset_dict["ground_truth"].append("")  # Add ground truth if available
    
    # Run RAGAS evaluation
    print("Running RAGAS evaluation...")
    results = evaluate_with_ragas(dataset_dict)
    
    print("\nRAGAS Evaluation Results:")
    print("=" * 60)
    
    # Display results
    for metric, score in results.items():
        if metric.endswith("_score"):
            print(f"{metric}: {score:.4f}")
    
    print("=" * 60)
    
    # Save results
    results.to_csv("ragas_evaluation_results.csv")
    print("\nResults saved to ragas_evaluation_results.csv")


if __name__ == "__main__":
    main()