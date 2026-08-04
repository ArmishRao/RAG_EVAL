# ragas_evaluate.py (Complete Fixed Version)

"""
Simple RAGAS evaluation script with proper result handling.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from app import ask_rag
import json
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info(f"Processing: {question[:50]}...")
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
        logger.info(f"Dataset saved to {output_file}")
    
    return dataset_dict


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Pakistan Legal Advisor - RAGAS Evaluation")
    print("=" * 60)
    
    # Test questions
    test_questions = [
        "What are the penalties for theft under Pakistani law?",
        "Explain the process of filing a civil suit in Pakistan.",
        "What are the rights of women under the Pakistani constitution?",
        "What is the punishment for murder in Pakistan?",
        "How does the legal system handle property disputes in Pakistan?",
    ]
    
    # Prepare dataset
    dataset_dict = {
        "question": test_questions,
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # Get RAG responses
    print("\nGenerating responses...")
    for q in test_questions:
        print(f"  Processing: {q[:50]}...")
        result = ask_rag(q)
        dataset_dict["answer"].append(result["Response"])
        dataset_dict["contexts"].append([doc.page_content for doc in result["sources"]])
        dataset_dict["ground_truth"].append("")
    
    # Try to evaluate
    try:
        from ragas_evaluator import evaluate_with_ragas
        
        print("\nRunning RAGAS evaluation...")
        results = evaluate_with_ragas(dataset_dict)
        
        print("\nRAGAS Evaluation Results:")
        print("=" * 60)
        
        # ============================================================
        # FIXED: Properly handle the results object
        # ============================================================
        
        # Method 1: If results is a DataFrame
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            print("\nDataFrame columns:", df.columns.tolist())
            
            # Print all score columns
            for col in df.columns:
                if col.endswith("_score") or "score" in col.lower():
                    try:
                        mean_score = df[col].mean()
                        print(f"{col}: {mean_score:.4f}")
                    except:
                        print(f"{col}: {df[col].iloc[0]}")
            
            # Save results
            df.to_csv("ragas_evaluation_results.csv", index=False)
            print("\nResults saved to ragas_evaluation_results.csv")
        
        # Method 2: If results is a dictionary
        elif isinstance(results, dict):
            print("\nResults Dictionary:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
                elif isinstance(value, list) and value:
                    try:
                        avg = sum(value) / len(value)
                        print(f"{key}: {avg:.4f}")
                    except:
                        print(f"{key}: {value}")
        
        # Method 3: If results has items() method
        elif hasattr(results, 'items'):
            print("\nResults Items:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
                elif isinstance(value, list) and value:
                    try:
                        avg = sum(value) / len(value)
                        print(f"{key}: {avg:.4f}")
                    except:
                        print(f"{key}: {value}")
        
        # Method 4: Try to access as attribute
        else:
            print("\nAttempting to access attributes...")
            for attr in dir(results):
                if not attr.startswith('_'):
                    try:
                        val = getattr(results, attr)
                        if isinstance(val, (int, float)):
                            print(f"{attr}: {val:.4f}")
                        elif isinstance(val, list) and val:
                            try:
                                avg = sum(val) / len(val)
                                print(f"{attr}: {avg:.4f}")
                            except:
                                print(f"{attr}: {val}")
                    except:
                        pass
        
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nRAGAS import error: {e}")
        print("\nUsing simplified evaluation...")
        
        # Simple evaluation
        scores = []
        for i, q in enumerate(test_questions):
            answer = dataset_dict["answer"][i]
            contexts = dataset_dict["contexts"][i]
            
            score = {
                'question': q[:50],
                'answer_length': len(answer),
                'num_contexts': len(contexts),
                'has_answer': len(answer) > 50,
                'context_exists': len(contexts) > 0
            }
            scores.append(score)
        
        df = pd.DataFrame(scores)
        print("\nSimplified Evaluation Results:")
        print(df.to_string(index=False))
        df.to_csv("simple_evaluation_results.csv", index=False)
        print("\nResults saved to simple_evaluation_results.csv")
        
    except Exception as e:
        print(f"\nError in evaluation: {e}")
        print("\nSaving raw results...")
        
        # Save raw results
        raw_df = pd.DataFrame({
            "Question": dataset_dict["question"],
            "Answer": dataset_dict["answer"],
            "Num_Contexts": [len(c) for c in dataset_dict["contexts"]]
        })
        raw_df.to_csv("raw_evaluation_results.csv", index=False)
        print("Raw results saved to raw_evaluation_results.csv")


if __name__ == "__main__":
    main()