"""
Run RAG Evaluation

This script:
1. Loads the evaluation dataset
2. Runs every question through the local RAG
3. Disables web search (to evaluate only your indexed documentation)
4. Saves all outputs for RAGAS evaluation

"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import json
import traceback
from pathlib import Path
from tqdm import tqdm

from app import ask_rag


DATASET_PATH = Path("evaluation/eval_dataset.json")
OUTPUT_PATH = Path("evaluation/results_4.json")

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 60)
    print(f"Loaded {len(dataset)} evaluation questions.")
    print("=" * 60)

    return dataset


def evaluate_question(sample):

    question = sample["question"]

    try:

        result = ask_rag(
            question,
            enable_web_search=False
        )

        generated_answer = result.get("Response", "")

        retrieved_contexts = []

        retrieved_sources = []

        for doc in result.get("sources", []):

            retrieved_contexts.append(doc.page_content)

            retrieved_sources.append(
                {
                    "source_file": doc.metadata.get(
                        "source_file",
                        ""
                    ),
                    "source_url": doc.metadata.get(
                        "source_url",
                        ""
                    ),
                    "doc_type": doc.metadata.get(
                        "doc_type",
                        ""
                    ),
                    "heading": doc.metadata.get(
                        "heading",
                        ""
                    ),
                    "title": doc.metadata.get(
                        "title",
                        ""
                    )
                }
            )

        output = {

            "question_id":
                sample["question_id"],

            "question":
                question,

            "ground_truth":
                sample["ground_truth"],

            "generated_answer":
                generated_answer,

            "retrieved_contexts":
                retrieved_contexts,

            "retrieved_sources":
                retrieved_sources,

            "documentation_source":
                sample.get(
                    "documentation_source",
                    ""
                ),

            "source_document":
                sample.get(
                    "source_document",
                    ""
                ),

            "category":
                sample.get(
                    "category",
                    ""
                ),

            "difficulty":
                sample.get(
                    "difficulty",
                    ""
                ),

            "source_type":
                result.get(
                    "source_type",
                    "unknown"
                )
        }

        return output

    except Exception as e:

        traceback.print_exc()

        return {

            "question_id":
                sample["question_id"],

            "question":
                sample["question"],

            "ground_truth":
                sample["ground_truth"],

            "generated_answer":
                "",

            "retrieved_contexts":
                [],

            "retrieved_sources":
                [],

            "documentation_source":
                sample.get(
                    "documentation_source",
                    ""
                ),

            "source_document":
                sample.get(
                    "source_document",
                    ""
                ),

            "category":
                sample.get(
                    "category",
                    ""
                ),

            "difficulty":
                sample.get(
                    "difficulty",
                    ""
                ),

            "source_type":
                "error",

            "error":
                str(e)
        }




def main():

    dataset = load_dataset()

    results = []

    print("\nStarting Evaluation...\n")

    for sample in tqdm(dataset):

        result = evaluate_question(sample)

        results.append(result)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nEvaluation Complete")
    print("=" * 60)
    print(f"Questions Evaluated : {len(results)}")
    print(f"Results Saved       : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()