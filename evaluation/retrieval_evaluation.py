import os
import json
import math
import csv

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

# Your annotated evaluation dataset
DATASET_PATH = "evaluation/evaluation_dataset_annotated.json"

# Folder containing index.faiss and index.pkl
# CHANGE THIS if your FAISS folder has a different name.
FAISS_INDEX_PATH = "D:\legal-advisor-vectorstore"

# Output files
JSON_RESULTS_PATH = "evaluation/retrieval_results_1.json"
CSV_RESULTS_PATH = "evaluation/retrieval_results_1.csv"

# K values to evaluate
K_VALUES = [1, 3, 5, 6]

# Primary K for the easy-to-read summary
PRIMARY_K = 5

# Mistral embedding model used to create the FAISS index
EMBEDDING_MODEL = "mistral-embed"


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if PRIMARY_K not in K_VALUES:
    raise ValueError(
        f"PRIMARY_K ({PRIMARY_K}) must be included in K_VALUES."
    )


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY was not found. "
        "Add it to your .env file."
    )


# ============================================================
# LOAD MISTRAL EMBEDDINGS
# ============================================================

print("Loading Mistral embeddings...")

embeddings = MistralAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=MISTRAL_API_KEY
)


# ============================================================
# LOAD FAISS
# ============================================================

print("Loading FAISS vector store...")

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS loaded successfully.")


# ============================================================
# CREATE RETRIEVER
# ============================================================

MAX_K = max(K_VALUES)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": MAX_K}
)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading evaluation dataset...")

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} evaluation questions.")


# ============================================================
# HELPER: GET CHUNK ID
# ============================================================

def get_chunk_id(doc):
    """
    Get the chunk_id from the document metadata.
    """

    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id:
        return chunk_id

    chunk_id = doc.metadata.get("id")

    if chunk_id:
        return chunk_id

    return getattr(doc, "id", None)


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(expected, retrieved, k):
    """
    Recall@K =
        relevant expected chunks retrieved in top K
        --------------------------------------------
        total expected relevant chunks
    """

    expected_set = set(expected)

    if not expected_set:
        return 0.0

    retrieved_k = retrieved[:k]

    hits = sum(
        1
        for chunk in retrieved_k
        if chunk in expected_set
    )

    return hits / len(expected_set)


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(expected, retrieved, k):
    """
    Precision@K =
        relevant retrieved chunks in top K
        --------------------------------
        number of retrieved chunks in K
    """

    expected_set = set(expected)
    retrieved_k = retrieved[:k]

    if not retrieved_k:
        return 0.0

    hits = sum(
        1
        for chunk in retrieved_k
        if chunk in expected_set
    )

    return hits / len(retrieved_k)


# ============================================================
# HIT RATE@K
# ============================================================

def hit_rate_at_k(expected, retrieved, k):
    """
    Hit Rate@K =
        1 if at least one relevant chunk appears in top K
        0 otherwise
    """

    expected_set = set(expected)
    retrieved_k = retrieved[:k]

    return float(
        any(
            chunk in expected_set
            for chunk in retrieved_k
        )
    )


# ============================================================
# RECIPROCAL RANK / MRR
# ============================================================

def reciprocal_rank(expected, retrieved):
    """
    Reciprocal Rank =
        1 / rank of the first relevant chunk

    If no relevant chunk is retrieved:
        0
    """

    expected_set = set(expected)

    for rank, chunk in enumerate(
        retrieved,
        start=1
    ):
        if chunk in expected_set:
            return 1.0 / rank

    return 0.0


# ============================================================
# DCG
# ============================================================

def dcg_at_k(relevances, k):
    score = 0.0

    for rank, relevance in enumerate(
        relevances[:k],
        start=1
    ):
        score += (
            (2 ** relevance - 1)
            / math.log2(rank + 1)
        )

    return score


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(expected, retrieved, k):
    """
    Binary NDCG:

        expected chunk     -> relevance 1
        non-expected chunk -> relevance 0

    This evaluates whether relevant chunks are ranked
    toward the top.
    """

    expected_set = set(expected)

    if not expected_set:
        return 0.0

    actual_relevances = [
        1 if chunk in expected_set else 0
        for chunk in retrieved[:k]
    ]

    ideal_relevances = [
        1
        for _ in range(
            min(len(expected_set), k)
        )
    ]

    ideal_relevances += [
        0
        for _ in range(
            k - len(ideal_relevances)
        )
    ]

    dcg = dcg_at_k(
        actual_relevances,
        k
    )

    idcg = dcg_at_k(
        ideal_relevances,
        k
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# RUN EVALUATION
# ============================================================

all_results = []


for index, item in enumerate(dataset):

    question_id = item["question_id"]
    question = item["question"]

    expected_chunks = item.get(
        "expected_chunks",
        []
    )

    print("\n")
    print("=" * 80)
    print(
        f"[{index + 1}/{len(dataset)}] {question_id}"
    )
    print("=" * 80)

    print(question)

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    try:

        documents = retriever.invoke(
            question
        )

    except Exception as e:

        print(
            f"ERROR retrieving {question_id}: {e}"
        )

        metrics = {}

        for k in K_VALUES:

            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"hit_rate@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0

        metrics["mrr"] = 0.0

        all_results.append({
            "question_id": question_id,
            "question": question,
            "expected_chunks": expected_chunks,
            "retrieved_chunks": [],
            "metrics": metrics,
            "error": str(e)
        })

        continue

    # --------------------------------------------------------
    # EXTRACT RETRIEVED CHUNK IDs
    # --------------------------------------------------------

    retrieved_chunks = []
    retrieved_details = []

    for rank, doc in enumerate(
        documents,
        start=1
    ):

        chunk_id = get_chunk_id(doc)

        if not chunk_id:

            print(
                f"WARNING: rank {rank} has no chunk_id."
            )

            continue

        retrieved_chunks.append(chunk_id)

        retrieved_details.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "source_file": doc.metadata.get(
                "source_file",
                ""
            ),
            "heading": doc.metadata.get(
                "heading",
                ""
            ),
            "content": doc.page_content
        })

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    metrics = {}

    for k in K_VALUES:

        metrics[f"recall@{k}"] = recall_at_k(
            expected_chunks,
            retrieved_chunks,
            k
        )

        metrics[f"precision@{k}"] = precision_at_k(
            expected_chunks,
            retrieved_chunks,
            k
        )

        metrics[f"hit_rate@{k}"] = hit_rate_at_k(
            expected_chunks,
            retrieved_chunks,
            k
        )

        metrics[f"ndcg@{k}"] = ndcg_at_k(
            expected_chunks,
            retrieved_chunks,
            k
        )

    metrics["mrr"] = reciprocal_rank(
        expected_chunks,
        retrieved_chunks
    )

    # --------------------------------------------------------
    # PRINT PER QUESTION PRIMARY METRICS
    # --------------------------------------------------------

    print("\nPRIMARY METRICS (K=5)")
    print("-" * 50)

    print(
        f"Recall@{PRIMARY_K}:    "
        f"{metrics[f'recall@{PRIMARY_K}']:.4f}"
    )

    print(
        f"Precision@{PRIMARY_K}: "
        f"{metrics[f'precision@{PRIMARY_K}']:.4f}"
    )

    print(
        f"Hit Rate@{PRIMARY_K}:  "
        f"{metrics[f'hit_rate@{PRIMARY_K}']:.4f}"
    )

    print(
        f"NDCG@{PRIMARY_K}:       "
        f"{metrics[f'ndcg@{PRIMARY_K}']:.4f}"
    )

    print(
        f"MRR:                  "
        f"{metrics['mrr']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    all_results.append({
        "question_id": question_id,
        "question": question,
        "expected_chunks": expected_chunks,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_details": retrieved_details,
        "metrics": metrics
    })


# ============================================================
# REMOVE FAILED QUESTIONS FROM AVERAGE
# ============================================================

successful_results = [
    result
    for result in all_results
    if "error" not in result
]

failed_results = [
    result
    for result in all_results
    if "error" in result
]


if not successful_results:

    raise RuntimeError(
        "No successful evaluation results were produced."
    )


# ============================================================
# OVERALL AVERAGES FOR ALL K VALUES
# ============================================================

overall_all_k = {}

for k in K_VALUES:

    overall_all_k[f"recall@{k}"] = (
        sum(
            result["metrics"][f"recall@{k}"]
            for result in successful_results
        )
        / len(successful_results)
    )

    overall_all_k[f"precision@{k}"] = (
        sum(
            result["metrics"][f"precision@{k}"]
            for result in successful_results
        )
        / len(successful_results)
    )

    overall_all_k[f"hit_rate@{k}"] = (
        sum(
            result["metrics"][f"hit_rate@{k}"]
            for result in successful_results
        )
        / len(successful_results)
    )

    overall_all_k[f"ndcg@{k}"] = (
        sum(
            result["metrics"][f"ndcg@{k}"]
            for result in successful_results
        )
        / len(successful_results)
    )


overall_all_k["mrr"] = (
    sum(
        result["metrics"]["mrr"]
        for result in successful_results
    )
    / len(successful_results)
)


# ============================================================
# PRIMARY OVERALL AVERAGES
# ============================================================

overall_primary = {
    f"recall@{PRIMARY_K}":
        overall_all_k[f"recall@{PRIMARY_K}"],

    f"precision@{PRIMARY_K}":
        overall_all_k[f"precision@{PRIMARY_K}"],

    f"hit_rate@{PRIMARY_K}":
        overall_all_k[f"hit_rate@{PRIMARY_K}"],

    f"ndcg@{PRIMARY_K}":
        overall_all_k[f"ndcg@{PRIMARY_K}"],

    "mrr":
        overall_all_k["mrr"]
}


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL EVALUATION SUMMARY")
print("=" * 80)

print(
    f"\nQuestions evaluated: "
    f"{len(successful_results)}/{len(dataset)}"
)

print(
    f"Failed questions: "
    f"{len(failed_results)}"
)


print("\nPRIMARY METRICS AVERAGE (K=5)")
print("-" * 60)

print(
    f"Recall@{PRIMARY_K}:      "
    f"{overall_primary[f'recall@{PRIMARY_K}']:.4f}"
)

print(
    f"Precision@{PRIMARY_K}:   "
    f"{overall_primary[f'precision@{PRIMARY_K}']:.4f}"
)

print(
    f"Hit Rate@{PRIMARY_K}:    "
    f"{overall_primary[f'hit_rate@{PRIMARY_K}']:.4f}"
)

print(
    f"NDCG@{PRIMARY_K}:         "
    f"{overall_primary[f'ndcg@{PRIMARY_K}']:.4f}"
)

print(
    f"MRR:                    "
    f"{overall_primary['mrr']:.4f}"
)


print("\nALL K VALUES")
print("-" * 60)

for name, value in overall_all_k.items():

    print(
        f"{name:20s}: {value:.4f}"
    )


# ============================================================
# BUILD JSON OUTPUT
# ============================================================

json_output = {

    "summary": {

        "dataset_size": len(dataset),

        "successful_questions":
            len(successful_results),

        "failed_questions":
            len(failed_results),

        "primary_k":
            PRIMARY_K,

        "primary_average_scores":
            overall_primary,

        "average_scores_all_k":
            overall_all_k
    },

    "per_question": all_results
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    JSON_RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE CSV
# ============================================================

# ------------------------------------------------------------
# CSV contains:
#
# 1. One row per question
# 2. One final OVERALL_AVERAGE row
#
# This makes it easy to open in Excel.
# ------------------------------------------------------------

csv_columns = [
    "question_id",

    "recall@1",
    "precision@1",
    "hit_rate@1",
    "ndcg@1",

    "recall@3",
    "precision@3",
    "hit_rate@3",
    "ndcg@3",

    "recall@5",
    "precision@5",
    "hit_rate@5",
    "ndcg@5",

    "recall@6",
    "precision@6",
    "hit_rate@6",
    "ndcg@6",

    "mrr"
]


with open(
    CSV_RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    # Header
    writer.writerow(csv_columns)

    # --------------------------------------------------------
    # Per-question rows
    # --------------------------------------------------------

    for result in all_results:

        metrics = result["metrics"]

        row = [
            result["question_id"]
        ]

        for column in csv_columns[1:]:

            row.append(
                round(
                    metrics.get(
                        column,
                        0.0
                    ),
                    4
                )
            )

        writer.writerow(row)

    # --------------------------------------------------------
    # Overall average row
    # --------------------------------------------------------

    overall_row = [
        "OVERALL_AVERAGE"
    ]

    for column in csv_columns[1:]:

        overall_row.append(
            round(
                overall_all_k.get(
                    column,
                    0.0
                ),
                4
            )
        )

    writer.writerow(overall_row)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"\nJSON:\n{JSON_RESULTS_PATH}"
)

print(
    f"\nCSV:\n{CSV_RESULTS_PATH}"
)

print("\nYour JSON contains:")
print("  ✓ Overall average for K=5")
print("  ✓ Overall average for every K")
print("  ✓ Per-question metrics")
print("  ✓ Retrieved chunk IDs")
print("  ✓ Expected chunk IDs")

print("\nYour CSV contains:")
print("  ✓ One row per question")
print("  ✓ OVERALL_AVERAGE final row")
print("  ✓ Recall@K")
print("  ✓ Precision@K")
print("  ✓ Hit Rate@K")
print("  ✓ NDCG@K")
print("  ✓ MRR")