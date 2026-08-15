import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI


# ============================================================
# CONFIGURATION
# ============================================================

# Your evaluation dataset
INPUT_DATASET = "evaluation/eval_dataset.json"

# New output file
OUTPUT_DATASET = "evaluation/evaluation_dataset_annotated.json"

# Detailed inspection file
OUTPUT_DETAILS = "evaluation/chunk_annotation_details.json"

# ------------------------------------------------------------
# IMPORTANT:
# Change this to the directory containing your FAISS index.
#
# Example:
# FAISS_INDEX_PATH = "faiss_index"
#
# If your project has:
#
# vectorstore/
#     index.faiss
#     index.pkl
#
# then use:
#
# FAISS_INDEX_PATH = "vectorstore"
# ------------------------------------------------------------

FAISS_INDEX_PATH = "D:\legal-advisor-vectorstore"


# ------------------------------------------------------------
# Number of chunks to retrieve for annotation.
#
# Your current system retrieves 6, so we'll use 6.
# ------------------------------------------------------------

RETRIEVAL_K = 6


# ------------------------------------------------------------
# Mistral models
# ------------------------------------------------------------

EMBEDDING_MODEL = "mistral-embed"

CHAT_MODEL = "mistral-small-latest"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY was not found.\n"
        "Please put it in your .env file."
    )


# ============================================================
# INITIALIZE MISTRAL EMBEDDINGS
# ============================================================

print("\nLoading Mistral embeddings...")

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

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K
    }
)


# ============================================================
# INITIALIZE MISTRAL JUDGE
# ============================================================

print("Loading Mistral annotation judge...")

llm = ChatMistralAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=MISTRAL_API_KEY
)


# ============================================================
# LOAD EVALUATION DATASET
# ============================================================

print("\nLoading evaluation dataset...")

with open(
    INPUT_DATASET,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


print(f"Loaded {len(dataset)} evaluation questions.")


# ============================================================
# HELPER: GET CHUNK ID
# ============================================================

def get_chunk_id(doc):

    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id:
        return chunk_id

    # Fallback if chunk_id doesn't exist
    return getattr(doc, "id", None)


# ============================================================
# HELPER: CLEAN JSON FROM LLM RESPONSE
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"Could not find JSON in LLM response:\n{text}"
        )

    json_text = text[start:end + 1]

    return json.loads(json_text)


# ============================================================
# ASK MISTRAL TO ANNOTATE CHUNKS
# ============================================================

def annotate_chunks(
    question,
    ground_truth,
    source_document,
    candidate_chunks
):

    # --------------------------------------------------------
    # Build candidate chunk text
    # --------------------------------------------------------

    chunks_text = ""

    for candidate in candidate_chunks:

        chunks_text += f"""
============================================================
CHUNK RANK: {candidate["rank"]}
CHUNK ID: {candidate["chunk_id"]}
SOURCE FILE: {candidate["source_file"]}
HEADING: {candidate["heading"]}

CONTENT:
{candidate["content"]}

"""

    # --------------------------------------------------------
    # Annotation prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert evaluator for a Retrieval-Augmented Generation
(RAG) system.

Your job is to identify which retrieved documentation chunks
are genuinely relevant for answering the question.

IMPORTANT:

A chunk is RELEVANT if its content directly contains information
that supports the ground-truth answer.

A chunk is NOT relevant merely because:
- it comes from the same documentation source
- it contains similar keywords
- it is about the same general topic
- it was retrieved by the search system

Evaluate the actual CONTENT of each chunk.

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
GROUND-TRUTH ANSWER
------------------------------------------------------------

{ground_truth}

------------------------------------------------------------
EXPECTED SOURCE DOCUMENT
------------------------------------------------------------

{source_document}

------------------------------------------------------------
CANDIDATE CHUNKS
------------------------------------------------------------

{chunks_text}

------------------------------------------------------------
TASK
------------------------------------------------------------

For every candidate chunk, assign:

3 = Highly relevant
    The chunk directly contains information necessary to answer
    the question or strongly supports the ground truth.

2 = Relevant
    The chunk contains useful information related to answering
    the question, but is not as directly useful as a 3.

1 = Weakly relevant
    The chunk is related to the topic but provides little
    information needed for the answer.

0 = Not relevant
    The chunk does not provide useful evidence for answering
    the question.

Then determine the expected_chunks.

IMPORTANT:
Only include chunks with relevance >= 2 in expected_chunks.

Return ONLY valid JSON in exactly this format:

{{
    "chunk_evaluations": [
        {{
            "chunk_id": "exact chunk id",
            "relevance": 3,
            "reason": "short explanation"
        }}
    ],
    "expected_chunks": [
        "exact chunk id",
        "exact chunk id"
    ]
}}

Do not modify chunk IDs.
"""


    # --------------------------------------------------------
    # Call Mistral
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    response_text = response.content

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    result = extract_json(response_text)

    return result


# ============================================================
# MAIN ANNOTATION LOOP
# ============================================================

annotated_dataset = []

annotation_details = []


for index, item in enumerate(dataset):

    question_id = item.get(
        "question_id",
        f"Q{index + 1:04d}"
    )

    question = item["question"]

    ground_truth = item.get(
        "ground_truth",
        ""
    )

    source_document = item.get(
        "source_document",
        ""
    )


    print("\n")
    print("=" * 80)
    print(
        f"[{index + 1}/{len(dataset)}] "
        f"Processing {question_id}"
    )
    print("=" * 80)

    print("Question:")
    print(question)

    print("\nRetrieving candidate chunks...")


    # ========================================================
    # RETRIEVE CHUNKS
    # ========================================================

    try:

        documents = retriever.invoke(
            question
        )

    except Exception as e:

        print(
            f"ERROR retrieving chunks for {question_id}: {e}"
        )

        item["expected_chunks"] = []

        annotated_dataset.append(item)

        continue


    print(
        f"Retrieved {len(documents)} candidate chunks."
    )


    # ========================================================
    # PREPARE CANDIDATE INFORMATION
    # ========================================================

    candidates = []


    for rank, doc in enumerate(
        documents,
        start=1
    ):

        chunk_id = get_chunk_id(doc)

        source_file = doc.metadata.get(
            "source_file",
            ""
        )

        heading = doc.metadata.get(
            "heading",
            ""
        )

        content = doc.page_content


        candidate = {

            "rank": rank,

            "chunk_id": chunk_id,

            "source_file": source_file,

            "heading": heading,

            "content": content

        }


        candidates.append(candidate)


        # ----------------------------------------------------
        # Print candidate
        # ----------------------------------------------------

        print("\n------------------------------")

        print(
            f"Rank: {rank}"
        )

        print(
            f"Chunk ID: {chunk_id}"
        )

        print(
            f"Source: {source_file}"
        )

        print(
            f"Heading: {heading}"
        )

        print(
            "Content:"
        )

        print(
            content[:500]
        )


    # ========================================================
    # ASK MISTRAL TO ANNOTATE
    # ========================================================

    print("\nAsking Mistral to annotate chunks...")


    try:

        annotation = annotate_chunks(
            question=question,
            ground_truth=ground_truth,
            source_document=source_document,
            candidate_chunks=candidates
        )


    except Exception as e:

        print(
            f"\nERROR annotating {question_id}: {e}"
        )

        item["expected_chunks"] = []

        annotated_dataset.append(item)

        continue


    # ========================================================
    # VALIDATE EXPECTED CHUNK IDS
    # ========================================================

    valid_chunk_ids = {
        candidate["chunk_id"]
        for candidate in candidates
    }


    expected_chunks = []

    for chunk_id in annotation.get(
        "expected_chunks",
        []
    ):

        if chunk_id in valid_chunk_ids:

            expected_chunks.append(
                chunk_id
            )

        else:

            print(
                "\nWARNING:"
                f" Mistral returned an unknown chunk ID:"
                f" {chunk_id}"
            )


    # ========================================================
    # ADD EXPECTED CHUNKS TO DATASET
    # ========================================================

    updated_item = item.copy()

    updated_item["expected_chunks"] = expected_chunks

    annotated_dataset.append(
        updated_item
    )


    # ========================================================
    # SAVE DETAILED ANNOTATION
    # ========================================================

    detail = {

        "question_id": question_id,

        "question": question,

        "ground_truth": ground_truth,

        "source_document": source_document,

        "retrieved_chunks": candidates,

        "annotation": annotation,

        "validated_expected_chunks": expected_chunks

    }


    annotation_details.append(
        detail
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n")
    print("RESULT")
    print("-" * 50)

    print(
        "Expected chunks:"
    )

    for chunk_id in expected_chunks:

        print(
            f"  ✓ {chunk_id}"
        )

    print(
        f"\nNumber of expected chunks: "
        f"{len(expected_chunks)}"
    )


# ============================================================
# SAVE ANNOTATED DATASET
# ============================================================

print("\n")
print("=" * 80)
print("Saving annotated dataset...")
print("=" * 80)


with open(
    OUTPUT_DATASET,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        annotated_dataset,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

with open(
    OUTPUT_DETAILS,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        annotation_details,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("ANNOTATION COMPLETE")
print("=" * 80)

print(
    f"\nAnnotated dataset:"
    f"\n{OUTPUT_DATASET}"
)

print(
    f"\nDetailed annotation results:"
    f"\n{OUTPUT_DETAILS}"
)

print(
    f"\nQuestions processed:"
    f" {len(annotated_dataset)}"
)

print("\nYou can now inspect:")
print(
    OUTPUT_DATASET
)

print(
    "\nThen use expected_chunks for "
    "Recall@K, Precision@K, Hit Rate@K, MRR and NDCG."
)