# Documentation RAG System

An AI-powered documentation assistant that uses **Retrieval-Augmented Generation (RAG)** to answer technical questions from official documentation.

The system combines **hybrid retrieval, reranking, LLM-based generation, RAGAS evaluation, LangSmith tracing, and security testing** to evaluate and improve the reliability of documentation-based RAG applications.

## Overview

The system indexes technical documentation from multiple sources and retrieves the most relevant information before generating an answer.

Currently supported documentation includes:

* Python
* LangChain
* FastAPI
* PyTorch

Instead of relying only on the LLM's pretrained knowledge, the system retrieves relevant documentation and uses it as context for generating grounded answers.

## Key Features

* **Documentation-Based RAG** — Answers questions using indexed technical documentation
* **Hybrid Retrieval** — Combines dense vector search and BM25 sparse retrieval
* **FAISS Vector Search** — Efficient semantic similarity search
* **MMR Retrieval** — Improves diversity among retrieved documents
* **Relevance Filtering** — Removes low-quality or irrelevant retrieved chunks
* **Reranking** — Reranks candidate documents before sending context to the LLM
* **LLM Generation** — Generates answers using retrieved documentation
* **RAGAS Evaluation** — Measures answer and retrieval quality
* **LangSmith Integration** — Provides tracing and evaluation capabilities
* **Security Evaluation** — Tests common RAG and LLM security vulnerabilities
* **Benchmark Dataset** — 100-question evaluation dataset across different difficulty levels
* **Local LLM Support** — Supports local inference through Ollama
* **API-Based LLM Support** — Supports Groq as a fallback generation provider

---

## System Architecture

```text
                Technical Documentation
                         |
                         v
              Document Loading & Cleaning
                         |
                         v
                    Chunking
                         |
                         v
              Embedding Generation
                         |
              +----------+----------+
              |                     |
              v                     v
        FAISS Vector Store       BM25 Index
              |                     |
              +----------+----------+
                         |
                         v
                  Hybrid Retrieval
                         |
                         v
                    MMR Filtering
                         |
                         v
                  Relevance Filtering
                         |
                         v
                     Reranking
                         |
                         v
               Relevant Context
                         |
                         v
                       LLM
                         |
                         v
                 Generated Answer
```

---

## Documentation Sources

The current knowledge base contains documentation from:

| Source    | Purpose                                            |
| --------- | -------------------------------------------------- |
| Python    | Python language and standard library documentation |
| LangChain | LLM and RAG framework documentation                |
| FastAPI   | API development documentation                      |
| PyTorch   | Deep learning framework documentation              |

Documentation files are processed and converted into searchable chunks before being stored in the retrieval system.

---

## Retrieval Pipeline

The retrieval system uses multiple techniques rather than relying on a single vector search method.

### 1. Dense Retrieval

Documentation chunks are converted into embeddings and stored in a **FAISS vector database**.

This allows the system to retrieve semantically similar content even when the wording of the query differs from the documentation.

### 2. Sparse Retrieval

The system also uses **BM25** to perform keyword-based retrieval.

This is useful when the question contains specific technical terms, function names, classes, or API terminology.

### 3. Hybrid Retrieval

Dense and sparse retrieval results are combined to improve retrieval coverage.

```text
Dense Retrieval
      +
BM25 Retrieval
      |
      v
Hybrid Candidate Set
```

### 4. MMR

Maximum Marginal Relevance is used to reduce redundant results and improve the diversity of retrieved documentation chunks.

### 5. Reranking

Retrieved candidates are filtered and reranked before being passed to the LLM.

This helps ensure that the final context contains the most useful documentation for answering the question.

---

## Evaluation

A major part of this project is evaluating the RAG system rather than only measuring whether it produces an answer.

The evaluation framework measures:

* Retrieval quality
* Context quality
* Answer relevance
* Faithfulness
* Groundedness
* Security robustness

### RAGAS Evaluation Results

| Metric              |    Score |
| ------------------- | -------: |
| Faithfulness        | **0.85** |
| Answer Relevancy    | **0.86** |
| Context Precision   | **0.75** |
| Context Recall      | **0.91** |
| Overall RAGAS Score | **0.84** |

These results indicate that the system generally produces answers that are well-supported by retrieved documentation, while retrieval precision remains an area for further improvement.

---

## Retrieval Evaluation

Retrieval was evaluated independently using a manually defined ground-truth evaluation dataset.

| Metric      |    Score |
| ----------- | -------: |
| Recall@1    | **0.26** |
| Recall@3    | **0.56** |
| Recall@5    | **0.75** |
| Recall@6    | **0.78** |
| Precision@5 | **0.27** |
| Hit Rate@5  | **0.96** |
| MRR         | **0.64** |
| NDCG@5      | **0.59** |

The high Hit Rate@5 shows that the relevant information is usually retrieved within the top candidates, while the lower Precision@5 indicates that the retrieval pipeline can still return unnecessary chunks.

---

## Evaluation Dataset

The project includes a benchmark containing **100 technical questions**.

The questions are distributed across different difficulty levels:

```text
Easy      → 30%
Medium    → 50%
Hard      → 20%
```

Each evaluation sample contains information such as:

```text
question_id
question
ground_truth
documentation_source
source_document
category
difficulty
expected_chunks
```

The dataset is used to evaluate both retrieval and generation performance.

---

## Security Evaluation

The RAG system was also tested against common security risks affecting LLM and RAG applications.

### Security Results

| Security Test            |    Score |
| ------------------------ | -------: |
| Prompt Injection         | **0.99** |
| Jailbreak Resistance     | **0.96** |
| PII Leakage              | **1.00** |
| Prompt Leakage           | **0.96** |
| Code Injection Detection | **0.95** |
| Retrieval Poisoning      | **0.71** |
| Overall Security         | **0.93** |

### Retrieval Poisoning

Retrieval poisoning was identified as the weakest security area.

The evaluation includes malicious or instruction-like content embedded inside retrieved documentation chunks to test whether the model incorrectly treats retrieved content as trusted instructions.

This highlights an important RAG security principle:

> Retrieved documents should be treated as data, not as instructions.

---

## RAGAS + LangSmith

The project uses **RAGAS** for automated evaluation of the RAG pipeline and **LangSmith** for tracing and analyzing model interactions.

The evaluation workflow can be summarized as:

```text
Evaluation Questions
        |
        v
       RAG
        |
        +------------------+
        |                  |
        v                  v
 Retrieved Context     Generated Answer
        |                  |
        +---------+--------+
                  |
                  v
              RAGAS
                  |
                  v
        Evaluation Metrics
```

LangSmith can additionally be used to inspect individual traces and analyze retrieval and generation behavior.

---

## Technology Stack

### Programming

* Python

### RAG & Retrieval

* LangChain
* FAISS
* BM25
* MMR
* Sentence Transformers

### LLMs

* Ollama
* Qwen 2.5
* Groq

### Evaluation

* RAGAS
* LangSmith

### Interface

* Streamlit
* CLI

---

## Project Structure

```text
documentation-rag/
│
├── data/
│   └── documentation/
│       ├── python/
│       ├── langchain/
│       ├── fastapi/
│       └── pytorch/
│
├── evaluation/
│   ├── evaluation_questions.json
│   ├── results.json
│   └── security_tests.json
│
├── vectorstore/
│   ├── FAISS index
│   └── document metadata
│
├── ingest.py
├── app.py
├── run.py
├── evaluate.py
├── security_eval.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

* Python 3.10+
* Git
* Ollama (optional)
* Groq API key (optional)
* LangSmith API key (optional)

### Clone the Repository

```bash
git clone https://github.com/armishrao/legal-advisor.git

cd legal-advisor
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=documentation-rag
```

Only the API keys required by your selected configuration need to be added.

---

## Build the Knowledge Base

Place the documentation files inside the appropriate data directories and run:

```bash
python ingest.py
```

The ingestion pipeline:

1. Loads documentation
2. Cleans the content
3. Splits documents into chunks
4. Generates embeddings
5. Builds the FAISS index
6. Builds the BM25 index
7. Stores document metadata

---

## Run the Application

```bash
python run.py
```

The application can be accessed through the available Streamlit interface or CLI.

---

## Evaluation

To evaluate the RAG system against the benchmark dataset:

```bash
python evaluate.py
```

The evaluation generates metrics for:

* Faithfulness
* Answer relevancy
* Context precision
* Context recall

---

## Security Testing

Run the security evaluation using:

```bash
python security_eval.py
```

The security evaluation tests:

* Prompt injection
* Jailbreak attempts
* PII leakage
* Prompt leakage
* Retrieval poisoning
* Code injection

---

## Future Improvements

Planned improvements include:

* Improving retrieval precision
* Better handling of retrieval poisoning
* More advanced reranking models
* Larger evaluation datasets
* Query expansion
* Hybrid retrieval optimization
* Automated failure analysis
* Improved chunking strategies
* More robust security evaluation
* Support for additional technical documentation
* Evaluation of multiple LLMs and embedding models

---

## Project Objective

The goal of this project is to build more than a basic question-answering chatbot.

The project focuses on understanding **how RAG systems behave, how their retrieval can be evaluated, how answer quality can be measured, and how security vulnerabilities can be identified**.

By combining retrieval optimization, automated evaluation, observability, and security testing, the project provides an end-to-end approach to building and evaluating documentation-based RAG systems.
