#  Pakistan Legal Advisor - RAG System

An AI-powered legal assistant that provides instant answers about Pakistani law using Retrieval-Augmented Generation (RAG) with multi-source search capabilities.

##  Features

-  **Local Legal Database** - FAISS vector store with Pakistani laws
-  **Multi-Source Search** - Local DB + Web + Fallback knowledge base
-  **Smart Retrieval** - Hybrid search with relevance filtering
-  **RAGAS Evaluation** - Comprehensive performance metrics
-  **LangSmith Integration** - Traceable evaluation
-  **Dual Interface** - Streamlit Web UI + CLI
-  **Privacy-First** - All processing done locally

##  Evaluation Metrics

| Metric | Description | Score |
|--------|-------------|-------|
| Faithfulness | Factual consistency | 0.80 |
| Answer Relevancy | How well answer addresses question | 0.9 |
| Context Relevancy | Utility of retrieved context | 0.58 |

##  Installation

### Prerequisites
- Python 3.10+
- Groq API Key (free)
- (Optional) LangSmith API Key

### Setup

```bash
# Clone repository
git clone https://github.com/armishrao/legal-advisor.git
cd legal-advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# Ingest legal data
python ingest.py

# Run the application
python run.py