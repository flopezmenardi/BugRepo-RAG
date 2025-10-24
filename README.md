# 🐛 BugRepo-RAG

A **Retrieval-Augmented Generation (RAG)** system for intelligent bug report analysis using vector embeddings and language models. The system finds similar bugs from Mozilla Bugzilla data and generates contextual analysis reports.

## 🚀 Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

Create `.env` file:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=bugrepo
```

### 2. Index Bug Database (One-time setup)
Creates embeddings for all bugs in `data/bugs_since.csv` and stores them in Pinecone:

```bash
python src/embeddings/indexer.py
```

This processes bugs in batches, converts summaries to 512-dimensional embeddings using OpenAI, and indexes them with metadata (bug_id, product, component, classification, etc.).

### 3. Analyze a New Bug
Process a bug report and get similar bugs with analysis:

```bash
python src/new_bug_pipeline.py test_bug.json
```

**Input**: `test_bug.json` (bug report with summary, product, component)  
**Output**: `outputs/bug_report_test_001.txt` (similar bugs + AI-generated analysis)

### 4. Evaluate RAG Quality
Run RAGAS metrics to evaluate retrieval and generation quality:

```bash
python -m src.evaluation.metrics --limit 10
```

**Output**: Precision, recall, relevancy, and faithfulness scores printed to console and saved as CSV.

## 🛠️ Tech Stack
- **OpenAI**: Embeddings (text-embedding-3-small) & GPT models
- **Pinecone**: Vector database for similarity search  
- **Python**: pandas, requests, ragas for evaluation
