

# ResearchLens AI — Domain-Aware Research Paper Retrieval

ResearchLens AI is a research paper retrieval system designed to retrieve relevant academic papers for domain-specific queries using lexical and semantic search techniques.

The project compares **BM25-based lexical retrieval** with **Sentence Transformer embeddings and FAISS-based semantic retrieval** across five research domains.

## Project Overview

ResearchLens evaluates two retrieval approaches:

- **BM25:** Keyword-based lexical retrieval using paper titles, abstracts, topics, and keywords.
- **Semantic Retrieval:** Sentence Transformer embeddings with FAISS similarity search to identify semantically relevant papers.

The system is evaluated using domain-specific queries and human relevance judgments.

## Research Domains and Queries

| Domain | Query |
|---|---|
| Healthcare | AI for early disease detection |
| Agriculture | Machine learning for crop disease prediction |
| Computer Science | Explainable AI for cybersecurity |
| Education | AI-based personalized learning |
| Environmental Science | AI for climate prediction |

## Dataset

The research corpus contains **2,200 research papers** across five domains.

Each query is associated with a domain-specific subset of papers. The corpus includes metadata such as:

- Paper ID
- Title
- Abstract
- Topics
- Keywords

The corpus and associated retrieval outputs are maintained in the project repository.

## Methodology

### 1. Dataset Preparation
The research paper corpus is prepared and organized using paper metadata and domain-specific query definitions.

### 2. Lexical Retrieval — BM25
BM25 retrieves papers based on lexical similarity between the query and paper metadata fields:

- Title
- Abstract
- Topics
- Keywords

### 3. Semantic Retrieval
Paper representations are generated using the `all-MiniLM-L6-v2` Sentence Transformer model.

The embeddings are normalized and indexed using FAISS for similarity-based retrieval.

### 4. Retrieval Evaluation
The system retrieves the top 10 results for each of the five queries using both methods.

Evaluation includes:

- Precision@10
- Recall@10
- F1@10
- nDCG@10

Human relevance judgments are used for relevance evaluation. Independent review and disagreement adjudication are part of the validation process.

### 5. Latency Benchmarking
Latency is measured for:

- BM25 search-only
- BM25 end-to-end
- Semantic FAISS search-only
- Semantic FAISS end-to-end

The benchmark separates retrieval latency from setup costs such as corpus loading, embedding generation, and index construction.

## Technology Stack

- Python
- Pandas
- NumPy
- Sentence Transformers
- FAISS
- BM25 retrieval
- Scikit-learn
- CSV-based research data and evaluation outputs

## Repository Structure

```text
PBL-sem5/
├── ResearchLens_*.csv
├── ResearchLens_*.xlsx
├── ResearchLens_*.zip
├── benchmark_latency_reconciled.py
├── benchmark_latency_interleaved.py
├── benchmark_results_reconciled/
├── benchmark_results_interleaved/
├── ResearchLens_retrieval_evaluation_audit_bundle_PROVISIONAL/
├── reshal_relevance_audit.csv
├── reshal_retrieval_run_info.txt
└── README.md
```

The repository contains dataset-related files, retrieval evaluation materials, relevance review files, latency benchmark scripts, and supporting audit packages.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RESHAAL/PBL-sem5.git
cd PBL-sem5
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pandas numpy sentence-transformers faiss-cpu rank-bm25 scikit-learn
```

> Dependency versions and the exact execution entry point should be confirmed against the project scripts before reproducing the complete retrieval pipeline.

## Running the Latency Benchmark

The repository includes latency benchmark scripts.

```bash
python benchmark_latency_reconciled.py
```

For the interleaved benchmark:

```bash
python benchmark_latency_interleaved.py
```

The benchmark output files are stored in the corresponding result directories.

## Evaluation Status

The project includes preliminary relevance evaluation and human-review materials.

**Final evaluation metrics remain subject to independent relevance review and disagreement adjudication.** Suggested labels and provisional metric files should not be interpreted as final human-validated results.

Latency benchmark outputs are environment-specific and should be interpreted in the context of the hardware and software environment used.

## Reproducibility

For reproducibility, the following artifacts should be maintained:

- Dataset and query definitions
- Embedding generation configuration
- FAISS index and Paper ID mapping
- Retrieval outputs
- Relevance judgment files
- Evaluation scripts
- Latency benchmark scripts
- Environment and dependency information

## Project Team

ResearchLens AI is developed as part of the PBL Semester 5 project.

## License

Add the applicable project license before external reuse or redistribution.

