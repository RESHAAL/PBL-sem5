# ResearchLens AI

ResearchLens AI is a research-paper exploration project using semantic search and research-gap analysis.

## Features
- Semantic search using Sentence Transformers and FAISS
- Research-gap detection
- Paper recommendations
- SDG analysis
- OpenAlex data collection
- BM25 retrieval comparison

## Dataset
Dataset: data/research_papers_real.csv
Contains 2,200 records across 10 research domains. Publication dates range from 2020-01-01 to 2026-05-20. These are publication dates, not the dataset collection period.
Source: OpenAlex.

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the application
```powershell
streamlit run app.py
```

## Retrieval experiment
Run the preliminary BM25 vs. semantic retrieval comparison:
```powershell
python compare_retrieval.py
```
Results are written to retrieval_comparison_results.csv. The Relevant column requires manual assessment. Timing is preliminary, and the two methods currently use different candidate-generation procedures; do not treat this as a controlled benchmark.

## Configuration and models
Keep secrets in a local .env file; never commit secrets. Model artifacts may be excluded by .gitignore and may need to be regenerated.

## Preliminary Evaluation Results

Evaluation was performed on 5 research queries, with 10 retrieved results per query for each method (100 result rows total).

| Metric | BM25 | Semantic Search (MiniLM + FAISS) |
|---|---:|---:|
| Mean Precision@10 | 0.20 | 0.26 |
| Mean nDCG@10 | 0.38 | 0.69 |

These are preliminary, title-based relevance judgments and have not yet been independently validated. The results suggest that semantic search performed better on this small query set, particularly in ranking relevant results near the top. They do not establish general superiority.

**Limitations:** Only five queries were evaluated. Relevance labels require verification using paper titles and abstracts. The methods currently use different candidate-generation procedures, so this is not a controlled, apples-to-apples benchmark. Latency measurements are preliminary and should be repeated with warm-up runs and multiple trials. Recall is not reported because the total number of relevant papers per query has not been established.

Detailed per-query metrics are available in `retrieval_metrics.csv`.
