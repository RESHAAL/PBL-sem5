# ResearchLens AI: Semantic Research Paper Retrieval and Research Gap Analysis Across Multiple Domains

## Abstract
ResearchLens AI is a research discovery and analysis system designed to retrieve relevant academic papers, identify recurring research gaps, recommend future research directions, and summarize Sustainable Development Goal (SDG) alignment. The system uses Sentence-BERT embeddings with FAISS-based similarity search and compares semantic retrieval with BM25 keyword retrieval. Its dataset contains 2,200 research papers across ten academic domains. A preliminary evaluation used five queries and ten retrieved papers per method for each query. The recalculated results show mean Precision@10 of 0.28 for semantic retrieval and 0.20 for BM25. Mean nDCG@10 was 0.7062 for semantic retrieval and 0.3458 for BM25. These findings suggest that semantic retrieval performed better on average in this small evaluation, although performance varied considerably by query. The relevance judgments are provisional and have not been independently validated by multiple expert assessors. The results should therefore be interpreted as preliminary rather than conclusive evidence.

**Keywords:** Research discovery, semantic search, Sentence-BERT, FAISS, BM25, research gap analysis, SDGs

## 1. Introduction
The growing volume of academic publications makes it difficult for researchers to identify relevant literature, recognize underexplored research areas, and formulate future research directions. Keyword-based retrieval can identify papers containing matching terms but may miss relevant documents that express similar concepts using different terminology.

ResearchLens AI addresses this problem by combining semantic paper retrieval with research gap detection, future-direction recommendations, and SDG analysis. The system provides a unified interface for exploring research literature across multiple domains.

The objectives of this work are:
1. To develop a semantic retrieval pipeline for academic papers.
2. To compare semantic retrieval with BM25 keyword retrieval using a common set of domain-specific candidates.
3. To identify recurring research-gap patterns in retrieved literature.
4. To provide rule-based recommendations and SDG-related summaries.

## 2. Dataset
The dataset, stored in `data/research_papers_real.csv`, contains 2,200 records and 2,200 unique Paper_ID values. The records were collected from OpenAlex and cover ten domains, with 220 papers per domain:

Agriculture, Computer Science, Economics and Finance, Education, Energy, Engineering, Environmental Science, Healthcare, Materials Science, and Social Sciences.

The dataset contains publication dates from 1 January 2020 to 20 May 2026. The collection pipeline is configured to retrieve works within a date range ending 29 August 2026; this configuration should not be confused with the latest publication date observed in the current dataset.

The embedding pipeline combines each paper's title, abstract, topics, and keywords. Missing text values are replaced with empty strings before encoding.

## 3. Methodology

### 3.1 Semantic Embedding
The system uses Sentence-Transformers `all-MiniLM-L6-v2` to encode each paper's title, abstract, topics, and keywords into dense vectors. Missing values are replaced with empty strings. Encoding uses batch size 32 and `normalize_embeddings=True`. The saved embedding file stores vectors and corresponding Paper_ID values. The model is loaded by the application and query embeddings are normalized before similarity search.

### 3.2 FAISS Similarity Search
The normalized embeddings are indexed using FAISS CPU `IndexFlatIP`, an exact inner-product index. With unit-normalized document and query vectors, inner-product scores correspond to cosine similarity. The embedding dimension is determined from the generated vectors. The application retrieves an initial semantic candidate pool and then applies the selected-domain filter; consequently, the final number of returned results can be smaller than the requested top-k.

### 3.3 BM25 Baseline
BM25 provides a lexical retrieval baseline. It ranks papers according to query-term matching and term-frequency and document-length statistics. In the corrected retrieval experiment, both methods rank a shared domain-specific candidate set, improving the comparability of the ranking evaluation.

### 3.4 Research Gap Detection
Research gaps are detected through predefined regular-expression and text-pattern rules. The categories include limited datasets, limited sample sizes, limited generalizability, limited external validation, limited real-world validation or deployment, scalability, interpretability, longitudinal studies, diversity, resource constraints, comparative evaluation, clinical validation, and ethical concerns.

Gap frequencies are calculated from the papers analyzed. These patterns indicate textual evidence of potential limitations; they do not independently establish that a research gap objectively exists.

### 3.5 Recommendations and SDG Analysis
Future research recommendations are generated using predefined mappings between detected gap categories and suggested research directions. Consequently, recommendations are rule-based rather than generated through a learned personalization model.

The SDG module analyzes the SDG labels associated with papers and calculates their frequency within the analyzed subset. The resulting summaries describe dataset labels and should not be interpreted as independent assessments of a paper's real-world impact.

### 3.6 Retrieval Evaluation
The corrected evaluation uses five queries, ten results per query, and two retrieval methods, producing 100 query-result rows. The queries are:

1. AI for early disease detection — Healthcare
2. Machine learning for crop disease prediction — Agriculture
3. Explainable AI for cybersecurity — Computer Science
4. AI-based personalized learning — Education
5. Deep learning for climate prediction — Environmental Science

Precision@10 is the proportion of the ten retrieved results labeled relevant. The pooled recall estimate uses relevant documents observed in the combined top-10 result pools from both methods for each query. F1 is calculated from precision and this pooled recall estimate. Because the complete set of relevant documents is unknown, pooled recall and F1 are estimates, not corpus-wide metrics.

The relevance labels in the current evaluation are provisional and have not been independently validated by multiple expert assessors. Two previously conflicting labels were aligned across the corrected results and review files after inspecting the paper abstracts. This alignment does not replace independent relevance assessment.

### 3.7 Timing Evaluation
The repeated timing experiment includes five queries and five runs per query for each method. It measures ranking operations on prepared candidates. For semantic retrieval, the reported timing represents vector scoring and ranking, not complete end-to-end FAISS search. It excludes query embedding generation, model loading, and other application overhead. The timing results must not be presented as a direct end-to-end latency comparison.

## 4. Results

### 4.1 Software and Hardware Environment
The recorded experiment environment was Python 3.13.14 on Windows 11 (build 10.0.26200), using an Intel64 Family 6 Model 154 Stepping 3 CPU; no GPU was available. Recorded package versions were pandas 2.3.2, NumPy 2.3.3, faiss-cpu 1.15.0, sentence-transformers 5.6.1, Transformers 5.14.1, PyTorch 2.11.0, rank-bm25 0.2.2, and Streamlit 1.63.0. These versions are taken from `experiment_environment.txt`.



### 4.2 Aggregate Retrieval Performance

| Method | Mean Precision@10 | Mean Pooled Recall Estimate@10 | Mean Pooled F1 Estimate@10 | Mean nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.2000 | 0.3467 | 0.2488 | 0.3458 |
| Semantic (MiniLM + FAISS) | 0.2800 | 0.7429 | 0.3797 | 0.7062 |

Semantic retrieval achieved higher mean Precision@10 (0.2800 vs. 0.2000), pooled recall estimate (0.7429 vs. 0.3467), pooled F1 estimate (0.3797 vs. 0.2488), and nDCG@10 (0.7062 vs. 0.3458) than BM25 in this five-query evaluation. Recall and F1 use the relevant documents observed in the combined top-10 result pools; they are estimates, not corpus-wide metrics, because the complete relevant-document set is unknown. All relevance labels remain provisional pending independent assessment.

These results favor semantic retrieval on average for this small query set. The relevance labels remain provisional and have not been independently validated by multiple expert assessors. The findings should not be treated as conclusive.

### 4.3 Query-Level Results

| Query | Method | Relevant@10 | Precision@10 | Pooled Recall Estimate@10 | Pooled F1 Estimate@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| AI for early disease detection | BM25 | 1 | 0.1000 | 0.3333 | 0.1538 | 0.3562 |
| AI for early disease detection | Semantic | 3 | 0.3000 | 1.0000 | 0.4615 | 0.6183 |
| AI-based personalized learning | BM25 | 7 | 0.7000 | 1.0000 | 0.8235 | 0.9611 |
| AI-based personalized learning | Semantic | 5 | 0.5000 | 0.7143 | 0.5882 | 0.9747 |
| Deep learning for climate prediction | BM25 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Deep learning for climate prediction | Semantic | 1 | 0.1000 | 1.0000 | 0.1818 | 1.0000 |
| Explainable AI for cybersecurity | BM25 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Explainable AI for cybersecurity | Semantic | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Machine learning for crop disease prediction | BM25 | 2 | 0.2000 | 0.4000 | 0.2667 | 0.4118 |
| Machine learning for crop disease prediction | Semantic | 5 | 0.5000 | 1.0000 | 0.6667 | 0.9379 |
Semantic retrieval performed better for early disease detection, climate prediction, and crop disease prediction. BM25 performed better for personalized learning. Both methods returned no results labeled relevant for explainable AI for cybersecurity. This variation indicates that retrieval performance depends on the query and domain rather than one method uniformly outperforming the other.

### 4.4 Repeated Timing Results
The repeated ranking-only timing measurements recorded the following mean times:

| Method | Mean ranking time | Standard deviation |
|---|---:|---:|
| BM25 | 0.768540 ms | 0.252552 ms |
| Semantic vector scoring and ranking | 0.065664 ms | 0.023416 ms |

These values represent the measured ranking operations only. They do not establish that the complete semantic search pipeline is faster than BM25 because embedding generation, full FAISS search, and other overheads are excluded.

## 5. Discussion
The preliminary evaluation indicates that semantic representations can improve retrieval for several queries where relevant papers may use terminology different from the query wording. The higher aggregate Precision@10 and nDCG@10, along with the pooled F1 estimate, support further investigation of semantic retrieval for cross-domain research discovery. The pooled recall and F1 values are estimates based only on relevant documents observed in the combined top-10 result pools, not corpus-wide metrics.

Nevertheless, the personalized-learning query favored BM25, and the cybersecurity query produced no relevant results under either method. These cases emphasize the need for query diversity, careful corpus inspection, and stronger relevance assessment.

The gap-analysis component provides structured summaries of recurring limitation-related expressions. Since it uses predefined patterns, its output may be affected by wording, context, negation, and differences in academic writing style. The recommendations are mapped from gap categories rather than learned from researcher feedback. SDG analysis similarly depends on the quality and coverage of the available labels.

## 6. Limitations

Independent expert validation of the research-gap detection output has not been completed or documented. Therefore, no claim is made that the detected gaps have been validated by domain experts. The retrieval relevance labels are provisional. Two previously conflicting labels were aligned across the corrected results and review files after inspection of the paper abstracts, but independent multi-assessor validation has not been completed.
1. The retrieval evaluation contains only five queries and ten results per method per query.
2. Relevance labels are provisional and have not been independently validated by multiple expert assessors.
3. The cybersecurity query yielded no results labeled relevant for either method; additional query formulations and corpus checks are needed.
4. The pooled recall and F1 estimates use relevant documents observed in the combined top-10 result pools. They are not corpus-wide metrics because the complete relevant-document set for each query is unknown.
5. Timing results measure prepared-candidate ranking operations, not end-to-end retrieval latency.
6. The semantic model and gap patterns may not capture all domain-specific terminology or research limitations.
7. Gap recommendations and SDG explanations are rule-based and should not be treated as expert judgments.
8. The equal allocation of 220 papers per domain creates a balanced evaluation corpus but may not reflect the actual distribution of publications across disciplines.
9. The current evaluation does not establish statistical significance or generalization to a larger set of unseen queries.

## 7. Future Work
Future work should include a larger and more diverse query benchmark, independent relevance judgments by domain experts, and inter-annotator agreement analysis. The evaluation should define the complete relevant-document pool per query and report metrics such as nDCG@10 and Recall@10 only when their required relevance judgments are available.

The retrieval system could also be evaluated end to end, including query encoding, candidate generation, domain filtering, and ranking. Hybrid retrieval combining BM25 with semantic similarity could be tested. Research-gap detection could be improved through contextual classification and human validation, while recommendations could be evaluated for usefulness by researchers. The SDG module would benefit from validation against the original paper content.

## 8. Conclusion
ResearchLens AI combines semantic academic-paper retrieval, rule-based research-gap detection, future-direction recommendations, and SDG summaries in a single research discovery application. In a preliminary five-query evaluation, semantic retrieval achieved higher mean Precision@10, pooled F1 estimate, and nDCG@10 than BM25. The pooled F1 values are estimates based on relevant documents observed in the combined top-10 result pools, not corpus-wide metrics. The query-level results show that the advantage is not uniform. Due to the limited evaluation size, provisional relevance labels, and ranking-only timing measurements, these findings should be treated as preliminary. A broader benchmark and independent expert evaluation are necessary before drawing general conclusions about retrieval quality or research-gap detection accuracy.

## References
The following foundational references should be checked against the implementation and formatted to the target journal or conference style before submission:

1. Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP-IJCNLP.
2. Robertson, S., and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.
3. Johnson, J., Douze, M., and Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.
4. Priem, J., Piwowar, H., and Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint.

## Reproducibility Notes
- Dataset: `data/research_papers_real.csv`
- Embeddings: `models/embeddings.pkl`
- FAISS index: `models/faiss_index.bin`
- Embedding model: `all-MiniLM-L6-v2`
- Environment details: `experiment_environment.txt`
- Evaluation script: `corrected_experiment.py`
- Corrected retrieval rows: `corrected_retrieval_results.csv`
- Recalculated aggregate metrics: `recalculated_metrics_summary.csv`
- Recalculated query-level metrics: `recalculated_metrics_by_query.csv`
- Relevance review file: `relevance_review_completed.csv`
- Timing measurements: `repeated_timing_results.csv`


## Relevance-Label Discrepancy Audit
A comparison of `corrected_retrieval_results.csv` and `relevance_review_completed.csv` identified two conflicting labels for the same query, method, rank, and paper:
- Healthcare, "AI for early disease detection", semantic retrieval, rank 4, Paper_ID `W3036858161`: corrected results label 1; review file label 0.
- Environmental Science, "Deep learning for climate prediction", semantic retrieval, rank 1, Paper_ID `W4382315652`: corrected results label 0; review file label 1.

These disagreements must be resolved by checking the paper content and applying a consistent relevance criterion. Until then, the relevance labels and metrics should be described as provisional. No label has been silently changed in this draft.


## Final Metric Verification Status
The corrected retrieval file was checked for 100 rows, binary relevance labels, and ten ranked results per query-method pair. Metrics were recalculated and saved to `verified_metrics_by_query.csv` and `verified_metrics_summary.csv`. Two previously conflicting labels were aligned across the corrected retrieval and review files after inspecting the paper abstracts. These judgments remain provisional and have not been independently validated by multiple assessors; therefore, this is not a final validation of relevance judgments.


## Embedding Provenance Confirmation Status
The repository's `modules/embeddings.py` code constructs embedding text from `Title`, `Abstract`, `Topics`, and `Keywords`, replacing missing values with empty strings. This establishes the fields specified by the current code. Confirmation from the project team that the committed embedding artifact was generated from this exact version of the pipeline remains pending and must not be represented as completed.
