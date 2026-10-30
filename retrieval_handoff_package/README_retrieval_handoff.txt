ResearchLens retrieval handoff package

Files:
- document_level_retrieval_results.csv: 100 rows with query, domain, method, rank, Paper_ID, title, abstract, score, and current Relevant value.
- relevance_label_audit_sheet.csv: same results plus blank Audited_Relevant/reviewer fields. Existing Relevant values are preserved but are NOT certified as independently audited.
- total_relevant_documents_template.csv: one row per approved query. Fill total relevant papers in the entire assigned-domain corpus only after reviewing that corpus.
- calculate_retrieval_metrics.py: reproducible metric calculation script.

Current label status:
The uploaded file marks Audit_Status as "Needs human confirmation" for all rows. Therefore current labels and metrics remain provisional. Do not describe the label sheet as audited until Audited_Relevant has been reviewed and completed.

Why corpus counts are needed:
The retrieved top 10 cannot establish the total number of relevant documents in the 220-paper domain corpus. Recall@10 and F1@10 are intentionally left blank until those counts are supplied.

After audit:
1. Fill Audited_Relevant with 0/1 for every retrieved row after title+abstract review.
2. Replace the Relevant column in document_level_retrieval_results.csv with those audited values (or export the audited sheet using that column name).
3. Fill Total_Relevant_Documents_in_Domain_Corpus in total_relevant_documents_template.csv based on a documented relevance assessment of all 220 papers for each query.
4. Save the counts template as total_relevant_documents.csv.
5. Run: python calculate_retrieval_metrics.py document_level_retrieval_results.csv total_relevant_documents.csv

Metrics:
Precision@10 is relevant results among top 10 divided by 10.
Recall@10 is relevant results among top 10 divided by total relevant documents in the full assigned-domain corpus.
F1@10 is harmonic mean of Precision@10 and Recall@10.
Binary nDCG@10 is computed from the 0/1 relevance labels and rank positions.
