ResearchLens AI retrieval experiment handoff
Files:
01_correct_query_top10_results.csv — 100 top-10 rows from the exact five approved queries; labels blank in the source run.
02_relevance_label_audit_sheet.csv — current audit worksheet; existing Relevant values are provisional, Audited_Relevant is blank.
03_provisional_metrics_per_query.csv — metrics from provisional labels, not final ground truth.
04_labeling_criteria_and_limitations.txt — scope, criteria, limitations, reproducibility steps.
05_calculate_metrics.py — reproducible P@10 and binary nDCG@10 calculator; by default requires completed Audited_Relevant.
06_previous_results_comparison_caution.txt — comparison context and warnings.

Key limitation: Recall and F1 are not final because corpus-wide relevant-document totals are not verified. Finish title+abstract audit before reporting final metrics.
