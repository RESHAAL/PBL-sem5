ResearchLens rapid completion package
1. 01_audit_labels_AI_assisted.csv — 100 retrieval rows with AI-assisted labels, rationale, and explicit human-confirmation status.
2. 02_metrics_per_query.csv — per-query P@10 and nDCG@10.
3. 03_metrics_macro_summary.csv — macro summaries.
4. 04_criteria_limitations_comparison.txt — criteria, caveats, previous-results comparison note.
5. 04_calculate_metrics.py — reproducible metric calculator.

Important: Labels are AI-assisted, not independently human-verified. Recall/F1 are omitted because corpus-wide relevant totals are not verified. Do not report the current metrics as final ground truth without human validation.
