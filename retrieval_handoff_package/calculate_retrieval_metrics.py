#!/usr/bin/env python3
"""Calculate retrieval metrics from top-10 results and audited labels.

Usage:
  python calculate_retrieval_metrics.py document_level_retrieval_results.csv total_relevant_documents.csv

total_relevant_documents.csv must contain:
  Query,Total_Relevant_Documents_in_Domain_Corpus
Counts must be established by judging the entire assigned-domain evaluation corpus,
not inferred from the retrieved top 10. Leave counts blank until audited.
"""
import sys, math
import pandas as pd
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit("Usage: python calculate_retrieval_metrics.py results.csv total_relevant_documents.csv")
results_path, counts_path = sys.argv[1:]
df = pd.read_csv(results_path)
counts = pd.read_csv(counts_path)
required = {"Query", "Method", "Rank", "Relevant"}
if not required.issubset(df.columns):
    raise ValueError(f"Results CSV missing columns: {sorted(required-set(df.columns))}")
if "Total_Relevant_Documents_in_Domain_Corpus" not in counts.columns:
    raise ValueError("Counts CSV must have Total_Relevant_Documents_in_Domain_Corpus column")
df["Relevant"] = pd.to_numeric(df["Relevant"], errors="coerce")
if df["Relevant"].isna().any() or not df["Relevant"].isin([0,1]).all():
    raise ValueError("Relevant must contain only audited 0/1 labels; blank/non-binary labels found.")
rows=[]
for (query, method), group in df.groupby(["Query","Method"]):
    group=group.sort_values("Rank").head(10)
    rel=group["Relevant"].astype(int).tolist()
    k=len(rel)
    p=sum(rel)/10.0
    dcg=sum(r/math.log2(i+2) for i,r in enumerate(rel))
    ideal=sorted(rel, reverse=True)
    idcg=sum(r/math.log2(i+2) for i,r in enumerate(ideal))
    ndcg=dcg/idcg if idcg else np.nan
    match=counts[counts["Query"]==query]
    total=None
    if len(match)==1:
        val=match.iloc[0]["Total_Relevant_Documents_in_Domain_Corpus"]
        if pd.notna(val) and str(val).strip():
            total=float(val)
    recall=(sum(rel)/total) if total and total>0 else np.nan
    f1=(2*p*recall/(p+recall)) if pd.notna(recall) and (p+recall)>0 else np.nan
    rows.append({"Query":query,"Method":method,"Retrieved_at_10":k,"Relevant_at_10":sum(rel),
                 "Precision@10":p,"Total_relevant_in_corpus":total,"Recall@10":recall,
                 "F1@10":f1,"nDCG@10_binary":ndcg})
out=pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("retrieval_metrics_calculated.csv",index=False)
print("\\nSaved retrieval_metrics_calculated.csv")
if out["Recall@10"].isna().any():
    print("WARNING: Recall/F1 are blank where corpus-wide relevant-document counts are missing.")
