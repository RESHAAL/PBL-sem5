#!/usr/bin/env python
"""Recompute ResearchLens P@10, Recall@10, F1@10 and nDCG@10.

IMPORTANT: Suggested_Relevance is an automated provisional label, not a human judgment.
Usage:
 python recompute_researchlens_metrics.py retrieval_top100.csv corpus_suggested_labels.csv
"""
import sys, math
import pandas as pd

retrieval = pd.read_csv(sys.argv[1])
labels = pd.read_csv(sys.argv[2])
joined = retrieval.merge(
    labels[["Query", "Paper_ID", "Suggested_Relevance"]],
    on=["Query", "Paper_ID"], how="left", validate="many_to_one"
)
if joined["Suggested_Relevance"].isna().any():
    raise ValueError("Missing query-paper labels for retrieved results.")
if len(labels) != 1100 or labels.duplicated(["Query","Paper_ID"]).any():
    raise ValueError("Expected 1,100 unique query-paper judgments (220 x 5).")

corpus_rel = labels.groupby("Query")["Suggested_Relevance"].sum().to_dict()
rows = []
for (query, domain, method), group in joined.groupby(["Query","Domain","Method"]):
    group = group.sort_values("Rank")
    rels = group["Suggested_Relevance"].astype(int).tolist()
    hits = sum(rels)
    p = hits / 10
    total = int(corpus_rel[query])
    r = hits / total if total else float("nan")
    f = 2*p*r/(p+r) if total and p+r else (0.0 if total else float("nan"))
    dcg = sum(v/math.log2(rank+1) for rank,v in enumerate(rels,1))
    # IDCG from all 220 corpus items for this query, not the retrieved top 10.
    ideal = sorted(labels.loc[labels.Query == query, "Suggested_Relevance"].astype(int),
                   reverse=True)[:10]
    idcg = sum(v/math.log2(rank+1) for rank,v in enumerate(ideal,1))
    ndcg = dcg/idcg if idcg else float("nan")
    rows.append({"Query":query,"Domain":domain,"Method":method,"Relevant_in_Top10":hits,
                 "Corpus_suggested_relevant_n":total,"Precision@10":p,"Recall@10":r,
                 "F1@10":f,"DCG@10":dcg,"IDCG@10_corpus220":idcg,"nDCG@10":ndcg,
                 "Label_status":"AI-suggested; not human-verified"})
per = pd.DataFrame(rows)
per.to_csv("metrics_per_query.csv", index=False)
macro = per.groupby("Method").agg({
    "Precision@10":"mean", "Recall@10":"mean", "F1@10":"mean", "nDCG@10":"mean"
}).reset_index()
macro.to_csv("metrics_macro.csv", index=False)
print(per.to_string(index=False))
print(macro.to_string(index=False))
