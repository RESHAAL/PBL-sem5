#!/usr/bin/env python3
"""Compute P@10 and binary nDCG@10 from the audit CSV.
Uses Audited_Relevant when every row has it; otherwise refuses to label results final.
"""
import argparse
import numpy as np
import pandas as pd

def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))

p = argparse.ArgumentParser()
p.add_argument("audit_csv", help="Audit CSV with Query, Method, Rank, Audited_Relevant")
p.add_argument("--allow-provisional", action="store_true", help="Use Relevant column if audited labels are incomplete")
p.add_argument("--out", default="metrics.csv")
a = p.parse_args()
df = pd.read_csv(a.audit_csv)
label_col = "Audited_Relevant"
if label_col not in df or df[label_col].isna().any():
    if not a.allow_provisional:
        raise SystemExit("Audited_Relevant is missing/incomplete. Finish human audit or pass --allow-provisional.")
    label_col = "Relevant"
df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
if df[label_col].isna().any():
    raise SystemExit(f"{label_col} contains blank/non-numeric labels.")
rows=[]
for (q,m), g in df.groupby(["Query","Method"], sort=False):
    g=g.sort_values("Rank").head(10)
    rel=g[label_col].astype(int).tolist()
    ideal=sorted(rel, reverse=True)
    idcg=dcg(ideal)
    rows.append({"Query":q,"Method":m,"N":len(g),"Relevant_at_10":sum(rel),
                 "Precision@10":sum(rel)/10 if len(g)==10 else np.nan,
                 "nDCG@10":dcg(rel)/idcg if idcg>0 else np.nan,
                 "nDCG_status":"Defined" if idcg>0 else "Undefined: no relevant item in top 10",
                 "Label_source":label_col})
out=pd.DataFrame(rows)
macro=out.groupby("Method").agg(Precision_at_10_macro=("Precision@10","mean"),
    nDCG_at_10_defined_queries=("nDCG@10","mean")).reset_index()
out.to_csv(a.out,index=False)
macro.to_csv(a.out.replace(".csv","_macro.csv"),index=False)
print(out.to_string(index=False))
print("\nMacro averages:\n",macro.to_string(index=False))
print("\nWrote",a.out,"and",a.out.replace(".csv","_macro.csv"))
