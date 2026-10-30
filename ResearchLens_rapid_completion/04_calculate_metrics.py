#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def dcg(rels):
    return sum((2**int(r)-1)/np.log2(i+2) for i,r in enumerate(rels))

p=argparse.ArgumentParser()
p.add_argument("audit_csv")
p.add_argument("--out",default="recomputed_metrics.csv")
args=p.parse_args()
df=pd.read_csv(args.audit_csv)
col="Audited_Relevant"
if col not in df or df[col].isna().any():
    raise SystemExit("Audited_Relevant must be fully populated.")
df[col]=pd.to_numeric(df[col],errors="raise").astype(int)
if not df[col].isin([0,1]).all():
    raise SystemExit("Labels must be binary 0/1.")
rows=[]
for (q,m),g in df.groupby(["Query","Method"],sort=False):
    g=g.sort_values("Rank").head(10)
    rel=g[col].tolist()
    idcg=dcg(sorted(rel,reverse=True))
    rows.append({"Query":q,"Method":m,"N":len(g),"Relevant_at_10":sum(rel),
        "Precision@10":sum(rel)/10 if len(g)==10 else np.nan,
        "nDCG@10":dcg(rel)/idcg if idcg else np.nan,
        "nDCG_status":"Defined" if idcg else "Undefined (IDCG=0)",
        "Label_source":"Audited_Relevant; check audit provenance before reporting"})
out=pd.DataFrame(rows)
out.to_csv(args.out,index=False)
macro=out.groupby("Method").agg(Macro_Precision_at_10=("Precision@10","mean"),
    Macro_nDCG_at_10_defined_queries=("nDCG@10","mean"),
    Queries_with_defined_nDCG=("nDCG@10","count")).reset_index()
macro.to_csv(args.out.replace(".csv","_macro.csv"),index=False)
print(out.to_string(index=False))
print("\\nMacro summary:\\n",macro.to_string(index=False))
