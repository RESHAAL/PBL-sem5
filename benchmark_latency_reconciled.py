"""
Comparable latency benchmark for ResearchLens AI.
Run from project root:
  python benchmark_latency_reconciled.py

Measures:
- BM25 search-only: BM25 score lookup, query already tokenized.
- BM25 end-to-end query: tokenize query + BM25 score lookup.
- FAISS search-only: index.search, query vector already encoded.
- Semantic end-to-end query: query embedding + FAISS index.search.
Index construction, corpus loading, and model loading are offline setup and excluded
from both query-latency measurements. Same exact five queries, same domain-filtered
220-document subsets, 3 warmups and 30 timed repeats.
"""
from pathlib import Path
import re, time, pickle, platform
import numpy as np, pandas as pd, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'research_papers_real.csv'
EMB=ROOT/'models'/'embeddings.pkl'
OUT=ROOT/'benchmark_results_reconciled'
MODEL='all-MiniLM-L6-v2'
REPEATS=30
WARMUPS=3
K=10
QUERIES=[
 ('AI for early disease detection','Healthcare'),
 ('Machine learning for crop disease prediction','Agriculture'),
 ('Explainable AI for cybersecurity','Computer Science'),
 ('AI-based personalized learning','Education'),
 ('AI for climate prediction','Environmental Science'),
]
def tok(s): return re.findall(r'[a-z0-9]+',str(s).lower())
def measure(fn):
 for _ in range(WARMUPS): fn()
 vals=[]
 for _ in range(REPEATS):
  t=time.perf_counter(); fn(); vals.append((time.perf_counter()-t)*1000)
 return {'mean_ms':float(np.mean(vals)),'median_ms':float(np.median(vals)),
         'std_ms':float(np.std(vals,ddof=1)),'min_ms':float(np.min(vals)),
         'max_ms':float(np.max(vals))}

def main():
 df=pd.read_csv(DATA)
 with open(EMB,'rb') as f: stored=pickle.load(f)
 ids=[str(x) for x in stored['paper_ids']]
 vec=np.asarray(stored['embeddings'],dtype=np.float32)
 if df.Paper_ID.astype(str).tolist()!=ids: raise ValueError('Embedding IDs/order do not match corpus.')
 model=SentenceTransformer(MODEL)
 OUT.mkdir(exist_ok=True)
 rows=[]
 for q,domain in QUERIES:
  idx=np.flatnonzero(df.Domain.astype(str).str.strip().to_numpy()==domain)
  sub=df.iloc[idx]
  texts=(sub.Title.fillna('').astype(str)+' '+sub.Abstract.fillna('').astype(str)+' '+
         sub.Topics.fillna('').astype(str)+' '+sub.Keywords.fillna('').astype(str)).tolist()
  bm=BM25Okapi([tok(x) for x in texts])
  qt=tok(q)
  def bm_search(): return bm.get_scores(qt)
  def bm_e2e(): return bm.get_scores(tok(q))
  qv=np.asarray(model.encode([q],normalize_embeddings=True,convert_to_numpy=True),dtype=np.float32)
  dv=np.ascontiguousarray(vec[idx],dtype=np.float32)
  ix=faiss.IndexFlatIP(dv.shape[1]); ix.add(dv)
  def faiss_search(): return ix.search(qv,K)
  def semantic_e2e():
   qnew=np.asarray(model.encode([q],normalize_embeddings=True,convert_to_numpy=True),dtype=np.float32)
   return ix.search(qnew,K)
  for method,kind,fn in [
   ('BM25','search_only',bm_search),('BM25','end_to_end_query',bm_e2e),
   ('Semantic_FAISS','search_only',faiss_search),('Semantic_FAISS','end_to_end_query',semantic_e2e)]:
   s=measure(fn); rows.append({'Query':q,'Domain':domain,'Method':method,'Latency_type':kind,
                               'Corpus_size':len(sub),'Warmups':WARMUPS,'Repeats':REPEATS,**s})
 out=pd.DataFrame(rows); out.to_csv(OUT/'latency_reconciled.csv',index=False)
 (OUT/'environment.txt').write_text(
  f'Python: {platform.python_version()}\\nPlatform: {platform.platform()}\\n'
  f'Model: {MODEL}\\nCorpus: {len(df)}\\nIndex build/model load excluded from query timings.\\n',
  encoding='utf-8')
 print(out.to_string(index=False))
 print(f'\\nSaved: {OUT}/latency_reconciled.csv')
if __name__=='__main__': main()
