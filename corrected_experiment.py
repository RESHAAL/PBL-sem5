import os, pickle, time, platform, importlib.metadata as md
import numpy as np
import pandas as pd
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = "data/research_papers_real.csv"
EMB = "models/embeddings.pkl"
IDX = "models/faiss_index.bin"
QUERIES = [
    ("AI for early disease detection", "Healthcare"),
    ("Machine learning for crop disease prediction", "Agriculture"),
    ("Explainable AI for cybersecurity", "Computer Science"),
    ("AI-based personalized learning", "Education"),
    ("Deep learning for climate prediction", "Environmental Science"),
]
RUNS = 5
TOP_K = 10

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def tokens(s):
    import re
    return re.findall(r"\b\w+\b", str(s).lower())

def main():
    df = pd.read_csv(DATA).drop_duplicates("Paper_ID").reset_index(drop=True)
    for c in ["Title","Abstract","Topics","Keywords","Domain"]:
        if c not in df: raise ValueError(f"Missing column: {c}")
        df[c] = df[c].map(clean)
    df["search_text"] = (df["Title"]+" "+df["Abstract"]+" "+df["Topics"]+" "+df["Keywords"]).str.strip()
    df["Paper_ID"] = df["Paper_ID"].astype(str).str.strip()

    with open(EMB,"rb") as f: emb_data = pickle.load(f)
    ids = [str(x).strip() for x in emb_data["paper_ids"]]
    index = faiss.read_index(IDX)
    if len(ids) != index.ntotal or len(set(ids)) != len(ids):
        raise ValueError("Embedding IDs and FAISS index size/uniqueness mismatch")
    if not set(ids).issubset(set(df["Paper_ID"])):
        raise ValueError("Embedding paper IDs do not all map to dataset")
    if len(emb_data["embeddings"]) != len(ids):
        raise ValueError("Saved embeddings count does not match paper IDs")
    print("Dataset papers:",len(df),"Embedding vectors:",len(ids),"FAISS vectors:",index.ntotal)
    print("ID mapping verified. Text-field provenance must still be checked against the embedding-generation code.")

    # Use one shared, domain-specific candidate set for fair ranking comparison.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    id_to_row = df.set_index("Paper_ID",drop=False)
    id_to_vector = {pid: np.asarray(emb_data["embeddings"][i],dtype=np.float32) for i,pid in enumerate(ids)}
    rows, timings = [], []
    for query, domain in QUERIES:
        candidates = df[df["Domain"] == domain].copy()
        if candidates.empty: raise ValueError(f"No papers for domain {domain}")
        corpus = [tokens(x) for x in candidates["search_text"]]
        bm25 = BM25Okapi(corpus)
        q_tokens = tokens(query)
        q_vec = model.encode([query],normalize_embeddings=True).astype("float32")[0]
        missing = [pid for pid in candidates["Paper_ID"] if pid not in id_to_vector]
        if missing: raise ValueError(f"Missing saved vectors for {len(missing)} candidate papers")
        vectors = np.vstack([id_to_vector[pid] for pid in candidates["Paper_ID"]])
        if vectors.shape[1] != q_vec.shape[0]: raise ValueError("Embedding dimensions mismatch")
        # Same candidate papers for both methods; cosine similarity for semantic ranking.
        semantic_scores = vectors @ q_vec
        bm25_scores = bm25.get_scores(q_tokens)
        candidates = candidates.reset_index(drop=True)
        bm_order = np.argsort(bm25_scores)[::-1][:TOP_K]
        sem_order = np.argsort(semantic_scores)[::-1][:TOP_K]
        for method, order, scores in [
            ("BM25",bm_order,bm25_scores),
            ("Semantic (MiniLM + FAISS vectors)",sem_order,semantic_scores)
        ]:
            for rank, i in enumerate(order,1):
                p = candidates.iloc[int(i)]
                rows.append({"Query":query,"Domain":domain,"Method":method,"Rank":rank,
                    "Paper_ID":p["Paper_ID"],"Title":p["Title"],"Score":float(scores[i]),"Relevant":""})
        # Warm-up already done above; time repeated ranking on the same candidate set.
        for run in range(1,RUNS+1):
            t0=time.perf_counter(); bm25.get_scores(q_tokens); np.argsort(bm25_scores)[::-1][:TOP_K]
            timings.append({"Query":query,"Method":"BM25","Run":run,"Latency_ms":(time.perf_counter()-t0)*1000})
            t0=time.perf_counter(); vectors @ q_vec; np.argsort(semantic_scores)[::-1][:TOP_K]
            timings.append({"Query":query,"Method":"Semantic (MiniLM + FAISS vectors)","Run":run,"Latency_ms":(time.perf_counter()-t0)*1000})

    pd.DataFrame(rows).to_csv("corrected_retrieval_results.csv",index=False)
    pd.DataFrame(timings).to_csv("repeated_timing_results.csv",index=False)
    tdf=pd.DataFrame(timings).groupby(["Query","Method"])["Latency_ms"].agg(["mean","std"]).reset_index()
    tdf.to_csv("timing_summary.csv",index=False)
    with open("experiment_environment.txt","w",encoding="utf-8") as f:
        f.write(f"Python: {platform.python_version()}\nOS: {platform.platform()}\nCPU: {platform.processor()}\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}\n")
        for p in ["pandas","numpy","faiss-cpu","sentence-transformers","transformers","torch","rank-bm25","streamlit"]:
            try: f.write(f"{p}: {md.version(p)}\n")
            except md.PackageNotFoundError: f.write(f"{p}: not installed\n")
    print("Saved corrected_retrieval_results.csv")
    print("Saved repeated_timing_results.csv and timing_summary.csv")
    print("Saved experiment_environment.txt")
    print("Next: manually label Relevant in corrected_retrieval_results.csv before calculating relevance metrics.")

if __name__ == "__main__": main()
