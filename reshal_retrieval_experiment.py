"""
ResearchLens AI: reproducible retrieval run for the five approved queries.
Run from the project root where data/research_papers_real.csv and models/embeddings.pkl exist.

Outputs:
  reshal_retrieval_results_to_label.csv
  reshal_retrieval_run_info.txt

This script intentionally leaves Relevant blank. Relevance must be judged from title + abstract
using a consistent human-defined criterion; do not treat model similarity as relevance.
"""
import os
import re
import pickle
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATASET = os.path.join("data", "research_papers_real.csv")
EMBEDDINGS_FILE = os.path.join("models", "embeddings.pkl")
TOP_K = 10
MODEL_NAME = "all-MiniLM-L6-v2"

QUERIES = [
    ("AI for early disease detection", "Healthcare"),
    ("Machine learning for crop disease prediction", "Agriculture"),
    ("Explainable AI for cybersecurity", "Computer Science"),
    ("AI-based personalized learning", "Education"),
    ("AI for climate prediction", "Environmental Science"),
]

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def tokenize(text):
    return re.findall(r"[a-z0-9]+", clean(text).lower())

def build_text(row):
    # Keep the fields consistent with the project description.
    fields = [row.get("Title", ""), row.get("Abstract", ""),
              row.get("Topics", ""), row.get("Keywords", "")]
    return " ".join(clean(v) for v in fields if clean(v))

def main():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(f"Dataset not found: {DATASET}. Run from project root.")
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings not found: {EMBEDDINGS_FILE}. Run from project root.")

    df = pd.read_csv(DATASET)
    with open(EMBEDDINGS_FILE, "rb") as f:
        stored = pickle.load(f)

    required = {"Paper_ID", "Title", "Abstract", "Domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    if not isinstance(stored, dict) or not {"paper_ids", "embeddings"}.issubset(stored):
        raise ValueError("embeddings.pkl must contain 'paper_ids' and 'embeddings'.")

    paper_ids = [str(x) for x in stored["paper_ids"]]
    vectors = np.asarray(stored["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(paper_ids):
        raise ValueError("Embedding matrix shape does not match paper_ids.")
    if len(df) != len(paper_ids):
        raise ValueError(f"Dataset rows ({len(df)}) != embedding rows ({len(paper_ids)}).")
    if df["Paper_ID"].astype(str).tolist() != paper_ids:
        raise ValueError("Dataset Paper_ID order differs from embeddings.pkl; refusing to run.")

    # Use a stable row index and confirm one paper per ID.
    if df["Paper_ID"].duplicated().any():
        raise ValueError("Duplicate Paper_ID values found; resolve before evaluation.")

    print(f"Dataset rows: {len(df)} | embedding shape: {vectors.shape}")
    print("Loading SentenceTransformer model (first run may download model files)...")
    model = SentenceTransformer(MODEL_NAME)

    out = []
    for query, domain in QUERIES:
        subset_idx = np.flatnonzero(
            df["Domain"].astype(str).str.strip().to_numpy() == domain
        )
        if len(subset_idx) == 0:
            raise ValueError(f"No rows found for domain: {domain}")
        subset = df.iloc[subset_idx].copy()
        texts = [build_text(row) for _, row in subset.iterrows()]

        # BM25 top 10 within the query's assigned domain.
        tokenized = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        q_tokens = tokenize(query)
        bm25_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
        bm25_order = np.argsort(-bm25_scores, kind="stable")[:TOP_K]

        for rank, local_idx in enumerate(bm25_order, start=1):
            row = subset.iloc[int(local_idx)]
            out.append({
                "Query": query, "Domain": domain, "Method": "BM25",
                "Rank": rank, "Paper_ID": clean(row["Paper_ID"]),
                "Title": clean(row["Title"]), "Abstract": clean(row["Abstract"]),
                "Score": float(bm25_scores[int(local_idx)]), "Relevant": ""
            })

        # Semantic top 10 using a real FAISS IndexFlatIP over the same domain subset.
        q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        q_vec = np.asarray(q_vec, dtype=np.float32)
        d_vecs = np.ascontiguousarray(vectors[subset_idx], dtype=np.float32)
        index = faiss.IndexFlatIP(d_vecs.shape[1])
        index.add(d_vecs)
        scores, positions = index.search(q_vec, min(TOP_K, len(subset)))
        for rank, (score, local_idx) in enumerate(zip(scores[0], positions[0]), start=1):
            if int(local_idx) < 0:
                continue
            row = subset.iloc[int(local_idx)]
            out.append({
                "Query": query, "Domain": domain, "Method": "Semantic (MiniLM + FAISS)",
                "Rank": rank, "Paper_ID": clean(row["Paper_ID"]),
                "Title": clean(row["Title"]), "Abstract": clean(row["Abstract"]),
                "Score": float(score), "Relevant": ""
            })
        print(f"Completed: {query} [{domain}]")

    result = pd.DataFrame(out)
    result_path = "reshal_retrieval_results_to_label.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    info = [
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Dataset: {DATASET}",
        f"Dataset rows: {len(df)}",
        f"Embedding matrix shape: {vectors.shape}",
        f"Embedding model for query vectors: {MODEL_NAME}",
        "Retrieval scope: each query restricted to its assigned domain (220 papers expected per domain).",
        "BM25 text fields: Title, Abstract, Topics, Keywords; tokenization uses lowercase alphanumeric tokens.",
        "Semantic retrieval: normalized query embeddings + FAISS IndexFlatIP over stored document vectors in the same domain.",
        f"Top K: {TOP_K}",
        "Relevant labels are intentionally blank and require human audit using title + abstract.",
        f"Output: {result_path}",
    ]
    with open("reshal_retrieval_run_info.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print(f"\nSaved {result_path} ({len(result)} rows)")
    print("Saved reshal_retrieval_run_info.txt")

if __name__ == "__main__":
    main()
