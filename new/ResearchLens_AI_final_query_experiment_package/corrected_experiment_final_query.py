"""
Final controlled BM25 vs MiniLM/FAISS retrieval experiment.
Fifth query confirmed by the user: "AI for climate prediction".

Run from the project root:
    python corrected_experiment_final_query.py

Outputs:
    final_query_retrieval_results.csv
    final_query_timing_runs.csv
    final_query_timing_summary.csv
    final_query_environment.txt

Important:
- Uses the same domain-specific candidate set for BM25 and FAISS.
- Uses saved MiniLM document embeddings and freshly encoded query embeddings.
- Relevance labels are intentionally blank; manually review titles/abstracts
  before calculating evaluation metrics.
- Latency excludes model loading/query encoding and index construction.
"""
import os
import pickle
import time
import platform
import importlib.metadata as md

import numpy as np
import pandas as pd
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = "data/research_papers_real.csv"
EMB = "models/embeddings.pkl"
RUNS = 5
TOP_K = 10
MODEL_NAME = "all-MiniLM-L6-v2"

QUERIES = [
    ("AI for early disease detection", "Healthcare"),
    ("Machine learning for crop disease prediction", "Agriculture"),
    ("Explainable AI for cybersecurity", "Computer Science"),
    ("AI-based personalized learning", "Education"),
    ("AI for climate prediction", "Environmental Science"),
]

def clean(value):
    return "" if pd.isna(value) else str(value).strip()

def tokenize(value):
    import re
    return re.findall(r"\b\w+\b", str(value).lower())

def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    if not os.path.exists(EMB):
        raise FileNotFoundError(f"Embeddings not found: {EMB}")

    df = pd.read_csv(DATA).drop_duplicates("Paper_ID").reset_index(drop=True)
    required = ["Paper_ID", "Title", "Abstract", "Topics", "Keywords", "Domain"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing dataset columns: {missing_cols}")

    for col in required:
        df[col] = df[col].map(clean)
    df["Paper_ID"] = df["Paper_ID"].astype(str).str.strip()
    df["search_text"] = (
        df["Title"] + " " + df["Abstract"] + " " +
        df["Topics"] + " " + df["Keywords"]
    ).str.strip()

    # Load trusted project-local embedding artifact. Validate IDs and dimensions.
    with open(EMB, "rb") as handle:
        embedding_data = pickle.load(handle)

    paper_ids = [str(x).strip() for x in embedding_data["paper_ids"]]
    all_vectors = np.asarray(embedding_data["embeddings"], dtype=np.float32)

    if all_vectors.ndim != 2:
        raise ValueError("Saved embeddings must be a 2-D matrix.")
    if len(paper_ids) != len(all_vectors):
        raise ValueError("Embedding IDs and vectors count mismatch.")
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Duplicate Paper_IDs in embedding artifact.")
    if not set(paper_ids).issubset(set(df["Paper_ID"])):
        raise ValueError("Some embedding Paper_IDs are absent from dataset.")
    if not np.isfinite(all_vectors).all():
        raise ValueError("Embedding artifact contains non-finite values.")

    vector_by_id = {pid: all_vectors[i] for i, pid in enumerate(paper_ids)}
    model = SentenceTransformer(MODEL_NAME)

    result_rows = []
    timing_rows = []

    print(f"Dataset rows after Paper_ID deduplication: {len(df)}")
    print(f"Saved embeddings: {len(paper_ids)}; dimensions: {all_vectors.shape[1]}")
    print("Confirmed fifth query: AI for climate prediction")

    for query, domain in QUERIES:
        candidates = df[df["Domain"] == domain].copy().reset_index(drop=True)
        if len(candidates) != 220:
            print(f"WARNING: {domain} has {len(candidates)} candidates, not 220.")

        ids = candidates["Paper_ID"].tolist()
        absent = [pid for pid in ids if pid not in vector_by_id]
        if absent:
            raise ValueError(f"{len(absent)} candidate papers lack embeddings in {domain}.")

        doc_vectors = np.vstack([vector_by_id[pid] for pid in ids]).astype(np.float32)
        query_vector = model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        if doc_vectors.shape[1] != query_vector.shape[1]:
            raise ValueError("Document/query embedding dimensions mismatch.")

        # Ensure vectors are normalized because IndexFlatIP then implements cosine ranking.
        norms = np.linalg.norm(doc_vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "Saved document embeddings are not unit-normalized. "
                "Regenerate embeddings using normalize_embeddings=True."
            )

        faiss_index = faiss.IndexFlatIP(doc_vectors.shape[1])
        faiss_index.add(doc_vectors)

        bm25 = BM25Okapi([tokenize(x) for x in candidates["search_text"]])
        query_tokens = tokenize(query)

        bm_scores = bm25.get_scores(query_tokens)
        bm_order = np.argsort(bm_scores)[::-1][:TOP_K]
        faiss_scores, faiss_order = faiss_index.search(query_vector, TOP_K)
        faiss_order = faiss_order[0]
        faiss_scores = faiss_scores[0]

        for method, order, scores in [
            ("BM25", bm_order, bm_scores),
            ("MiniLM + FAISS IndexFlatIP", faiss_order, faiss_scores),
        ]:
            for rank, idx in enumerate(order, 1):
                row = candidates.iloc[int(idx)]
                result_rows.append({
                    "Query": query,
                    "Domain": domain,
                    "Method": method,
                    "Rank": rank,
                    "Paper_ID": row["Paper_ID"],
                    "Title": row["Title"],
                    "Abstract": row["Abstract"],
                    "Score": float(scores[rank - 1] if method != "BM25" else scores[int(idx)]),
                    "Relevant": ""
                })

        # Query-time benchmark only: scoring + top-10 selection/search.
        # Excludes model loading, query encoding, and index construction.
        for run in range(1, RUNS + 1):
            t0 = time.perf_counter()
            bm_scores_run = bm25.get_scores(query_tokens)
            np.argsort(bm_scores_run)[::-1][:TOP_K]
            timing_rows.append({
                "Query": query, "Domain": domain, "Method": "BM25",
                "Run": run, "Latency_ms": (time.perf_counter() - t0) * 1000
            })

            t0 = time.perf_counter()
            faiss_index.search(query_vector, TOP_K)
            timing_rows.append({
                "Query": query, "Domain": domain,
                "Method": "MiniLM + FAISS IndexFlatIP",
                "Run": run, "Latency_ms": (time.perf_counter() - t0) * 1000
            })

        print(f"Completed: {domain} | {query}")

    results = pd.DataFrame(result_rows)
    timings = pd.DataFrame(timing_rows)
    results.to_csv("final_query_retrieval_results.csv", index=False)
    timings.to_csv("final_query_timing_runs.csv", index=False)
    timings.groupby(["Query", "Domain", "Method"])["Latency_ms"].agg(
        mean_ms="mean", std_ms="std", median_ms="median"
    ).reset_index().to_csv("final_query_timing_summary.csv", index=False)

    with open("final_query_environment.txt", "w", encoding="utf-8") as f:
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"OS: {platform.platform()}\n")
        f.write(f"CPU: {platform.processor()}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}\n")
        f.write(f"Embedding model: {MODEL_NAME}\n")
        f.write("Semantic index: FAISS IndexFlatIP, domain-specific candidate vectors\n")
        f.write("Timing: BM25 scoring+top-10 vs FAISS search; excludes query encoding and index construction\n")
        for package in [
            "pandas", "numpy", "faiss-cpu", "sentence-transformers",
            "transformers", "torch", "rank-bm25"
        ]:
            try:
                f.write(f"{package}: {md.version(package)}\n")
            except md.PackageNotFoundError:
                f.write(f"{package}: not installed\n")

    print("\nSaved:")
    print("- final_query_retrieval_results.csv")
    print("- final_query_timing_runs.csv")
    print("- final_query_timing_summary.csv")
    print("- final_query_environment.txt")
    print("\nIMPORTANT: Review and fill the Relevant column for all 100 rows before metrics.")

if __name__ == "__main__":
    main()
