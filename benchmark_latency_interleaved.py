"""
ResearchLens interleaved latency benchmark (repeat on the project machine).

Run from project root:
  python benchmark_latency_interleaved.py

Requires the project's existing dependencies:
  pandas numpy faiss-cpu rank-bm25 sentence-transformers

Protocol:
- Same five approved exact queries; each query uses its domain's 220 papers.
- BM25 search-only: score an already-tokenized query.
- BM25 end-to-end: tokenize query + score.
- FAISS search-only: index.search with an already-encoded normalized query vector.
- Semantic end-to-end: encode query with all-MiniLM-L6-v2 + FAISS search.
- Model loading, corpus loading, embedding loading, index construction are setup and excluded.
- 5 warmups per measurement path, then 50 timed repetitions.
- Four measurement paths are interleaved in randomized order on each repeat to reduce
  systematic order/thermal effects. Raw per-repeat timings are exported.
"""

from pathlib import Path
import re, time, pickle, platform, random, sys
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "research_papers_real.csv"
EMB = ROOT / "models" / "embeddings.pkl"
OUT = ROOT / "benchmark_results_interleaved"
MODEL = "all-MiniLM-L6-v2"
WARMUPS = 5
REPEATS = 50
K = 10
SEED = 20261031

QUERIES = [
    ("AI for early disease detection", "Healthcare"),
    ("Machine learning for crop disease prediction", "Agriculture"),
    ("Explainable AI for cybersecurity", "Computer Science"),
    ("AI-based personalized learning", "Education"),
    ("AI for climate prediction", "Environmental Science"),
]

def tokenize(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())

def main():
    if not DATA.exists() or not EMB.exists():
        raise FileNotFoundError(
            f"Expected {DATA} and {EMB}. Put this script in the project root."
        )

    df = pd.read_csv(DATA)
    with open(EMB, "rb") as f:
        saved = pickle.load(f)

    ids = [str(x) for x in saved["paper_ids"]]
    vectors = np.asarray(saved["embeddings"], dtype=np.float32)
    if df["Paper_ID"].astype(str).tolist() != ids:
        raise ValueError("Dataset Paper_ID order does not match saved embedding IDs.")
    if vectors.shape != (len(df), 384):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

    model = SentenceTransformer(MODEL)
    OUT.mkdir(exist_ok=True)
    raw_rows = []
    summary_rows = []
    rng = random.Random(SEED)

    for query, domain in QUERIES:
        mask = df["Domain"].astype(str).str.strip().eq(domain).to_numpy()
        positions = np.flatnonzero(mask)
        sub = df.iloc[positions]
        if len(sub) != 220:
            raise ValueError(f"{domain}: expected 220 papers, found {len(sub)}")

        docs = (
            sub["Title"].fillna("").astype(str) + " " +
            sub["Abstract"].fillna("").astype(str) + " " +
            sub["Topics"].fillna("").astype(str) + " " +
            sub["Keywords"].fillna("").astype(str)
        ).tolist()
        bm25 = BM25Okapi([tokenize(x) for x in docs])
        query_tokens = tokenize(query)

        doc_vectors = np.ascontiguousarray(vectors[positions], dtype=np.float32)
        index = faiss.IndexFlatIP(doc_vectors.shape[1])
        index.add(doc_vectors)
        query_vector = np.asarray(
            model.encode([query], normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32
        )

        def bm25_search_only():
            return bm25.get_scores(query_tokens)

        def bm25_end_to_end():
            return bm25.get_scores(tokenize(query))

        def faiss_search_only():
            return index.search(query_vector, K)

        def semantic_end_to_end():
            qv = np.asarray(
                model.encode([query], normalize_embeddings=True, convert_to_numpy=True),
                dtype=np.float32
            )
            return index.search(qv, K)

        funcs = {
            ("BM25", "search_only"): bm25_search_only,
            ("BM25", "end_to_end_query"): bm25_end_to_end,
            ("Semantic_FAISS", "search_only"): faiss_search_only,
            ("Semantic_FAISS", "end_to_end_query"): semantic_end_to_end,
        }

        # Warm each path separately, outside timed samples.
        for fn in funcs.values():
            for _ in range(WARMUPS):
                fn()

        samples = {key: [] for key in funcs}
        # Interleave all four paths on each repeat in a deterministic shuffled order.
        keys = list(funcs)
        for rep in range(1, REPEATS + 1):
            order = keys.copy()
            rng.shuffle(order)
            for method, latency_type in order:
                start = time.perf_counter_ns()
                funcs[(method, latency_type)]()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                samples[(method, latency_type)].append(elapsed_ms)
                raw_rows.append({
                    "Query": query, "Domain": domain, "Method": method,
                    "Latency_type": latency_type, "Repeat": rep,
                    "Order_within_repeat": order.index((method, latency_type)) + 1,
                    "Latency_ms": elapsed_ms,
                })

        for (method, latency_type), vals in samples.items():
            a = np.asarray(vals, dtype=float)
            summary_rows.append({
                "Query": query, "Domain": domain, "Method": method,
                "Latency_type": latency_type, "Corpus_size": len(sub),
                "Warmups": WARMUPS, "Repeats": REPEATS,
                "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                "std_ms": float(a.std(ddof=1)), "min_ms": float(a.min()),
                "max_ms": float(a.max()),
            })

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(OUT / "latency_raw_interleaved.csv", index=False)
    summary.to_csv(OUT / "latency_summary_interleaved.csv", index=False)

    env = (
        f"Python: {platform.python_version()}\n"
        f"Platform: {platform.platform()}\n"
        f"NumPy: {np.__version__}\n"
        f"Model: {MODEL}\n"
        f"Queries: {len(QUERIES)}\n"
        f"Corpus: {len(df)}; 220 papers per domain subset\n"
        f"Warmups per path/query: {WARMUPS}\n"
        f"Timed repeats per path/query: {REPEATS}\n"
        "Order: four timing paths randomized/interleaved per repeat.\n"
        "Excluded: corpus load, embeddings load, model load, BM25 build, FAISS index build.\n"
    )
    (OUT / "environment.txt").write_text(env, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved raw timings and summary in: {OUT.resolve()}")

if __name__ == "__main__":
    main()
