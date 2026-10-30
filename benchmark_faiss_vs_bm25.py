
"""
BM25 vs FAISS Benchmark
Dataset: data/research_papers_real.csv

Outputs:
    benchmark_results/benchmark_query_results.csv
    benchmark_results/benchmark_summary.csv

This script builds a temporary FAISS index from the selected CSV
and does not overwrite existing project indexes or result files.
"""

from pathlib import Path
import re
import time
import warnings

import numpy as np
import pandas as pd
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "data" / "research_papers_real.csv"

TEXT_COLUMN = "Abstract"
TITLE_COLUMN = "Title"
ID_COLUMN = "Paper_ID"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = PROJECT_DIR / "benchmark_results"

TOP_K = 10
REPEATS = 30
WARMUP_RUNS = 3
BATCH_SIZE = 32

# None = use every valid paper in the CSV.
# Set to 220 only after selecting the exact 220-paper subset
# used by the original experiment.
MAX_PAPERS = None

# Replace these with the exact five queries from your experiment.
QUERIES = [
    "machine learning methods for research paper retrieval",
    "deep learning applications and evaluation",
    "information retrieval using vector similarity",
    "natural language processing for document search",
    "semantic search using dense embeddings",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_corpus():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n"
            "Check that the CSV exists inside the data folder."
        )

    print(f"Loading dataset: {DATASET_PATH.name}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = [TEXT_COLUMN, TITLE_COLUMN]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TEXT_COLUMN]).copy()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()

    df = df[df[TEXT_COLUMN] != ""].reset_index(drop=True)

    if MAX_PAPERS is not None:
        if len(df) < MAX_PAPERS:
            warnings.warn(
                f"Requested {MAX_PAPERS} papers, "
                f"but only {len(df)} valid papers are available."
            )

        df = df.head(MAX_PAPERS).copy().reset_index(drop=True)

    titles = df[TITLE_COLUMN].fillna("").astype(str).tolist()

    if ID_COLUMN in df.columns:
        paper_ids = df[ID_COLUMN].astype(str).tolist()
    else:
        paper_ids = [str(i) for i in range(len(df))]

    texts = []

    for _, row in df.iterrows():
        title = str(row[TITLE_COLUMN]).strip()
        abstract = str(row[TEXT_COLUMN]).strip()

        # Include title in retrieval text for better matching.
        combined_text = f"{title}. {abstract}".strip()
        texts.append(combined_text)

    print(f"Valid papers loaded: {len(texts)}")
    print(f"Text column: {TEXT_COLUMN}")
    print(f"Title column: {TITLE_COLUMN}")

    return texts, titles, paper_ids


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def build_bm25(texts):
    print("\nBuilding BM25 index...")

    tokenized_corpus = [tokenize(text) for text in texts]

    return BM25Okapi(tokenized_corpus)


def bm25_search(index, query):
    scores = index.get_scores(tokenize(query))

    k = min(TOP_K, len(scores))

    indices = np.argsort(scores)[::-1][:k]

    return indices.tolist()


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def build_faiss(texts, model):
    print("\nGenerating document embeddings...")

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    dimension = vectors.shape[1]

    # Inner product on normalized vectors is equivalent to cosine
    # similarity. This is a newly built in-memory index.
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print(f"FAISS index contains {index.ntotal} vectors.")
    print(f"Embedding dimension: {dimension}")

    return index


def encode_query(query, model):
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vector, dtype=np.float32)


def faiss_search(index, query_vector):
    k = min(TOP_K, index.ntotal)

    _, indices = index.search(query_vector, k)

    return [int(i) for i in indices[0] if i >= 0]


# ============================================================
# REPEATED TIMING
# ============================================================

def measure_latency(search_function):
    # Warm-up runs are excluded from measured timings.
    for _ in range(WARMUP_RUNS):
        search_function()

    timings = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        search_function()

        elapsed_ms = (time.perf_counter() - start) * 1000

        timings.append(elapsed_ms)

    return np.asarray(timings, dtype=float)


def latency_statistics(timings):
    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "std_ms": float(np.std(timings, ddof=1))
        if len(timings) > 1 else 0.0,
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load papers
    texts, titles, paper_ids = load_corpus()

    if not texts:
        raise ValueError("No valid paper abstracts were found.")

    if not QUERIES:
        raise ValueError("Please configure at least one query.")

    # 2. Load embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # 3. Build both retrieval systems
    bm25_index = build_bm25(texts)
    faiss_index = build_faiss(texts, model)

    detail_rows = []
    summary_rows = []

    # 4. Run all queries
    for query_number, query in enumerate(QUERIES, start=1):
        print("\n" + "=" * 60)
        print(f"Query {query_number}: {query}")

        # Measure query embedding separately.
        embedding_start = time.perf_counter()

        query_vector = encode_query(query, model)

        embedding_ms = (
            time.perf_counter() - embedding_start
        ) * 1000

        # Measure retrieval latency independently.
        bm25_times = measure_latency(
            lambda: bm25_search(bm25_index, query)
        )

        faiss_times = measure_latency(
            lambda: faiss_search(faiss_index, query_vector)
        )

        # Obtain ranked results.
        bm25_ids = bm25_search(bm25_index, query)
        faiss_ids = faiss_search(faiss_index, query_vector)

        bm25_stats = latency_statistics(bm25_times)
        faiss_stats = latency_statistics(faiss_times)

        # Top-K overlap measures agreement, not relevance.
        overlap = len(set(bm25_ids) & set(faiss_ids))

        # Save individual retrieved papers.
        for method, ranked_ids in [
            ("BM25", bm25_ids),
            ("FAISS", faiss_ids),
        ]:
            for rank, paper_index in enumerate(ranked_ids, start=1):
                detail_rows.append({
                    "query_id": query_number,
                    "query": query,
                    "method": method,
                    "rank": rank,
                    "paper_index": paper_index,
                    "paper_id": paper_ids[paper_index],
                    "paper_title": titles[paper_index],
                })

        # Summarize both methods for this query.
        summary_rows.append({
            "query_id": query_number,
            "query": query,
            "method": "BM25",
            "papers_count": len(texts),
            "top_k": min(TOP_K, len(texts)),
            "repeats": REPEATS,
            "warmup_runs": WARMUP_RUNS,
            "mean_search_ms": bm25_stats["mean_ms"],
            "median_search_ms": bm25_stats["median_ms"],
            "std_search_ms": bm25_stats["std_ms"],
            "min_search_ms": bm25_stats["min_ms"],
            "max_search_ms": bm25_stats["max_ms"],
            "query_embedding_ms": np.nan,
            "mean_total_ms": bm25_stats["mean_ms"],
            "top_k_overlap": overlap,
        })

        summary_rows.append({
            "query_id": query_number,
            "query": query,
            "method": "FAISS",
            "papers_count": len(texts),
            "top_k": min(TOP_K, len(texts)),
            "repeats": REPEATS,
            "warmup_runs": WARMUP_RUNS,
            "mean_search_ms": faiss_stats["mean_ms"],
            "median_search_ms": faiss_stats["median_ms"],
            "std_search_ms": faiss_stats["std_ms"],
            "min_search_ms": faiss_stats["min_ms"],
            "max_search_ms": faiss_stats["max_ms"],
            "query_embedding_ms": embedding_ms,
            "mean_total_ms": (
                faiss_stats["mean_ms"] + embedding_ms
            ),
            "top_k_overlap": overlap,
        })

        print(
            f"BM25 mean retrieval time: "
            f"{bm25_stats['mean_ms']:.4f} ms"
        )

        print(
            f"FAISS mean index-search time: "
            f"{faiss_stats['mean_ms']:.4f} ms"
        )

        print(
            f"FAISS query embedding time: "
            f"{embedding_ms:.4f} ms"
        )

        print(
            f"FAISS mean total time (embedding + search): "
            f"{faiss_stats['mean_ms'] + embedding_ms:.4f} ms"
        )

        print(f"Top-{TOP_K} overlap: {overlap}")

    # 5. Save CSV outputs
    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = OUTPUT_DIR / "benchmark_query_results.csv"
    summary_path = OUTPUT_DIR / "benchmark_summary.csv"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # 6. Display overall summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    print(f"Dataset papers: {len(texts)}")
    print(f"Queries: {len(QUERIES)}")
    print(f"Repeated timing runs per query: {REPEATS}")

    overall = (
        summary_df.groupby("method")[
            ["mean_search_ms", "mean_total_ms"]
        ]
        .mean()
        .reset_index()
    )

    print("\nAverage results across queries:")
    print(overall.to_string(index=False))

    print(f"\nDetailed results saved to:\n{detail_path}")
    print(f"Summary results saved to:\n{summary_path}")

    print(
        "\nNote: Top-K overlap measures agreement between methods. "
        "It does not measure retrieval accuracy."
    )


if __name__ == "__main__":
    main()
