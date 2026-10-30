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


# ============================================================
# INPUT FILES
# ============================================================

DATA = "data/research_papers_real.csv"
EMB = "models/embeddings.pkl"
IDX = "models/faiss_index.bin"


# ============================================================
# EXPERIMENT QUERIES
# ============================================================

QUERIES = [
    ("AI for early disease detection", "Healthcare"),
    ("Machine learning for crop disease prediction", "Agriculture"),
    ("Explainable AI for cybersecurity", "Computer Science"),
    ("AI-based personalized learning", "Education"),
    ("AI for climate prediction", "Environmental Science"),
]


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RUNS = 5
TOP_K = 10


# ============================================================
# OUTPUT FILES
# Separate filenames preserve your previous results.
# ============================================================

RETRIEVAL_OUTPUT = "climate_corrected_retrieval_results.csv"
TIMING_OUTPUT = "climate_corrected_repeated_timing_results.csv"
SUMMARY_OUTPUT = "climate_corrected_timing_summary.csv"
ENV_OUTPUT = "climate_corrected_environment.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean(value):
    return "" if pd.isna(value) else str(value).strip()


def tokens(text):
    import re
    return re.findall(r"\b\w+\b", str(text).lower())


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Validate input files
    # --------------------------------------------------------

    for path in [DATA, EMB, IDX]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    # --------------------------------------------------------
    # 2. Load and validate dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA)

    if "Paper_ID" not in df.columns:
        raise ValueError(
            "Dataset is missing the Paper_ID column"
        )

    df["Paper_ID"] = (
        df["Paper_ID"].astype(str).str.strip()
    )

    df = df.drop_duplicates(
        subset="Paper_ID",
        keep="first"
    ).reset_index(drop=True)

    required_columns = [
        "Title",
        "Abstract",
        "Topics",
        "Keywords",
        "Domain"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}"
            )

        df[col] = df[col].map(clean)

    df["search_text"] = (
        df["Title"] + " " +
        df["Abstract"] + " " +
        df["Topics"] + " " +
        df["Keywords"]
    ).str.strip()

    # --------------------------------------------------------
    # 3. Load saved embeddings and FAISS index
    # --------------------------------------------------------

    with open(EMB, "rb") as file:
        emb_data = pickle.load(file)

    ids = [
        str(value).strip()
        for value in emb_data["paper_ids"]
    ]

    embedding_matrix = np.asarray(
        emb_data["embeddings"],
        dtype=np.float32
    )

    faiss_index = faiss.read_index(IDX)

    if (
        len(ids) != faiss_index.ntotal
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(
            "Embedding IDs and FAISS index size/"
            "uniqueness mismatch"
        )

    if not set(ids).issubset(set(df["Paper_ID"])):
        raise ValueError(
            "Some embedding Paper_IDs do not map "
            "to the dataset"
        )

    if len(embedding_matrix) != len(ids):
        raise ValueError(
            "Embedding count does not match Paper_ID count"
        )

    if embedding_matrix.ndim != 2:
        raise ValueError(
            "Saved embeddings must be a 2D matrix"
        )

    if not np.isfinite(embedding_matrix).all():
        raise ValueError(
            "Saved embeddings contain NaN or infinite values"
        )

    if embedding_matrix.shape[1] != faiss_index.d:
        raise ValueError(
            "Embedding dimensions do not match FAISS index"
        )

    print("Dataset papers:", len(df))
    print("Embedding vectors:", len(ids))
    print("FAISS vectors:", faiss_index.ntotal)

    print(
        "Embedding ID and dimension checks passed."
    )

    print(
        "Note: embedding-generation provenance and "
        "FAISS ID ordering must be verified separately."
    )

    id_to_vector = {
        paper_id: embedding_matrix[i]
        for i, paper_id in enumerate(ids)
    }

    # --------------------------------------------------------
    # 4. Load semantic embedding model
    # --------------------------------------------------------

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    retrieval_rows = []
    timing_rows = []

    # --------------------------------------------------------
    # 5. Process each query
    # --------------------------------------------------------

    for query, domain in QUERIES:

        print("\nProcessing query:", query)
        print("Domain:", domain)

        candidates = df[
            df["Domain"] == domain
        ].copy()

        if candidates.empty:
            raise ValueError(
                f"No papers found for domain: {domain}"
            )

        missing_ids = [
            paper_id
            for paper_id in candidates["Paper_ID"]
            if paper_id not in id_to_vector
        ]

        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} candidate papers "
                "have no saved embeddings"
            )

        candidates = candidates.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # 6. Prepare BM25
        # ----------------------------------------------------

        corpus = [
            tokens(text)
            for text in candidates["search_text"]
        ]

        bm25 = BM25Okapi(corpus)
        query_tokens = tokens(query)

        # ----------------------------------------------------
        # 7. Encode query for semantic retrieval
        # ----------------------------------------------------

        query_vector = model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)[0]

        vectors = np.vstack([
            id_to_vector[paper_id]
            for paper_id in candidates["Paper_ID"]
        ]).astype(np.float32)

        if vectors.shape[1] != query_vector.shape[0]:
            raise ValueError(
                f"Query embedding dimension mismatch: {query}"
            )

        # ----------------------------------------------------
        # 8. Calculate retrieval scores
        # ----------------------------------------------------

        bm25_scores = bm25.get_scores(
            query_tokens
        )

        semantic_scores = vectors @ query_vector

        bm25_order = np.argsort(
            bm25_scores
        )[::-1][:TOP_K]

        semantic_order = np.argsort(
            semantic_scores
        )[::-1][:TOP_K]

        # ----------------------------------------------------
        # 9. Save top-10 retrieval results
        # ----------------------------------------------------

        methods = [
            (
                "BM25",
                bm25_order,
                bm25_scores
            ),
            (
                "Semantic (MiniLM + FAISS vectors)",
                semantic_order,
                semantic_scores
            ),
        ]

        for method, order, scores in methods:

            for rank, idx in enumerate(
                order,
                start=1
            ):

                paper = candidates.iloc[int(idx)]

                retrieval_rows.append({
                    "Query": query,
                    "Domain": domain,
                    "Method": method,
                    "Rank": rank,
                    "Paper_ID": paper["Paper_ID"],
                    "Title": paper["Title"],
                    "Abstract": paper["Abstract"],
                    "Score": float(scores[idx]),
                    "Relevant": ""
                })

        # ----------------------------------------------------
        # 10. Warm up scoring operations
        # ----------------------------------------------------

        warmup_bm25 = bm25.get_scores(
            query_tokens
        )

        np.argsort(
            warmup_bm25
        )[::-1][:TOP_K]

        warmup_semantic = vectors @ query_vector

        np.argsort(
            warmup_semantic
        )[::-1][:TOP_K]

        # ----------------------------------------------------
        # 11. Measure repeated retrieval latency
        # ----------------------------------------------------

        for run_number in range(1, RUNS + 1):

            # BM25 timing
            start = time.perf_counter()

            fresh_bm25_scores = bm25.get_scores(
                query_tokens
            )

            np.argsort(
                fresh_bm25_scores
            )[::-1][:TOP_K]

            bm25_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            timing_rows.append({
                "Query": query,
                "Method": "BM25",
                "Run": run_number,
                "Latency_ms": bm25_latency_ms
            })

            # Semantic scoring timing
            start = time.perf_counter()

            fresh_semantic_scores = (
                vectors @ query_vector
            )

            np.argsort(
                fresh_semantic_scores
            )[::-1][:TOP_K]

            semantic_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            timing_rows.append({
                "Query": query,
                "Method": (
                    "Semantic (MiniLM + FAISS vectors)"
                ),
                "Run": run_number,
                "Latency_ms": semantic_latency_ms
            })

    # --------------------------------------------------------
    # 12. Convert results into DataFrames
    # --------------------------------------------------------

    retrieval_df = pd.DataFrame(
        retrieval_rows
    )

    timing_df = pd.DataFrame(
        timing_rows
    )

    expected_retrieval_rows = (
        len(QUERIES) * 2 * TOP_K
    )

    expected_timing_rows = (
        len(QUERIES) * 2 * RUNS
    )

    if len(retrieval_df) != expected_retrieval_rows:
        raise ValueError(
            f"Unexpected retrieval row count: "
            f"{len(retrieval_df)}"
        )

    if len(timing_df) != expected_timing_rows:
        raise ValueError(
            f"Unexpected timing row count: "
            f"{len(timing_df)}"
        )

    # --------------------------------------------------------
    # 13. Save retrieval and timing CSV files
    # --------------------------------------------------------

    retrieval_df.to_csv(
        RETRIEVAL_OUTPUT,
        index=False
    )

    timing_df.to_csv(
        TIMING_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # 14. Calculate timing summary
    # --------------------------------------------------------

    summary_df = (
        timing_df.groupby(
            ["Query", "Method"]
        )["Latency_ms"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # 15. Save environment information
    # --------------------------------------------------------

    with open(
        ENV_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Python: {platform.python_version()}\n"
        )

        file.write(
            f"OS: {platform.platform()}\n"
        )

        file.write(
            f"CPU: {platform.processor()}\n"
        )

        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Not available"
        )

        file.write(f"GPU: {gpu}\n")

        packages = [
            "pandas",
            "numpy",
            "faiss-cpu",
            "sentence-transformers",
            "transformers",
            "torch",
            "rank-bm25",
            "streamlit",
        ]

        for package in packages:

            try:
                version = md.version(package)

                file.write(
                    f"{package}: {version}\n"
                )

            except md.PackageNotFoundError:

                file.write(
                    f"{package}: not installed\n"
                )

    # --------------------------------------------------------
    # 16. Print completion information
    # --------------------------------------------------------

    print("\nExperiment completed successfully.")

    print(
        f"Retrieval output: {RETRIEVAL_OUTPUT}"
    )

    print(
        f"Repeated timing output: {TIMING_OUTPUT}"
    )

    print(
        f"Timing summary output: {SUMMARY_OUTPUT}"
    )

    print(
        f"Environment report: {ENV_OUTPUT}"
    )

    print(
        f"Retrieval rows: {len(retrieval_df)}"
    )

    print(
        f"Timing rows: {len(timing_df)}"
    )

    print(
        "\nTiming scope: scoring plus full argsort "
        "top-K selection."
    )

    print(
        "Query encoding and candidate preparation "
        "are excluded from timing."
    )

    print(
        "Semantic scoring uses NumPy matrix multiplication, "
        "not timed FAISS search."
    )

    print(
        "\nWARNING: Relevant labels are blank. "
        "Review relevance labels before calculating "
        "Precision@10, Recall, F1, or nDCG@10."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()