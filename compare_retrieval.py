import re
import time
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATASET = "data/research_papers_real.csv"
EMBEDDINGS_FILE = "models/embeddings.pkl"
INDEX_FILE = "models/faiss_index.bin"
TOP_K = 10
SEARCH_K = 500

QUERIES = [
    ("AI for early disease detection", "Healthcare"),
    ("Machine learning for crop disease prediction", "Agriculture"),
    ("Explainable AI for cybersecurity", "Computer Science"),
    ("AI-based personalized learning", "Education"),
    ("Deep learning for climate prediction", "Environmental Science"),
]

def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET).drop_duplicates(
        subset=["Paper_ID"], keep="first"
    ).reset_index(drop=True)

    for col in ["Title", "Abstract", "Topics", "Keywords", "Domain"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].map(clean)

    df["search_text"] = (
        df["Title"] + " " + df["Abstract"] + " " +
        df["Topics"] + " " + df["Keywords"]
    ).str.strip()

    print(f"Unique papers: {len(df)}")
    print("Building BM25 index...")
    corpus = [tokenize(text) for text in df["search_text"]]
    bm25 = BM25Okapi(corpus)

    print("Loading semantic model and FAISS index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(INDEX_FILE)

    # Map saved embedding order to paper IDs from the existing project file.
    import pickle
    with open(EMBEDDINGS_FILE, "rb") as f:
        embedding_data = pickle.load(f)
    paper_ids = list(embedding_data["paper_ids"])

    if len(paper_ids) != index.ntotal:
        raise ValueError(
            f"Embedding/index mismatch: {len(paper_ids)} IDs, "
            f"{index.ntotal} FAISS vectors"
        )

    paper_lookup = df.set_index("Paper_ID", drop=False)
    rows = []

    for query, domain in QUERIES:
        print(f"\nQuery: {query} | Domain: {domain}")

        # BM25: score all papers, then apply domain filter.
        start = time.perf_counter()
        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_results = []
        for i in bm25_order:
            paper = df.iloc[int(i)]
            if paper["Domain"] == domain:
                bm25_results.append((paper["Paper_ID"], float(bm25_scores[i])))
            if len(bm25_results) == TOP_K:
                break
        bm25_time = (time.perf_counter() - start) * 1000

        # Semantic: same model/index as project; retrieve candidates then filter.
        start = time.perf_counter()
        query_embedding = model.encode(
            [query], normalize_embeddings=True
        )
        search_k = min(SEARCH_K, index.ntotal)
        scores, indices = index.search(query_embedding, search_k)

        semantic_results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(paper_ids):
                continue
            paper_id = paper_ids[idx]
            if paper_id in seen or paper_id not in paper_lookup.index:
                continue
            paper = paper_lookup.loc[paper_id]
            if paper["Domain"] != domain:
                continue
            semantic_results.append((paper_id, float(score)))
            seen.add(paper_id)
            if len(semantic_results) == TOP_K:
                break
        semantic_time = (time.perf_counter() - start) * 1000

        print(f"BM25 results: {len(bm25_results)}")
        print(f"Semantic results: {len(semantic_results)}")
        print(f"BM25 retrieval time: {bm25_time:.2f} ms")
        print(f"Semantic retrieval time: {semantic_time:.2f} ms")

        for rank, (paper_id, score) in enumerate(bm25_results, 1):
            paper = paper_lookup.loc[paper_id]
            rows.append({
                "Query": query, "Domain": domain,
                "Method": "BM25", "Rank": rank,
                "Paper_ID": paper_id, "Title": paper["Title"],
                "Score": score, "Latency_ms": bm25_time,
                "Relevant": ""
            })

        for rank, (paper_id, score) in enumerate(semantic_results, 1):
            paper = paper_lookup.loc[paper_id]
            rows.append({
                "Query": query, "Domain": domain,
                "Method": "Semantic (MiniLM + FAISS)", "Rank": rank,
                "Paper_ID": paper_id, "Title": paper["Title"],
                "Score": score, "Latency_ms": semantic_time,
                "Relevant": ""
            })

    output = "retrieval_comparison_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved results to: {output}")
    print("Next: manually label Relevant as 1 or 0 for each result.")
    print("Do not interpret scores as accuracy; relevance metrics need labels.")

if __name__ == "__main__":
    main()
