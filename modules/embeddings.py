import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET = "data/research_papers_real.csv"
MODEL_DIR = "models"
OUTPUT_FILE = "models/embeddings.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATASET)

    print(f"Total papers: {len(df)}")

    # Combine title + abstract + topics + keywords
    texts = (
        df["Title"].fillna("").astype(str)
        + ". "
        + df["Abstract"].fillna("").astype(str)
        + ". Topics: "
        + df["Topics"].fillna("").astype(str)
        + ". Keywords: "
        + df["Keywords"].fillna("").astype(str)
    ).tolist()

    print("\nLoading Sentence-BERT model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("\n========================================")
    print("EMBEDDINGS GENERATED")
    print("========================================")

    print("Number of papers:", len(embeddings))
    print("Embedding dimension:", embeddings.shape[1])

    data = {
        "embeddings": embeddings,
        "paper_ids": df["Paper_ID"].tolist()
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print("\nEmbeddings saved successfully!")
    print("File:", OUTPUT_FILE)


if __name__ == "__main__":
    main()