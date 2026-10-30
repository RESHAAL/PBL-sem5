import pickle
import faiss
import os

EMBEDDINGS_FILE = "models/embeddings.pkl"
INDEX_FILE = "models/faiss_index.bin"


def main():

    print("Loading embeddings...")

    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]

    print("Number of papers:", len(embeddings))
    print("Embedding dimension:", embeddings.shape[1])

    print("\nCreating FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS index created!")
    print("Total vectors:", index.ntotal)

    faiss.write_index(
        index,
        INDEX_FILE
    )

    print("\nFAISS index saved successfully!")
    print("File:", INDEX_FILE)


if __name__ == "__main__":
    main()