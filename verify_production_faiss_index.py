"""
Run from the ResearchLens project root:
    python verify_production_faiss_index.py

This inspects the saved production FAISS index without modifying it.
"""
from pathlib import Path
import faiss

INDEX_PATH = Path("models/faiss_index.bin")

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"Production index not found: {INDEX_PATH.resolve()}")

index = faiss.read_index(str(INDEX_PATH))


print("Index file:", INDEX_PATH.resolve())
print("Concrete index class:", type(index).__name__)
print("FAISS dimension:", index.d)
print("Number of vectors (ntotal):", index.ntotal)
print("Metric type:", index.metric_type)
print("Metric name:", {
    faiss.METRIC_INNER_PRODUCT: "METRIC_INNER_PRODUCT",
    faiss.METRIC_L2: "METRIC_L2",
}.get(index.metric_type, f"Unknown metric ({index.metric_type})"))
print("FAISS version:", getattr(faiss, "__version__", "not exposed"))

if type(index).__name__ == "IndexFlatIP" and index.metric_type == faiss.METRIC_INNER_PRODUCT:
    print("VERDICT: Saved production index is IndexFlatIP.")
else:
    print("VERDICT: Saved production index is NOT confirmed as IndexFlatIP.")
