
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder
OUTPUT_DIR = "benchmark_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Benchmark results from your original benchmark
methods = ["BM25", "FAISS"]

# 1. Average search latency (milliseconds)
avg_search = [3.004311, 0.131978]

# 2. Query-wise search latency (milliseconds)
queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]

bm25_latency = [4.289, 2.629, 2.582, 3.232, 2.290]
faiss_latency = [0.137, 0.141, 0.126, 0.137, 0.119]

# 3. Reported average total latency (milliseconds)
avg_total = [3.004311, 10.907178]


def add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )


# GRAPH 1: Average Search Latency
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(methods, avg_search)

ax.set_title(
    "Average Search Latency: BM25 vs FAISS",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Retrieval Method")
ax.set_ylabel("Average Latency (ms)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

add_labels(ax, bars)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_average_search_latency.png"),
    dpi=300
)
plt.close()


# GRAPH 2: Query-wise Search Latency
x = np.arange(len(queries))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(
    x - width / 2,
    bm25_latency,
    width,
    label="BM25"
)

bars2 = ax.bar(
    x + width / 2,
    faiss_latency,
    width,
    label="FAISS"
)

ax.set_title(
    "Query-wise Search Latency Comparison",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Benchmark Queries")
ax.set_ylabel("Search Latency (ms)")
ax.set_xticks(x)
ax.set_xticklabels(queries)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

add_labels(ax, bars1)
add_labels(ax, bars2)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_query_wise_search_latency.png"),
    dpi=300
)
plt.close()


# GRAPH 3: Reported Average Total Latency
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(methods, avg_total)

ax.set_title(
    "Reported Average Total Latency",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Retrieval Method")
ax.set_ylabel("Reported Latency (ms)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

add_labels(ax, bars)

ax.text(
    0.5,
    0.95,
    "FAISS includes one query-embedding measurement + search.\n"
    "BM25 value represents search time only.",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=8
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "03_reported_total_latency.png"),
    dpi=300
)
plt.close()


print("\nAll graphs generated successfully!")
print("Saved inside:", OUTPUT_DIR)
print("1. Average Search Latency")
print("2. Query-wise Search Latency")
print("3. Reported Average Total Latency")
