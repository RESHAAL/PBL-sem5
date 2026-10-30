import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET = "data/research_papers_real.csv"
EMBEDDINGS_FILE = "models/embeddings.pkl"
INDEX_FILE = "models/faiss_index.bin"

TOP_K = 10
SEARCH_K = 500


def clean_value(value):
    """Convert NaN/empty values into a clean string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def format_sdg(sdg_text):
    """Clean SDG information for display."""
    sdg_text = clean_value(sdg_text)

    if not sdg_text:
        return "No SDG information available"

    return sdg_text


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATASET)

    print("Total papers:", len(df))

    # ---------------------------------------------------------
    # DOMAIN INFORMATION
    # ---------------------------------------------------------

    domains = sorted(
        df["Domain"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print("\nAvailable Research Domains:")

    for i, domain in enumerate(domains, 1):
        count = (
            df["Domain"]
            .astype(str)
            .str.strip()
            .eq(domain)
            .sum()
        )

        print(f"{i}. {domain} ({count} papers)")

    print(f"{len(domains) + 1}. All Domains")

    # ---------------------------------------------------------
    # LOAD EMBEDDINGS
    # ---------------------------------------------------------

    print("\nLoading embeddings...")

    with open(EMBEDDINGS_FILE, "rb") as f:
        embedding_data = pickle.load(f)

    paper_ids = embedding_data["paper_ids"]

    print("Embedding vectors:", len(paper_ids))

    # ---------------------------------------------------------
    # LOAD FAISS
    # ---------------------------------------------------------

    print("\nLoading FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    print("FAISS vectors:", index.ntotal)

    # ---------------------------------------------------------
    # LOAD SENTENCE-BERT
    # ---------------------------------------------------------

    print("\nLoading Sentence-BERT...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------------
    # USER QUERY
    # ---------------------------------------------------------

    query = input(
        "\nEnter research topic: "
    ).strip()

    if not query:
        print("ERROR: Research topic cannot be empty.")
        return

    # ---------------------------------------------------------
    # DOMAIN SELECTION
    # ---------------------------------------------------------

    choice = input(
        "\nEnter domain number: "
    ).strip()

    try:
        choice_num = int(choice)
    except ValueError:
        print("Invalid choice.")
        return

    if choice_num == len(domains) + 1:

        selected_domain = "All"

    elif 1 <= choice_num <= len(domains):

        selected_domain = domains[
            choice_num - 1
        ]

    else:

        print("Invalid domain choice.")
        return

    print(
        f"\nSelected Domain: {selected_domain}"
    )

    # ---------------------------------------------------------
    # QUERY EMBEDDING
    # ---------------------------------------------------------

    print("\nPerforming semantic search...")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # ---------------------------------------------------------
    # FAISS SEARCH
    # ---------------------------------------------------------

    search_k = min(
        SEARCH_K,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    # ---------------------------------------------------------
    # FAST PAPER LOOKUP
    # ---------------------------------------------------------

    paper_lookup = (
        df.drop_duplicates(
            subset=["Paper_ID"],
            keep="first"
        )
        .set_index("Paper_ID")
        .to_dict("index")
    )

    results = []
    used_ids = set()

    # ---------------------------------------------------------
    # FILTER RESULTS
    # ---------------------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(paper_ids):
            continue

        paper_id = paper_ids[idx]

        if paper_id in used_ids:
            continue

        if paper_id not in paper_lookup:
            continue

        paper = paper_lookup[paper_id]

        paper_domain = clean_value(
            paper.get("Domain")
        )

        # Domain filtering
        if selected_domain != "All":

            if paper_domain != selected_domain:
                continue

        title = clean_value(
            paper.get("Title")
        )

        abstract = clean_value(
            paper.get("Abstract")
        )

        topics = clean_value(
            paper.get("Topics")
        )

        keywords = clean_value(
            paper.get("Keywords")
        )

        sdgs = format_sdg(
            paper.get("SDGs")
        )

        results.append({

            "Paper_ID": paper_id,

            "Title": title,

            "Domain": paper_domain,

            "Year": paper.get("Year", ""),

            "Topics": topics,

            "Keywords": keywords,

            "Abstract": abstract,

            "SDGs": sdgs,

            "Similarity": round(
                float(score) * 100,
                2
            )
        })

        used_ids.add(paper_id)

        if len(results) >= TOP_K:
            break

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 90)

    print("SEMANTIC RESEARCH DISCOVERY")

    print("=" * 90)

    print(f"\nResearch Topic : {query}")

    print(
        f"Domain        : {selected_domain}"
    )

    print(
        f"Relevant Papers Found: {len(results)}"
    )

    print("\n" + "=" * 90)

    if not results:

        print(
            "\nNo relevant papers found "
            "for the selected domain."
        )

        return

    # ---------------------------------------------------------
    # PAPER CARDS
    # ---------------------------------------------------------

    for i, paper in enumerate(
        results,
        1
    ):

        print(
            f"\n{'─' * 90}"
        )

        print(
            f"PAPER {i}"
        )

        print(
            f"{'─' * 90}"
        )

        print(
            f"Paper ID    : {paper['Paper_ID']}"
        )

        print(
            f"Title       : {paper['Title']}"
        )

        print(
            f"Domain      : {paper['Domain']}"
        )

        print(
            f"Year        : {paper['Year']}"
        )

        print(
            f"Similarity  : {paper['Similarity']}%"
        )

        # Topics
        if paper["Topics"]:

            print(
                f"\nTopics:"
            )

            for topic in paper[
                "Topics"
            ].split(";"):

                topic = topic.strip()

                if topic:
                    print(
                        f"  • {topic}"
                    )

        # Keywords
        if paper["Keywords"]:

            print(
                f"\nKeywords:"
            )

            for keyword in paper[
                "Keywords"
            ].split(";"):

                keyword = keyword.strip()

                if keyword:
                    print(
                        f"  • {keyword}"
                    )

        # SDGs
        print(
            f"\nSDGs:"
        )

        print(
            f"  {paper['SDGs']}"
        )

        # Abstract
        print(
            f"\nAbstract:"
        )

        if paper["Abstract"]:

            print(
                f"  {paper['Abstract']}"
            )

        else:

            print(
                "  Abstract not available."
            )

    print(
        "\n" + "=" * 90
    )

    print(
        "SEMANTIC SEARCH COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()