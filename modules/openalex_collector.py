import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.openalex.org/works"
API_KEY = os.getenv("OPENALEX_API_KEY")

if not API_KEY:
    print("ERROR: OPENALEX_API_KEY not found in .env")
    exit()


# ============================================================
# SETTINGS
# ============================================================

TARGET_PER_DOMAIN = 220
TOTAL_TARGET = TARGET_PER_DOMAIN * 10

START_DATE = "2020-01-01"
END_DATE = "2026-08-29"


# ============================================================
# ABSTRACT
# ============================================================

def reconstruct_abstract(inverted_index):

    if not inverted_index:
        return ""

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda x: x[0])

    return " ".join(
        word for _, word in words
    )


# ============================================================
# AUTHORS
# ============================================================

def get_authors(work):

    authors = []

    for item in work.get("authorships", []):

        author = item.get("author", {})

        name = author.get("display_name")

        if name:
            authors.append(name)

    return "; ".join(authors)


# ============================================================
# TOPICS
# ============================================================

def get_topics(work):

    topics = []

    for item in work.get("topics", []):

        name = item.get("display_name")

        if name:
            topics.append(name)

    return "; ".join(topics)


# ============================================================
# KEYWORDS
# ============================================================

def get_keywords(work):

    keywords = []

    for item in work.get("keywords", []):

        name = item.get("display_name")

        if name:
            keywords.append(name)

    return "; ".join(keywords)


# ============================================================
# SDGs
# ============================================================

def get_sdgs(work):

    sdgs = []

    for item in work.get(
        "sustainable_development_goals",
        []
    ):

        name = item.get("display_name")

        if name:
            sdgs.append(name)

    return "; ".join(sdgs)


# ============================================================
# SOURCE
# ============================================================

def get_source(work):

    location = work.get(
        "primary_location",
        {}
    )

    source = location.get("source")

    if not source:
        return ""

    return source.get(
        "display_name",
        ""
    )


# ============================================================
# FETCH PAPERS
# ============================================================

def fetch_papers(
    domain,
    queries,
    target,
    global_seen_ids
):

    papers = []

    local_seen_ids = set()

    print("\n")
    print("=" * 60)
    print(f"DOMAIN: {domain}")
    print(f"TARGET: {target}")
    print("=" * 60)

    for query in queries:

        if len(papers) >= target:
            break

        cursor = "*"

        print(f"\nQuery: {query}")

        while cursor and len(papers) < target:

            params = {
                "search": query,

                "filter": (
                    "has_abstract:true,"
                    f"from_publication_date:{START_DATE},"
                    f"to_publication_date:{END_DATE}"
                ),

                "per-page": 100,

                "cursor": cursor,

                "api_key": API_KEY
            }

            try:

                response = requests.get(
                    BASE_URL,
                    params=params,
                    timeout=30
                )

                print(
                    "Status:",
                    response.status_code
                )

                response.raise_for_status()

                data = response.json()

            except Exception as e:

                print(
                    "Request error:",
                    e
                )

                time.sleep(2)

                continue

            results = data.get(
                "results",
                []
            )

            if not results:
                break

            for work in results:

                paper_id = (
                    work.get("id", "")
                    .split("/")[-1]
                )

                if not paper_id:
                    continue

                # GLOBAL duplicate protection
                if paper_id in global_seen_ids:
                    continue

                # LOCAL duplicate protection
                if paper_id in local_seen_ids:
                    continue

                year = work.get(
                    "publication_year"
                )

                if not year:
                    continue

                if year < 2020 or year > 2026:
                    continue

                title = work.get(
                    "title",
                    ""
                ).strip()

                if not title:
                    continue

                abstract = reconstruct_abstract(
                    work.get(
                        "abstract_inverted_index"
                    )
                )

                if len(abstract.strip()) < 100:
                    continue

                local_seen_ids.add(
                    paper_id
                )

                global_seen_ids.add(
                    paper_id
                )

                papers.append({

                    "Paper_ID":
                        paper_id,

                    "Title":
                        title,

                    "Abstract":
                        abstract,

                    "Year":
                        year,

                    "Publication_Date":
                        work.get(
                            "publication_date",
                            ""
                        ),

                    "Authors":
                        get_authors(work),

                    "DOI":
                        work.get(
                            "doi",
                            ""
                        ),

                    "Domain":
                        domain,

                    "Topics":
                        get_topics(work),

                    "Keywords":
                        get_keywords(work),

                    "SDGs":
                        get_sdgs(work),

                    "Citations":
                        work.get(
                            "cited_by_count",
                            0
                        ),

                    "Open_Access":
                        work.get(
                            "open_access",
                            {}
                        ).get(
                            "is_oa",
                            False
                        ),

                    "Source":
                        get_source(work),

                    "OpenAlex_URL":
                        work.get(
                            "id",
                            ""
                        )
                })

                if len(papers) >= target:
                    break

            print(
                f"{domain}: "
                f"{len(papers)}/{target}"
            )

            cursor = (
                data
                .get("meta", {})
                .get("next_cursor")
            )

            if not cursor:
                break

            time.sleep(0.15)

    return papers


# ============================================================
# MAIN
# ============================================================

def main():

    domains = {

        "Healthcare": [

            "artificial intelligence healthcare",
            "AI medical diagnosis",
            "AI disease detection",
            "machine learning healthcare",
            "deep learning medical imaging",
            "AI clinical decision support"
        ],

        "Agriculture": [

            "artificial intelligence agriculture",
            "AI crop disease",
            "AI precision agriculture",
            "machine learning agriculture",
            "deep learning crop monitoring",
            "AI smart farming"
        ],

        "Computer Science": [

            "artificial intelligence computer science",
            "machine learning algorithms",
            "deep learning computer science",
            "natural language processing",
            "computer vision AI",
            "generative artificial intelligence"
        ],

        "Engineering": [

            "artificial intelligence engineering",
            "machine learning engineering",
            "AI engineering systems",
            "intelligent engineering systems",
            "deep learning engineering",
            "AI industrial applications"
        ],

        "Environmental Science": [

            "artificial intelligence environment",
            "AI climate change",
            "machine learning environmental science",
            "AI environmental monitoring",
            "AI sustainability",
            "deep learning climate science"
        ],

        "Education": [

            "artificial intelligence education",
            "AI educational technology",
            "machine learning education",
            "AI personalized learning",
            "intelligent tutoring systems",
            "generative AI education"
        ],

        "Economics and Finance": [

            "artificial intelligence finance",
            "machine learning financial markets",
            "AI economic forecasting",
            "AI banking",
            "machine learning economics",
            "AI financial risk"
        ],

        "Social Sciences": [

            "artificial intelligence social sciences",
            "AI social science research",
            "machine learning social sciences",
            "AI human behavior",
            "AI society",
            "artificial intelligence public policy"
        ],

        "Energy": [

            "artificial intelligence energy",
            "AI renewable energy",
            "machine learning energy systems",
            "AI smart grid",
            "deep learning energy forecasting",
            "AI energy optimization"
        ],

        "Materials Science": [

            "artificial intelligence materials science",
            "machine learning materials discovery",
            "AI material design",
            "deep learning materials science",
            "AI nanomaterials",
            "machine learning materials engineering"
        ]
    }


    # --------------------------------------------------------
    # CREATE DATA DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )


    # --------------------------------------------------------
    # GLOBAL DUPLICATE SET
    # --------------------------------------------------------

    global_seen_ids = set()

    all_papers = []

    domain_counts = {}


    # --------------------------------------------------------
    # COLLECT
    # --------------------------------------------------------

    for domain, queries in domains.items():

        papers = fetch_papers(
            domain,
            queries,
            TARGET_PER_DOMAIN,
            global_seen_ids
        )

        domain_counts[domain] = len(
            papers
        )

        all_papers.extend(
            papers
        )


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_papers
    )


    if df.empty:

        print(
            "\nERROR: No papers collected."
        )

        return


    # --------------------------------------------------------
    # FINAL DUPLICATE CLEANING
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["Paper_ID"],
        keep="first"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "Domain",
            "Year"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        "data/research_papers_real.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL MULTI-DOMAIN DATASET")
    print("=" * 60)

    print(
        "Total papers:",
        len(df)
    )

    print(
        "Unique Paper IDs:",
        df["Paper_ID"].nunique()
    )

    print(
        "Target:",
        TOTAL_TARGET
    )

    print("\nDomain Distribution:")

    distribution = (
        df["Domain"]
        .value_counts()
        .sort_index()
    )

    for domain, count in distribution.items():

        print(
            f"{domain}: {count}"
        )

    print("\nYear Range:")

    print(
        df["Year"].min(),
        "-",
        df["Year"].max()
    )

    print("\nSDG Tagged Papers:")

    sdg_count = (
        df["SDGs"]
        .notna()
        .sum()
    )

    print(
        sdg_count
    )

    print("\nSaved:")

    print(
        output_file
    )


if __name__ == "__main__":
    main()