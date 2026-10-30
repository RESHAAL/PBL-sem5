import pandas as pd
import re

DATASET_PATH = "data/research_papers_real.csv"


SDG_MAP = {
    "No poverty": {
        "number": 1,
        "name": "No Poverty"
    },
    "Zero hunger": {
        "number": 2,
        "name": "Zero Hunger"
    },
    "Good health and well-being": {
        "number": 3,
        "name": "Good Health and Well-Being"
    },
    "Quality education": {
        "number": 4,
        "name": "Quality Education"
    },
    "Gender equality": {
        "number": 5,
        "name": "Gender Equality"
    },
    "Clean water and sanitation": {
        "number": 6,
        "name": "Clean Water and Sanitation"
    },
    "Affordable and clean energy": {
        "number": 7,
        "name": "Affordable and Clean Energy"
    },
    "Decent work and economic growth": {
        "number": 8,
        "name": "Decent Work and Economic Growth"
    },
    "Industry, innovation and infrastructure": {
        "number": 9,
        "name": "Industry, Innovation and Infrastructure"
    },
    "Reduced inequalities": {
        "number": 10,
        "name": "Reduced Inequalities"
    },
    "Sustainable cities and communities": {
        "number": 11,
        "name": "Sustainable Cities and Communities"
    },
    "Responsible consumption and production": {
        "number": 12,
        "name": "Responsible Consumption and Production"
    },
    "Climate action": {
        "number": 13,
        "name": "Climate Action"
    },
    "Life below water": {
        "number": 14,
        "name": "Life Below Water"
    },
    "Life on land": {
        "number": 15,
        "name": "Life on Land"
    },
    "Peace, justice and strong institutions": {
        "number": 16,
        "name": "Peace, Justice and Strong Institutions"
    },
    "Partnerships for the goals": {
        "number": 17,
        "name": "Partnerships for the Goals"
    }
}


def normalize_sdg_name(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    for key, data in SDG_MAP.items():

        if key.lower() in value:

            return {
                "number": data["number"],
                "name": data["name"]
            }

    return None


def get_sdg_distribution(df):

    if "SDGs" not in df.columns:
        return pd.DataFrame()

    records = []

    for value in df["SDGs"].dropna():

        for item in str(value).split(";"):

            sdg = normalize_sdg_name(item)

            if sdg:
                records.append(sdg)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    result = (
        result
        .groupby(["number", "name"])
        .size()
        .reset_index(name="papers")
        .sort_values(
            "papers",
            ascending=False
        )
    )

    total = len(df)

    result["percentage"] = (
        result["papers"] / total * 100
    ).round(2)

    return result


def get_topic_sdg_relevance(topic, domain, papers_df):

    """
    Finds SDGs actually represented in the papers
    relevant to the current topic.
    """

    if papers_df is None or papers_df.empty:
        return []

    distribution = get_sdg_distribution(
        papers_df
    )

    if distribution.empty:
        return []

    results = []

    for _, row in distribution.iterrows():

        sdg_number = int(row["number"])
        sdg_name = row["name"]

        if sdg_number == 1:
            explanation = (
                "Relevant when research addresses poverty reduction "
                "or access to essential resources."
            )

        elif sdg_number == 2:
            explanation = (
                "Relevant when the research improves food production, "
                "crop productivity, agricultural efficiency or food security."
            )

        elif sdg_number == 3:
            explanation = (
                "Relevant when the research improves disease detection, "
                "healthcare delivery, diagnosis, treatment or well-being."
            )

        elif sdg_number == 4:
            explanation = (
                "Relevant when AI or technology improves teaching, "
                "learning, accessibility or educational outcomes."
            )

        elif sdg_number == 5:
            explanation = (
                "Relevant when research addresses gender equality "
                "or reduces gender-related disparities."
            )

        elif sdg_number == 6:
            explanation = (
                "Relevant when research improves water management, "
                "water efficiency or sanitation."
            )

        elif sdg_number == 7:
            explanation = (
                "Relevant when research improves clean-energy generation, "
                "efficiency, storage or access."
            )

        elif sdg_number == 8:
            explanation = (
                "Relevant when research contributes to employment, "
                "productivity, economic growth or decent work."
            )

        elif sdg_number == 9:
            explanation = (
                "Relevant when research contributes to technological "
                "innovation, infrastructure or resilient systems."
            )

        elif sdg_number == 10:
            explanation = (
                "Relevant when research improves accessibility "
                "and reduces social or economic inequalities."
            )

        elif sdg_number == 11:
            explanation = (
                "Relevant when research improves sustainable urban "
                "systems, smart cities or community resilience."
            )

        elif sdg_number == 12:
            explanation = (
                "Relevant when research improves resource efficiency, "
                "waste reduction or sustainable production."
            )

        elif sdg_number == 13:
            explanation = (
                "Relevant when research supports climate mitigation, "
                "adaptation or climate resilience."
            )

        elif sdg_number == 14:
            explanation = (
                "Relevant when research protects marine ecosystems "
                "and aquatic resources."
            )

        elif sdg_number == 15:
            explanation = (
                "Relevant when research protects terrestrial ecosystems, "
                "biodiversity, forests or land resources."
            )

        elif sdg_number == 16:
            explanation = (
                "Relevant when research improves governance, "
                "institutions, transparency or justice."
            )

        else:
            explanation = (
                "Relevant when research supports collaboration "
                "and partnerships for sustainable development."
            )

        if row["percentage"] >= 10:
            relevance = "High"
        elif row["percentage"] >= 5:
            relevance = "Medium"
        else:
            relevance = "Low"

        results.append({
            "number": sdg_number,
            "name": sdg_name,
            "relevance": relevance,
            "papers": int(row["papers"]),
            "percentage": float(row["percentage"]),
            "explanation": explanation
        })

    return results


def analyze_sdg(domain):

    df = pd.read_csv(DATASET_PATH)

    if domain != "All Domains":
        df = df[
            df["Domain"] == domain
        ]

    distribution = get_sdg_distribution(df)

    print("\n" + "=" * 70)
    print("SDG ANALYSIS")
    print("=" * 70)

    print(f"\nDomain: {domain}")
    print(f"Papers analyzed: {len(df)}")

    if distribution.empty:

        print("\nNo valid SDG information available.")
        return

    print("\nSDG Distribution:\n")

    for _, row in distribution.iterrows():

        print(
            f"SDG {int(row['number'])} - "
            f"{row['name']}: "
            f"{int(row['papers'])} papers "
            f"({row['percentage']:.2f}%)"
        )

    top = distribution.iloc[0]

    print("\n" + "=" * 70)
    print("DOMINANT SDG")
    print("=" * 70)

    print(
        f"\nSDG {int(top['number'])} - "
        f"{top['name']}"
    )


if __name__ == "__main__":

    print("=" * 70)
    print("AI RESEARCH SDG ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    domains = sorted(
        df["Domain"]
        .dropna()
        .unique()
        .tolist()
    )

    print("\nAvailable Research Domains:")

    for i, domain in enumerate(
        domains,
        1
    ):
        count = (
            df["Domain"] == domain
        ).sum()

        print(
            f"{i}. {domain} "
            f"({count} papers)"
        )

    print(
        f"{len(domains) + 1}. All Domains"
    )

    choice = input(
        "\nEnter domain number: "
    ).strip()

    try:
        choice = int(choice)

        if choice == len(domains) + 1:
            domain = "All Domains"
        else:
            domain = domains[choice - 1]

    except:
        domain = "All Domains"

    analyze_sdg(domain)