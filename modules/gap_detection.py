import pandas as pd
import re
from collections import Counter

DATASET = "data/research_papers_real.csv"

GAP_PATTERNS = {
    "Limited Dataset": [
        "small dataset",
        "limited dataset",
        "insufficient dataset",
        "limited data",
        "lack of data",
        "data scarcity",
        "insufficient data"
    ],

    "Limited Sample Size": [
        "small sample",
        "limited sample size",
        "small sample size",
        "limited number of participants"
    ],

    "Limited Generalizability": [
        "limited generalizability",
        "poor generalizability",
        "lack of generalizability",
        "limited generalisation",
        "poor generalisation"
    ],

    "Limited External Validation": [
        "lack of external validation",
        "limited external validation",
        "no external validation",
        "without external validation"
    ],

    "Limited Real-World Deployment": [
        "limited real-world",
        "lack of real-world",
        "real-world deployment",
        "real time deployment",
        "real-time deployment",
        "not deployed",
        "limited deployment",
        "clinical deployment"
    ],

    "Poor Scalability": [
        "poor scalability",
        "limited scalability",
        "lack of scalability",
        "scalability remains"
    ],

    "Limited Interpretability": [
        "lack of interpretability",
        "limited interpretability",
        "poor interpretability",
        "lack of explainability",
        "limited explainability"
    ],

    "Limited Longitudinal Studies": [
        "limited longitudinal",
        "lack of longitudinal",
        "few longitudinal",
        "longitudinal studies are limited"
    ],

    "Limited Diversity": [
        "limited diversity",
        "insufficient diversity",
        "lack of diversity",
        "limited demographic diversity",
        "limited geographic diversity"
    ],

    "Resource Constraints": [
        "resource-constrained",
        "resource constrained",
        "limited computational resources",
        "low-resource",
        "resource limitations"
    ],

    "Limited Comparative Evaluation": [
        "limited comparative",
        "lack of comparative",
        "few comparative studies",
        "limited comparison"
    ],

    "Limited Clinical Validation": [
        "limited clinical validation",
        "lack of clinical validation",
        "clinical validation is limited",
        "limited validation",
        "insufficient clinical validation"
    ],

    "Ethical Concerns": [
        "ethical concerns",
        "ethical challenges",
        "lack of ethical",
        "ethical issues",
        "privacy concerns",
        "privacy challenges",
        "data privacy"
    ]
}


def find_gap(text):
    """
    Detect predefined research-gap signals from text.
    """

    if pd.isna(text):
        return []

    text = str(text).lower()

    found = []

    for gap_name, patterns in GAP_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                r"\b" + re.escape(pattern) + r"\b",
                text
            ):
                found.append(gap_name)
                break

    return found


def analyze_gaps(df, relevant_indices=None):
    """
    Analyze gaps for the complete dataset or a selected
    subset of relevant papers.

    Returns:
        DataFrame containing gap, count, percentage and strength.
    """

    if relevant_indices is not None:
        working_df = df.loc[relevant_indices].copy()
    else:
        working_df = df.copy()

    gap_records = []

    for idx, row in working_df.iterrows():

        text = (
            str(row.get("Title", "")) + " " +
            str(row.get("Abstract", "")) + " " +
            str(row.get("Keywords", "")) + " " +
            str(row.get("Topics", ""))
        )

        gaps = find_gap(text)

        for gap in gaps:
            gap_records.append({
                "index": idx,
                "gap": gap
            })

    if not gap_records:
        return pd.DataFrame(
            columns=[
                "Gap",
                "Evidence_Count",
                "Percentage",
                "Strength"
            ]
        )

    gap_df = pd.DataFrame(gap_records)

    counts = gap_df.groupby("gap")["index"].nunique()

    total = max(len(working_df), 1)

    results = []

    for gap, count in counts.sort_values(
        ascending=False
    ).items():

        percentage = (count / total) * 100

        if percentage >= 20:
            strength = "High"
        elif percentage >= 10:
            strength = "Medium"
        else:
            strength = "Low"

        results.append({
            "Gap": gap,
            "Evidence_Count": int(count),
            "Percentage": round(percentage, 2),
            "Strength": strength
        })

    return pd.DataFrame(results)


def get_gap_evidence(df, gap_name, relevant_indices=None, limit=5):
    """
    Return papers providing evidence for a specific gap.
    """

    if relevant_indices is not None:
        working_df = df.loc[relevant_indices].copy()
    else:
        working_df = df.copy()

    evidence = []

    for idx, row in working_df.iterrows():

        text = (
            str(row.get("Title", "")) + " " +
            str(row.get("Abstract", "")) + " " +
            str(row.get("Keywords", "")) + " " +
            str(row.get("Topics", ""))
        )

        detected = find_gap(text)

        if gap_name in detected:

            evidence.append({
                "Paper_ID": row.get("Paper_ID", ""),
                "Title": row.get("Title", ""),
                "Year": row.get("Year", ""),
                "Abstract": row.get("Abstract", "")
            })

        if len(evidence) >= limit:
            break

    return evidence


def get_gap_summary(df, relevant_indices=None):
    """
    Returns a dashboard-ready summary.
    """

    analysis = analyze_gaps(
        df,
        relevant_indices
    )

    if analysis.empty:
        return {
            "primary_gap": "No strong predefined gap detected",
            "evidence_count": 0,
            "strength": "Low",
            "why_gap": [
                "The retrieved literature does not contain enough explicit gap signals.",
                "Broader semantic analysis should be used for deeper gap discovery."
            ],
            "all_gaps": analysis
        }

    primary = analysis.iloc[0]

    why_gap = {
        "Limited Dataset": [
            "Many studies rely on small or restricted datasets.",
            "Limited data can reduce model robustness.",
            "Results may not generalize to unseen populations or environments."
        ],

        "Limited Sample Size": [
            "Studies use relatively small participant or experimental groups.",
            "Small samples can reduce statistical confidence.",
            "Larger studies are needed for stronger validation."
        ],

        "Limited Real-World Deployment": [
            "Many proposed systems remain at research or prototype level.",
            "Real-world operational testing is limited.",
            "Deployment constraints may not be fully evaluated."
        ],

        "Limited Clinical Validation": [
            "Models may show promising experimental performance.",
            "Independent clinical validation is limited.",
            "Real-world clinical effectiveness remains insufficiently established."
        ],

        "Limited External Validation": [
            "Models are often evaluated on the same or closely related datasets.",
            "Independent datasets are less frequently used.",
            "External validation is needed to establish robustness."
        ],

        "Limited Interpretability": [
            "Some AI models provide predictions without clear explanations.",
            "Users may have difficulty understanding model decisions.",
            "Explainability can improve trust and practical adoption."
        ],

        "Limited Generalizability": [
            "Results may be dependent on specific datasets or environments.",
            "Performance across different populations or settings is uncertain.",
            "Cross-dataset evaluation is needed."
        ],

        "Ethical Concerns": [
            "Privacy, fairness and responsible AI issues remain important.",
            "Ethical risks can limit real-world adoption.",
            "Transparent and accountable AI practices are required."
        ],

        "Poor Scalability": [
            "Some approaches may struggle with larger datasets or users.",
            "Computational requirements can limit deployment.",
            "Scalable architectures need further investigation."
        ],

        "Resource Constraints": [
            "High computational or infrastructure requirements can restrict adoption.",
            "Resource-efficient approaches are needed.",
            "Edge and lightweight AI may improve accessibility."
        ]
    }

    return {
        "primary_gap": primary["Gap"],
        "evidence_count": int(primary["Evidence_Count"]),
        "percentage": float(primary["Percentage"]),
        "strength": primary["Strength"],
        "why_gap": why_gap.get(
            primary["Gap"],
            [
                "Existing studies indicate a recurring limitation.",
                "Further validation and investigation are required."
            ]
        ),
        "all_gaps": analysis
    }


def main():

    print("=" * 70)
    print("AI RESEARCH GAP ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET)

    print("Total papers:", len(df))

    analysis = analyze_gaps(df)

    df["Detected_Gaps"] = ""

    for idx, row in df.iterrows():

        text = (
            str(row.get("Title", "")) + " " +
            str(row.get("Abstract", "")) + " " +
            str(row.get("Keywords", "")) + " " +
            str(row.get("Topics", ""))
        )

        df.at[idx, "Detected_Gaps"] = "; ".join(
            find_gap(text)
        )

    df.to_csv(
        DATASET,
        index=False,
        encoding="utf-8-sig"
    )

    papers_with_gaps = (
        df["Detected_Gaps"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        "\nPapers with detected gaps:",
        papers_with_gaps
    )

    print("\n" + "=" * 70)
    print("RESEARCH GAP ANALYSIS")
    print("=" * 70)

    if analysis.empty:

        print("\nNo predefined research gaps detected.")

    else:

        print("\nIdentified Research Gaps:\n")

        for _, row in analysis.iterrows():

            print(
                f"- {row['Gap']}: "
                f"{row['Evidence_Count']} papers "
                f"({row['Percentage']:.2f}%) "
                f"[{row['Strength']}]"
            )

        primary = analysis.iloc[0]

        print("\n" + "=" * 70)
        print("DOMINANT RESEARCH GAP")
        print("=" * 70)

        print(
            f"\n{primary['Gap']}"
        )

        print(
            f"Found in {primary['Evidence_Count']} paper(s)"
        )

        print(
            f"Strength: {primary['Strength']}"
        )

    print("\nDataset updated:")
    print(DATASET)


if __name__ == "__main__":
    main()