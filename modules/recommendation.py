import pandas as pd

DATASET_PATH = "data/research_papers_real.csv"


RECOMMENDATION_MAP = {

    "Limited Dataset": {
        "direction": [
            "Build larger and more diverse datasets.",
            "Combine data from multiple sources.",
            "Evaluate performance across independent datasets."
        ],
        "methodology": [
            "Multi-source data integration",
            "Data augmentation",
            "Transfer learning",
            "Cross-dataset validation"
        ],
        "contribution": [
            "Improved robustness",
            "Better generalization",
            "More representative research findings"
        ]
    },

    "Limited Sample Size": {
        "direction": [
            "Increase the study population.",
            "Include multiple institutions or geographical regions.",
            "Perform stronger statistical validation."
        ],
        "methodology": [
            "Multi-center data collection",
            "Statistical power analysis",
            "Cross-validation",
            "External testing"
        ],
        "contribution": [
            "Higher statistical reliability",
            "More generalizable results"
        ]
    },

    "Limited Real-World Deployment": {
        "direction": [
            "Move promising approaches from experimental settings toward real-world deployment.",
            "Evaluate performance under realistic operational conditions.",
            "Study practical implementation barriers."
        ],
        "methodology": [
            "Prototype deployment",
            "Field testing",
            "Real-world performance evaluation",
            "User and system evaluation"
        ],
        "contribution": [
            "Improved deployment readiness",
            "Better practical usability",
            "Evidence for real-world adoption"
        ]
    },

    "Limited Clinical Validation": {
        "direction": [
            "Validate AI systems using independent clinical environments.",
            "Evaluate performance across different patient populations.",
            "Compare AI-assisted decisions with established clinical practices."
        ],
        "methodology": [
            "External clinical validation",
            "Multi-center evaluation",
            "Prospective testing",
            "Clinical performance comparison"
        ],
        "contribution": [
            "Stronger clinical evidence",
            "Improved reliability",
            "Greater confidence in practical adoption"
        ]
    },

    "Limited External Validation": {
        "direction": [
            "Test models on independent datasets.",
            "Evaluate robustness across different environments.",
            "Investigate performance degradation outside the training distribution."
        ],
        "methodology": [
            "Cross-dataset evaluation",
            "External validation",
            "Out-of-distribution testing"
        ],
        "contribution": [
            "Improved reliability",
            "Better generalization"
        ]
    },

    "Limited Interpretability": {
        "direction": [
            "Develop models whose decisions can be understood by end users.",
            "Investigate interpretable and explainable AI methods.",
            "Evaluate whether explanations improve user trust."
        ],
        "methodology": [
            "Explainable AI",
            "SHAP",
            "LIME",
            "Feature importance analysis"
        ],
        "contribution": [
            "Improved transparency",
            "Higher user trust",
            "Better responsible AI adoption"
        ]
    },

    "Limited Generalizability": {
        "direction": [
            "Evaluate models across different populations and environments.",
            "Test robustness across multiple datasets.",
            "Investigate domain adaptation techniques."
        ],
        "methodology": [
            "Cross-domain validation",
            "Multi-source datasets",
            "Domain adaptation",
            "External testing"
        ],
        "contribution": [
            "More robust models",
            "Improved real-world applicability"
        ]
    },

    "Ethical Concerns": {
        "direction": [
            "Develop transparent and responsible AI systems.",
            "Evaluate privacy, fairness and bias.",
            "Investigate ethical requirements for real-world deployment."
        ],
        "methodology": [
            "Fairness evaluation",
            "Bias analysis",
            "Privacy-preserving AI",
            "Explainable AI"
        ],
        "contribution": [
            "Improved trust",
            "Reduced algorithmic bias",
            "More responsible AI adoption"
        ]
    },

    "Poor Scalability": {
        "direction": [
            "Develop architectures capable of handling larger datasets and users.",
            "Reduce computational requirements.",
            "Evaluate scalability under realistic workloads."
        ],
        "methodology": [
            "Distributed processing",
            "Model optimization",
            "Cloud deployment",
            "Edge computing"
        ],
        "contribution": [
            "Improved scalability",
            "Lower computational cost",
            "Better deployment performance"
        ]
    },

    "Resource Constraints": {
        "direction": [
            "Develop lightweight solutions suitable for constrained environments.",
            "Reduce memory and computational requirements.",
            "Investigate efficient deployment strategies."
        ],
        "methodology": [
            "Model compression",
            "Quantization",
            "Knowledge distillation",
            "Edge AI"
        ],
        "contribution": [
            "Efficient deployment",
            "Lower infrastructure requirements",
            "Greater accessibility"
        ]
    }
}


def generate_recommendation(
    topic,
    domain,
    gaps
):

    if gaps is None or gaps.empty:

        return {
            "main_gap": "No strong predefined gap detected",
            "future_direction": [
                "Expand the literature search.",
                "Investigate limitations that are not captured by predefined patterns.",
                "Use deeper semantic analysis for unexplored research opportunities."
            ],
            "methodology": [
                "Systematic literature review",
                "Semantic analysis",
                "Comparative evaluation"
            ],
            "contribution": [
                "Better understanding of unexplored research areas."
            ]
        }

    main_gap = gaps.iloc[0]["Gap"]

    result = RECOMMENDATION_MAP.get(
        main_gap,
        {
            "direction": [
                f"Further investigate the identified gap: {main_gap}.",
                "Validate existing findings using broader evidence."
            ],
            "methodology": [
                "Comparative experiments",
                "Independent validation",
                "Systematic evaluation"
            ],
            "contribution": [
                "Improved understanding and practical applicability."
            ]
        }
    )

    return {
        "main_gap": main_gap,
        "future_direction": result["direction"],
        "methodology": result["methodology"],
        "contribution": result["contribution"]
    }


def main():

    print("=" * 70)
    print("AI RESEARCH RECOMMENDATION SYSTEM")
    print("=" * 70)

    topic = input(
        "\nEnter research topic: "
    ).strip()

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
        print(
            f"{i}. {domain}"
        )

    print(
        f"{len(domains) + 1}. All Domains"
    )

    choice = int(
        input(
            "\nEnter domain number: "
        )
    )

    if choice == len(domains) + 1:
        domain = "All Domains"
    else:
        domain = domains[choice - 1]

    if domain != "All Domains":
        df = df[
            df["Domain"] == domain
        ]

    if "Detected_Gaps" in df.columns:

        records = []

        for value in df["Detected_Gaps"].dropna():

            for gap in str(value).split(";"):

                gap = gap.strip()

                if gap:
                    records.append(gap)

        if records:

            counts = (
                pd.Series(records)
                .value_counts()
            )

            gaps = pd.DataFrame({
                "Gap": counts.index,
                "Evidence_Count": counts.values
            })

        else:
            gaps = pd.DataFrame()

    else:
        gaps = pd.DataFrame()

    print(
        "\nGenerating research recommendations..."
    )

    result = generate_recommendation(
        topic,
        domain,
        gaps
    )

    print("\n" + "=" * 70)
    print("RESEARCH RECOMMENDATION")
    print("=" * 70)

    print(
        f"\nMain Research Gap:\n"
        f"{result['main_gap']}"
    )

    print("\nFuture Research Direction:")

    for item in result[
        "future_direction"
    ]:
        print(f"• {item}")

    print("\nSuggested Methodology:")

    for item in result[
        "methodology"
    ]:
        print(f"• {item}")

    print("\nExpected Contribution:")

    for item in result[
        "contribution"
    ]:
        print(f"• {item}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()