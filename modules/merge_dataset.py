import pandas as pd

HEALTHCARE_FILE = "data/healthcare_papers.csv"
AGRICULTURE_FILE = "data/agriculture_papers.csv"
OUTPUT_FILE = "data/research_papers_real.csv"


def main():

    print("Loading datasets...")

    healthcare = pd.read_csv(HEALTHCARE_FILE)
    agriculture = pd.read_csv(AGRICULTURE_FILE)

    print(f"Healthcare papers loaded: {len(healthcare)}")
    print(f"Agriculture papers loaded: {len(agriculture)}")

    # Force correct domain labels
    healthcare["Domain"] = "Healthcare"
    agriculture["Domain"] = "Agriculture"

    # Combine directly
    df = pd.concat(
        [healthcare, agriculture],
        ignore_index=True
    )

    # Remove completely identical rows only
    df = df.drop_duplicates(
        keep="first"
    )

    df = df.reset_index(drop=True)

    # Save final dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========================================")
    print("FINAL DATASET")
    print("========================================")

    print(f"Total papers: {len(df)}")

    print(
        f"Healthcare: "
        f"{(df['Domain'] == 'Healthcare').sum()}"
    )

    print(
        f"Agriculture: "
        f"{(df['Domain'] == 'Agriculture').sum()}"
    )

    print(
        f"SDG tagged: "
        f"{df['SDGs'].fillna('').astype(str).str.strip().ne('').sum()}"
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()