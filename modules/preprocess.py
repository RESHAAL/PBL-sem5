import pandas as pd
import os


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset():
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "research_gap_dataset.csv"
    )

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully!")
    print(f"Total papers: {len(df)}")

    return df


# -----------------------------
# Clean Dataset
# -----------------------------
def clean_dataset(df):

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate papers
    df = df.drop_duplicates(subset=["Paper_ID"])

    # Fill missing values
    text_columns = [
        "Title",
        "Abstract",
        "Keywords"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("")

    return df


# -----------------------------
# Create Text for Embeddings
# -----------------------------
def create_combined_text(df):

    df["combined_text"] = (
        df["Title"].astype(str)
        + ". "
        + df["Abstract"].astype(str)
        + ". Keywords: "
        + df["Keywords"].astype(str)
    )

    return df


# -----------------------------
# Main Preprocessing Function
# -----------------------------
def preprocess_data():

    df = load_dataset()

    df = clean_dataset(df)

    df = create_combined_text(df)

    print(f"After preprocessing: {len(df)} papers")

    return df


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    df = preprocess_data()

    print("\nFirst paper:")
    print(df.iloc[0]["combined_text"])