import pandas as pd

# Load both datasets (adjust paths if needed)
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

def print_data():

    # ----- Basic info -----
    print("=== Dataset 1 (Bat Landings) ===")
    df1.info()
    print("\nShape:", df1.shape)
    print("\nFirst 5 rows:\n", df1.head())

    print("\n=== Dataset 2 (Rat Arrivals) ===")
    df2.info()
    print("\nShape:", df2.shape)
    print("\nFirst 5 rows:\n", df2.head())

    # ----- Quick numeric summaries -----
    print("\nNumeric summary – Dataset 1:")
    print(df1.describe().T)

    print("\nNumeric summary – Dataset 2:")
    print(df2.describe().T)

    # ----- Count missing values -----
    print("\nMissing values per column (Dataset 1):")
    print(df1.isnull().sum())

    print("\nMissing values per column (Dataset 2):")
    print(df2.isnull().sum())

    # ----- Check categorical variables (first few unique values) -----
    for col in df1.select_dtypes(include="object").columns:
        print(f"\nUnique values in {col} (Dataset 1):")
        print(df1[col].unique()[:10])

    for col in df2.select_dtypes(include="object").columns:
        print(f"\nUnique values in {col} (Dataset 2):")
        print(df2[col].unique()[:10])


if __name__ == "__main__":
    print_data()
# This script provides a basic understanding and cleaning of two datasets: