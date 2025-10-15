import pandas as pd
import numpy as np
import os
import sys

# Ensure console can print UTF-8 (for emojis)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_data(df1, df2):
    # 1️⃣ Convert date/time columns to datetime objects
    date_cols_df1 = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
    date_cols_df2 = ['time']

    for col in date_cols_df1:
        df1[col] = pd.to_datetime(df1[col], format='%d/%m/%Y %H:%M', errors='coerce')

    for col in date_cols_df2:
        df2[col] = pd.to_datetime(df2[col], format='%d/%m/%Y %H:%M', errors='coerce')

    print("Time columns converted successfully")

    # 2️⃣ Handle missing values in 'habit'
    print(f"Missing 'habit' before: {df1['habit'].isna().sum()}")
    df1['habit'] = df1['habit'].fillna('unknown')
    print(f"Missing 'habit' after: {df1['habit'].isna().sum()}")

    # 3️⃣ Remove duplicates if any
    before = df1.shape[0]
    df1 = df1.drop_duplicates()
    after = df1.shape[0]
    print(f"Removed {before - after} duplicate rows from Dataset 1")

    # 4️⃣ Standardise season codes
    df1['season_label'] = df1['season'].map({0: 'winter', 1: 'spring'})
    print("Season label counts:")
    print(df1['season_label'].value_counts())

    # 5️⃣ Quick outlier range checks
    numeric_cols_df1 = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
    print("Numeric columns summary:")
    print(df1[numeric_cols_df1].describe())

    # 6️⃣ Save cleaned data
    save_dir = os.path.join(os.getcwd(), "cleaned_data")
    os.makedirs(save_dir, exist_ok=True)

    dataset1_clean_path = os.path.join(save_dir, "dataset1_clean.csv")
    dataset2_clean_path = os.path.join(save_dir, "dataset2_clean.csv")

    df1.to_csv(dataset1_clean_path, index=False)
    df2.to_csv(dataset2_clean_path, index=False)

    print(f"Cleaned Dataset 1 saved to: {dataset1_clean_path}")
    print(f"Cleaned Dataset 2 saved to: {dataset2_clean_path}")

    return df1, df2


if __name__ == "__main__":
    # Load datasets
    df1 = pd.read_csv("dataset1.csv")
    df2 = pd.read_csv("dataset2.csv")

    # Clean datasets
    df1_clean, df2_clean = clean_data(df1, df2)
