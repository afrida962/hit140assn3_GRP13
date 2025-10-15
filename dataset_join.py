import pandas as pd
import numpy as np
import os

clean_dir = os.path.join(os.getcwd(), "cleaned_data")

def merge_datasets(df1, df2):
    # Create 30-minute period end for dataset2
    df2['period_end'] = df2['time'] + pd.Timedelta(minutes=30)

    # Sort both by time
    df1 = df1.sort_values('start_time')
    df2 = df2.sort_values('time')

    # Merge
    merged = pd.merge_asof(df1, df2, left_on='start_time', right_on='time', direction='backward')
    merged = merged[merged['start_time'] < merged['period_end']].reset_index(drop=True)

    print(f"Merged dataset created with shape: {merged.shape}")

    # 🔹 Fix: handle duplicate column names
    # Rename for clarity
    if 'hours_after_sunset_x' in merged.columns:
        merged.rename(columns={'hours_after_sunset_x': 'hours_after_sunset'}, inplace=True)
    if 'hours_after_sunset_y' in merged.columns:
        merged.drop(columns=['hours_after_sunset_y'], inplace=True)

    # 1️⃣ Rat present flag
    merged['rat_present_at_landing'] = (
        (merged['rat_period_start'].notna()) &
        (merged['start_time'] >= merged['rat_period_start']) &
        (merged['start_time'] <= merged['rat_period_end'])
    ).astype(int)

    # 2️⃣ Log transforms
    for col in ['rat_minutes', 'rat_arrival_number', 'food_availability']:
        if col in merged.columns:
            merged[f'log_{col}'] = np.log1p(merged[col])

    # 3️⃣ Cyclic time
    merged['hour_sin'] = np.sin(2 * np.pi * merged['hours_after_sunset'] / 24)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['hours_after_sunset'] / 24)

    # 4️⃣ Interaction terms
    merged['rat_minutes_x_food'] = merged['rat_minutes'] * merged['food_availability']
    merged['rat_minutes_x_hours'] = merged['rat_minutes'] * merged['hours_after_sunset']

    print("Feature engineering complete")
    print(merged[['rat_present_at_landing', 'log_rat_minutes', 'hour_sin', 'hour_cos']].head())

    # Save merged dataset
    save_path = os.path.join(clean_dir, "merged_dataset.csv")
    merged.to_csv(save_path, index=False)
    print(f"Merged and feature-engineered dataset saved to: {save_path}")


if __name__ == "__main__":
    df1 = pd.read_csv(os.path.join(clean_dir, "dataset1_clean.csv"),
                      parse_dates=['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time'])
    df2 = pd.read_csv(os.path.join(clean_dir, "dataset2_clean.csv"),
                      parse_dates=['time'])

    print("Cleaned data loaded successfully")
    merge_datasets(df1, df2)
