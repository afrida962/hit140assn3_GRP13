import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_data(merged_df):
    """
    Perform Exploratory Data Analysis (EDA) on the merged and feature-engineered dataset.
    This method includes summary statistics, correlations, and visualisations.
    """

    print("=== Exploratory Data Analysis Started ===")

    # ---- 1️⃣ Basic Info ----
    print(f"Dataset shape: {merged_df.shape}")
    print("Column names:", merged_df.columns.tolist())
    print("\n=== Summary Statistics ===")
    print(merged_df.describe(include='all').T)

    # ---- 2️⃣ Numeric Summary ----
    numeric_cols = [
        'bat_landing_to_food', 'seconds_after_rat_arrival', 'rat_minutes',
        'rat_arrival_number', 'food_availability', 'hours_after_sunset',
        'rat_minutes_x_food', 'rat_minutes_x_hours'
    ]

    # Check columns exist
    numeric_cols = [col for col in numeric_cols if col in merged_df.columns]
    print("\nNumeric Columns Used:", numeric_cols)

    print("\n=== Numeric Summary ===")
    print(merged_df[numeric_cols].describe().T)

    # ---- 3️⃣ Correlation Heatmap ----
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        merged_df[numeric_cols].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", square=True
    )
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    plt.show()

    # ---- 4️⃣ Histograms ----
    merged_df[numeric_cols].hist(bins=20, figsize=(12, 10))
    plt.suptitle("Distributions of Numeric Variables", fontsize=14)
    plt.show()

    # ---- 5️⃣ Bat Behaviour vs Rat Presence ----
    if 'rat_present_at_landing' in merged_df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=merged_df, x='rat_present_at_landing', y='bat_landing_to_food')
        plt.title("Bat Landing-to-Food Time vs Rat Presence")
        plt.xlabel("Rat Present (1 = Yes, 0 = No)")
        plt.ylabel("Bat Landing to Food (seconds)")
        plt.show()

    # ---- 6️⃣ Seasonal Comparison ----
    if 'season_label' in merged_df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=merged_df, x='season_label', y='bat_landing_to_food', palette='Set2')
        plt.title("Bat Landing-to-Food Time by Season")
        plt.xlabel("Season")
        plt.ylabel("Bat Landing to Food (seconds)")
        plt.show()

    # ---- 7️⃣ Scatter Plot: Bat vs Rat Minutes ----
    if 'rat_minutes' in merged_df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=merged_df,
            x='rat_minutes',
            y='bat_landing_to_food',
            hue='rat_present_at_landing',
            alpha=0.7
        )
        plt.title("Bat Landing-to-Food vs Rat Minutes")
        plt.xlabel("Rat Minutes (per 30-min window)")
        plt.ylabel("Bat Landing to Food (seconds)")
        plt.legend(title='Rat Present')
        plt.show()

    # ---- 8️⃣ Pairplot (optional, can be slow) ----
    selected_cols = [
        'bat_landing_to_food', 'rat_minutes', 'food_availability',
        'seconds_after_rat_arrival', 'hours_after_sunset', 'rat_arrival_number'
    ]
    selected_cols = [c for c in selected_cols if c in merged_df.columns]
    if len(selected_cols) > 3:
        sns.pairplot(
            merged_df[selected_cols],
            diag_kind="kde",
            plot_kws={'alpha': 0.6, 's': 40}
        )
        plt.suptitle("Pairwise Relationships of Key Variables", y=1.02)
        plt.show()

    print("Exploratory Data Analysis Complete")


if __name__ == "__main__":
    # Load merged dataset
    clean_dir = os.path.join(os.getcwd(), "cleaned_data")
    merged_path = os.path.join(clean_dir, "merged_dataset.csv")

    merged_df = pd.read_csv(merged_path)
    print("Merged dataset loaded successfully for EDA.")

    explore_data(merged_df)
