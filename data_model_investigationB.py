import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from math import sqrt


def run_investigationB(df):
    """
    Investigation B: Compare bat behaviour between Winter and Spring
    using separate Linear Regression models.
    """

    print("=== Investigation B: Linear Regression (Seasonal Comparison) Started ===")

    # 1️⃣ Ensure season labels exist
    if 'season_label' not in df.columns:
        df['season_label'] = df['season'].map({0: 'winter', 1: 'spring'})

    # 2️⃣ Define response and predictor variables
    response_var = 'bat_landing_to_food'
    predictors = [
        'seconds_after_rat_arrival',
        'rat_present_at_landing',
        'rat_minutes',
        'rat_arrival_number',
        'food_availability',
        'hours_after_sunset',
        'risk',
        'reward',
        'habit'
    ]

    predictors = [col for col in predictors if col in df.columns]

    # 3️⃣ Split by season
    winter_df = df[df['season_label'] == 'winter'].copy()
    spring_df = df[df['season_label'] == 'spring'].copy()

    print(f"Winter observations: {len(winter_df)}")
    print(f"Spring observations: {len(spring_df)}")

    results = {}

    for season_name, season_df in [('winter', winter_df), ('spring', spring_df)]:
        print(f"\n--- Training model for {season_name.upper()} ---")

        season_df = season_df.dropna(subset=[response_var])
        X = season_df[predictors]
        y = season_df[response_var]

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit model
        lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
        lr_pipeline.fit(X_train, y_train)

        y_pred = lr_pipeline.predict(X_test)

        # Performance metrics
        r2 = r2_score(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results[season_name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}

        print(f"{season_name.capitalize()} R²: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

        # --- OLS model for coefficient insight ---
        X_design = pd.get_dummies(X, drop_first=True)

        # Ensure all data is numeric
        X_design = X_design.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # Drop any rows with NaN or non-numeric entries
        combined = pd.concat([X_design, y], axis=1).dropna()
        X_design = combined.drop(columns=[response_var])
        y = combined[response_var]

        # Add constant for intercept
        X_design = sm.add_constant(X_design)

        # Fit OLS model
        ols_model = sm.OLS(y.values.astype(float), X_design.values.astype(float)).fit()
        print(f"\n{season_name.capitalize()} OLS Summary:")
        print(ols_model.summary().tables[1])  # only coefficient table

        # Save coefficients for comparison
        coef_df = pd.DataFrame({
            'Variable': X_design.columns,
            f'{season_name}_Coef': ols_model.params
        }).reset_index(drop=True)
        results[season_name]['Coefficients'] = coef_df

    # 4️⃣ Compare coefficients between Winter and Spring
    print("\n=== Coefficient Comparison Between Seasons ===")
    winter_coef = results['winter']['Coefficients']
    spring_coef = results['spring']['Coefficients']
    coef_compare = pd.merge(winter_coef, spring_coef, on='Variable', how='inner')
    coef_compare['Difference'] = coef_compare['spring_Coef'] - coef_compare['winter_Coef']

    print(coef_compare.head(10))

    # Save results
    save_dir = os.path.join(os.getcwd(), "cleaned_data")
    coef_compare.to_csv(os.path.join(save_dir, "seasonal_coefficients_comparison.csv"), index=False)
    print(f"\nCoefficient comparison saved to: {save_dir}/seasonal_coefficients_comparison.csv")

    # 5️⃣ Visualize differences
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_compare.sort_values('Difference', ascending=False)[:15],
                x='Difference', y='Variable', palette='coolwarm')
    plt.title("Top 15 Coefficient Differences (Spring - Winter)")
    plt.xlabel("Difference in Coefficient Value")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()

    # === 6️⃣ Model Performance Summary ===
    summary_rows = []
    for season_name, metrics in results.items():
        summary_rows.append({
            'Model': 'Linear Regression (B)',
            'Season': season_name.capitalize(),
            'R²': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })

    performance_B = pd.DataFrame(summary_rows)
    print("\n=== Investigation B Seasonal Performance ===")
    print(performance_B)

    # Optional: save
    save_dir = os.path.join(os.getcwd(), "cleaned_data")
    performance_B.to_csv(os.path.join(save_dir, "performance_investigationB.csv"), index=False)

    print("Investigation B: Seasonal Linear Regression Comparison Complete")
    return results


if __name__ == "__main__":
    clean_dir = os.path.join(os.getcwd(), "cleaned_data")
    merged_path = os.path.join(clean_dir, "merged_dataset.csv")

    df = pd.read_csv(merged_path)
    print("Merged dataset loaded successfully for Investigation B.")

    run_investigationB(df)
