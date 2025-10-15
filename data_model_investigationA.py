import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_linear_regression(df):
    """
    Investigation A: Linear Regression model to test whether bats perceive rats as predators.
    Uses 'bat_landing_to_food' as the response variable (y) and rat-related variables as predictors (X).
    """

    print("=== Investigation A: Linear Regression Modelling Started ===")

    # ----- 1️⃣ Select Response and Predictor Variables -----
    response_var = 'bat_landing_to_food'

    candidate_predictors = [
        'seconds_after_rat_arrival',
        'rat_present_at_landing',
        'rat_minutes',
        'rat_arrival_number',
        'food_availability',
        'hours_after_sunset',
        'risk',
        'reward',
        'habit',
        'month'
    ]

    # Keep only columns that exist
    predictors = [col for col in candidate_predictors if col in df.columns]
    print(f"Predictors used: {predictors}")

    # Drop rows with missing response
    df = df.dropna(subset=[response_var])
    X = df[predictors]
    y = df[response_var]

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # ----- 2️⃣ Preprocessing -----
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

    # ----- 3️⃣ Split Data -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----- 4️⃣ Baseline Linear Regression -----
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)

    print("\n=== Linear Regression Performance ===")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    # ----- 5️⃣ Feature Importance (Statsmodels) -----
    print("\n=== OLS Detailed Summary (Statsmodels) ===")

    # Convert categorical to dummies and align with numeric predictors
    X_design = pd.get_dummies(X, drop_first=True)

    # Add constant manually for intercept
    X_design = sm.add_constant(X_design)

    # Ensure all are numeric and aligned with response
    X_design = X_design.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop any rows with NaNs in either X or y
    valid_idx = X_design.dropna().index.intersection(y.dropna().index)
    X_design = X_design.loc[valid_idx]
    y = y.loc[valid_idx]

    # Convert to numpy arrays (avoids dtype=object issue entirely)
    X_array = np.asarray(X_design, dtype=float)
    y_array = np.asarray(y, dtype=float)

    # Fit OLS model safely
    ols_model = sm.OLS(y_array, X_array).fit()
    print(ols_model.summary())

    # ----- 6️⃣ Multicollinearity Check (VIF) -----
    print("\n=== Variance Inflation Factor (VIF) ===")

    # Keep only numeric and finite columns for VIF
    X_vif = X_design.select_dtypes(include=[np.number]).copy()
    X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    # Drop constant if present (since it has infinite VIF)
    if 'const' in X_vif.columns:
        X_vif = X_vif.drop(columns=['const'])

    # Calculate VIF values
    vif_data = []
    for i in range(X_vif.shape[1]):
        try:
            vif_value = variance_inflation_factor(X_vif.values, i)
            vif_data.append((X_vif.columns[i], vif_value))
        except Exception:
            vif_data.append((X_vif.columns[i], np.nan))

    vif_df = pd.DataFrame(vif_data, columns=["Feature", "VIF"])
    vif_df = vif_df.sort_values("VIF", ascending=False)

    print(vif_df.head(10))

    # ----- 7️⃣ Regularized Models -----
    ridge_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
    ridge_pipe.fit(X_train, y_train)
    ridge_pred = ridge_pipe.predict(X_test)
    print(f"\nRidge Best Alpha: {ridge_pipe.named_steps['ridge'].alpha_:.4f}")
    print(f"Ridge R²: {r2_score(y_test, ridge_pred):.3f}")

    lasso_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('lasso', LassoCV(alphas=None, cv=5, max_iter=5000, random_state=42))])
    lasso_pipe.fit(X_train, y_train)
    lasso_pred = lasso_pipe.predict(X_test)
    print(f"Lasso Best Alpha: {lasso_pipe.named_steps['lasso'].alpha_:.4f}")
    print(f"Lasso R²: {r2_score(y_test, lasso_pred):.3f}")

    # ----- 8️⃣ Residual Plot -----
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()

    # ----- 9️⃣ QQ Plot -----
    sm.qqplot(residuals, line='45', fit=True)
    plt.title("QQ Plot of Residuals")
    plt.show()

    print("Investigation A: Linear Regression Modelling Complete")

    # === 10️⃣ Model Performance Summary Table ===
    results_A = {
        'Model': ['Linear Regression (A)', 'Ridge (A)', 'Lasso (A)'],
        'Season': ['Combined', 'Combined', 'Combined'],
        'R²': [
            r2_score(y_test, y_pred),
            r2_score(y_test, ridge_pred),
            r2_score(y_test, lasso_pred)
        ],
        'RMSE': [
            sqrt(mean_squared_error(y_test, y_pred)),
            sqrt(mean_squared_error(y_test, ridge_pred)),
            sqrt(mean_squared_error(y_test, lasso_pred))
        ],
        'MAE': [
            mean_absolute_error(y_test, y_pred),
            mean_absolute_error(y_test, ridge_pred),
            mean_absolute_error(y_test, lasso_pred)
        ]
    }

    performance_A = pd.DataFrame(results_A)
    print("\n=== Investigation A Model Performance ===")
    print(performance_A)

    # Optional: save to CSV
    save_dir = os.path.join(os.getcwd(), "cleaned_data")
    performance_A.to_csv(os.path.join(save_dir, "performance_investigationA.csv"), index=False)

    # Optional: return model objects for further analysis
    return lr_pipeline, ols_model, ridge_pipe, lasso_pipe


if __name__ == "__main__":
    # Load merged dataset
    clean_dir = os.path.join(os.getcwd(), "cleaned_data")
    merged_path = os.path.join(clean_dir, "merged_dataset.csv")

    df = pd.read_csv(merged_path)
    print("Merged dataset loaded successfully for Investigation A.")

    run_linear_regression(df)
