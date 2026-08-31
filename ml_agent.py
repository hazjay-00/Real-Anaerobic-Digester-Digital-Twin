import kagglehub
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_and_export_brain():
    print("Downloading full-scale industrial data from Kaggle...")
    dataset_dir = kagglehub.dataset_download("d4rklucif3r/full-scale-waste-water-treatment-plant-data")

    # Auto-detect the exact CSV file in the folder
    csv_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("Could not find any CSV data file inside the downloaded Kaggle directory.")

    csv_path = os.path.join(dataset_dir, csv_files[0])
    print(f"Loading dataset file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean whitespaces and force lowercase
    df.columns = df.columns.str.strip().str.lower()
    print("Available CSV columns:", list(df.columns))

    # Flexible column mapping to handle name variations in the dataset
    def find_column(candidates):
        for candidate in candidates:
            for col in df.columns:
                if candidate in col:
                    return col
        return None

    col_inflow = find_column(['avg_inflow', 'inflow', 'flow', 'q_in'])
    col_cod = find_column(['cod', 'cod_in', 's0'])
    col_ammonia = find_column(['am', 'nh4', 'ammonia', 'n_in'])
    col_temp = find_column(['t', 'temp', 'temperature'])
    col_target = find_column(['total_grid', 'grid', 'power', 'energy', 'kwh'])

    # Verify that all required columns were found
    matched_cols = {
        'avg_inflow': col_inflow,
        'cod': col_cod,
        'am': col_ammonia,
        't': col_temp,
        'total_grid': col_target
    }

    missing = [k for k, v in matched_cols.items() if v is None]
    if missing:
        raise KeyError(f"Could not map dataset columns for: {missing}. Available columns: {list(df.columns)}")

    # Rename matched columns to standard model names
    df = df.rename(columns={
        col_inflow: 'avg_inflow',
        col_cod: 'cod',
        col_ammonia: 'am',
        col_temp: 't',
        col_target: 'total_grid'
    })

    feature_columns = ['avg_inflow', 'cod', 'am', 't']
    target_column = 'total_grid'

    # Clean up data strings or missing slots
    for col in feature_columns + [target_column]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter out offline maintenance plant logs and drop missing records
    cleaned_df = df[(df['avg_inflow'] > 5000) & (df['total_grid'] > 0)][feature_columns + [target_column]].dropna()
    print(f"Cleaned SCADA dataset contains {len(cleaned_df):,} active plant operational logs.")

    # Train-Test Split (80/20 standard protocol)
    X = cleaned_df[feature_columns]
    y = cleaned_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Regressor on active operational logs...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Performance Verification
    y_pred = model.predict(X_test)
    real_r2 = r2_score(y_test, y_pred) * 100
    print(f"\nModel Verification Complete!")
    print(f"Real-World Cross-Validated R² Score: {real_r2:.2f}%")

    # Export artifacts cleanly
    artifacts = {
        "model": model,
        "r2_score": real_r2
    }

    with open("wastewater_brain_cod.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("Real industrial brain exported successfully to 'wastewater_brain_cod.pkl'!")

if __name__ == "__main__":
    train_and_export_brain()
